import json
import logging
import time as time_module
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

from litellm import completion

from awms.config import LLM_CONFIG, MAX_TOKENS, TEMPERATURE
from awms.logging import logger
from awms.utils import Message, ProblemState, SubproblemResult


class MessageBus:
    """A message bus for passing messages between agents."""

    def __init__(self):
        self.agents: Dict[str, "Agent"] = {}
        self.queue: deque = deque()
        self.message_log: List[Dict[str, Any]] = []

    def register_agent(self, agent: "Agent"):
        self.agents[agent.agent_id] = agent
        logger.info(
            f"[MessageBus] Agent '{agent.agent_id}' registered.",
            extra={
                "event": "agent_registered",
                "problem_id": "system",
                "parent_id": None,
                "depth": 0,
                "sender_id": "MessageBus",
                "receiver_id": agent.agent_id,
                "content": {},
            },
        )

    def send_message(self, message: Message):
        self.queue.append(message)
        logger.info(
            f"[MessageBus] Message sent from '{message.sender_id}' to '{message.receiver_id}'.",
            extra={
                "event": "sent",
                "problem_id": message.content.get("problem_id", "unknown"),
                "parent_id": message.content.get("parent_id"),
                "depth": message.depth,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "content": message.content,
            },
        )

        self.log_message(message, event="sent")

    def dispatch_messages(self):
        while self.queue:
            message = self.queue.popleft()
            receiver = self.agents.get(message.receiver_id)
            if receiver:
                try:
                    receiver.process_message(message)
                    self.log_message(message, event="processed", processor_id=receiver.agent_id)
                except Exception as e:
                    logger.error(f"Error processing message {message} by {receiver.agent_id}: {e}")
            else:
                logger.warning(f"[MessageBus] No agent found with ID '{message.receiver_id}'.")
                logger.info(
                    f"[MessageBus] Message could not be delivered.",
                    extra={
                        "event": "undelivered",
                        "problem_id": message.content.get("problem_id", "unknown"),
                        "parent_id": message.content.get("parent_id"),
                        "depth": message.depth,
                        "sender_id": message.sender_id,
                        "receiver_id": message.receiver_id,
                        "content": message.content,
                    },
                )

    def log_message(self, message: Message, event: str, processor_id: Optional[str] = None):
        message_dict = {
            "event": event,
            "timestamp": time_module.time(),
            "sender_id": message.sender_id,
            "receiver_id": message.receiver_id,
            "content": self.serialize_content(message.content),
            "depth": message.depth,
            "hierarchy": message.hierarchy,
            "problem_id": message.content.get("problem_id", "unknown"),
        }
        if processor_id:
            message_dict["processor_id"] = processor_id
        if "tool_output" in message.content:
            message_dict["tool_output"] = message.content["tool_output"]
        if "pal_code" in message.content:
            message_dict["pal_code"] = message.content["pal_code"]
        self.message_log.append(message_dict)

    def serialize_content(self, content):
        try:
            json.dumps(content)
            return content
        except TypeError:
            if isinstance(content, dict):
                return {k: self.serialize_content(v) for k, v in content.items()}
            elif isinstance(content, list):
                return [self.serialize_content(item) for item in content]
            elif isinstance(content, set):
                return list(content)
            elif isinstance(content, (int, float, str, bool)) or content is None:
                return content
            else:
                return str(content)

    def build_problem_tree(self):
        problem_tree = {}

        for message in self.message_log:
            hierarchy = message.get("hierarchy", [])
            problem_id = message.get("problem_id", "unknown")
            depth = message.get("depth", 0)

            # Starting from the root
            current_level = problem_tree

            # Traverse the hierarchy
            for parent_problem_id in hierarchy:
                if parent_problem_id not in current_level:
                    # Create a new node
                    current_level[parent_problem_id] = {
                        "problem_id": parent_problem_id,
                        "events": [],
                        "subproblems": {},
                    }
                current_level = current_level[parent_problem_id]["subproblems"]

            # Now at the level for the current problem
            if problem_id not in current_level:
                current_level[problem_id] = {"problem_id": problem_id, "events": [], "subproblems": {}}
            # Add the message to the events of the current problem
            current_level[problem_id]["events"].append(message)

        return problem_tree

    def save_messages(self, filepath: str):
        def serialize(obj):
            """Helper function to convert non-serializable objects into JSON-compatible types."""
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, SubproblemResult):  # Custom class example
                return obj.__dict__  # Serialize its attributes as a dictionary
            elif isinstance(obj, ProblemState):
                return {
                    "problem_id": obj.problem_id,
                    "problem": obj.problem,
                    "depth": obj.depth,
                    "hierarchy": obj.hierarchy,
                    "prior_results": serialize(obj.prior_results),  # Recursively serialize
                    "attempted_problems": serialize(obj.attempted_problems),
                    "subproblem_solutions": serialize(obj.subproblem_solutions),
                }
            else:
                return str(obj)  # Convert other types to string

        problem_tree = self.build_problem_tree()
        problem_tree_serialized = serialize(problem_tree)
        with open(filepath, "w") as f:
            json.dump(problem_tree_serialized, f, indent=2)

    def load_messages(self, filepath: str):
        with open(filepath, "r") as f:
            self.message_log = json.load(f)


class Agent(ABC):
    """A base agent class."""

    def __init__(self, agent_id: str, message_bus: "MessageBus"):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.message_bus.register_agent(self)

    @abstractmethod
    def process_message(self, message: Message):
        pass

    def send_message(self, receiver_id: str, content: dict, depth: int = 0, hierarchy: list = None):
        if hierarchy is None:
            hierarchy = []
        message = Message(self.agent_id, receiver_id, content, depth=depth, hierarchy=hierarchy)
        self.message_bus.send_message(message)


class LLMAgent(Agent):
    """A base agent class that uses LLMs."""

    def __init__(self, agent_id: str, message_bus: "MessageBus"):
        super().__init__(agent_id, message_bus)
        self.prior_results = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process_message(self, message: Message):
        pass

    def llm_call(self, prompt: str):
        try:
            response = completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                **LLM_CONFIG,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM API call: {e}")
            return "An error occurred while calling the LLM API."
