import json
import logging
import os
import time as time_module
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from awms.logging import logger
from awms.utils import Message, ProblemState, SubproblemResult

# Load environment variables for LLM
load_dotenv()
job_id = os.getenv("JOB_ID")

if job_id:
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    model_type = "vllm"
    base_url = "https://" + job_id + "-8000.job.console.elementai.com/v1"
    llm_client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
else:
    model_type = "gpt"
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("API_VERSION")
    model_name = os.getenv("MODEL_NAME")
    llm_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


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
        self.llm = llm_client

    @abstractmethod
    def process_message(self, message: Message):
        pass

    def llm_call(self, prompt: str):
        try:
            if model_type == "vllm":
                prediction = self.llm.completions.create(
                    model=model_name,
                    prompt=f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
                    max_tokens=1500,
                    temperature=0.1,
                )
                response = [p.model_dump()["text"].strip() for p in prediction.choices]
                llm_output = response[0]
            elif model_type == "gpt":
                prediction = self.llm.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.1,
                )
                llm_output = prediction.choices[0].message.content
            return llm_output
        except Exception as e:
            logger.error(f"Error in LLM API call: {e}")
            return "An error occurred while calling the LLM API."
