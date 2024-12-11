from collections import defaultdict
from typing import List, Optional

import numpy as np

from awms.bases.agent import LLMAgent, MessageBus
from awms.logging import logger
from awms.tools import TOOL_REGISTRY
from awms.utils import Message


class ToolSelectionAgent(LLMAgent):
    """Agent to select the appropriate tool using a Contextual Multi-Armed Bandit approach with Thompson Sampling."""

    def __init__(self, agent_id: str, message_bus: "MessageBus", task: "Task"):
        super().__init__(agent_id, message_bus)
        self.task = task
        self.tool_options = list(TOOL_REGISTRY.keys())  # Get tool names from the registry
        # Tracking performance per problem type
        self.tool_successes = defaultdict(lambda: defaultdict(int))  # problem_type -> tool -> successes
        self.tool_failures = defaultdict(lambda: defaultdict(int))  # problem_type -> tool -> failures
        self.tool_retries = defaultdict(lambda: defaultdict(int))  # problem_type -> tool -> retries
        self.decay_factor = 0.9  # Decay factor for past successes and failures

    def process_message(self, message: Message):
        content = message.content
        if content.get("request") == "select_tool":
            problem_type = content["problem_type"]
            attempted_tools = content["attempted_tools"]
            selected_tool = self.select_tool(problem_type, attempted_tools)
            response = {"selected_tool": selected_tool, "problem_id": content["problem_id"]}
            self.send_message(message.sender_id, response, depth=message.depth, hierarchy=message.hierarchy)
        elif content.get("request") == "update_performance":
            # Ensure we have all required fields
            if all(k in content for k in ["problem_type", "tool", "success", "problem_id"]):
                problem_type = content["problem_type"]
                tool = content["tool"]
                success = content["success"]
                retry = content.get("retry", False)
                self.update_tool_performance(problem_type, tool, success, retry)
            else:
                logger.warning("[ToolSelectionAgent] Incomplete performance update message received")

    def select_tool(self, problem_type: str, attempted_tools: List[str]) -> Optional[str]:
        """Selects a tool based on Thompson Sampling."""
        available_tools = [tool for tool in self.tool_options if tool not in attempted_tools]
        if not available_tools:
            return None  # No tools left to try

        # Sample from Beta distribution for each tool
        sampled_values = {}
        for tool in available_tools:
            successes = self.tool_successes[problem_type][tool]
            failures = self.tool_failures[problem_type][tool]
            retries = self.tool_retries[problem_type][tool]
            alpha = successes + 1
            # beta = failures + retries + 1
            beta = failures + 1
            sampled_value = np.random.beta(alpha, beta)
            sampled_values[tool] = sampled_value

        # Select the tool with the highest sampled value
        selected_tool = max(sampled_values, key=sampled_values.get)
        logger.info(
            f"[ToolSelectionAgent] Selected tool for problem type '{problem_type}': {selected_tool} with Thompson Sampling"
            f" (sampled values: {sampled_values}), alpha: {alpha}, beta: {beta}"
        )
        return selected_tool

    def update_tool_performance(self, problem_type: str, tool: str, success: bool, retry: bool = False):
        """Update tool performance based on feedback."""
        # Apply decay
        for t in self.tool_options:
            self.tool_successes[problem_type][t] *= self.decay_factor
            self.tool_failures[problem_type][t] *= self.decay_factor
            self.tool_retries[problem_type][t] *= self.decay_factor

        # Update performance metrics
        if success:
            self.tool_successes[problem_type][tool] += 1
        else:
            self.tool_failures[problem_type][tool] += 1
            if retry:
                self.tool_retries[problem_type][tool] += 1

        logger.info(
            f"[ToolSelectionAgent] Updated tool performance for '{tool}' in problem type '{problem_type}' with success: {success}, retry: {retry}"
        )
