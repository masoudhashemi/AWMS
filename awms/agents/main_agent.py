from awms.bases.agent import Agent, MessageBus
from awms.logging import logger
from awms.utils import Message


class MainAgent(Agent):
    """Agent to receive final solutions from the ExecutionAgent."""

    def __init__(self, agent_id: str, message_bus: "MessageBus"):
        super().__init__(agent_id, message_bus)
        self.final_solutions = {}

    def process_message(self, message: Message):
        problem_id = message.content.get("problem_id")
        final_solution = message.content.get("final_solution")
        explanation = message.content.get("explanation")
        # Store the result
        self.final_solutions[problem_id] = (final_solution, explanation)
        logger.info(f"[MainAgent] Received final solution for problem {problem_id}.")
