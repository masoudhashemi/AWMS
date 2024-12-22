from awms.bases import LLMAgent, MessageBus, Task
from awms.logging import logger
from awms.utils import Message


class FeedbackAgent(LLMAgent):
    """Agent to evaluate solutions and provide feedback."""

    def __init__(self, agent_id: str, message_bus: "MessageBus", task: "Task"):
        super().__init__(agent_id, message_bus)
        self.task = task

    def process_message(self, message: Message):
        content = message.content
        if content.get("request") == "evaluate_solution":
            problem = content["problem"]
            solution = content["solution"]
            sender_id = message.sender_id

            # Evaluate the solution using the task's evaluation method
            feedback, success = self.task.evaluate_solution(problem, solution, self)
            retry = False
            if not success:
                # Use LLM to decide if a retry is possible or necessary
                retry_prompt = (
                    f"The following solution was incorrect. Can it be retried with improvements using the same method, or should another method be used? "
                    f"Answer 'Retry' or 'Switch' and provide a brief explanation.\n"
                    f"Problem:\n{problem}\nSolution:\n{solution}\nFeedback:\n{feedback}"
                )
                decision = self.llm_call(retry_prompt)
                retry = "retry" in decision.lower()

            # Send feedback back to the sender
            feedback_content = {
                "feedback": feedback,
                "success": success,
                "retry": retry,
                "problem": problem,
                "solution": solution,
                "problem_id": content["problem_id"],
            }
            self.send_message(sender_id, feedback_content, depth=message.depth, hierarchy=message.hierarchy)
            logger.info(
                f"[FeedbackAgent] Sent feedback to '{sender_id}': {feedback}, Success: {success}, Retry: {retry}"
            )
