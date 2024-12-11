from typing import Any, Dict, List

from awms.bases.agent import LLMAgent
from awms.logging import logger
from awms.utils import Message


class ExplanationAgent(LLMAgent):
    def process_message(self, message: Message):
        content = message.content
        if content.get("request") == "generate_explanation":
            self.handle_generate_explanation(message)
        else:
            logger.warning(f"[ExplanationAgent] Unrecognized message content: {content}")

    def handle_generate_explanation(self, message: Message):
        content = message.content
        problem = content["problem"]
        solution = content["solution"]
        subproblem_results = content.get("subproblem_results", [])
        problem_id = content["problem_id"]
        original_sender = content.get("original_sender", "Main")

        # Generate the explanation using the LLM
        explanation = self.generate_explanation(problem, solution, subproblem_results)

        # Send the explanation back to the ExecutionAgent
        response_content = {"problem_id": problem_id, "explanation": explanation}
        self.send_message("ExecutionAgent", response_content)

    def generate_explanation(self, problem: str, solution: str, subproblem_results: List[Dict[str, Any]]) -> str:
        # Construct the prompt for the LLM
        prompt = f"Problem:\n{problem}\n\n"

        if subproblem_results:
            prompt += "The problem was broken down into the following subproblems and solutions:\n"
            for result in subproblem_results:
                question = result["question"]
                answer = result["answer"]
                success = result["success"]
                if success:
                    prompt += f"Subproblem: {question}\nSolution: {answer}\n\n"
                else:
                    prompt += f"Subproblem: {question}\nSolution: [Failed to Solve]\nFeedback: {answer}\n\n"

        prompt += (
            f"The final solution is:\n{solution}\n\nProvide a detailed explanation of how the problem was solved, incorporating the subproblems and their solutions. "
            "Keep the description in natural language without code. If necessary add a highlevel description of what the code does. "
            "You MUST write the final answer in LaTeX format, enclosed in \\boxed{{}} at the very end. Only the answer must be in the boxed.\n"
        )

        explanation = self.llm_call(prompt)
        return explanation.strip()
