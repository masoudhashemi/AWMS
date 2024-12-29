import json
import re
from typing import List, Tuple

from awms.bases.agent import LLMAgent
from awms.bases.task import Task
from awms.logging import logger
from awms.utils import SubproblemResult


class MathProblemTask(Task):
    """A task class for solving math problems."""

    def should_break_down_problem(self, problem: str, llm_agent: "LLMAgent") -> bool:
        # Use a more robust approach to decide whether to break down the problem
        prompt = (
            "Decide whether the following math problem requires breaking down into subproblems to solve effectively. Be conservative in breaking down problems. "
            "Consider the complexity of the problem and whether it can be solved directly. "
            "Only break down the problem if it is complex and would benefit from it. "
            "Answer 'Yes' if it is complex and would benefit from breaking down, or 'No' if it is simple enough or there are known methods to solve directly. "
            "If the problem can be answered without breaking down with known code or algorithms, answer with 'No'.\n"
            f"Provide ONLY 'Yes' or 'No' as the answer.\nProblem:\n{problem}"
        )
        response = llm_agent.llm_call(prompt)
        return "yes" in response.lower()

    def break_down_problem(self, problem: str, llm_agent: "LLMAgent") -> List[str]:
        """Breaks down a complex problem into subproblems using an LLM."""
        prompt = (
            "Break down the following problem into minimum number of sequential, detailed subproblems. "
            "Each subproblem MUST be solvable independently and must contribute to solving the original problem. "
            "You must include all the necessary information needed to solve each subproblem from the original problem. "
            "If subproblems are dependent on each other, provide them in the correct order. "
            "If two subproblems are independent, provide them in the order they appear in the original problem. "
            "Do not ask for inputs (i.e., refrain from using Python CLI input). "
            "Note: If the problem can be solved without breaking down, provide a single subproblem with the original problem text. "
            "Provide the subproblems as a JSON array of strings. Output ONLY the JSON array and nothing else.\n"
            f"Problem:\n{problem}\nSubproblems:"
        )
        response = llm_agent.llm_call(prompt)
        logger.info(f"[MathProblemTask] Subproblems: {response}")
        
        try:
            # First try direct JSON parsing
            if isinstance(response, str):
                if "```json" in response:
                    # Extract JSON from markdown
                    pattern = r"```json\n(.*?)```"
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        json_str = match.group(1).strip()
                        subproblems = json.loads(json_str)
                    else:
                        subproblems = [response]
                else:
                    # Try parsing response directly as JSON
                    subproblems = json.loads(response)
            else:
                # Response is already parsed
                subproblems = response if isinstance(response, list) else [str(response)]
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response}")
            subproblems = [response]
        
        subproblems = self.filter_subproblems(problem, subproblems, llm_agent)
        logger.info(f"[MathProblemTask] Subproblems parsed: {subproblems}")
        return subproblems

    def filter_subproblems(self, problem: str, subproblems: List[str], llm_agent: "LLMAgent") -> List[str]:
        unique = list(dict.fromkeys(subproblems))
        relevant = []
        for sp in unique:
            if self.is_subproblem_useful(problem, sp, llm_agent):
                relevant.append(sp)
        return relevant

    def is_subproblem_useful(self, problem: str, subproblem: str, llm_agent: "LLMAgent") -> bool:
        prompt = f"Check if this subproblem is distinct and useful in solving the original problem. Answer 'Yes' or 'No'.\nOriginal Problem: {problem}\nSubproblem:\n{subproblem}"
        response = llm_agent.llm_call(prompt)
        return "yes" in response.lower()

    def classify_problem(self, problem: str, llm_agent: "LLMAgent") -> str:
        """Classifies the problem into a problem type using an LLM."""
        prompt = (
            f"Classify the following math problem into one of the following types: 'Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Prealgebra', 'Precalculus'. "
            f"Provide ONLY the type as the answer.\nProblem:\n{problem}"
        )
        classification = llm_agent.llm_call(prompt)
        classification = classification.split(".")[0].strip()
        logger.info(f"[MathProblemTask] Problem classified as: {classification}")
        return classification

    def evaluate_solution(self, problem: str, solution: str, llm_agent: "LLMAgent") -> Tuple[str, bool]:
        """Evaluates the solution using an LLM."""
        prompt = (
            f"Evaluate the following solution to the problem and determine if it is correct. Only check the correctness of the final result. Ignore the units and formatting. "
            f"Answer 'Correct' or 'Incorrect' and provide a brief justification.\nProblem:\n{problem}\nSolution:\n{solution}"
        )
        evaluation = llm_agent.llm_call(prompt)
        if "incorrect" in evaluation.lower():
            return evaluation, False
        else:
            return evaluation, True

    def merge_solutions(
        self, subproblem_results: List[SubproblemResult], original_problem: str, agent: LLMAgent
    ) -> str:
        # Prepare a summary of subproblem results
        summary = ""
        for result in subproblem_results:
            if result.success:
                summary += f"Solution to '{result.question}':\n{result.answer}\n\n"
            else:
                summary += f"Failed to solve '{result.question}'. Feedback:\n{result.answer}\n\n"

        prompt = (
            f"Using the solutions to the subproblems below, provide a final solution to the original problem.\n"
            f"Original Problem:\n{original_problem}\n"
            f"Subproblem Solutions:\n{summary}\n"
            f"Final Solution:"
        )

        final_solution = agent.llm_call(prompt)
        return final_solution.strip()

    def get_all_tools_for_problem_type(self, problem_type: str) -> List[str]:
        return ["CoT", "PAL"]
