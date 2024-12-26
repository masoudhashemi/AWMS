from typing import Dict, List

from awms.bases.tool import Tool


class CoTTool(Tool):
    """Tool for Chain-of-Thought reasoning."""

    def generate_prompt(self, problem: str, prior_results: List[Dict[str, str]]) -> str:
        prior_text = ""
        for item in prior_results:
            if item.success:
                prior_text += f"Question: {item.question}\nAnswer: {item.answer}\n\n"
            else:
                prior_text += f"Question: {item.question}\nAnswer: [Failed to Solve]\nFeedback: {item.answer}\n\n"
        prompt = (
            f"{prior_text}"
            f"Solve the following problem step-by-step using natural language reasoning.\n"
            f"Problem:\n{problem}"
        )
        return prompt

    def process_llm_output(self, llm_output: str) -> str:
        return {"execution_result": llm_output.strip(), "cot_reasoning": llm_output.strip()}
