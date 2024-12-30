from typing import Dict, List

from awms.bases.tool import Tool


class CoTTool(Tool):
    """Tool for Chain-of-Thought reasoning."""

    def generate_prompt(
        self, problem: str, prior_results: List[Dict[str, str]], similar_examples: List[Dict[str, str]] = None
    ) -> str:
        # Format prior results
        prior_text = ""
        for item in prior_results:
            if item.get("success"):
                prior_text += f"Question: {item['question']}\nAnswer: {item['answer']}\n\n"
            else:
                prior_text += f"Question: {item['question']}\nAnswer: [Failed to Solve]\nFeedback: {item['answer']}\n\n"

        # Format similar examples if provided
        examples_text = ""
        if similar_examples:
            examples_text = "\nSimilar examples:\n"
            for example in similar_examples:
                examples_text += f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"

        prompt = (
            f"{prior_text}"
            f"{examples_text}"
            f"Solve the following problem step-by-step using natural language reasoning.\n"
            f"Problem:\n{problem}"
        )
        return prompt

    def process_llm_output(self, llm_output: str) -> str:
        return {"execution_result": llm_output.strip(), "cot_reasoning": llm_output.strip()}
