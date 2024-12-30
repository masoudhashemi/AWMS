from typing import Dict, List

from awms.bases.tool import Tool
from awms.utils import execute_python_code


class PALTool(Tool):
    """Tool for Program-aided LLM (PAL) that generates Python code and executes it."""

    def generate_prompt(self, problem: str, prior_results: List[Dict[str, str]]) -> str:
        prior_text = ""
        for item in prior_results:
            if item.success:
                prior_text += f"Question: {item.question}\nAnswer: {item.answer}\n\n"
            else:
                prior_text += f"Question: {item.question}\nAnswer: [Failed to Solve]\nFeedback: {item.answer}\n\n"

        examples_text = ""
        if similar_examples:
            examples_text = "\nSimilar examples:\n"
            for example in similar_examples:
                examples_text += f"Example Question: {example['question']}\nPython Solution:\n{example['answer']}\n\n"

        prompt = (
            f"{examples_text}"
            "Using the information from previous steps (if there are any),"
            f"{prior_text}"
            f"Write Python code to solve the following problem."
            f"Make sure to import all necessary functions and libraries. Only use well-known packages.\n"
            f"Problem:\n{problem}\n"
            f"Provide the code within ```python``` code blocks.\n"
            "At the end of the code, you MUST print the final result (e.g., end with print(result))."
        )
        return prompt

    def process_llm_output(self, llm_output: str) -> str:
        execution_result = execute_python_code(llm_output) or llm_output
        return {"execution_result": execution_result, "pal_code": llm_output}
