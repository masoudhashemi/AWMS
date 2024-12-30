from abc import ABC, abstractmethod
from typing import Dict, List

from awms.bases.agent import LLMAgent
from awms.logging import logger


class Tool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    def generate_prompt(
        self, problem: str, prior_results: List[Dict[str, str]], similar_examples: List[Dict[str, str]] = None
    ) -> str:
        pass

    @abstractmethod
    def process_llm_output(self, llm_output: str) -> str:
        pass

    def solve(
        self,
        problem: str,
        llm_agent: LLMAgent,
        prior_results: List[Dict[str, str]],
        similar_examples: List[Dict[str, str]] = None,
    ):
        prompt = self.generate_prompt(problem, prior_results, similar_examples)
        logger.info(f"[Tool] Prompt: {prompt}")
        llm_output = llm_agent.llm_call(prompt)

        result = self.process_llm_output(llm_output)
        logger.info(f"[Tool] LLM output: {llm_output} & Result: {result}")
        return result
