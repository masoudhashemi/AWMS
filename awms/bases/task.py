from abc import ABC, abstractmethod
from typing import List, Tuple

from awms.bases.agent import LLMAgent


class Task(ABC):
    """An abstract base class for tasks."""

    @abstractmethod
    def should_break_down_problem(self, problem: str, llm_agent: "LLMAgent") -> bool:
        """Determine if the problem should be broken down into subproblems."""
        pass

    @abstractmethod
    def break_down_problem(self, problem: str, llm_agent: "LLMAgent") -> List[str]:
        pass

    @abstractmethod
    def classify_problem(self, problem: str, llm_agent: "LLMAgent") -> str:
        pass

    @abstractmethod
    def evaluate_solution(self, problem: str, solution: str, llm_agent: "LLMAgent") -> Tuple[str, bool]:
        pass

    @abstractmethod
    def merge_solutions(self, subproblem_solutions: List[str], problem: str, llm_agent: "LLMAgent") -> str:
        pass
