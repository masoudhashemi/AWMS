from typing import Dict, List, Optional, Union

from awms.bases.agent import LLMAgent
from awms.logging import logger
from awms.utils import Message


class MemoryAgent(LLMAgent):
    """
    Agent responsible for storing and retrieving information about problems and their solutions.

    This agent maintains a structured memory of:
    - Problem contexts
    - Subproblem breakdowns
    - Tool outputs and solutions
    - Relationships between problems
    """

    def __init__(self, agent_id: str, message_bus: "MessageBus"):
        super().__init__(agent_id, message_bus)
        self.memory: Dict[str, Dict] = {}

    def process_message(self, message: Message):
        """Process incoming memory-related requests."""
        content = message.content
        if "request" not in content:
            logger.warning(f"[MemoryAgent] Received message without request type: {content}")
            return

        request = content["request"]
        if request == "store":
            self.handle_store(content)
        elif request == "retrieve":
            self.handle_retrieve(message)
        elif request == "search":
            self.handle_search(message)
        else:
            logger.warning(f"[MemoryAgent] Unknown request type: {request}")

    def handle_store(self, content: Dict):
        """Store information in memory."""
        problem_id = content.get("problem_id")
        if not problem_id:
            logger.error("[MemoryAgent] Attempted to store without problem_id")
            return

        if problem_id not in self.memory:
            self.memory[problem_id] = {
                "problem": content.get("problem"),
                "problem_type": content.get("problem_type"),
                "subproblems": [],
                "tool_outputs": [],
                "solutions": [],
                "parent_id": content.get("parent_id"),
                "hierarchy": content.get("hierarchy", []),
                "context": content.get("context", {}),
            }

        memory_type = content.get("memory_type")
        data = content.get("data")

        if memory_type == "subproblem":
            self.memory[problem_id]["subproblems"].append(data)
        elif memory_type == "tool_output":
            self.memory[problem_id]["tool_outputs"].append(data)
        elif memory_type == "solution":
            self.memory[problem_id]["solutions"].append(data)
        elif memory_type == "context":
            self.memory[problem_id]["context"].update(data)

        logger.info(f"[MemoryAgent] Stored {memory_type} for problem {problem_id}")

    def handle_retrieve(self, message: Message):
        """Retrieve specific information from memory."""
        content = message.content
        problem_id = content.get("problem_id")
        memory_type = content.get("memory_type")

        if not problem_id or not memory_type:
            logger.error("[MemoryAgent] Missing problem_id or memory_type in retrieve request")
            return

        result = self._retrieve(problem_id, memory_type)

        response = {"problem_id": problem_id, "memory_type": memory_type, "data": result}
        self.send_message(message.sender_id, response, depth=message.depth, hierarchy=message.hierarchy)

    def handle_search(self, message: Message):
        """Search memory for relevant information using LLM."""
        content = message.content
        query = content.get("query")
        context = content.get("context", {})

        if not query:
            logger.error("[MemoryAgent] Missing query in search request")
            return

        results = self._search_memory(query, context)

        response = {"query": query, "results": results}
        self.send_message(message.sender_id, response, depth=message.depth, hierarchy=message.hierarchy)

    def _retrieve(self, problem_id: str, memory_type: str) -> Optional[Union[Dict, List]]:
        """Internal method to retrieve information from memory."""
        if problem_id not in self.memory:
            return None

        memory = self.memory[problem_id]
        if memory_type == "all":
            return memory
        return memory.get(memory_type, [])

    def _search_memory(self, query: str, context: Dict) -> List[Dict]:
        """
        Search memory for relevant information using LLM.
        Returns a list of relevant memory entries.
        """
        # Construct prompt for LLM to search memory
        memory_context = "\n".join(
            f"Problem {pid}: {mem.get('problem', '')}\n"
            f"Type: {mem.get('problem_type', '')}\n"
            f"Solutions: {mem.get('solutions', [])}\n"
            for pid, mem in self.memory.items()
        )

        prompt = (
            f"Given the following memory context and query, identify the most relevant information.\n"
            f"Memory Context:\n{memory_context}\n"
            f"Additional Context: {context}\n"
            f"Query: {query}\n"
            f"Return the problem IDs and relevant information that best match the query."
        )

        response = self.llm_call(prompt)

        # Parse LLM response to extract relevant problem IDs
        # This is a simple implementation - you might want to make it more sophisticated
        relevant_problems = []
        for pid in self.memory:
            if pid in response:
                relevant_problems.append({"problem_id": pid, "data": self.memory[pid]})

        return relevant_problems
