import uuid
from enum import Enum
from typing import Dict, List, Optional

from awms.bases.agent import LLMAgent, MessageBus
from awms.logging import logger
from awms.tools import get_tool
from awms.utils import Message, ProblemState, SubproblemResult


class MessageType(Enum):
    SOLVE_PROBLEM = "solve_problem"
    SELECTED_TOOL = "selected_tool"
    EXPLANATION = "explanation"
    SUBPROBLEM_SOLVED = "subproblem_solved"
    SUBPROBLEM_FAILED = "subproblem_failed"
    GENERATE_EXPLANATION = "generate_explanation"
    EVALUATE_SOLUTION = "evaluate_solution"
    SELECT_TOOL = "select_tool"
    UPDATE_PERFORMANCE = "update_performance"
    FEEDBACK = "feedback"
    STORE_MEMORY = "store_memory"
    RETRIEVE_MEMORY = "retrieve_memory"
    SEARCH_MEMORY = "search_memory"


class ExecutionAgent(LLMAgent):
    """
    Agent responsible for executing problem-solving tasks.

    This agent manages a hierarchy of problems and subproblems, decides whether to break them down,
    selects appropriate tools to solve them, and merges subproblem results. It ensures that parent-child
    relationships are well-defined, and all states are tracked in a central dictionary `self.state`.

    Key Responsibilities:
    - Handle requests to solve a root problem.
    - Possibly break down problems into subproblems and recursively solve them.
    - Choose tools to solve problems directly when no further breakdown is needed.
    - Merge subproblem solutions into a final solution.
    - Communicate results to other agents like FeedbackAgent and ExplanationAgent.
    - Maintain a coherent problem hierarchy, ensuring no unknown parent_id or problem_id references.
    """

    def __init__(self, agent_id: str, message_bus: "MessageBus", task, max_depth=2, max_tries=3):
        super().__init__(agent_id, message_bus)
        self.task = task
        self.state: Dict[str, ProblemState] = {}
        self.MAX_DEPTH = max_depth
        self.MAX_TRIES = max_tries
        self.metrics = {
            "problems_solved": 0,
            "problems_failed": 0,
            "average_solution_time": 0,
            "tool_success_rates": {},
        }

    def process_message(self, message: Message):
        """Processes incoming messages and routes them to appropriate handlers."""
        content = message.content
        problem_id = content.get("problem_id")
        logger.info(f"[ExecutionAgent] Processing message for problem_id: {problem_id}")

        try:
            if "request" in content:
                request = content["request"]
                if request == MessageType.SOLVE_PROBLEM.value:
                    self.handle_solve_problem(message)
                else:
                    logger.warning(f"[ExecutionAgent] Unknown request type: {request}")
            elif MessageType.SELECTED_TOOL.value in content:
                self.process_tool_selection(message)
            elif "feedback" in content:
                self.handle_feedback(message)
            elif "explanation" in content:
                self.handle_explanation(message)
            elif "subproblem_solved" in content:
                self.handle_subproblem_solved(content)
            elif "subproblem_failed" in content:
                self.handle_subproblem_failed(content)
            else:
                logger.warning(f"[ExecutionAgent] Unrecognized message content: {content}")
        except Exception as e:
            logger.error(f"[ExecutionAgent] Error processing message for problem_id: {problem_id}: {e}")

    def handle_solve_problem(self, message: Message):
        """
        Handles solving the main problem and setting up subproblem states.

        Steps:
        - Create a ProblemState for the current problem.
        - If depth exceeds MAX_DEPTH, directly select a tool and attempt a solution.
        - Otherwise, determine if the problem should be broken down.
          - If yes, create subproblem states and start solving them.
          - If no, select a tool and attempt a solution.
        """
        content = message.content
        problem = content["problem"]
        problem_id = content["problem_id"]
        depth = content.get("depth", 0)
        hierarchy = content.get("hierarchy", [])
        prior_results = content.get("prior_results", [])
        attempted_problems = set(content.get("attempted_problems", []))
        parent_id = content.get("parent_id")
        root_problem_id = content.get("root_problem_id", problem_id)

        # If this is a subproblem, inherit subproblem_solutions from the parent state if needed
        subproblem_solutions = content.get("subproblem_solutions", []) if parent_id else []

        # Initialize problem state
        state = ProblemState(
            problem_id=problem_id,
            problem=problem,
            depth=depth,
            hierarchy=hierarchy,
            prior_results=prior_results,
            attempted_problems=attempted_problems,
            parent_id=parent_id,
            root_problem_id=root_problem_id,
        )
        state.subproblem_solutions = subproblem_solutions
        state.retry_count = 0
        self.state[problem_id] = state

        # Store initial problem in memory
        self.store_in_memory(
            state, "context", {"initial_problem": problem, "attempted_problems": list(attempted_problems)}
        )

        # Check for maximum depth
        if depth >= self.MAX_DEPTH:
            logger.warning(f"[ExecutionAgent] Maximum depth reached for problem_id: {problem_id}")
            self.select_tool_and_solve(state)
            return

        # Decide whether to break down the problem
        if self.task.should_break_down_problem(problem, self):
            logger.info(f"[ExecutionAgent] Breaking down problem: {problem}")
            subproblems = self.task.break_down_problem(problem, self)

            # Filter out any empty or None subproblems
            subproblems = [sp for sp in subproblems if sp]

            if not subproblems:
                logger.warning(f"[ExecutionAgent] No valid subproblems generated for problem_id: {problem_id}")
                self.select_tool_and_solve(state)
                return

            state.subproblems = subproblems
            state.current_subproblem_index = 0
            self.solve_next_subproblem(state)
        else:
            self.select_tool_and_solve(state)

    def create_subproblem_state(self, parent_state: ProblemState, subproblem: str) -> ProblemState:
        """
        Creates a new ProblemState for a given subproblem, linking it to its parent.
        """
        if not subproblem:
            logger.error(
                f"[ExecutionAgent] Attempted to create subproblem state with empty subproblem for parent {parent_state.problem_id}"
            )
            return None

        subproblem_id = str(uuid.uuid4())
        logger.info(f"[ExecutionAgent] Creating new subproblem {subproblem_id} for parent {parent_state.problem_id}")

        new_hierarchy = parent_state.hierarchy + [parent_state.problem_id]
        subproblem_state = ProblemState(
            problem_id=subproblem_id,
            problem=subproblem,
            depth=parent_state.depth + 1,
            hierarchy=new_hierarchy,
            prior_results=parent_state.prior_results + parent_state.subproblem_solutions,
            attempted_problems=parent_state.attempted_problems.copy(),
            parent_id=parent_state.problem_id,
            root_problem_id=parent_state.root_problem_id,
        )
        self.state[subproblem_id] = subproblem_state
        parent_state.attempted_problems.add(subproblem)
        return subproblem_state

    def finish_subproblem(self, subproblem_id: str, success: bool, solution: str):
        """
        Finalizes a subproblem after it's solved or failed:
        - Retrieves its state and its parent's state.
        - Updates parent's subproblem_solutions.
        - Removes the subproblem from self.state.
        - Continues to solve next subproblem or merges results if all are done.
        """
        subproblem_state = self.state.get(subproblem_id)
        if not subproblem_state:
            logger.error(f"[ExecutionAgent] No state found for subproblem_id: {subproblem_id}.")
            return

        parent_id = subproblem_state.parent_id
        parent_state = self.state.get(parent_id)
        if not parent_state:
            logger.error(f"[ExecutionAgent] Parent state not found for problem_id: {parent_id}.")
            # Clean up subproblem state anyway to avoid dangling states
            del self.state[subproblem_id]
            return

        # Record the subproblem result in the parent
        subproblem_question = subproblem_state.problem
        parent_state.subproblem_solutions.append(
            SubproblemResult(question=subproblem_question, answer=solution, success=success)
        )
        # Remove the completed subproblem state
        del self.state[subproblem_id]

        # Move on to the next subproblem or merge results if done
        if parent_state.current_subproblem_index < len(parent_state.subproblems):
            self.solve_next_subproblem(parent_state)
        else:
            # All subproblems solved (or attempted), merge solutions
            merged_solution = self.task.merge_solutions(parent_state.subproblem_solutions, parent_state.problem, self)
            parent_state.final_solution = merged_solution
            self.handle_problem_solved(parent_state)

    def handle_subproblem_solved(self, content: dict):
        """
        Handles a message indicating a subproblem of the current problem has been solved.
        The content should contain 'problem_id' of the parent and 'solution'.
        """
        parent_problem_id = content["problem_id"]
        solution = content["solution"]
        success = content.get("success", True)
        parent_state = self.state.get(parent_problem_id)

        if not parent_state:
            logger.warning(f"[ExecutionAgent] Parent state not found for problem_id: {parent_problem_id}")
            return

        subproblem_id = content.get("subproblem_id")
        if not subproblem_id or subproblem_id not in self.state:
            # If for some reason we don't have subproblem_id, log error
            logger.error("[ExecutionAgent] subproblem_id missing or not in state during handle_subproblem_solved.")
            return

        self.finish_subproblem(subproblem_id, success, solution)

    def handle_subproblem_failed(self, content: dict):
        """
        Handles a message indicating a subproblem has failed.
        In this case, we treat the subproblem as solved but unsuccessful.
        """
        parent_problem_id = content["problem_id"]
        feedback = content.get("feedback", "Subproblem failed without specific feedback.")
        parent_state = self.state.get(parent_problem_id)

        if not parent_state:
            logger.warning(f"[ExecutionAgent] Parent state not found for problem_id: {parent_problem_id}")
            return

        # Similarly to handle_subproblem_solved, we assume 'subproblem_id' is included:
        subproblem_id = content.get("subproblem_id")
        if not subproblem_id or subproblem_id not in self.state:
            logger.error("[ExecutionAgent] subproblem_id missing or not in state during handle_subproblem_failed.")
            return

        self.finish_subproblem(subproblem_id, False, feedback)

    def solve_next_subproblem(self, state: ProblemState):
        """
        Solves the next subproblem in line or merges results if all are solved.
        """
        if state.current_subproblem_index >= len(state.subproblems):
            logger.info(f"[ExecutionAgent] All subproblems processed for problem_id: {state.problem_id}")
            merged_solution = self.task.merge_solutions(state.subproblem_solutions, state.problem, self)
            state.final_solution = merged_solution
            self.handle_problem_solved(state)
            return

        subproblem = state.subproblems[state.current_subproblem_index]
        if not subproblem:
            logger.error(
                f"[ExecutionAgent] Empty subproblem found at index {state.current_subproblem_index} for problem {state.problem_id}"
            )
            state.current_subproblem_index += 1
            self.solve_next_subproblem(state)
            return

        state.current_subproblem_index += 1

        # Create the subproblem state
        subproblem_state = self.create_subproblem_state(state, subproblem)
        if not subproblem_state:
            logger.error(f"[ExecutionAgent] Failed to create subproblem state for problem {state.problem_id}")
            self.solve_next_subproblem(state)
            return

        # Check depth again
        if subproblem_state.depth >= self.MAX_DEPTH:
            logger.warning(f"[ExecutionAgent] Maximum depth reached for problem_id: {subproblem_state.problem_id}")
            self.select_tool_and_solve(subproblem_state)
            return

        # Decide if we need to break down this subproblem further
        if self.task.should_break_down_problem(subproblem, self):
            logger.info(f"[ExecutionAgent] Breaking down subproblem: {subproblem}")
            sub_subproblems = self.task.break_down_problem(subproblem, self)
            if not sub_subproblems:
                logger.warning(f"[ExecutionAgent] No sub-subproblems for problem_id: {subproblem_state.problem_id}")
                self.select_tool_and_solve(subproblem_state)
                return
            subproblem_state.subproblems = sub_subproblems
            subproblem_state.current_subproblem_index = 0
            self.solve_next_subproblem(subproblem_state)
        else:
            self.select_tool_and_solve(subproblem_state)

    def select_tool_and_solve(self, state: ProblemState):
        """
        Sends a request to the ToolSelectionAgent to choose a tool. Once a tool is selected,
        the solution attempt will proceed in process_tool_selection.
        """
        problem_type = self.task.classify_problem(state.problem, self)
        state.problem_type = problem_type
        logger.info(f"[ExecutionAgent] Problem classified as: {problem_type}")

        content = {
            "request": "select_tool",
            "problem_type": problem_type,
            "attempted_tools": state.attempted_tools,
            "problem_id": state.problem_id,
        }
        self.send_message("ToolSelectionAgent", content, depth=state.depth, hierarchy=state.hierarchy)

    def process_tool_selection(self, message: Message):
        """
        After a tool is selected by the ToolSelectionAgent, use it to solve the problem.
        If the tool fails, send feedback to handle retries or switching tools.
        """
        content = message.content
        problem_id = content["problem_id"]
        selected_tool = content[MessageType.SELECTED_TOOL.value]
        state = self.state.get(problem_id)

        if not state:
            logger.warning(f"[ExecutionAgent] State not found for problem_id: {problem_id}")
            return

        state.selected_tool = selected_tool
        state.attempted_tools.append(selected_tool)

        try:
            tool = get_tool(selected_tool)
            tool_output = tool.solve(state.problem, self, state.prior_results)
            state.solution = tool_output["execution_result"]

            # Log tool output and reasoning/code
            logger.info(
                f"[ExecutionAgent] Tool '{selected_tool}' output for problem_id '{problem_id}': {state.solution}"
            )
            if "pal_code" in tool_output:
                logger.info(f"[ExecutionAgent] PAL code for problem_id '{problem_id}': {tool_output['pal_code']}")
                self.store_in_memory(state, "pal_code", {"tool": selected_tool, "pal_code": tool_output["pal_code"]})
            if "cot_reasoning" in tool_output:
                logger.info(
                    f"[ExecutionAgent] CoT reasoning for problem_id '{problem_id}': {tool_output['cot_reasoning']}"
                )
                self.store_in_memory(
                    state, "cot_reasoning", {"tool": selected_tool, "cot_reasoning": tool_output["cot_reasoning"]}
                )

            # Store tool output in memory
            self.store_in_memory(state, "tool_output", {"tool": selected_tool, "output": state.solution})

            # Update tool success rate
            if selected_tool not in self.metrics["tool_success_rates"]:
                self.metrics["tool_success_rates"][selected_tool] = {"successes": 0, "total": 0}
            self.metrics["tool_success_rates"][selected_tool]["total"] += 1

        except Exception as e:
            logger.error(f"[ExecutionAgent] Error solving problem_id: {problem_id} with tool {selected_tool}: {e}")
            # Send feedback that this attempt failed
            self.send_message(
                self.agent_id,
                {"problem_id": problem_id, "success": False, "retry": False, "feedback": str(e)},
                depth=state.depth,
                hierarchy=state.hierarchy,
            )
            return

        # Send the solution to FeedbackAgent for evaluation
        feedback_content = {
            "request": "evaluate_solution",
            "problem": state.problem,
            "solution": state.solution,
            "problem_id": state.problem_id,
        }
        if "pal_code" in tool_output:
            feedback_content["pal_code"] = tool_output["pal_code"]
        if "cot_reasoning" in tool_output:
            feedback_content["cot_reasoning"] = tool_output["cot_reasoning"]
        self.send_message("FeedbackAgent", feedback_content, depth=state.depth, hierarchy=state.hierarchy)

    def handle_feedback(self, message: Message):
        """
        Handles feedback for a problem or subproblem. Updates tool performance, decides on retrying or switching tools.
        If final success, calls handle_problem_solved; if fail and no more tools, fail the problem.
        """
        content = message.content
        problem_id = content.get("problem_id")

        if not problem_id:
            logger.error("[ExecutionAgent] Received feedback message without problem_id")
            return

        success = content.get("success", False)
        retry = content.get("retry", False)
        feedback = content.get("feedback", "")

        state = self.state.get(problem_id)
        if not state:
            logger.warning(f"[ExecutionAgent] State not found for problem_id: {problem_id}")
            return

        # Only send performance update if we have both problem type and selected tool
        if state.problem_type and state.selected_tool:
            update_content = {
                "request": "update_performance",
                "problem_type": state.problem_type,
                "tool": state.selected_tool,
                "success": success,
                "retry": retry,
                "problem_id": problem_id,  # Add problem_id to track context
            }
            self.send_message("ToolSelectionAgent", update_content, depth=state.depth, hierarchy=state.hierarchy)

        if success:
            # Update tool success metrics
            if state.selected_tool in self.metrics["tool_success_rates"]:
                self.metrics["tool_success_rates"][state.selected_tool]["successes"] += 1

            state.final_solution = state.solution
            self.handle_problem_solved(state)
        else:
            # Failure handling
            if retry and state.retry_count < self.MAX_TRIES:
                # Retry same tool with feedback appended to prior_results
                logger.info(f"[ExecutionAgent] Retrying problem_id: {problem_id}, attempt {state.retry_count + 1}")
                state.retry_count += 1
                state.prior_results.append(SubproblemResult(question=state.problem, answer=feedback, success=False))
                self.attempt_solution(state)
            else:
                # Try another tool if available
                all_tools = self.task.get_all_tools_for_problem_type(state.problem_type)
                remaining_tools = [t for t in all_tools if t not in state.attempted_tools]
                if remaining_tools:
                    new_tool = remaining_tools[0]
                    logger.info(f"[ExecutionAgent] Switching to new tool: {new_tool} for problem_id: {problem_id}")
                    state.selected_tool = new_tool
                    self.select_tool_and_solve(state)
                else:
                    # No remaining tools
                    self.fail_problem(state, feedback)

    def fail_problem(self, state: ProblemState, feedback: str):
        """
        Handles problem failures by notifying the parent if it exists, otherwise notifying the Main agent.
        """
        # Add metrics update
        self.metrics["problems_failed"] += 1

        if state.parent_id:
            # Notify parent that this subproblem failed
            content = {
                "subproblem_failed": True,
                "problem_id": state.parent_id,
                "feedback": feedback,
                "subproblem_id": state.problem_id,
            }
            self.send_message(self.agent_id, content, depth=state.depth - 1, hierarchy=state.hierarchy[:-1])
        else:
            # Top-level problem failed, notify Main
            failure_content = {
                "problem_id": state.problem_id,
                "problem": state.problem,
                "status": "failed",
                "feedback": feedback,
            }
            self.send_message("Main", failure_content, depth=state.depth, hierarchy=state.hierarchy)

        # Clean up state
        if state.problem_id in self.state:
            del self.state[state.problem_id]

    def attempt_solution(self, state: ProblemState):
        """
        Attempts to solve the problem again with the currently selected tool.
        This is used when retrying the same tool after feedback.
        """
        try:
            tool = get_tool(state.selected_tool)
            solution = tool.solve(state.problem, self, state.prior_results)
            state.solution = solution
        except Exception as e:
            logger.error(f"[ExecutionAgent] Error while retrying solution for problem_id: {state.problem_id}: {e}")
            # No retry this time, just fail
            self.send_message(
                self.agent_id,
                {"problem_id": state.problem_id, "success": False, "retry": False, "feedback": str(e)},
                depth=state.depth,
                hierarchy=state.hierarchy,
            )
            return

        content = {
            "request": "evaluate_solution",
            "problem": state.problem,
            "solution": solution,
            "problem_id": state.problem_id,
        }
        self.send_message("FeedbackAgent", content, depth=state.depth, hierarchy=state.hierarchy)

    def handle_problem_solved(self, state: ProblemState):
        """
        Handles when a problem is solved. If it has a parent, notify the parent.
        If it is a root problem, request an explanation and then finalize.
        """
        # Add metrics update
        self.metrics["problems_solved"] += 1

        if state.parent_id:
            # Notify parent that this subproblem is solved
            content = {
                "subproblem_solved": True,
                "solution": state.final_solution,
                "success": True,
                "problem_id": state.parent_id,
                "subproblem_id": state.problem_id,
            }
            self.send_message(self.agent_id, content, depth=state.depth - 1, hierarchy=state.hierarchy[:-1])
            # The parent's handling code (finish_subproblem) will remove this child's state
        else:
            # Root problem solved, ask for explanation
            self.request_explanation(state)

        # Store final solution in memory
        self.store_in_memory(
            state,
            "solution",
            {
                "final_solution": state.final_solution,
                "subproblem_solutions": [s.to_dict() for s in state.subproblem_solutions],
            },
        )

    def request_explanation(self, state: ProblemState):
        """
        Requests an explanation from the ExplanationAgent after the final solution is found.
        """
        content = {
            "request": "generate_explanation",
            "problem": state.problem,
            "solution": state.final_solution,
            "subproblem_results": [res.__dict__ for res in state.subproblem_solutions],
            "problem_id": state.problem_id,
        }
        self.send_message("ExplanationAgent", content, depth=state.depth, hierarchy=state.hierarchy)

    def handle_explanation(self, message: Message):
        """
        Receives the explanation from ExplanationAgent and sends final solution to Main.
        """
        content = message.content
        problem_id = content["problem_id"]
        explanation = content["explanation"]

        state = self.state.get(problem_id)
        if not state:
            logger.warning(f"[ExecutionAgent] State not found for problem_id: {problem_id}")
            return

        final_content = {
            "final_solution": state.final_solution,
            "explanation": explanation,
            "problem": state.problem,
            "problem_id": state.root_problem_id,
        }
        self.send_message("Main", final_content, depth=state.depth, hierarchy=state.hierarchy)
        del self.state[state.problem_id]

    def store_in_memory(self, state: ProblemState, memory_type: str, data: Dict):
        """Store information in the MemoryAgent."""
        content = {
            "request": "store",
            "problem_id": state.problem_id,
            "problem": state.problem,
            "problem_type": state.problem_type,
            "parent_id": state.parent_id,
            "hierarchy": state.hierarchy,
            "memory_type": memory_type,
            "data": data,
        }
        self.send_message("MemoryAgent", content, depth=state.depth, hierarchy=state.hierarchy)

    def retrieve_from_memory(self, state: ProblemState, memory_type: str):
        """Retrieve information from the MemoryAgent."""
        content = {"request": "retrieve", "problem_id": state.problem_id, "memory_type": memory_type}
        self.send_message("MemoryAgent", content, depth=state.depth, hierarchy=state.hierarchy)

    def search_in_memory(self, state: ProblemState, query: str):
        """Search for information in the MemoryAgent."""
        content = {"request": "search", "problem_id": state.problem_id, "query": query}
        self.send_message("MemoryAgent", content, depth=state.depth, hierarchy=state.hierarchy)
