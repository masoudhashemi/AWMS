import logging
import pickle
import time as time_module
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from awms.agents.execution import ExecutionAgent
from awms.agents.explanation import ExplanationAgent
from awms.agents.feedback import FeedbackAgent
from awms.agents.main_agent import MainAgent
from awms.agents.memory import MemoryAgent
from awms.agents.tool_selection import ToolSelectionAgent
from awms.bases import MessageBus
from awms.logging import logger
from awms.tasks.math import MathProblemTask
from awms.utils import Message

# Initialize the message bus and agents for this process
message_bus = MessageBus()
task = MathProblemTask()

main_agent = MainAgent("Main", message_bus)
memory_agent = MemoryAgent("MemoryAgent", message_bus)
tool_selection_agent = ToolSelectionAgent("ToolSelectionAgent", message_bus, task)
feedback_agent = FeedbackAgent("FeedbackAgent", message_bus, task)
explanation_agent = ExplanationAgent("ExplanationAgent", message_bus)
execution_agent = ExecutionAgent("ExecutionAgent", message_bus, task, max_depth=2)


def solve_problem(idx: int, problem: str):
    """Solve a single problem by initializing MessageBus and agents in each process."""
    # Problem-solving logic
    problem_id = f"problem_{idx}"
    message_content = {
        "request": "solve_problem",
        "problem": problem,
        "problem_id": problem_id,
        "depth": 0,
        "hierarchy": [],
        "prior_results": [],
        "attempted_problems": set(),
        "root_problem_id": problem_id,
    }
    # Send initial message to ExecutionAgent
    message_bus.send_message(Message("Main", "ExecutionAgent", message_content, depth=0, hierarchy=[]))

    # Dispatch messages until the problem is solved
    while True:
        message_bus.dispatch_messages()
        if problem_id in main_agent.final_solutions:
            final_solution, explanation = main_agent.final_solutions[problem_id]
            break
        time_module.sleep(0.1)  # Small delay to prevent tight loop

    return idx, final_solution, explanation


def save_intermediate_results(
    solutions, explanations, solutions_output_path, explanations_output_path, messages_log_path
):
    """Save intermediate results to disk."""
    with open(solutions_output_path, "wb") as f:
        pickle.dump(solutions, f)
    with open(explanations_output_path, "wb") as f:
        pickle.dump(explanations, f)

    message_bus.save_messages(messages_log_path)
    logger.info("Intermediate results saved.")


def main():
    # Paths for intermediate results
    solutions_output_path = "solutions.pkl"
    explanations_output_path = "explanations.pkl"
    messages_log_path = "messages_log.json"
    index_path = "index.pkl"

    # Load your problems
    problems_math = pd.read_json("./data/problems/gsm8k/test.jsonl", orient="records", lines=True)["question"].tolist()

    # Load the previously processed indexes to skip
    try:
        with open(index_path, "rb") as f:
            index = pickle.load(f)
    except FileNotFoundError:
        index = []

    # Filter problems not already processed
    problems_math = [problem for idx, problem in enumerate(problems_math) if idx not in index]

    # Initialize lists to store solutions and explanations
    all_explanations = []
    all_final_solutions = []

    # Configure parallel execution
    num_workers = 4  # Adjust based on available CPU cores or requirements
    logger.info("\n--- Running Parallel Recursive Problem Solver ---")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = {executor.submit(solve_problem, idx, problem): idx for idx, problem in enumerate(problems_math)}

        # Collect results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                idx, final_solution, explanation = future.result()
                all_final_solutions.append((idx, final_solution))
                all_explanations.append((idx, explanation))

                # Display results for debugging purposes
                logger.info(f"\n=== Solved Problem {idx + 1} ===")
                logger.info("\n--- Final Solution ---\n%s", final_solution)
                logger.info("\n--- Explanation ---\n%s", explanation)
            except Exception as exc:
                logger.error(f"[Main] Exception in problem {idx}: {exc}")
                all_final_solutions.append((idx, "Exception occurred"))
                all_explanations.append((idx, str(exc)))

            # Save intermediate results periodically
            if (idx + 1) % 5 == 0:  # Save every 5 problems
                save_intermediate_results(
                    all_final_solutions,
                    all_explanations,
                    solutions_output_path,
                    explanations_output_path,
                    messages_log_path,
                )

    # Final saving of results after all problems are solved
    save_intermediate_results(
        all_final_solutions, all_explanations, solutions_output_path, explanations_output_path, messages_log_path
    )
    logger.info("\n--- All Problems Solved ---")


if __name__ == "__main__":
    main()
