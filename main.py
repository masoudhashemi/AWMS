import json
import pickle
import time as time_module

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


def main():
    # Initialize the message bus and the task
    message_bus = MessageBus()
    task = MathProblemTask()

    # Instantiate agents
    main_agent = MainAgent("Main", message_bus)
    memory_file = "agent_memory.json"
    memory_agent = MemoryAgent("MemoryAgent", message_bus, memory_file)
    tool_selection_agent = ToolSelectionAgent("ToolSelectionAgent", message_bus, task)
    feedback_agent = FeedbackAgent("FeedbackAgent", message_bus, task)
    explanation_agent = ExplanationAgent("ExplanationAgent", message_bus)
    execution_agent = ExecutionAgent("ExecutionAgent", message_bus, task, max_depth=2)

    # Initialize data paths and variables
    solutions_output_path = "solutions.pkl"
    explanations_output_path = "explanations.pkl"
    messages_log_path = "messages_log.json"

    # Load your problems (assuming you have them in a list)
    problems_math = [
        (
            "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  "
            "He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  "
            "He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours?"
        ),
        # (
        #     "A car travels 120 miles at a certain speed.  If the car had gone 20 mph faster, the trip would have taken 1 hour less.  "
        #     "What was the speed of the car?"
        # ),
        # a problem needing its subproblems to be broken down further
        # (
        #     "A car travels 120 miles at a certain speed.  If the car had gone 20 mph faster, the trip would have taken 1 hour less.  "
        #     "How long does it take to travel 120 miles with the original speed?"
        # ),
    ]

    # problems_math = pd.read_json("./data/problems/gsm8k/test.jsonl", orient="records", lines=True)["question"].tolist()

    # Initialize lists to store solutions and explanations
    all_explanations = []
    all_final_solutions = []

    # Function to save intermediate results
    def save_intermediate_results(solutions, explanations):
        with open(solutions_output_path, "wb") as f:
            pickle.dump(solutions, f)
        with open(explanations_output_path, "wb") as f:
            pickle.dump(explanations, f)
        # Save messages and memory
        message_bus.save_messages(messages_log_path)
        memory_agent.save_memory()
        logger.info("Intermediate results, messages, and memory saved.")

    # Helper function for solving problems
    def solve_problem(idx: int, problem: str):
        """Sends a message to ExecutionAgent to solve the problem and collects the result."""
        problem_id = f"problem_{idx}"
        message_content = {
            "request": "solve_problem",
            "problem": problem,
            "problem_id": problem_id,
            "depth": 0,
            "hierarchy": [],
            "prior_results": [],
            "attempted_problems": set(),
            "root_problem_id": problem_id,  # Set root_problem_id to the initial problem_id
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

    # Main execution loop
    logger.info("\n--- Running Recursive Problem Solver ---")
    for idx, problem in enumerate(problems_math):
        try:
            idx, final_solution_, explanation_ = solve_problem(idx, problem)
            all_final_solutions.append((idx, final_solution_))
            all_explanations.append((idx, explanation_))

            # Display results for debugging purposes
            logger.info(f"\n=== Solved Problem {idx + 1} ===")
            logger.info("\n--- Final Solution ---\n%s", final_solution_)
            logger.info("\n--- Explanation ---\n%s", explanation_)

        except Exception as exc:
            logger.error(f"[Main] Exception in problem {idx}: {exc}")
            all_final_solutions.append((idx, "Exception occurred"))
            all_explanations.append((idx, str(exc)))

        # Save intermediate results periodically
        if (idx + 1) % 2 == 0:  # Save every 5 problems
            save_intermediate_results(all_final_solutions, all_explanations)

    # Final saving of results after all problems are solved
    save_intermediate_results(all_final_solutions, all_explanations)
    logger.info("\n--- All Problems Solved ---")

    final_tree = logger.handlers[0].get_logs()  # assuming the only handler is your HierarchicalLoggingHandler
    # final_tree is now a nested dictionary representing your problem and its subproblems
    # print(json.dumps(final_tree, indent=2))


if __name__ == "__main__":
    main()
