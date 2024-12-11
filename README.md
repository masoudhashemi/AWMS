# Message Passing Framework for Recursive Solution of Word-Based Math Problems

This repository presents a **recursive message-passing framework** that coordinates specialized agents to solve complex word-based math problems by breaking them down into smaller sub-problems. The framework recursively applies agents on each sub-problem, enabling efficient and accurate solutions through adaptive tool selection.

## Overview

1. **Recursive Problem Solving**: Each problem is recursively broken down into smaller sub-problems, enabling finer-grained application of agents at each step. This recursive approach allows for handling complex, multi-step problems by tackling smaller, manageable parts first and then integrating their solutions.

2. **Agent Roles**:
    - **Problem Classification Agent** - Analyzes each sub-problem to determine its type (e.g., algebra, calculus) and recommends the initial tool (Chain of Thought (CoT), or Program-aided Language (PAL)).
    - **Tool Selection Agent** - Chooses the optimal tool based on recommendations and past performance, balancing exploration and exploitation through either randomization or bandit algorithms.
    - **Execution Agent** - Executes the chosen tool for each sub-problem, generating initial solutions that can be further refined.
    - **Feedback Agent** - Validates the solution of each sub-problem, providing feedback and suggesting adjustments or alternative tools if necessary.
    - **Explanation Agent** - Compiles detailed explanations for each step or the final solution, ensuring interpretability and clarity for the user.

## Key Features

- **Recursive Solution Generation**: Each complex math problem is decomposed into sub-problems. The message-passing agents are then recursively applied to each sub-problem, allowing for precise, step-by-step solutions.
  
- **Adaptive Tool Selection**: Uses an exploration-exploitation strategy (randomization or bandit algorithms) to select the most suitable tool for each sub-problem. This adaptive selection enhances problem-solving efficiency by learning from feedback.
  
- **Flexible, Modular Architecture**: The framework's agents communicate through a message-passing protocol, allowing dynamic iteration and retooling based on feedback. Each agent’s responsibilities are modular, making it easy to expand the framework with additional tools or more complex recursive logic.

## Example Workflow

Consider the problem \( x^3 - 6x^2 + 11x - 6 = 0 \):

1. **Recursive Breakdown**: The problem is initially classified as requiring symbolic computation. If a smaller problem within the initial solution is identified (e.g., breaking down polynomial factors), the framework recursively applies agents to solve each factor individually.
2. **Agent Steps**:
    - **Problem Classification Agent** identifies it as an algebraic equation requiring SymPy for root computation.
    - **Tool Selection Agent** confirms the use of SymPy or explores alternatives if the solution requires further breakdown.
    - **Execution Agent** runs SymPy to compute roots, yielding intermediate solutions.
    - **Feedback Agent** verifies the correctness of roots and may suggest further steps if explanations or refinements are needed.
    - **Explanation Agent** provides a clear, recursive breakdown of each factorization step.

3. **Feedback-Driven Iteration**: The Feedback Agent may direct the Tool Selection Agent to switch tools or adjust parameters if initial solutions are incomplete or unclear, recursively refining until each sub-problem converges on an optimal solution.

## Future Work

- **Dynamic Parameter Tuning**: Implement strategies to adjust exploration parameters in real-time based on system performance.
- **Enhanced Bandit Algorithms**: Test advanced bandit algorithms like Thompson Sampling for more strategic tool selection.
- **Multi-Objective Optimization**: Integrate additional performance metrics, such as computation time or accuracy, into the selection strategy.
- **Reinforcement Learning Integration**: Explore reinforcement learning to refine the recursive decision-making process.
- **Training Phase**: Incorporate a training round to optimize tool performance on specific problem types before recursive application. 

This recursive, feedback-driven approach makes the framework robust, scalable, and highly adaptable to a wide range of word-based math problems, providing both efficient computation and detailed explanations for each solution.

### Project Structure Overview

``` plaintext
awms/
├── agents/
├── bases/
├── notebooks/
├── problem_solver.py
├── tasks/
├── tools/
└── utils.py
```

### Folder Descriptions

1. **agents/**
   - Contains the different types of agents involved in solving problems and managing tasks. Each agent has a specific role, such as executing a solution, generating explanations, providing feedback, or selecting the most appropriate tool for the task.
   - **Files**:
     - `execution.py`: Handles task execution and manages the primary logic for problem-solving.
     - `explanation.py`: Provides explanations or detailed reasoning steps for each solution.
     - `feedback.py`: Collects feedback on solutions, which could be used for improving future problem-solving attempts.
     - `tool_selection.py`: Chooses the best tool or approach for each task based on the problem type.

2. **bases/**
   - Defines base classes and shared structures for the main components like agents, tasks, and tools. These base classes provide common functionalities and structures, ensuring consistency across different agents and tools.
   - **Files**:
     - `agent.py`: Base class for defining agent-specific behavior and attributes.
     - `task.py`: Base class for defining tasks, which are units of work an agent can execute.
     - `tool.py`: Base class for defining tools, with reusable logic for any tool used in problem-solving.

3. **notebooks/**
   - A folder for Jupyter notebooks, which can be used for experimentation, testing, or demonstrating the problem-solving process in an interactive environment.

4. **problem_solver.py**
   - This script is the main problem-solving logic, coordinating the agents, tasks, and tools to solve problems. It acts as a central hub, orchestrating the flow of tasks between different components.

5. **tasks/**
   - Contains specific task definitions that the problem solver will handle. Each task defines a particular problem type or domain that the agents are set to solve.
   - **Files**:
     - `math.py`: Defines mathematical tasks, likely including problem structures, rules, and requirements specific to solving math-related problems.

6. **tools/**
   - Houses specific tools or methods that agents can employ to solve tasks. Tools might include predefined solution strategies, reasoning methods, or problem-solving algorithms.
   - **Files**:
     - `cot.py`: Implements the Chain of Thought (CoT) method, a step-by-step reasoning approach for complex problems.
     - `pal.py`: Implements the PAL (Program-Aided Language) method, which aids in problem-solving through programming techniques or symbolic logic.

7. **utils.py**
   - Contains helper functions or utilities used throughout the project. These might include functions for logging, data manipulation, saving/loading files, or handling exceptions.
