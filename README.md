# tsp_rl_metaheuristic

## Summary
This repository contains implementations that combine metaheuristic algorithms with reinforcement learning to solve the Traveling Salesman Problem (TSP). The code aims to explore and optimize solution strategies using advanced machine learning techniques.

## Remarks
- The code uses instances from [Math TSP Data](https://www.math.uwaterloo.ca/tsp/data/index.html).
- Additional benchmark data is available from [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/).

## Getting Started

### Prerequisites
- Python 3.x
- pip package manager

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tsp_rl_metaheuristic.git
    cd tsp_rl_metaheuristic
    ```
2. (Optional) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuring Pre-commit Hooks

To ensure code quality and consistency, this repository uses pre-commit hooks.

1. Install pre-commit (if not already installed):
    ```bash
    pip install pre-commit
    ```
2. Make sure the `.pre-commit-config.yaml` file is present in the repository root.
3. Install the pre-commit hooks:
    ```bash
    pre-commit install
    ```
4. Run the hooks manually on all files:
    ```bash
    pre-commit run --all-files
    ```

Integrating pre-commit into your workflow helps maintain consistent code standards and prevent common issues.

## Usage

Refer to the project's documentation for details on running experiments and configuring parameters. Further instructions and use cases may be added as the project evolves.

## Contributing

Contributions are welcome! Please follow our coding standards and include tests where applicable. For any questions, open an issue on the repository.

## License

Distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.
