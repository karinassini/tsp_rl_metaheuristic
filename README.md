# Metaheuristic - TSP

Implementations that combine reinforcement learning-inspired heuristics with classic metaheuristics to tackle the Traveling Salesman Problem (TSP). The toolbox ships with ready-to-run greedy, exact, and Variable Neighborhood Search (VNS) solvers plus utilities for analysing solutions.

## Key Features
- **Graph loader**: Parse TSPLIB coordinate and distance-matrix files into a unified `Graph` structure (`src/structures/graph.py`).
- **Solver suite**: Run greedy baselines, a VNS metaheuristic, or an exact Gurobi model (`src/solver/*`).
- **Constraint checks**: Validate candidate tours with reusable constraints (`src/constraints/tsp_constraint.py`).
- **Result tracking**: Persist JSON summaries, logs, and comparison plots directly under `outputs/`.
- **Benchmark ready**: Sample TSPLIB instances live inside `instances/tsplib/` with hooks for adding new datasets.

## Project Layout
```text
├── instances/          # Benchmark TSP instances (TSPLIB format)
├── outputs/            # Solver artefacts (plots, JSON solutions, logs)
├── src/
│   ├── constraints/    # Feasibility checks for tours
│   ├── solver/         # Greedy, exact (Gurobi), and VNS implementations
│   ├── structures/     # Graph abstraction and parsers
│   └── utils/          # Plotting helpers for solution comparisons
├── main.py             # Example entry point wiring graph + solvers
├── pyproject.toml      # Poetry configuration and dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8–3.11
- [Poetry](https://python-poetry.org/) ≥ 1.5 (project developed with 2.1.3)
- (Optional) [Gurobi](https://www.gurobi.com/) license if you intend to run the exact solver (installs via `gurobipy`).

### Environment Setup
```bash
git clone https://github.com/yourusername/tsp_rl_metaheuristic.git
cd tsp_rl_metaheuristic
poetry install
poetry shell  # or `poetry run <command>` for one-off execution
```

If you prefer `venv`, replace the `poetry` commands with your virtual environment workflow and install dependencies using `pip install -r requirements.txt` generated via `poetry export`.

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
Hooks enforce formatting (Black) and linting rules defined in `.pre-commit-config.yaml` before every commit.

## Running Solvers

### 1. Select an Instance
Drop additional TSPLIB-formatted problems under `instances/tsplib/`. The project includes `dj38.tsp`, `dj38_simplified.tsp`, and `swiss42.tsp` as examples.

### 2. Launch from `main.py`
Edit the `instance`, solver choice, and parameters in `main.py`, then run:
```bash
poetry run python main.py
```
The example script demonstrates how to:
- Load a graph from TSPLIB files.
- Switch between greedy, exact, and VNS solvers (`test_greedy_solver`, `test_exact`, `test_vns_solver`).
- Persist solutions through `SolutionSaver`, including optional plots.

### 3. Analyse Results
- JSON outputs and logs appear under `outputs/solutions/<instance>/<solver>/`.
- Use `SolutionComparisonPlotter` (`src/utils/plot_comparison_from_json.py`) to compare experiments:
  ```bash
  poetry run python -c "from src.utils.plot_comparison_from_json import SolutionComparisonPlotter as P; P('outputs/solutions/swiss42/vns').run()"
  ```
- Graph visualisations can be generated via `Graph.visualize(...)` when running experiments.

## Adding New Solvers or Experiments
- Implement a solver in `src/solver/` and expose an entry function.
- Register experiment helpers inside `main.py` or create a dedicated driver script under `src/`.
- Reuse `TSPConstraint` to ensure feasibility and `SolutionSaver` to keep artefacts consistent.

## Benchmarks and selected instances 
The repository currently focuses on TSPLIB data. You can extend `Graph.from_tsplib_math` or add new loaders for problem variants. References:
- [Math TSP Data](https://www.math.uwaterloo.ca/tsp/data/index.html)
- [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)

Selected instances according with TSPLIB


swiss42
gr48
berlin52
prl76 -> pr226
gr120
ch150
si175
a280
pcb442

## Contributing
Issues, bug fixes, and feature contributions are welcome. Please include relevant tests (`pytest`) and ensure formatting checks pass before submitting a pull request.

## License
Distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
