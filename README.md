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

## Reading from TSPLIB

The loader `Graph.from_tsplib(...)` accepts any TSPLIB-formatted `.tsp` file and reconstructs the full distance matrix required by the solvers. The parser follows the TSPLIB specification step by step:

- **Header scan** – Iterate through the file collecting metadata such as `DIMENSION`, `EDGE_WEIGHT_TYPE`, and `EDGE_WEIGHT_FORMAT`. These flags describe how distances are provided (explicit matrix versus coordinates) and how many nodes to expect.
- **Section dispatch** – When the parser encounters markers like `EDGE_WEIGHT_SECTION` or `NODE_COORD_SECTION`, it switches into the appropriate reader. While the weights/coordinates may be split across multiple physical lines, they form a single continuous sequence that the loader reassembles.
- **Explicit matrices** – For `EDGE_WEIGHT_TYPE: EXPLICIT`, the reader converts the flattened list of weights into a symmetric matrix. Formats such as `FULL_MATRIX`, `UPPER_DIAG_ROW`, `LOWER_ROW`, etc., control whether the file lists every entry or only the upper/lower triangle. The loader validates the expected number of values and mirrors whatever portion is provided to recover the complete matrix.
- **Coordinate-based problems** – When the file supplies coordinates (`EUC_2D`, `CEIL_2D`), the parser computes pairwise Euclidean distances, rounding or ceiling as mandated by the TSPLIB naming. The result is again a dense matrix with distance from each node *i* to *j*.
- **Graph construction** – With the matrix in hand, the loader builds the `Graph` object storing: the ordered node list, an edge dictionary `(i, j) → distance`, and a 2D adjacency matrix. All solvers consume this unified structure regardless of the original TSPLIB format.

Because TSPLIB allows long sequences of numbers to wrap across lines, the parser relies on counting rather than line breaks. For instance, an `UPPER_DIAG_ROW` matrix for a 175-node instance provides `175 + 174 + … + 1 = 15,400` numbers in total; the loader simply streams them in order and slices the sequence according to the format definition.

Example usage:

```python
from src.structures.graph import Graph

graph = Graph.from_tsplib("instances/tsplib/si175.tsp")
distance_matrix = graph.get_distance_matrix()
```

This call automatically interprets the header, unpacks the explicit upper-triangular weights, mirrors them to the lower triangle, and returns a ready-to-use graph.

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

## Local Search Operators

The VNS pipeline embeds a Variable Neighborhood Descent (VND) stage that alternates between complementary neighbourhoods:
- **2-opt first improvement** scans non-adjacent edge pairs and returns as soon as a swap shortens the tour. After every improving swap the evaluation restarts from the first pair, quickly reaching a 2-opt local optimum.
- **1-insertion first improvement** removes a node from position *i* and reinserts it before position *j*, exploring insertion neighbourhoods cited in interchange heuristics. The move skips the depot/start city and stops at the first improvement.
- **2-exchange first improvement** swaps two internal vertices (classic interchange move) while keeping the start node fixed. It explores permutations that 2-opt and 1-insertion cannot reach and exits on the first improvement.
- **Double-bridge first improvement** enumerates four cut points, reconnecting the tour segments in the order 1–3–2–4. This larger jump helps escape plateaus left by the previous neighbourhoods; the operator stops at the first improving reconnection.

Whenever any operator finds a better route, the VND loop restarts with 2-opt, guaranteeing that fine-grained improvements are exhausted before escalating to broader neighbourhood kicks.

