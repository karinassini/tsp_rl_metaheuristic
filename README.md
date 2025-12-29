# Metaheuristic TSP Toolkit

Metaheuristics and reinforcement learning for the Traveling Salesman Problem. The toolbox provides greedy baselines, an exact solver, Variable Neighborhood Search (VNS), and analysis utilities ready to run on TSPLIB instances.

## Contents
- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Working with TSPLIB](#working-with-tsplib)
- [Running Experiments](#running-experiments)
- [Analysis Utilities](#analysis-utilities)
- [Extending the Project](#extending-the-project)
- [Local Search Operators](#local-search-operators)
- [Q-Learning Primer](#q-learning-primer)
- [Logging Tips](#logging-tips)
- [Benchmark Instances](#benchmark-instances)
- [Contributing](#contributing)
- [License](#license)

## Highlights
- **Unified graph loader** converts TSPLIB coordinates or explicit matrices into the `Graph` abstraction (`src/structures/graph.py`).
- **Solver suite** includes greedy heuristics, VNS, and an optional Gurobi-based exact model (`src/solver/`).
- **Constraints** ensure feasible tours via `TSPConstraint`.
- **Experiment tracking** stores JSON metrics, plots, and logs under `outputs/`.
- **Plotting helpers** compare objective trends and time-to-target curves.

## Repository Structure
```text
├── instances/          # TSPLIB benchmark data
├── outputs/            # Logs, JSON summaries, plots
├── src/
│   ├── constraints/    # Feasibility checks
│   ├── solver/         # Greedy, exact, VNS, RL integrations
│   ├── structures/     # Graph loaders and utilities
│   └── utils/          # Plotting and reporting helpers
├── main.py             # Example experiment runner
├── pyproject.toml      # Dependencies via Poetry
└── README.md
```

## Quick Start
### Prerequisites
- Python 3.8–3.11
- [Poetry](https://python-poetry.org/) ≥ 1.5 (developed with 2.1.3)
- Optional: [Gurobi](https://www.gurobi.com/) license when using the exact solver

### Installation
```bash
git clone https://github.com/yourusername/tsp_rl_metaheuristic.git
cd tsp_rl_metaheuristic
poetry install
poetry shell
```

Prefer `venv`? Export dependencies via `poetry export --format requirements.txt` and install with `pip install -r requirements.txt`.

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Working with TSPLIB
`Graph.from_tsplib(path)` reads any TSPLIB `.tsp` file and reconstructs a distance matrix. The parser:
- scans headers for dimension, weight type, and format;
- dispatches to the correct section reader (`EDGE_WEIGHT_SECTION`, `NODE_COORD_SECTION`, etc.);
- rebuilds explicit matrices from formats such as `FULL_MATRIX`, `UPPER_DIAG_ROW`, or `LOWER_ROW`;
- derives Euclidean distances for coordinate-based problems (`EUC_2D`, `CEIL_2D`).

Example:
```python
from src.structures.graph import Graph

graph = Graph.from_tsplib("instances/tsplib/si175.tsp")
distance_matrix = graph.get_distance_matrix()
```

## Running Experiments
1. **Pick an instance**: add `.tsp` files under `instances/tsplib/`.
2. **Configure `main.py`**: choose solver (`test_greedy_solver`, `test_exact`, `test_vns_solver`), tweak parameters, and set output paths.
3. **Execute**:
   ```bash
   poetry run python main.py
   ```

## Analysis Utilities
- `SolutionSaver` records JSON histories, objective plots, and time-to-target curves.
- `src/utils/plot_comparison_from_json.py` overlays results across experiments:
  ```bash
  poetry run python -c "from src.utils.plot_comparison_from_json import SolutionComparisonPlotter as Plot; Plot('outputs/solutions/swiss42/vns').run()"
  ```
- `Graph.visualize()` renders route diagrams for quick inspection.

## Extending the Project
- Implement new heuristics in `src/solver/` and wire them into `main.py` or a custom driver.
- Share constraint logic via `TSPConstraint` to guarantee valid tours.
- Store artefacts with `SolutionSaver` for consistency across experiments.

## Local Search Operators
The VNS pipeline embeds a Variable Neighborhood Descent loop cycling through complementary neighbourhoods:
- **2-opt first improvement** flips non-adjacent edges and restarts on every improvement.
- **1-insertion first improvement** removes a vertex at position `i` and reinserts it before `j`.
- **2-exchange first improvement** swaps two interior vertices while keeping the depot fixed.
- **Double-bridge first improvement** reconnects four segments in the order `1-3-2-4` to escape plateaus.

Each improvement resets the sequence so finer-grained moves finish before broader perturbations.

## Q-Learning Primer
The RL-assisted initialisation builds a reward matrix from the distance matrix and trains a Q-table using the Bellman update:
```python
q_table[current, action] = (1 - cfg.alpha) * q_table[current, action] \
    + cfg.alpha * (reward + cfg.gamma * max_future)
current = next_state
```

Where:
- `cfg.alpha` – learning rate;
- `cfg.gamma` – discount factor for future rewards;
- `reward` – immediate payoff for the chosen edge;
- `max_future` – best Q-value available from the next state.

The agent follows an epsilon-greedy policy, decays `epsilon` toward `epsilon_min`, and explicitly reinforces the closing edge to complete the tour.

Equivalent mathematical form:

\[
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a')\Big]
\]

or, expressed as the incremental update:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]
\]

Parameter meanings:

| Symbol | Code reference | Description |
| --- | --- | --- |
| \(Q(s, a)\) | `q_table[current, action]` | Expected cumulative reward for taking action `a` in state `s`. |
| \(\alpha\) | `cfg.alpha` | Learning rate controlling how strongly new information overwrites old values. |
| \(r\) | `reward` | Immediate payoff after executing the action. |
| \(\gamma\) | `cfg.gamma` | Discount factor weighting future rewards. |
| \(\max_{a'} Q(s', a')\) | `max_future` | Best available Q-value from the successor state. |
| \(s'\) | `next_state` | State reached after applying the action. |
| \(s\) | `current` | State before the action. |

## Logging Tips
Logging is cheap in Python but file I/O dominates once messages hit disk. To measure the impact:
- time representative solver sections with `time.perf_counter()`;
- rerun with `logging.WARNING` or without the handler;
- compare elapsed times to decide whether to batch or down-sample log entries.

## Benchmark Instances
Bundled TSPLIB problems:
- swiss42
- gr48
- berlin52
- prl76 (redirected to pr226)
- gr120
- ch150
- si175
- a280
- pcb442

Refer to [TSPLIB95]( http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and [Math TSP Data](https://www.math.uwaterloo.ca/tsp/data/index.html) for additional instances.

## Contributing
Please open issues or pull requests with accompanying tests (`pytest`) and formatting checks. Run `pre-commit run --all-files` before submitting.

## License
Licensed under the MIT License. See [LICENSE](LICENSE).

## Recommended Parameters
- Validated across TSPLIB instances from swiss42 through ch150.

```python
class QLearningConfig:
  alpha: float = 0.4
  gamma: float = 0.8
  epsilon: float = 0.4
  epsilon_min: float = 0.05
  epsilon_decay: float = 0.995
  episodes: int = 3000
  cache_dir: Optional[Path] = None


class RLLocalSearchConfig:
  rl_alpha: float = 0.3
  rl_gamma: float = 0.5
  rl_epsilon: float = 0.5
  rl_epsilon_min: float = 0.05
  rl_epsilon_decay: float = 0.98
  max_local_search_iterations: Optional[int] = 80
  negative_reward_scale: float = 1.5
  operator_failure_limit: int = 10


class VNSMainConfig:
  instances: List[str] = field(default_factory=lambda: ["swiss42.tsp"])
  method: str = "nearest_neighbour"  # Options: "random", "q_learning", "nearest_neighbour"
  iteration_max: Optional[int] = 200
  max_non_improving_iterations: int = 15
  repeats: int = 30
  start_city: int = 0
  k_max: int = 2
  save_dir: Optional[str] = None
  q_learning_cfg: QLearningConfig = field(default_factory=QLearningConfig)
  rl_local_search_cfg: RLLocalSearchConfig = field(default_factory=RLLocalSearchConfig)
```

Match both configurations during experiments to preserve exploration versus exploitation timing.

### Q-learning initial solution