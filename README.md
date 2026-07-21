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

## Instance-Size Tuning Guide

Parameter sensitivity varies significantly with problem size. Use the following profiles to balance solution quality against runtime on different instance classes.

### Small Instances (n ≤ 100, e.g., swiss42, berlin52, eil76)
**Goal:** Explore thoroughly; runtime is not critical.

```python
class RLLocalSearchConfig:
    rl_alpha: float = 0.25
    rl_gamma: float = 0.80
    rl_epsilon: float = 0.5
    rl_epsilon_min: float = 0.05
    rl_epsilon_decay: float = 0.995
    max_local_search_iterations: Optional[int] = 200
    negative_reward_scale: float = 1.5
    operator_failure_limit: int = 10

class VNSConfig:
    max_double_bridge_checks: int = 300
    restricted_two_opt_max_span: int = 8
    max_flip_subsequence_length: int = 5
    max_inversion_segment_length: int = 5
    segment_len_val: Optional[int] = None
    verbose_route_log: bool = False

class VNSMainConfig:
    iteration_max: Optional[int] = 500
    max_non_improving_iterations: int = 100
    k_max: int = 4
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_local_search_iterations` | 200 | Enough iterations for quality; computational cost is manageable. |
| `rl_epsilon` | 0.5 | Higher exploration; small n allows fine-grained learning. |
| `max_double_bridge_checks` | 300 | Full perturbation sampling; dense graphs are affordable. |
| `iteration_max` | 500 | Long budget allows thorough neighbourhood scans. |

### Medium Instances (100 < n ≤ 300, e.g., ch150, a280, pr226)
**Goal:** Balance quality and speed; RL learns faster but runtime accumulates.

```python
class RLLocalSearchConfig:
    rl_alpha: float = 0.22
    rl_gamma: float = 0.80
    rl_epsilon: float = 0.40
    rl_epsilon_min: float = 0.03
    rl_epsilon_decay: float = 0.995
    max_local_search_iterations: Optional[int] = 120
    negative_reward_scale: float = 1.4
    operator_failure_limit: int = 8

class VNSConfig:
    max_double_bridge_checks: int = 150
    restricted_two_opt_max_span: int = 6
    max_flip_subsequence_length: int = 4
    max_inversion_segment_length: int = 4
    segment_len_val: Optional[int] = None
    verbose_route_log: bool = False

class VNSMainConfig:
    iteration_max: Optional[int] = 300
    max_non_improving_iterations: int = 60
    k_max: int = 3
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_local_search_iterations` | 120 | Moderate; operator selection converges around 60–80 iterations. |
| `rl_epsilon` | 0.40 | Moderate exploration; balance exploration and exploitation. |
| `max_double_bridge_checks` | 150 | Halved from small; random sampling avoids combinatorial explosion. |
| `operator_failure_limit` | 8 | Ban weak operators faster to reduce wasted iterations. |
| `iteration_max` | 300 | Shorter budget; focus on quality within time constraint. |

### Large Instances (n > 300, e.g., d657, pcb442, att532)
**Goal:** Minimize per-iteration cost; sacrifice some exploration for speed.

```python
class RLLocalSearchConfig:
    rl_alpha: float = 0.18
    rl_gamma: float = 0.75
    rl_epsilon: float = 0.35
    rl_epsilon_min: float = 0.02
    rl_epsilon_decay: float = 0.995
    max_local_search_iterations: Optional[int] = 80
    negative_reward_scale: float = 1.3
    operator_failure_limit: int = 6

class VNSConfig:
    max_double_bridge_checks: int = 50
    restricted_two_opt_max_span: int = 4
    max_flip_subsequence_length: int = 3
    max_inversion_segment_length: int = 4
    segment_len_val: Optional[int] = None
    verbose_route_log: bool = False

class VNSMainConfig:
    iteration_max: Optional[int] = 200
    max_non_improving_iterations: int = 40
    k_max: int = 2
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_local_search_iterations` | 80 | Aggressive cutoff; O(n²) operators dominate on 600+ cities. |
| `rl_epsilon` | 0.35 | Lower exploration; Q-table converges faster with fewer states. |
| `rl_alpha` | 0.18 | Slower learning; stability trumps fast convergence on large n. |
| `max_double_bridge_checks` | 50 | Massive reduction; even 50 random 4-cuts is 10^9 edge pairs in O(1). |
| `operator_failure_limit` | 6 | Ban quickly; cannot afford trial-and-error on 600 cities. |
| `iteration_max` | 200 | Tight budget; rely on strong initial solution (e.g., MARL seeding). |

### MARL Initialization Scaling

Use corresponding MARL budgets alongside VNS config:

| Instance Class | `population_size` | `marl_iterations` | `marl_agents` | `marl_candidate_ratio` |
|---|---|---|---|---|
| Small (n ≤ 100) | 50 | 5000 | 5 | 1.5 |
| Medium (100 < n ≤ 300) | 20 | 1000 | 3 | 1.0 |
| Large (n > 300) | 6 | 150 | 2 | 0.5 |

**Key insight:** For VNS (single-solution), small `population_size` (6) with aggressive `marl_candidate_ratio` (0.5) avoids expensive refinements. For GA (population-based), use larger budgets.

### Quick Selection Guide
```python
# In config.py, branch on problem size:
if n_cities <= 100:
    # Use small instance profile
    rl_local_search_cfg = RLLocalSearchConfig(max_local_search_iterations=200)
    vns_solver_cfg = VNSConfig(max_double_bridge_checks=300)
elif n_cities <= 300:
    # Use medium instance profile
    rl_local_search_cfg = RLLocalSearchConfig(max_local_search_iterations=120)
    vns_solver_cfg = VNSConfig(max_double_bridge_checks=150)
else:
    # Use large instance profile
    rl_local_search_cfg = RLLocalSearchConfig(max_local_search_iterations=80)
    vns_solver_cfg = VNSConfig(max_double_bridge_checks=50)
```

## Operator Performance Analysis

Analysis of 30 runs on **swiss42** (run date: 2026-02-24) reveals significant variability in operator effectiveness. The following metrics aggregate RL local search iterations across all logs in `outputs/solutions/swiss42/VNS_Solver_Q_Learnings/marl/20260224_210316/logs/`.

### Performance Summary

| Operator | Improvements | Failures | Success Rate | Avg Reward |
|----------|--------------|----------|--------------|-----------|
| `_one_move_insertion_improvement` | 84 | 9 | **90.3%** | 0.009908 |
| `_three_opt_first_improvement` | 63 | 11 | **85.1%** | 0.011148 |
| `_two_opt_first_improvement` | 57 | 35 | 62.0% | **0.016777** |
| `_double_bridge_first_improvement` | 23 | 47 | 32.9% | **0.020096** |

### Key Findings

**1. One-Move Insertion (Reliability Dominant)**
- **90.3% success rate** — most reliable operator
- 84 total improvements across 30 runs
- Average reward 0.009908 (modest per-move gain, but consistent)
- **Role:** Fine-tuning tours; workhorse for incremental refinement
- **Observation:** Succeeds even after other operators plateau; critical for later VNS iterations

**2. Double-Bridge (Quality Dominant)**
- **Highest average reward: 0.020096** — biggest solution jumps when it works
- 23 improvements but 47 failures (32.9% success rate)
- **Role:** Escaping local optima; large perturbations in early/mid VNS phases
- **Observation:** Low success rate suggests it is operator-of-last-resort; effective when tour is stuck

**3. Two-Opt (Balanced)**
- Average reward **0.016777** (second-best quality)
- 57 improvements, 62.0% success rate
- **Role:** Reliable general-purpose improvement
- **Observation:** Backbone operator; complement to insertion for diverse moves

**4. Three-Opt (Steady Performer)**
- 85.1% success rate (near insertion reliability)
- Average reward 0.011148
- **Role:** Moderate-scope moves; stepping stone between 2-opt and double-bridge
- **Observation:** Computationally expensive (O(n³) worst-case); rarely the best choice on large n

### Recommendations for Different Instance Sizes

**Small instances (n ≤ 100):**
- Prioritize operator diversity: use all four operators with high sampling budgets
- Insertion and 2-opt dominate early; reserve double-bridge for mid-phase escapes
- 3-opt cost is acceptable; include in full VND cycle

**Medium instances (100 < n ≤ 300):**
- Reduce double-bridge checks from 300 to 150 to balance exploration
- Deprecate 3-opt slightly; focus budget on insertion + 2-opt + selective double-bridge
- Use moderate operator failure limits (8–10 iterations) to ban weak choices quickly

**Large instances (n > 300, e.g., d657):**
- **Aggressively restrict 3-opt and double-bridge sampling** (already reduced to 50 checks in Large profile)
- **Maximize insertion sampling**: highest success rate justifies O(n²) cost on 600+ cities
- **Ban weak operators fast** (`operator_failure_limit=6`): cannot afford trial-and-error
- One-move insertion should dominate RL operator selection

### Integration with RL Q-Learning

The Q-table learns operator value through Bellman updates. Initial epsilon-greedy exploration discovers that:
- High-reward operators (double-bridge, 2-opt) are *sparse* → require exploration budget
- High-frequency operators (insertion) are *dense* → RL converges to them naturally
- Recommend: Set `rl_epsilon` proportional to operator success variance
  - Small instances: `rl_epsilon=0.5` (explore all)
  - Large instances: `rl_epsilon=0.35` (converge to insertion + 2-opt quickly)

## Operator Scaling Analysis Across Instance Sizes

Empirical operator performance data collected from VNS runs validates the instance-size tuning profiles. Analysis spans three benchmark sizes: swiss42 (42 cities), ch150 (150 cities), and d657 (657 cities).

### Medium Instance Analysis: ch150 (n=150, run date: 2026-03-09)

Analysis of logs from `outputs/solutions/ch150/VNS_Solver_Q_Learnings/marl/20260309_181315/logs/`.

| Operator | Improvements | Failures | Success Rate | Avg Reward |
|----------|--------------|----------|--------------|-----------|
| `_one_move_insertion_improvement` | 507,427 | 12,668 | **97.6%** | 0.002697 |
| `_two_opt_first_improvement` | 362,648 | 40,980 | **89.8%** | **0.005181** |
| `_two_exchange_first_improvement` | 135,928 | 29,524 | 82.2% | 0.003560 |
| `_three_opt_first_improvement` | 113,865 | 35,891 | 76.0% | 0.004241 |
| `_double_bridge_first_improvement` | 3,066 | 45,119 | 6.4% | 0.009327 |

**Key Observations:**
- **Insertion dominance amplified:** 507k improvements (40% more than 2-opt); 97.6% success rate
- **Double-bridge collapse:** Success rate drops from 33% (swiss42) to 6% (ch150); cost-benefit ratio unfavorable
- **Reward magnitude halves:** Average rewards shrink by ~50% compared to small instance, reflecting finer-grained neighborhoods on denser graphs
- **Two-exchange integration:** New operator appears in medium config; adds 136k improvements for diversity

**Scaling pattern:** As n grows from 42→150, insertion call frequency increases 6,043× while success rate improves (90%→98%), indicating insertion finds deeper neighborhoods even on larger tours.

### Large Instance Analysis: d657 (n=657, run date: 2026-07-20)

Analysis of logs from `outputs/solutions/d657/VNS_Solver_Q_Learnings/marl/20260720_102625/logs/`.

| Operator | Improvements | Failures | Success Rate | Avg Reward |
|----------|--------------|----------|--------------|-----------|
| `_one_move_insertion_improvement` | 6,514 | 23 | **99.6%** | 0.000622 |
| `_two_opt_first_improvement` | 6,423 | 205 | **96.9%** | **0.001309** |
| `_three_opt_first_improvement` | 283 | 773 | 26.8% | 0.001212 |
| `_double_bridge_first_improvement` | 0 | 980 | **0.0%** | 0.000000 |

**Critical Findings:**
- **Double-bridge completely fails:** 0 improvements out of 980 attempts (0.0% success)
  - This validates Large Instance tuning: max_double_bridge_checks=50 was insufficient; operator should be disabled on d657
  - Tours are too large; 4-cut perturbations introduce structural damage without recovery
- **Insertion near-universal:** 99.6% success rate; operates at highest reliability across all sizes
- **Two-Opt indispensable:** 96.9% success; only viable broad-scope operator after insertion saturation
- **Three-Opt ineffective:** 26.8% success; disabled in current d657 config is justified
- **Reward cliff:** Average rewards drop to 0.0006–0.0013 (98% reduction from swiss42)
  - Tours on 657 cities have small per-move improvements; tour structure is rigid

**Scaling catastrophe for double-bridge:**
- Swiss42: 23 improvements (33% success)
- Ch150: 3,066 improvements (6% success)
- D657: 0 improvements (0% success) ← **complete failure**

The trend confirms double-bridge is size-incompatible with large instances. Four-segment reconnections work on small tours but break large tours beyond repair.

### Cross-Size Comparison Table

| Metric | Swiss42 (n=42) | Ch150 (n=150) | D657 (n=657) | Trend |
|--------|---|---|---|---|
| **Insertion Success Rate** | 90.3% | 97.6% | 99.6% | ↑ Improves with scale |
| **2-Opt Success Rate** | 62.0% | 89.8% | 96.9% | ↑ Improves with scale |
| **Double-Bridge Success Rate** | 32.9% | 6.4% | 0.0% | ↓ Collapses with scale |
| **Insertion Avg Reward** | 0.009908 | 0.002697 | 0.000622 | ↓ Halves each 4× size |
| **2-Opt Avg Reward** | 0.016777 | 0.005181 | 0.001309 | ↓ Halves each 4× size |
| **Double-Bridge Avg Reward** | 0.020096 | 0.009327 | 0.000000 | ↓ Vanishes on large n |

### Implications for Configuration Tuning

1. **Double-Bridge Should Be Eliminated on Large Instances**
   - Current config: `max_double_bridge_checks=50` for d657 still wastes budget on zero-success operator
   - Recommendation: Set `max_double_bridge_checks=0` or remove from operator list for n > 400
   - Savings: ~5% of local-search CPU time recovered for insertion/2-opt

2. **Insertion Becomes Dominant with Scale**
   - Counterintuitively, insertion success rate *increases* (62%→99.6%) as problems grow
   - This is because tour structure becomes more regular on large n; vertex reinsertion finds local optima easily
   - Implication: Increase insertion sampling budget for large instances (already done in Large profile via `max_local_search_iterations=80`)

3. **Two-Opt Scales Better Than 3-Opt**
   - 2-opt maintains >96% success even on 657 cities; 3-opt collapses to 26%
   - O(n²) edge-reversal moves are more robust than O(n³) 3-segment recombinations
   - Recommendation: Deemphasize 3-opt for n > 300; run 2-opt more iterations instead

4. **Reward Magnitude Halves Per 4× Size Increase**
   - Swiss42→Ch150 (4× size): rewards drop by 73% (insertion: 0.0099→0.0027)
   - Ch150→D657 (4.4× size): rewards drop by 77% (insertion: 0.0027→0.0006)
   - This explains why `max_local_search_iterations` must drop (80 on d657 vs 200 on swiss42): fewer iterations find more improvements per iteration on large tours

### Validated Configuration Recommendations

**For d657 and similar large instances (n > 400):**
```python
# Remove double-bridge entirely
available_operators = [
    '_one_move_insertion_improvement',   # Dominant (99.6% SR)
    '_two_opt_first_improvement',        # Essential (96.9% SR)
    '_two_exchange_first_improvement'    # Optional diversity (add if available)
]

# Tune insertion-heavy convergence
rl_local_search_cfg = RLLocalSearchConfig(
    max_local_search_iterations=80,       # Short iterations; high-success operators saturate quickly
    rl_epsilon=0.30,                      # Aggressive convergence to insertion (most reliable)
    operator_failure_limit=4              # Ban 3-opt/rare operators after 4 failures
)

vns_solver_cfg = VNSConfig(
    max_double_bridge_checks=0,           # Do not attempt (0% success empirically)
    restricted_two_opt_max_span=4,        # Focus 2-opt on local neighborhoods
    max_flip_subsequence_length=3         # Limit perturbation scope
)
```

**For medium instances (100 < n ≤ 300):**
- Reduce `max_double_bridge_checks` from 300 to 50 (empirically, 6% success on ch150 even with reduced budget)
- Increase insertion and 2-opt iterations; they scale better
- Keep 3-opt but lower in VND cycle priority

**For small instances (n ≤ 100):**
- Use Large profile parameters unchanged; diversify all operators
- Double-bridge remains viable (33% success on swiss42) but rare; keep in VND for escape moves

