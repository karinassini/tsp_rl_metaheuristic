from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class QLearningConfig:
	alpha: float = 0.3
	gamma: float = 0.5
	epsilon: float = 0.5
	epsilon_min: float = 0.1
	epsilon_decay: float = 0.98
	epsilon_reset_interval: Optional[int] = 500  # episodes between epsilon resets
	epsilon_reset_value: Optional[float] = 0.35  # fallback value; defaults to initial epsilon
	episodes: int = 5000
	cache_dir: Optional[Path] = None
	monitor_interval: Optional[int] = None  # episodes between Q-table delta logs


@dataclass
class RLLocalSearchConfig:
	rl_alpha: float = 0.3
	rl_gamma: float = 0.5
	rl_epsilon: float = 0.5
	rl_epsilon_min: float = 0.05
	rl_epsilon_decay: float = 0.98
	max_local_search_iterations: Optional[int] = 80
	negative_reward_scale: float = 1.5
	operator_failure_limit: int = 10


@dataclass
class VNSMainConfig:
	"""Top-level configuration for running VNS experiments via ``main.py``."""

	instances: List[str] = field(default_factory=lambda: ["gr48.tsp", "ch150.tsp"]) # swiss42.tsp, berlin52.tsp, gr48.tsp, ch150.tsp, gr120.tsp, si175.tsp, pr226.tsp, a280.tsp, pr226.tsp
	method: List[str] = field(default_factory=lambda: ["rcl", "q_learning", "nearest_neighbor" ]) # q_learning, random , nearest_neighbor
	start_city: int | None = None
	iteration_max: Optional[int] = 500
	max_non_improving_iterations: int = 250
	repeats: int = 30
	k_max: int = 4
	save_dir: Optional[str] = None
	local_search: List[str] = field(default_factory=lambda: ["VNS_Solver_Q_Learnings", "VNS_Solver"])  # Options: "VNS_Solver", "VNS_Solver_Q_Learnings"
	q_learning_cfg: QLearningConfig = field(default_factory=QLearningConfig)
	rl_local_search_cfg: RLLocalSearchConfig = field(default_factory=RLLocalSearchConfig)


@dataclass
class GAMainConfig:
	"""Configuration for running the Genetic Algorithm experiments."""

	instances: List[str] = field(default_factory=lambda: ["eil51.tsp"])
	repeats: int = 3
	save_dir: Optional[str] = None
	log_root: str = "outputs/logs/ga"
	ga_config_kwargs: Dict[str, Any] = field(
		default_factory=lambda: {
			"population_size": 100,
			"max_generations": 5000,
			"mutation_rate": 0.2,
			"crossover_rate": 0.9,
			"tournament_size": 4,
			"elitism": True,
			"elite_fraction": 0.05,
			"seed": None,
			"stagnation_limit": None,
			"selection_method": "roulette",
			"rank_selection_pressure": 1.7,
			"truncation_ratio": 0.3,
			"initialization_method": "marl",
			"marl_iterations": 5000,
			"marl_agents": 5,
			"marl_epsilon": 0.85,
			"marl_softmax_beta": 2.0,
			"marl_learning_rate": 0.75,
			"marl_discount": 0.7,
			"marl_reward": 1.0,
			"marl_candidate_ratio": 1.2,
			"marl_two_opt_passes": 5,
			"marl_top_k": 3,
			"log_dir": None,
		}
	)