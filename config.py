from dataclasses import dataclass, field
from typing import List, Optional
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

	instances: List[str] = field(default_factory=lambda: ["swiss42.tsp"]) # swiss42.tsp, berlin52.tsp, gr48.tsp, ch150.tsp, gr120.tsp, si175.tsp, pr226.tsp, a280.tsp, pr226.tsp
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