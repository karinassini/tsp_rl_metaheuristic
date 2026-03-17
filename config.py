from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class QLearningConfig:
	alpha: float = 0.3
	gamma: float = 0.65
	epsilon: float = 0.5
	epsilon_min: float = 0.1
	epsilon_decay: float = 0.995
	epsilon_reset_interval: Optional[int] = 500  # episodes between epsilon resets
	epsilon_reset_value: Optional[float] = 0.35  # fallback value; defaults to initial epsilon
	episodes: int = 5000
	cache_dir: Optional[Path] = None
	monitor_interval: Optional[int] = None  # episodes between Q-table delta logs


@dataclass
class RLLocalSearchConfig:
	rl_alpha: float = 0.22
	rl_gamma: float = 0.80
	rl_epsilon: float = 0.4
	rl_epsilon_min: float = 0.02
	rl_epsilon_decay: float = 0.995
	max_local_search_iterations: Optional[int] = 600 #100 small
	negative_reward_scale: float = 1.4
	operator_failure_limit: int = 10


@dataclass
class VNSConfig:
	"""Solver-level knobs for VNS neighbourhood strength and sampling."""

	max_double_bridge_checks: int = 300 #large size 300, small size 150
	restricted_two_opt_max_span: int = 8
	max_flip_subsequence_length: int = 5
	max_inversion_segment_length: int = 5
	segment_len_val: Optional[int] = None #large size 14, small 5


@dataclass
class VNSMainConfig:
	"""Top-level configuration for running VNS experiments via ``main.py``."""

	# rodar o gr48 novamente
	instances: List[str] = field(default_factory=lambda: ["pr226.tsp", "a280.tsp"]) # swiss42.tsp, berlin52.tsp, gr48.tsp, kroA100.tsp, kroB100.tsp, ch150.tsp, gr120.tsp, si175.tsp, pr226.tsp, a280.tsp,
	method: List[str] = field(default_factory=lambda: ["rcl", "marl"]) # q_learning, random , nearest_neighbor, marl
	start_city: int | None = None
	iteration_max: Optional[int] = 800
	max_non_improving_iterations: int = 200 #300 small
	repeats: int = 30
	k_max: int = 4 # 4 small
	save_dir: Optional[str] = None
	vns_solver_cfg: VNSConfig = field(default_factory=VNSConfig)
	local_search: List[str] = field(default_factory=lambda: ["VNS_Solver_Q_Learnings", "VNS_Solver"])  # Options: "VNS_Solver", "VNS_Solver_Q_Learnings"
	q_learning_cfg: QLearningConfig = field(default_factory=QLearningConfig)
	rl_local_search_cfg: RLLocalSearchConfig = field(default_factory=RLLocalSearchConfig)
	marl_config_kwargs: Dict[str, Any] = field(
		default_factory=lambda: {
			"population_size": 50, #small and medium 100, large 50
			"marl_iterations": 5000, #small 10000, large
			"marl_agents": 4, #small 5, super large 3
			"marl_epsilon": 0.15,
			"marl_softmax_beta": 3.0,
			"marl_learning_rate": 0.70,
			"marl_discount": 0.7,
			"marl_reward": 1,
			"marl_candidate_ratio": 1.2,
			"marl_two_opt_passes": 5,
			"marl_top_k": 3,
			"marl_policy_epsilon_greedy_prob": 0.85,
		}
	)

@dataclass
class GAMainConfig:
	"""Configuration for running the Genetic Algorithm experiments."""

	instances: List[str] = field(default_factory=lambda: ["pr226.tsp", "a280.tsp", "pr226.tsp"])
	ga_solver: str = "standard"  # options: "standard" (fixed crossover), "q_learning" (learned crossover selection)
	repeats: int = 30
	save_dir: Optional[str] = None
	log_root: str = "outputs/logs/ga"
	ga_config_kwargs: Dict[str, Any] = field(
		default_factory=lambda: {
			"population_size": 100,
			"max_generations": 10000,
			"mutation_rate": 0.1,
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
			"marl_iterations": 10000,
			"marl_agents": 5,
			"marl_epsilon": 0.85,
			"marl_softmax_beta": 2.0,
			"marl_learning_rate": 0.75,
			"marl_discount": 0.7,
			"marl_reward": 1.2,
			"marl_candidate_ratio": 1.2,
			"marl_two_opt_passes": 5,
			"marl_top_k": 3,
			"marl_policy_epsilon_greedy_prob": 1.0,
			"log_dir": None,
			"survivor_selection_method": "best_improves",  # options: "tournament", "best_improves"
			"crossover_method": "er" #er, smx
		}
	)