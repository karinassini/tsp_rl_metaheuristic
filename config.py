from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VNSMainConfig:
	"""Top-level configuration for running VNS experiments via ``main.py``."""

	instances: List[str] = field(default_factory=lambda: ["swiss42.tsp"]) # swiss42.tsp, berlin52.tsp, gr48.tsp, ch150.tsp, gr120.tsp, si175.tsp, pr226.tsp, a280.tsp, pr226.tsp
	method: List[str] = field(default_factory=lambda: ["nearest_neighbor"]) # q_learning, random , nearest_neighbor
	start_city: int | None = 0
	iteration_max: Optional[int] = 200
	max_non_improving_iterations: int = 200
	repeats: int = 30
	k_max: int = 3
	save_dir: Optional[str] = None
	local_search: List[str] = field(default_factory=lambda: [ "VNS_Solver"])  # Options: "VNS_Solver", "VNS_Solver_Q_Learnings"