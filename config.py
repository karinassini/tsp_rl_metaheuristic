from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VNSMainConfig:
	"""Top-level configuration for running VNS experiments via ``main.py``."""

	instances: List[str] = field(default_factory=lambda: ["swiss42.tsp"])
	method: str = "random"
	iteration_max: Optional[int] = 200
	max_non_improving_iterations: int = 10
	repeats: int = 10
	start_city: int = 0
	k_max: int = 2
	save_dir: Optional[str] = None
