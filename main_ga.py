import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root, "src")
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

if root not in sys.path:
    sys.path.insert(0, root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import GAMainConfig
from src.structures.graph import Graph
from src.solver.genetic_algorithm import GeneticTSPSolver, GeneticAlgorithmConfig
from src.solver.solution_saver import SolutionSaver, VNSSummaryTracker

SUMMARY_TRACKER = VNSSummaryTracker()


def _prepare_ga_config(base_kwargs: dict, log_dir: str | None, run_idx: int) -> GeneticAlgorithmConfig:
    """Build a GA config for the current run, injecting log dir and per-run seed tweaks."""
    kwargs = dict(base_kwargs or {})
    if log_dir:
        kwargs["log_dir"] = log_dir
    base_seed = kwargs.get("seed")
    if base_seed is not None:
        kwargs["seed"] = base_seed + run_idx
    return GeneticAlgorithmConfig(**kwargs)


def _build_log_dir(config: GAMainConfig, instance_name: str, init_method: str, current_timestamp: str, run_idx: int) -> str | None:
    if not config.log_root:
        return None
    path = os.path.join(
        config.log_root,
        instance_name,
        init_method,
        current_timestamp,
        f"run_{run_idx + 1}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def run_ga_experiments(graph: Graph, instance: str, config: GAMainConfig, current_timestamp: str) -> None:
    instance_name = Path(instance).stem
    base_kwargs = dict(config.ga_config_kwargs or {})

    for run_idx in range(config.repeats):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        placeholder_init = base_kwargs.get("initialization_method", "nearest_random")
        print(f"Starting GA run {run_idx + 1}/{config.repeats} | instance={instance_name} | init={placeholder_init}")
        #log_dir = _build_log_dir(config, instance_name, placeholder_init, current_timestamp, run_idx)
        solver_config = _prepare_ga_config(base_kwargs, None, run_idx)
        init_method = solver_config.initialization_method

        save_root = config.save_dir or os.path.join(
            "outputs",
            "solutions",
            instance_name,
            "GeneticTSPSolver",
            init_method,
            current_timestamp,
        )
        os.makedirs(save_root, exist_ok=True)
        saver = SolutionSaver(instance_name=instance_name, save_dir=save_root)

        solver = GeneticTSPSolver(graph, save_dir=save_root, config=solver_config)
        start_time = time.perf_counter()
        tour, total_distance, history = solver.run()
        elapsed = time.perf_counter() - start_time

        initial_stats = history[0] if history else {}
        init_mean = initial_stats.get("mean_distance")
        init_std = initial_stats.get("std_distance")

        print(
            f"GA run completed | instance={instance_name} | init={init_method} | "
            f"distance={total_distance:.2f} | time={elapsed:.2f}s"
        )

        saver.save(
            tour,
            total_distance,
            timestamp=run_timestamp,
            solver=type(solver).__name__,
            method=init_method,
            iteration_max=solver_config.max_generations,
            runtime_seconds=elapsed,
            history=history,
            solver_params=asdict(solver_config),
            initial_population_mean_distance=init_mean,
            initial_population_std_distance=init_std,
        )

        SUMMARY_TRACKER.record_run(
            instance_name=instance_name,
            method=init_method,
            iteration_max=solver_config.max_generations,
            solver_name=type(solver).__name__,
            total_distance=total_distance,
            exploration_time=elapsed,
            exploitation_time=0.0,
            route=tour,
            save_dir=save_root,
            best_known=saver.best_total_distance,
            iteration_first_reach=None,
            start_city_initialization=None,
        )


def main() -> None:
    config = GAMainConfig()
    tsplib_root = os.path.join(root, "instances", "tsplib")
    graphs: dict[str, Graph] = {}

    for instance in config.instances:
        graph_path = os.path.join(tsplib_root, instance)
        graph = Graph.from_tsplib(graph_path)
        print(f"Loaded GA instance {instance}: nodes={graph.n_nodes}")
        graphs[instance] = graph

    for instance in config.instances:
        graph = graphs[instance]
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_ga_experiments(graph, instance, config, current_timestamp)


if __name__ == "__main__":
    main()
