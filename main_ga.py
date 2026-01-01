import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

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


def _plot_learning_iterations_curve(
    records: list[dict[str, float | int]],
    instance_name: str,
    save_dir: str,
) -> None:
    if not records:
        return

    grouped: dict[int, list[dict[str, float | int]]] = {}
    for entry in records:
        key = int(entry["marl_iterations"])
        grouped.setdefault(key, []).append(entry)

    xs = sorted(grouped.keys())
    avg_distances: list[float] = []
    avg_runtimes: list[float] = []
    std_distances: list[float] = []
    run_counts: list[int] = []

    for key in xs:
        bucket = grouped[key]
        distances = [float(item["distance"]) for item in bucket]
        runtimes = [float(item["runtime"]) for item in bucket]
        run_counts.append(len(bucket))
        mean_distance = sum(distances) / max(1, len(distances))
        avg_distances.append(mean_distance)
        if len(distances) > 1:
            variance = sum((value - mean_distance) ** 2 for value in distances) / (len(distances) - 1)
            std_distances.append(variance ** 0.5)
        else:
            std_distances.append(0.0)
        avg_runtimes.append(sum(runtimes) / max(1, len(runtimes)))

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, avg_distances, color="black", marker="s", linewidth=1.8)
    for x, y, runtime in zip(xs, avg_distances, avg_runtimes):
        plt.scatter([x], [y], color="black")
        plt.text(x, y + 0.4, f"{runtime:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.title("Average tour length vs. MNLI", fontsize=12)
    plt.xlabel("Maximum number of learning iterations (MNLI)", fontsize=11)
    plt.ylabel(f"Average tour length ({instance_name})", fontsize=11)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "learning_iterations_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    payload = {
        "instance": instance_name,
        "points": [
            {
                "marl_iterations": int(x),
                "avg_distance": float(y),
                "std_distance": float(std),
                "avg_runtime_seconds": float(rt),
                "runs": count,
            }
            for x, y, std, rt, count in zip(xs, avg_distances, std_distances, avg_runtimes, run_counts)
        ],
    }
    json_path = os.path.join(save_dir, "learning_iterations_curve.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Saved learning-iteration curve to {plot_path}")


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
    learning_curve_records: list[dict[str, float | int]] = []
    aggregate_save_root: str | None = None

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
        aggregate_save_root = save_root
        saver = SolutionSaver(instance_name=instance_name, save_dir=save_root)

        solver = GeneticTSPSolver(
            graph,
            save_dir=save_root,
            config=solver_config,
            best_known_distance=saver.best_total_distance,
        )
        tour, total_distance, history, elapsed = solver.run()

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
            iteration_first_reach=solver.best_known_hit_generation,
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
            iteration_first_reach=solver.best_known_hit_generation,
            start_city_initialization=None,
        )

        learning_curve_records.append(
            {
                "marl_iterations": solver_config.marl_iterations,
                "distance": total_distance,
                "runtime": elapsed,
            }
        )

    if aggregate_save_root:
        _plot_learning_iterations_curve(
            learning_curve_records,
            instance_name,
            aggregate_save_root,
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
