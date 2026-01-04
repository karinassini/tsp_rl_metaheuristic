import os
from dataclasses import replace
from typing import List

from src.structures.graph import Graph
from src.solver.genetic_algorithm import GeneticTSPSolver, GeneticAlgorithmConfig


def debug_marl_initial_population(
    graph: Graph,
    base_config: GeneticAlgorithmConfig,
    save_dir: str | None = None,
) -> List[List[int]]:
    """Generate and inspect a MARL initial population without running the GA loop."""

    debug_save_dir = save_dir or os.path.join("outputs", "debug_marl_initial_population")
    os.makedirs(debug_save_dir, exist_ok=True)

    cfg = replace(base_config, initialization_method="marl")
    solver = GeneticTSPSolver(graph, save_dir=debug_save_dir, config=cfg)
    try:
        population = solver._marl_initial_population()
        distances = [solver._tour_distance(individual) for individual in population]
        print(distances)
    finally:
        solver._stop_logging()

    if distances:
        best = min(distances)
        worst = max(distances)
        avg = sum(distances) / len(distances)
    else:
        best = worst = avg = float("inf")

    print(
        "[MARL Init Debug] individuals=%d best=%.3f worst=%.3f mean=%.3f"
        % (len(population), best, worst, avg)
    )
    return [ind.tolist() for ind in population]

if __name__ == "__main__":
    from config import GAMainConfig

    config = GAMainConfig()
    tsplib_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances", "tsplib")
    print(config.instances)
    for instance in config.instances:
        graph_path = os.path.join(tsplib_root, instance)
        graph = Graph.from_tsplib(graph_path)
        print(f"Loaded graph {instance} with {graph.n_nodes} nodes.")

        base_ga_config = GeneticAlgorithmConfig(**config.ga_config_kwargs)
        debug_marl_initial_population(
            graph,
            base_ga_config,
            save_dir=os.path.join(config.log_root, f"debug_marl_init_{instance.split('.')[0]}"),
        )