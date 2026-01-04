import sys
import os
from itertools import product

from src.structures.graph import Graph
from src.solver.greedy_solver import TSPSolver
from src.solver.solution_saver import SolutionSaver, VNSSummaryTracker
from src.solver.exact_solver import TSPSolver as ExactTSPSolver
from src.solver.vns_solver import VNS_Solver
from src.solver.vns_solver_q_learning import VNS_Solver_Q_Learnings
from utils.plot_comparison_from_json import SolutionComparisonPlotter
from datetime import datetime
from config import VNSMainConfig

# Add the 'src' directory to the Python module search path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

SUMMARY_TRACKER = VNSSummaryTracker()


def _ensure_iterable(value):
    if value is None:
        return [None]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def test_vns_solver(
    graph,
    instance,
    save_dir=None,
    method="greedy",
    max_non_improving_iterations=20,
    iteration_max=None,
    k_max=2,
    current_timestamp=None,
    should_plot: bool = True,
    start_city: int | None = None,
    local_search: str = 'VNS_Solver_Q_Learnings',
    q_learning_cfg=None,
    rl_local_search_cfg=None,
    marl_params=None,
):
    """Run a single VNS experiment and persist metrics/plots.

    The helper builds a save directory, executes the solver, stores the JSON
    solution via `SolutionSaver`, updates the rolling `VNSSummaryTracker`, and
    optionally renders the objective trend and time-to-target plots when
    `should_plot` is True and the best reference is available.
    """

    if local_search == 'VNS_Solver_Q_Learnings':
         local_searcher = VNS_Solver_Q_Learnings
    else:
        local_searcher = VNS_Solver

    name_without_extension = instance.split(".")[0]
    save_dir = (
        save_dir
        if save_dir
        else f"outputs/solutions/{name_without_extension}/{local_searcher.__name__}/{method}/{current_timestamp}/"
    )
    if iteration_max is None:
        iteration_max = 500 if graph.n_nodes <= 50 else 1000
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saver = SolutionSaver(instance_name=name_without_extension, save_dir=save_dir)
    
    solver_kwargs = {
        "graph": graph,
        "save_dir": save_dir,
        "method": method,
        "q_learning_cfg": q_learning_cfg,
        "best_known_distance": saver.best_total_distance,
        "timestamp": timestamp,
    }
    if method.lower() == "marl":
        solver_kwargs["marl_params"] = marl_params
    if local_searcher is VNS_Solver_Q_Learnings and rl_local_search_cfg is not None:
        solver_kwargs["rl_local_search_cfg"] = rl_local_search_cfg

    solver = local_searcher(**solver_kwargs)
    solver_name = type(solver).__name__
    solver_params = solver.export_parameters()
    
    tour, total_distance, exploration_time, exploitation_time = solver.vns_solve(
        start=start_city,
        iteration_max=iteration_max,
        k_max=k_max,
        max_non_improving_iterations=max_non_improving_iterations,
    )

    print("VNS Tour:", tour)
    print("VNS Total distance:", total_distance)
    print("VNS Exploration time:", exploration_time)
    print("VNS Exploitation time:", exploitation_time)

    saver.save(
        tour,
        total_distance,
        timestamp=timestamp,
        exploration_time=exploration_time,
        exploitation_time=exploitation_time,
        iteration_max=iteration_max,
        method=method,
        solver=solver_name,
        iteration_first_reach=solver.best_known_hit_iteration,
        iterations_executed=solver.iterations_executed,
        start_city_initialization=start_city,
        initial_solution=solver.initial_solution,
        solver_params=solver_params,
    )  # saves as JSON
    # saver.plot_solution_graph(tour, graph.get_distance_matrix(), filename=f"outputs/solutions/{name_without_extension}/vns/solution_graph.png")

    summary = SUMMARY_TRACKER.record_run(
        instance_name=name_without_extension,
        method=method,
        iteration_max=iteration_max,
        solver_name=solver_name,
        total_distance=total_distance,
        exploration_time=exploration_time,
        exploitation_time=exploitation_time,
        route=tour,
        save_dir=save_dir,
        best_known=saver.best_total_distance,
        iteration_first_reach=solver.best_known_hit_iteration,
        start_city_initialization=start_city,
    )
    if should_plot:
        try:
            SUMMARY_TRACKER.plot_objective_trend(
                instance_name=name_without_extension,
                method=method,
                iteration_max=iteration_max,
                save_dir=save_dir,
                solver_name=solver_name,
            )
            SUMMARY_TRACKER.plot_time_to_target(
                instance_name=name_without_extension,
                method=method,
                iteration_max=iteration_max,
                save_dir=save_dir,
                solver_name=solver_name,
                store_key=f"{method}|start_{'null' if start_city is None else start_city}",
            )
        except ValueError as exc:
            print(f"Skipping time-to-target plot: {exc}")

    print(
        "VNS aggregated summary: runs={runs}, mean={mean_total_distance:.4f}, "
        "std={std_total_distance:.4f}, best={best_total_distance:.4f}, "
        "best_known_hits={best_known_hits}".format(**summary)
    )


def plot_solution_comparison(directory):
    """Compare JSON solutions within `directory` using the plotting utility."""
    plotter = SolutionComparisonPlotter(directory)
    plotter.run()


def main():
    """Entry point used for quick local experiments with the available solvers."""
    config = VNSMainConfig()
    tsplib_root = os.path.join(root, "instances", "tsplib")
    graph_cache: dict[str, Graph] = {}
    for instance in config.instances:
        graph_path = os.path.join(tsplib_root, instance)
        graph = Graph.from_tsplib(graph_path)
        print(f"Loaded instance {instance}: nodes={graph.nodes}, edges={len(graph.edges)}")
        graph_cache[instance] = graph

    for instance, method, local_search in product(
        config.instances,
        config.method,
        config.local_search,
    ):
        graph = graph_cache[instance]
        print(f"Processing instance={instance}, method={method}, local_search={local_search}")
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for run_idx in range(config.repeats):
            is_last_run = run_idx == config.repeats - 1
            test_vns_solver(
                graph,
                instance,
                save_dir=config.save_dir,
                iteration_max=config.iteration_max,
                method=method,
                current_timestamp=current_timestamp,
                max_non_improving_iterations=config.max_non_improving_iterations,
                k_max=config.k_max,
                start_city=config.start_city,
                should_plot=is_last_run,
                local_search=local_search,
                q_learning_cfg=config.q_learning_cfg,
                rl_local_search_cfg=config.rl_local_search_cfg,
                marl_params=config.marl_config_kwargs,
            )

    # print(f"outputs/solutions/{instance.split('.')[0]}/")
    # #plot_solution_comparison(f"outputs/solutions/{instance.split('.')[0]}/vns/")


if __name__ == "__main__":
    main()
