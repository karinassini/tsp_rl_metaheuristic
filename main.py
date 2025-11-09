import sys
import os

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


def test_vns_solver(
    graph,
    instance,
    save_dir=None,
    method="greedy",
    max_non_improving_iterations=20,
    iteration_max=None,
    k_max=2,
    start_city=0,
    current_timestamp=None,

):
    """Run a single VNS experiment and persist metrics/plots.

    The helper builds a save directory, executes the solver, stores the JSON
    solution via `SolutionSaver`, updates the rolling `VNSSummaryTracker`, and
    renders both the objective trend and time-to-target plots (when the best
    reference is available).
    """
    name_without_extension = instance.split(".")[0]
    save_dir = (
        save_dir
        if save_dir
        else f"outputs/solutions/{name_without_extension}/vns/{method}/{current_timestamp}/"
    )
    if iteration_max is None:
        iteration_max = 500 if graph.n_nodes <= 50 else 1000
    solver = VNS_Solver(graph, save_dir, method=method)
    solver_name = type(solver).__name__
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

    saver = SolutionSaver(instance_name=name_without_extension, save_dir=save_dir)
    saver.save(
        tour,
        total_distance,
        exploration_time=exploration_time,
        exploitation_time=exploitation_time,
        iteration_max=iteration_max,
        method=method,
        solver=solver_name,
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
    )

    plot_path = SUMMARY_TRACKER.plot_objective_trend(
        instance_name=name_without_extension,
        method=method,
        iteration_max=iteration_max,
        save_dir=save_dir,
        solver_name=solver_name,
    )

    try:
        ttt_plot_path = SUMMARY_TRACKER.plot_time_to_target(
            instance_name=name_without_extension,
            method=method,
            iteration_max=iteration_max,
            save_dir=save_dir,
            solver_name=solver_name,
            store_key=method,
        )
    except ValueError as exc:
        ttt_plot_path = None
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

    for instance in config.instances:
        print(f"Processing instance: {instance}")
        graph = Graph.from_tsplib(f"{root}/instances/tsplib/{instance}")

        print("Nodes:", graph.nodes)
        print("Number of edges:", len(graph.edges))

        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for _ in range(config.repeats):
            test_vns_solver(
                graph,
                instance,
                save_dir=config.save_dir,
                iteration_max=config.iteration_max,
                method=config.method,
                current_timestamp=current_timestamp,
                max_non_improving_iterations=config.max_non_improving_iterations,
                k_max=config.k_max,
                start_city=config.start_city,
            )

    # print(f"outputs/solutions/{instance.split('.')[0]}/")
    # #plot_solution_comparison(f"outputs/solutions/{instance.split('.')[0]}/vns/")


if __name__ == "__main__":
    main()
