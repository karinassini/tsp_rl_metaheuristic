import sys
import os
from typing import List, Optional

from src.structures.graph import Graph
from src.solver.greedy_solver import TSPSolver
from src.solver.solution_saver import SolutionSaver, VNSSummaryTracker
from src.solver.exact_solver import TSPSolver as ExactTSPSolver
from src.solver.vns_solver import VNS_Solver
from src.utils.plot_comparison_from_json import SolutionComparisonPlotter
from datetime import datetime

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
    k_max=None,
    no_improvement_patience=None,
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
    if k_max is not None:
        k_max = k_max
    else:
        k_max = 500 if graph.n_nodes <= 50 else 1000
    solver = VNS_Solver(graph, save_dir, method=method)
    tour, total_distance, exploration_time, exploitation_time = solver.vns_solve(
        start=0, k_max=k_max, no_improvement_patience=no_improvement_patience
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
        k_max=k_max,
        method=method,
    )  # saves as JSON
    # saver.plot_solution_graph(tour, graph.get_distance_matrix(), filename=f"outputs/solutions/{name_without_extension}/vns/solution_graph.png")

    summary = SUMMARY_TRACKER.record_run(
        instance_name=name_without_extension,
        method=method,
        k_max=k_max,
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
        k_max=k_max,
        save_dir=save_dir,
    )

    try:
        ttt_plot_path = SUMMARY_TRACKER.plot_time_to_target(
            instance_name=name_without_extension,
            method=method,
            k_max=k_max,
            save_dir=save_dir,
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
    # Creating the graph object
    # instance = "dj38_simplified.tsp"
    # graph = Graph.from_tsplib_math(f'instances/tsplib/{instance}')
    # for i in ["swiss42.tsp", "a280.tsp", "berlin52.tsp", "ch130.tsp", "gr48.tsp", "gr120.tsp", "pcb442.tsp", "pr226.tsp", "si175.tsp"]:

    # for i in ["berlin52.tsp","gr48.tsp","gr120.tsp" , "ch150.tsp","si175.tsp","pr226.tsp","a280.tsp", "pcb442.tsp"]:
    instance = "swiss42.tsp"
    print(f"Processing instance: {instance}")
    graph = Graph.from_tsplib(f"{root}/instances/tsplib/{instance}")

    # Print nodes and number of edges
    print("Nodes:", graph.nodes)
    print("Number of edges:", len(graph.edges))

    # graph.visualize(save_dir=f"outputs/plots/{instance}_first_plot")
    # test_greedy_solver(graph, instance)
    # test_exact(graph, instance)

    custom_method = "random"
    k_max = 200
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(30):
        test_vns_solver(
            graph,
            instance,
            k_max=k_max,
            method=custom_method,
            no_improvement_patience=25,
            current_timestamp=current_timestamp,
        )

    # print(f"outputs/solutions/{instance.split('.')[0]}/")
    # #plot_solution_comparison(f"outputs/solutions/{instance.split('.')[0]}/vns/")


if __name__ == "__main__":
    main()
