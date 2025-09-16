import sys
import os

from src.structures.graph import Graph
from src.solver.greedy_solver import TSPSolver
from src.solver.solution_saver import SolutionSaver
from src.solver.exact_solver import TSPSolver as ExactTSPSolver
from src.solver.vns_solver import VNS_Solver

# Add the 'src' directory to the Python module search path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def test_exact(graph, instance):

    name_without_extension = instance.split('.')[0]
    dist = graph.build_distance_dict()
    solver = ExactTSPSolver(dist, graph.nodes)
    solver.model.update()
    output_dir = f"outputs/solutions/{name_without_extension}/exact"
    os.makedirs(output_dir, exist_ok=True)

    solver.solve()
    solver.get_solution()
    solver.save_solution(f"{output_dir}/solution.json")


def test_greedy_solver(graph, instance):
    solver = TSPSolver(graph)
    tour, total_distance = solver.greedy_solve(start=0)
    print("Tour:", tour)
    print("Total distance:", total_distance)

    name_without_extension = instance.split('.')[0]
    saver = SolutionSaver(instance_name=name_without_extension, save_dir=f"outputs/solutions/{name_without_extension}/greedy/")
    saver.save(tour, total_distance)      # saves as JSON
    saver.plot_solution_graph(tour, graph.get_distance_matrix(), filename=f"outputs/solutions/{name_without_extension}/greedy/solution_graph.png")


def test_vns_solver(graph, instance, save_dir=None, method="greedy", k_max=None):
    name_without_extension = instance.split('.')[0]
    save_dir = save_dir if save_dir else f"outputs/solutions/{name_without_extension}/vns/"
    if k_max is not None:
        k_max = k_max
    else:
        k_max = 500 if graph.n_nodes <= 50 else 1000
    solver = VNS_Solver(graph, save_dir, method=method)
    tour, total_distance, exploration_time, exploitation_time = solver.vns_solve(start=0, k_max=k_max)
    print("VNS Tour:", tour)
    print("VNS Total distance:", total_distance)
    print("VNS Exploration time:", exploration_time)
    print("VNS Exploitation time:", exploitation_time)

    
    saver = SolutionSaver(instance_name=name_without_extension, save_dir=save_dir)
    saver.save(tour, total_distance,     exploration_time=exploration_time,
    exploitation_time=exploitation_time,
    k_max=k_max, method=method)      # saves as JSON
    #saver.plot_solution_graph(tour, graph.get_distance_matrix(), filename=f"outputs/solutions/{name_without_extension}/vns/solution_graph.png")


def main():

    # Creating the graph object
    #instance = "dj38_simplified.tsp"
    #graph = Graph.from_tsplib_math(f'instances/tsplib/{instance}')
    instance = "swiss42.tsp"
    graph = Graph.from_tsplib(f'instances/tsplib/{instance}')

    # Print nodes and number of edges
    print("Nodes:", graph.nodes)
    print("Number of edges:", len(graph.edges))
    print(graph) 

    #graph.visualize(save_dir=f"outputs/plots/{instance}_first_plot")
    #test_greedy_solver(graph, instance)
    #test_exact(graph, instance)

    custom_method = "greedy"  
    k_max = 500  
    for _ in range(30):
        test_vns_solver(graph, instance, k_max=k_max, method=custom_method)

if __name__ == "__main__":
    main()





