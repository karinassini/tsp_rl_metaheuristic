import sys
import os

from src.structures.graph import Graph
from src.solver.greedy_solver import TSPSolver
from src.solver.solution_saver import SolutionSaver
from src.solver.exact_solver import TSPSolver as ExactTSPSolver

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
    saver = SolutionSaver(save_dir=f"outputs/solutions/{name_without_extension}/greedy/")
    saver.save(tour, total_distance)      # saves as JSON
    saver.plot_solution_graph(tour, graph.get_distance_matrix(), filename=f"outputs/solutions/{name_without_extension}/greedy/solution_graph.png")



def main():

    # Creating the graph object
    instance = "dj38_simplified.tsp"
    graph = Graph.from_tsplib(f'instances/tsplib/{instance}')

    # Print nodes and number of edges
    print("Nodes:", graph.nodes)
    print("Number of edges:", len(graph.edges))
    print(graph)  # Uses __repr__

    #graph.visualize(save_dir=f"outputs/plots/{instance}_first_plot")
    test_greedy_solver(graph, instance)
    test_exact(graph, instance)

if __name__ == "__main__":
    main()





