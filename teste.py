import sys
import os

from src.structures.graph import Graph
from src.solver.greedy_solver import TSPSolver
from src.solver.exact_solver import TSPSolver as ExactTSPSolver
from src.solver.solution_saver import SolutionSaver

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

    #solver.model.write(f"output_dir/file.lp")



def main():

    # Creating the graph object
    instance = "dj38_simplified.tsp"
    graph = Graph.from_tsplib(f'instances/tsplib/{instance}')

    # Print nodes and number of edges
    print("Nodes:", graph.nodes)
    print("Number of edges:", len(graph.edges))
    print(graph)  # Uses __repr__

    test_exact(graph, instance)


if __name__ == "__main__":
    main()





