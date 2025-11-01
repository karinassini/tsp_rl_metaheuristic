from src.structures.graph import Graph
from src.constraints.tsp_constraint import TSPConstraint
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TSPSolver:
    """
    A class to solve the Traveling Salesman Problem (TSP) using greedy algorithms.
    Accepts a Graph object for flexibility.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the TSP solver with a Graph object.
        :param graph: Graph instance representing the TSP.
        """
        self.graph = graph
        self.distance_matrix = graph.get_distance_matrix()
        self.n_cities = graph.n_nodes
        self.constraint = TSPConstraint(self.n_cities)

    def greedy_solve(self, start=0):
        """
        Solve the TSP using a greedy nearest neighbor heuristic.
        :param start: Index of the starting city.
        :return: (tour, total_distance)
        """
        visited = [False] * self.n_cities
        tour = [start]
        visited[start] = True
        total_distance = 0
        current_city = start

        for initial_city in range(self.n_cities - 1):
            next_city = None
            min_dist = float("inf")
            for city in range(initial_city + 1, self.n_cities):
                if (
                    not visited[city]
                    and self.distance_matrix[current_city][city] < min_dist
                ):
                    min_dist = self.distance_matrix[current_city][city]
                    next_city = city
            if next_city is None:
                break
            tour.append(next_city)
            visited[next_city] = True
            total_distance += min_dist
            current_city = next_city

        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        tour.append(start)

        # Check constraints
        if not self.constraint.is_valid_tour(tour):
            logger.warning("Solution does not satisfy TSP constraints.")
        if isinstance(total_distance, np.int64):
            total_distance = int(total_distance)
        return tour, total_distance
