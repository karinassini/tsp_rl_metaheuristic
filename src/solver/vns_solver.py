import logging
import random
import time
import numpy as np
from src.structures.graph import Graph
from src.constraints.tsp_constraint import TSPConstraint

METHOD = "greedy"  # Change to "random" to use a random initial solution

logging.basicConfig(level=logging.INFO)
timestamp = time.strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(f"vns_steps_{timestamp}_{METHOD}.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

class VNS_Solver:
    """
    A class to solve the Traveling Salesman Problem (TSP) using Variable Neighborhood Search (VNS).
    Accepts a Graph object for flexibility.
    """
    def __init__(self, graph: Graph):
        """
        Initialize the VNS solver with a Graph object.
        :param graph: Graph instance representing the TSP.
        """
        self.graph = graph
        self.distance_matrix = graph.get_distance_matrix()
        self.n_cities = graph.n_nodes
        self.constraint = TSPConstraint(self.n_cities)

    @staticmethod
    def tour_distance(tour, dist_matrix):
        tour_shifted = np.roll(tour, -1)
        return np.sum(np.asarray(dist_matrix)[tour, tour_shifted])

    @staticmethod
    def two_opt(tour, i, j):
        new_tour = np.concatenate((tour[:i], tour[i:j+1][::-1], tour[j+1:]))
        return new_tour

    @staticmethod
    def three_opt(tour, i, j, k):
        new_tour = np.concatenate((tour[:i], tour[j:k], tour[i:j], tour[k:]))
        return new_tour

    def local_search(self, tour, operator):
        better_solution_found = True
        original_distance = self.tour_distance(tour, self.distance_matrix)
        while better_solution_found:
            better_solution_found = False
            for i in range(1, len(tour) - 1):
                for j in range(i+1, len(tour)):
                    if j-i == 1: 
                        continue
                    new_tour = operator(tour, i, j)
                    new_distance = self.tour_distance(new_tour, self.distance_matrix)
                    if new_distance < original_distance:
                        tour = new_tour
                        original_distance = new_distance
                        better_solution_found = True
        return tour

    def shaking(self, tour, k):
        new_tour = tour.copy()
        for _ in range(k):
            i, j = sorted(random.sample(range(1, len(tour)-1), 2))
            new_tour = self.two_opt(new_tour, i, j)
        return new_tour

    def vns_solve(self, start=0, k_max=500, operator=None):
        """
        Solve the TSP using Variable Neighborhood Search (VNS) and log the process.
        :param start: Index of the starting city.
        :param k_max: Maximum number of neighborhoods.
        :param operator: Local search operator (default: two_opt).
        :return: (tour, total_distance, total_exploration_time, total_exploitation_time)
        """
        if operator is None:
            operator = self.two_opt
        # Choose initialization method: "greedy" or "random"
        if METHOD == "greedy":
            # Greedy initial solution: starting from the given start or a random city
            start_city = start if start is not None else random.randint(0, self.n_cities - 1)
            unvisited = list(range(self.n_cities))
            tour = [start_city]
            unvisited.remove(start_city)
            current_city = start_city
            while unvisited:
                # Find the nearest unvisited city
                next_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
                tour.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
            tour.append(tour[0])  # complete the cycle
            tour = np.array(tour)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            logging.info(f"Initial greedy tour: {tour.tolist()}, distance: {total_distance:.2f}")

        else:
            # Random initial solution ensuring a valid tour (cycle)
            tour = np.arange(self.n_cities)
            np.random.shuffle(tour)
            tour = np.append(tour, tour[0])
            tour = np.array(tour)
        total_distance = self.tour_distance(tour, self.distance_matrix)
        logging.info(f"Initial {METHOD} tour: {tour.tolist()}, distance: {total_distance:.2f}")

        k = 1
        total_exploration_time = 0
        total_exploitation_time = 0
        iteration = 1
        while k <= k_max:
            logging.info(f"Iteration {iteration} - k: {k}")
            time_start = time.time()
            k_tour = self.shaking(tour, k)
            time_end = time.time()
            exploration_time = time_end - time_start
            total_exploration_time += exploration_time
            logging.info(f"Shaking phase completed in {exploration_time:.4f} seconds.")
            time_start = time.time()
            new_tour = self.local_search(k_tour, operator)
            time_end = time.time()
            exploitation_time = time_end - time_start
            total_exploitation_time += exploitation_time
            logging.info(f"Local search phase completed in {exploitation_time:.4f} seconds.")
            old_distance = self.tour_distance(tour, self.distance_matrix)
            new_distance = self.tour_distance(new_tour, self.distance_matrix)
            logging.info(f"Old distance: {old_distance:.2f}, New distance: {new_distance:.2f}")
            if new_distance < old_distance:
                logging.info("Found an improved tour. Resetting k to 1.")
                tour = new_tour
                k = 1
            else:
                k += 1
                logging.info("No improvement, increasing k.")
            iteration += 1
        if not self.constraint.is_valid_tour(tour):
            logging.warning("Solution does not satisfy TSP constraints.")
        total_distance = self.tour_distance(tour, self.distance_matrix)
        logging.info(f"Final tour: {tour.tolist()}, total distance: {total_distance:.2f}")
        total_distance = float(total_distance)
        total_exploration_time = float(total_exploration_time)
        total_exploitation_time = float(total_exploitation_time)
        return tour.tolist(), total_distance, total_exploration_time, total_exploitation_time
