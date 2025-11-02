import logging
import os
import random
import time
import numpy as np
from src.structures.graph import Graph
from src.constraints.tsp_constraint import TSPConstraint
from src.solver.initial_solution import nearest_neighbour_tour, q_learning_tour, QLearningConfig


class VNS_Solver:
    """
    A class to solve the Traveling Salesman Problem (TSP) using Variable Neighborhood Search (VNS).
    Accepts a Graph object for flexibility.
    """

    def __init__(
        self,
        graph: Graph,
        save_dir: str = None,
        method="random",
        max_double_bridge_checks: int = 250,
    ):
        """
        Initialize the VNS solver with a Graph object.
        :param graph: Graph instance representing the TSP.
        :param max_double_bridge_checks: Maximum sampled double-bridge moves evaluated per local-search call.
        """
        self.graph = graph
        # Store distance matrix once as NumPy array to avoid repeated conversions during evaluation.
        self.distance_matrix = np.asarray(graph.get_distance_matrix(), dtype=float)
        self.n_cities = graph.n_nodes
        self.constraint = TSPConstraint(self.n_cities)
        self.save_dir = save_dir
        self.method = method  # Change to "random" to use a random initial solution
        self.max_double_bridge_checks = max(0, max_double_bridge_checks)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.save_dir:
            logs_dir = os.path.join(self.save_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(
                logs_dir, f"vns_steps_{timestamp}_{self.method}.log"
            )
        else:
            log_path = f"vns_steps_{timestamp}_{self.method}.log"
        self.logger = logging.getLogger(f"vns_solver.{timestamp}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @staticmethod
    def tour_distance(tour, dist_matrix):
        tour = np.asarray(tour, dtype=int)
        tour_shifted = np.roll(tour, -1)
        lower = np.minimum(tour, tour_shifted)
        upper = np.maximum(tour, tour_shifted)
        return float(np.sum(dist_matrix[lower, upper]))

    @staticmethod
    def two_opt(tour, i, j):
        new_tour = np.concatenate((tour[:i], tour[i : j + 1][::-1], tour[j + 1 :]))
        return new_tour

    @staticmethod
    def three_opt(tour, i, j, k):
        new_tour = np.concatenate((tour[:i], tour[j:k], tour[i:j], tour[k:]))
        return new_tour

    @staticmethod
    def two_exchange(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Swap two vertex positions in the tour (2-exchange move)."""
        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        new_core = core.copy()
        new_core[i], new_core[j] = new_core[j], new_core[i]
        if is_closed:
            return np.concatenate((new_core, [new_core[0]]))
        return new_core

    @staticmethod
    def one_insertion(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Remove the vertex at position i and insert it before position j."""
        if i == j:
            return tour.copy()

        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        new_core = core.copy()
        city = new_core[i]
        new_core = np.delete(new_core, i)
        if j > i:
            j -= 1
        new_core = np.insert(new_core, j, city)
        if is_closed:
            return np.concatenate((new_core, [new_core[0]]))
        return new_core

    @staticmethod
    def double_bridge_move(
        tour: np.ndarray, a: int, b: int, c: int, d: int
    ) -> np.ndarray:
        """Apply a double-bridge move defined by four cut indices."""
        if len(tour) < 6:
            return tour.copy()

        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        n = len(core)
        if n < 6:
            return tour.copy()

        segment_1 = core[:a]
        segment_2 = core[a:b]
        segment_3 = core[b:c]
        segment_4 = core[c:d]
        segment_5 = core[d:]

        new_core = np.concatenate(
            (segment_1, segment_3, segment_2, segment_4, segment_5)
        )
        if is_closed:
            new_core = np.concatenate((new_core, [new_core[0]]))
        return new_core

    @classmethod
    def random_double_bridge(cls, tour: np.ndarray) -> np.ndarray:
        """Generate a random double-bridge perturbation."""
        is_closed = tour[0] == tour[-1]
        core_length = len(tour) - 1 if is_closed else len(tour)
        if core_length < 6:
            return tour.copy()

        a, b, c, d = sorted(random.sample(range(1, core_length), 4))
        return cls.double_bridge_move(tour, a, b, c, d)

    def local_search(
        self, tour: np.ndarray, current_distance: float, operators=None
    ) -> np.ndarray:
        """Variable Neighborhood Descent over 2-opt, 2-exchange, and double bridge moves."""
        if operators is None:
            operators = [
                self._two_opt_first_improvement,
                self._one_insertion_first_improvement,
                self._two_exchange_first_improvement,
                self._double_bridge_first_improvement,
            ]

        current_tour = tour.copy()
        improved = True
        i = 0
        while improved:
            improved = False
            for operator in operators:
                self.logger.info(
                    f"Starting local search iteration trial {i} operator {operator.__name__}"
                )
                i += 1
                old_distance = current_distance
                candidate, was_improved, current_distance = operator(
                    current_tour, current_distance
                )
                if was_improved:
                    self.logger.info(
                        f"Local search improvement found with {operator.__name__} -> old_distance: {old_distance:.2f} vs current_distance: {current_distance:.2f}"
                    )
                    current_tour = candidate
                    improved = True
                    break  # restart from the first operator in the next iteration
        return current_tour

    def _two_opt_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        """Return the first improving 2-opt move using incremental edge deltas.

        The tour may be open or explicitly closed by repeating the starting city. The
        inner loops enumerate non-adjacent edge pairs (a,b) and (c,d), skipping
        consecutive indices so that the swap remains valid. For each candidate swap we
        reuse the upper-triangular distance matrix and compute the local delta
        ``(a,b) + (c,d) -> (a,c) + (b,d)`` without recomputing the full tour length. If a
        negative delta is found the segment [i,j] is reversed, and the updated tour and
        distance are returned immediately.
        """
        self.logger.info("2-opt start distance: %.2f", current_distance)
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)

        def edge_cost(u: int, v: int) -> float:
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        core = tour[:-1] if is_closed else tour

        for i in range(1, length - 1):
            for j in range(i + 1, length):
                if j - i == 1:
                    continue

                a, b = core[i - 1], core[i]
                c, d = core[j], core[(j + 1) % length]

                removed = edge_cost(a, b) + edge_cost(c, d)
                added = edge_cost(a, c) + edge_cost(b, d)
                delta = added - removed

                if delta < 0:
                    new_distance = current_distance + delta
                    candidate = self.two_opt(tour, i, j)
                    return candidate, True, new_distance

        return tour, False, current_distance

    def _two_exchange_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool]:
        self.logger.info("2-exchange start distance: %.2f", current_distance)
        length = len(tour) - 1 if tour[0] == tour[-1] else len(tour)
        if length < 3:
            return tour, False

        for i in range(1, length - 1):
            for j in range(i + 1, length):
                candidate = self.two_exchange(tour, i, j)
                candidate_distance = self.tour_distance(candidate, self.distance_matrix)
                if candidate_distance < current_distance:
                    return candidate, True, candidate_distance
        return tour, False, current_distance

    def _one_insertion_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool]:
        self.logger.info("1-insertion start distance: %.2f", current_distance)
        length = len(tour) - 1 if tour[0] == tour[-1] else len(tour)
        if length < 3:
            return tour, False, current_distance

        for i in range(1, length):
            for j in range(1, length + 1):
                if i == j or j == i + 1:
                    continue
                candidate = self.one_insertion(tour, i, j)
                candidate_distance = self.tour_distance(candidate, self.distance_matrix)
                if candidate_distance < current_distance:
                    return candidate, True, candidate_distance
        return tour, False, current_distance

    def _double_bridge_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool]:
        self.logger.info("Double-bridge start distance: %.2f", current_distance)
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 6:
            return tour, False, current_distance

        # Exhaustive enumeration is O(n^4). Instead, sample a bounded number of candidate quadruples.
        max_checks = self.max_double_bridge_checks
        if max_checks == 0:
            return tour, False, current_distance

        indices = list(range(1, length))
        tried: set[tuple[int, int, int, int]] = set()
        checks = 0

        # We keep sampling unique quadruples until we reach the configured limit.
        while checks < max_checks and len(tried) < max_checks:
            a, b, c, d = sorted(random.sample(indices, 4))
            key = (a, b, c, d)
            if key in tried:
                continue
            tried.add(key)
            checks += 1

            candidate = self.double_bridge_move(tour, a, b, c, d)
            candidate_distance = self.tour_distance(candidate, self.distance_matrix)
            if candidate_distance < current_distance:
                return candidate, True, candidate_distance
        return tour, False, current_distance

    def shaking(self, tour, k):
        new_tour = tour.copy()
        for _ in range(k):
            i, j = sorted(random.sample(range(1, len(tour) - 1), 2))
            new_tour = self.two_opt(new_tour, i, j)
        if k >= 2:
            new_tour = self.random_double_bridge(new_tour)
        return new_tour

    def vns_solve(self, start=0, k_max=500, no_improvement_patience: int | None = None):
        """
        Solve the TSP using Variable Neighborhood Search (VNS) and log the process.
        :param start: Index of the starting city.
        :param k_max: Maximum number of neighborhoods.
        :param no_improvement_patience: Optional cap on consecutive non-improving iterations before early stop.
        :return: (tour, total_distance, total_exploration_time, total_exploitation_time)
        """
        method_key = self.method.lower() if isinstance(self.method, str) else "random"
        if method_key in {"greedy", "nearest_neighbour", "nearest_neighbor", "nrnbr"}:
            seed = start if start is not None else random.randint(0, self.n_cities - 1)
            tour = nearest_neighbour_tour(self.graph, start=seed)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial {method_key} tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
        elif method_key in {"q_learning", "qlearning", "q-learn"}:
            tour, full_distance_matrix = q_learning_tour(self.graph, start=start)
            total_distance = self.tour_distance(tour, full_distance_matrix)
            self.logger.info(
                f"Initial q_learning tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
        elif method_key == "random":
            tour = np.arange(self.n_cities)
            np.random.shuffle(tour)
            tour = np.append(tour, tour[0])
            tour = np.array(tour)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial random tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
        else:
            raise ValueError(
                f"Unknown initialisation method '{self.method}'. "
                "Supported values: 'greedy', 'nearest_neighbour', 'random', 'q_learning'."
            )

        k = 1
        total_exploration_time = 0
        total_exploitation_time = 0
        iteration = 1
        consecutive_non_improvements = 0
        patience = None
        if no_improvement_patience is not None:
            patience = max(1, int(no_improvement_patience))
            self.logger.info(
                "Early stopping enabled: patience set to %d consecutive non-improving iterations.",
                patience,
            )
        while k <= k_max:
            self.logger.info(f"Iteration {iteration} - k: {k}")
            time_start = time.time()
            k_tour = self.shaking(tour, k)
            time_end = time.time()
            exploration_time = time_end - time_start
            total_exploration_time += exploration_time
            shaken_distance = self.tour_distance(k_tour, self.distance_matrix)
            self.logger.info(
                "Shaking phase completed in %.4f seconds (distance %.2f)",
                exploration_time,
                shaken_distance,
            )
            time_start = time.time()
            new_tour = self.local_search(k_tour, shaken_distance)
            time_end = time.time()
            exploitation_time = time_end - time_start
            total_exploitation_time += exploitation_time
            self.logger.info(
                f"Local search phase completed in {exploitation_time:.4f} seconds."
            )
            old_distance = self.tour_distance(tour, self.distance_matrix)
            new_distance = self.tour_distance(new_tour, self.distance_matrix)
            self.logger.info(
                f"Old distance: {old_distance:.2f}, New distance: {new_distance:.2f}"
            )
            if new_distance < old_distance:
                self.logger.info("Found an improved tour. Resetting k to 1.")
                tour = new_tour
                k = 1
                consecutive_non_improvements = 0
            else:
                consecutive_non_improvements += 1
                k += 1
                self.logger.info("No improvement, increasing k.")
                if patience is not None and consecutive_non_improvements >= patience:
                    self.logger.info(
                        "Early stopping triggered after %d consecutive non-improving iterations.",
                        consecutive_non_improvements,
                    )
                    break
            iteration += 1
        if not self.constraint.is_valid_tour(tour):
            self.logger.warning("Solution does not satisfy TSP constraints.")
        total_distance = self.tour_distance(tour, self.distance_matrix)
        self.logger.info(
            f"Final tour: {tour.tolist()}, total distance: {total_distance:.2f}"
        )
        total_distance = float(total_distance)
        total_exploration_time = float(total_exploration_time)
        total_exploitation_time = float(total_exploitation_time)
        return (
            tour.tolist(),
            total_distance,
            total_exploration_time,
            total_exploitation_time,
        )
