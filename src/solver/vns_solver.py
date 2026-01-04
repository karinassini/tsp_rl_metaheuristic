import logging
import os
import random
import time
import itertools
from queue import Queue
from dataclasses import asdict

import numpy as np
from logging.handlers import QueueHandler, QueueListener

from src.structures.graph import Graph
from src.constraints.tsp_constraint import TSPConstraint
from src.solver.initial_solution import (
    QLearningConfig,
    nearest_neighbour_tour,
    q_learning_tour,
    rcl_nearest_neighbour_tour,
    marl_initial_population,
)


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
        restricted_two_opt_max_span: int = 8,
        max_flip_subsequence_length: int = 5,
        max_inversion_segment_length: int = 5,
        q_learning_cfg: QLearningConfig | None = None,
        marl_params: dict | None = None,
        best_known_distance: float | None = None,
        timestamp: str | None = None,
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
        self.restricted_two_opt_max_span = max(2, restricted_two_opt_max_span)
        self.max_flip_subsequence_length = max(2, max_flip_subsequence_length)
        self.max_inversion_segment_length = max(2, max_inversion_segment_length)
        self.q_learning_cfg = q_learning_cfg
        self.marl_params = marl_params or {}
        self.best_known_distance = best_known_distance
        self.best_known_hit_iteration: int | None = None
        self.iterations_executed: int = 0
        self.timestamp = timestamp if timestamp is not None else time.strftime("%Y%m%d_%H%M%S")

        if self.save_dir:
            logs_dir = os.path.join(self.save_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(
                logs_dir, f"vns_steps_{timestamp}_{self.method}_{os.getpid()}_{id(self)}.log"
            )
        else:
            log_path = f"vns_steps_{self.timestamp}_{self.method}_{os.getpid()}_{id(self)}.log"
        self.logger = logging.getLogger(f"vns_solver.{self.timestamp}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        self._log_queue = Queue(maxsize=0)
        self._queue_handler = QueueHandler(self._log_queue)
        self._queue_handler.setLevel(logging.INFO)
        self.logger.addHandler(self._queue_handler)

        self._queue_listener = QueueListener(self._log_queue, file_handler)
        self._queue_listener.start()
        self._queue_listener_stopped = False

    def export_parameters(self) -> dict[str, object]:
        """Expose core solver configuration for persistence/debugging."""
        return {
            "method": self.method,
            "max_double_bridge_checks": self.max_double_bridge_checks,
            "restricted_two_opt_max_span": self.restricted_two_opt_max_span,
            "max_flip_subsequence_length": self.max_flip_subsequence_length,
            "max_inversion_segment_length": self.max_inversion_segment_length,
            "best_known_distance": self.best_known_distance,
            "q_learning_cfg": asdict(self.q_learning_cfg) if self.q_learning_cfg else None,
        }

    def _stop_logging(self) -> None:
        """Flush and stop the asynchronous logging listener."""
        if getattr(self, "_queue_listener_stopped", True):
            return
        try:
            self._queue_listener.stop()
        finally:
            for handler in getattr(self._queue_listener, "handlers", ()):  # pragma: no branch
                try:
                    handler.close()
                except Exception:  # pragma: no cover - best effort clean-up
                    pass
            self._queue_listener = None
            if getattr(self, "_queue_handler", None) is not None:
                try:
                    self.logger.removeHandler(self._queue_handler)
                except ValueError:
                    pass
                try:
                    self._queue_handler.close()
                except Exception:
                    pass
                self._queue_handler = None
            self._log_queue = None
            self._queue_listener_stopped = True

    def _record_best_known_hit(self, distance: float, iteration_marker: int) -> None:
        """Persist the first iteration when the best-known distance is matched."""
        if self.best_known_distance is None:
            return
        if self.best_known_hit_iteration is not None:
            return
        if abs(distance - self.best_known_distance) <= 1e-6:
            self.best_known_hit_iteration = iteration_marker

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self._stop_logging()
        except Exception:
            pass

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
                self._limited_subsequence_flip_first_improvement,
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
                old_distance_ls = current_distance
                candidate, was_improved, current_distance = operator(
                    current_tour, current_distance
                )
                if was_improved:
                    self.logger.info(
                        f"Local search improvement found with {operator.__name__} -> old_distance: {old_distance_ls:.2f} vs current_distance: {current_distance:.2f}"
                    )
                    current_tour = candidate
                    improved = True
                    break  # restart from the first operator in the next iteration
        return current_tour, current_distance

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
    ) -> tuple[np.ndarray, bool, float]:
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 3:
            return tour, False, current_distance

        def edge_cost(u: int | None, v: int | None) -> float:
            if u is None or v is None:
                return 0.0
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        core = tour[:-1] if is_closed else tour

        for i in range(1, length - 1):
            for j in range(i + 1, length):
                city_i = core[i]
                city_j = core[j]

                prev_i = core[i - 1] if i > 0 else (core[-1] if is_closed else None)
                next_i = core[(i + 1) % length] if (is_closed or i + 1 < length) else None

                prev_j = core[j - 1] if j > 0 else (core[-1] if is_closed else None)
                next_j = core[(j + 1) % length] if (is_closed or j + 1 < length) else None

                adjacent = j == i + 1
                if adjacent:
                    removed = edge_cost(prev_i, city_i) + edge_cost(city_j, next_j)
                    added = edge_cost(prev_i, city_j) + edge_cost(city_i, next_j)
                else:
                    removed = (
                        edge_cost(prev_i, city_i)
                        + edge_cost(city_i, next_i)
                        + edge_cost(prev_j, city_j)
                        + edge_cost(city_j, next_j)
                    )
                    added = (
                        edge_cost(prev_i, city_j)
                        + edge_cost(city_j, next_i)
                        + edge_cost(prev_j, city_i)
                        + edge_cost(city_i, next_j)
                    )

                delta = added - removed
                if delta < 0:
                    new_distance = current_distance + delta
                    candidate = self.two_exchange(tour, i, j)
                    return candidate, True, new_distance

        return tour, False, current_distance

    def _one_insertion_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 3:
            return tour, False, current_distance

        def edge_cost(u: int, v: int) -> float:
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        core = tour[:-1] if is_closed else tour

        for i in range(1, length):
            city = core[i]
            prev_i = core[i - 1] if i > 0 else (core[-1] if is_closed else None)
            next_i = core[(i + 1) % length] if (is_closed or i + 1 < length) else None

            removal_delta = 0.0
            if prev_i is not None:
                removal_delta -= edge_cost(prev_i, city)
            if next_i is not None:
                removal_delta -= edge_cost(city, next_i)
                if prev_i is not None:
                    removal_delta += edge_cost(prev_i, next_i)

            core_removed = np.delete(core, i)
            reduced_len = len(core_removed)
            if reduced_len == 0:
                continue

            for j in range(1, length + 1):
                if i == j or j == i + 1:
                    continue

                insert_idx = j
                if insert_idx > i:
                    insert_idx -= 1

                if insert_idx < 0 or insert_idx > reduced_len:
                    continue

                if is_closed:
                    prev_new = core_removed[(insert_idx - 1) % reduced_len]
                    next_new = core_removed[insert_idx % reduced_len]
                else:
                    prev_new = core_removed[insert_idx - 1] if insert_idx > 0 else None
                    next_new = core_removed[insert_idx] if insert_idx < reduced_len else None

                insertion_delta = 0.0
                if prev_new is not None and next_new is not None:
                    insertion_delta -= edge_cost(prev_new, next_new)
                if prev_new is not None:
                    insertion_delta += edge_cost(prev_new, city)
                if next_new is not None:
                    insertion_delta += edge_cost(city, next_new)

                delta = removal_delta + insertion_delta
                if delta < 0:
                    new_distance = current_distance + delta
                    candidate = self.one_insertion(tour, i, j)
                    return candidate, True, new_distance

        return tour, False, current_distance

    def _double_bridge_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 6:
            return tour, False, current_distance

        def edge_cost(u: int | None, v: int | None) -> float:
            if u is None or v is None:
                return 0.0
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        # Exhaustive enumeration is O(n^4). Instead, sample a bounded number of candidate quadruples.
        max_checks = self.max_double_bridge_checks
        if max_checks == 0:
            return tour, False, current_distance

        core = tour[:-1] if is_closed else tour
        indices = list(range(1, length))
        all_combinations = list(itertools.combinations(indices, 4))
        selected_combination = random.sample(all_combinations, int(len(all_combinations)*min(1, max_checks / len(all_combinations))))

        # We keep sampling unique quadruples until we reach the configured limit.
        while selected_combination:
            a, b, c, d = sorted(selected_combination.pop())
            
            prev_a = core[a - 1]
            head_s2 = core[a]
            tail_s2 = core[b - 1]
            head_s3 = core[b]
            tail_s3 = core[c - 1]
            head_s4 = core[c]

            removed = (
                edge_cost(prev_a, head_s2)
                + edge_cost(tail_s2, head_s3)
                + edge_cost(tail_s3, head_s4)
            )
            added = (
                edge_cost(prev_a, head_s3)
                + edge_cost(tail_s3, head_s2)
                + edge_cost(tail_s2, head_s4)
            )

            delta = added - removed
            if delta < 0:
                new_distance = current_distance + delta
                candidate = self.double_bridge_move(tour, a, b, c, d)
                return candidate, True, new_distance
        return tour, False, current_distance

    def _restricted_two_opt_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        """First improvement 2-opt limited to short spans to focus on local noise."""

        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        span_limit = min(self.restricted_two_opt_max_span, max(2, length - 1))

        def edge_cost(u: int, v: int) -> float:
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        core = tour[:-1] if is_closed else tour

        for i in range(1, length - 1):
            max_j = min(length, i + span_limit)
            for j in range(i + 1, max_j):
                if j - i == 1:
                    continue

                a, b = core[i - 1], core[i]
                c, d = core[j], core[(j + 1) % length]

                removed = edge_cost(a, b) + edge_cost(c, d)
                added = edge_cost(a, c) + edge_cost(b, d)
                delta = added - removed

                if delta < 0:
                    candidate = self.two_opt(tour, i, j)
                    new_distance = current_distance + delta
                    return candidate, True, new_distance

        return tour, False, current_distance

    def _limited_subsequence_flip_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        """Reverse only short contiguous subsequences (<= max_flip_subsequence_length)."""

        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 4:
            return tour, False, current_distance

        max_len = min(self.max_flip_subsequence_length, length - 1)

        def edge_cost(u: int | None, v: int | None) -> float:
            if u is None or v is None:
                return 0.0
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        core = tour[:-1] if is_closed else tour

        for start in range(1, length - 1):
            local_max = min(max_len, length - start)
            for seg_len in range(2, local_max + 1):
                end = start + seg_len - 1
                prev_city = core[start - 1]
                first_city = core[start]
                last_city = core[end]
                if is_closed or end + 1 < length:
                    next_city = core[(end + 1) % length]
                else:
                    next_city = None

                removed = edge_cost(prev_city, first_city) + edge_cost(last_city, next_city)
                added = edge_cost(prev_city, last_city) + edge_cost(first_city, next_city)
                delta = added - removed

                if delta < 0:
                    candidate = self.two_opt(tour, start, end)
                    new_distance = current_distance + delta
                    return candidate, True, new_distance

        return tour, False, current_distance

    def _sample_inversion_insertion(self, tour: np.ndarray) -> np.ndarray:
        core = tour[:-1] if tour[0] == tour[-1] else tour
        length = len(core)
        if length < 5:
            return tour.copy()

        max_len = min(self.max_inversion_segment_length, length - 1)
        start = random.randrange(1, length - 1)
        seg_len = random.randrange(2, min(max_len, length - start) + 1)
        end = start + seg_len

        segment = core[start:end][::-1]
        remainder = np.concatenate((core[:start], core[end:]))

        insert_idx = random.randrange(1, remainder.size + 1)
        new_core = np.concatenate((remainder[:insert_idx], segment, remainder[insert_idx:]))
        if tour[0] == tour[-1]:
            new_core = np.concatenate((new_core, [new_core[0]]))
        return new_core

    def shaking(self, tour, k):
        new_tour = tour.copy()
        if k == 1:
            i, j = sorted(random.sample(range(1, len(tour) - 1), 2))
            new_tour = self.two_opt(new_tour, i, j)
        if k == 2:
            is_closed = new_tour[0] == new_tour[-1]
            length = len(new_tour) - 1 if is_closed else len(new_tour)
            if length > 3:
                i = random.randrange(1, length)
                j = random.randrange(1, length + 1)
                # Avoid no-op or immediate reinsertion positions.
                while j == i or j == i + 1:
                    j = random.randrange(1, length + 1)
                new_tour = self.one_insertion(new_tour, i, j)
        if k == 3:
            new_tour = self.random_double_bridge(new_tour)
        if k == 4:
            new_tour = self._sample_inversion_insertion(new_tour)
        return new_tour

    def vns_solve(
        self,
        iteration_max: int = 500,
        k_max: int = 2,
        max_non_improving_iterations: int = 10,
        start: int | None = None,
    ):
        """
        Solve the TSP using Variable Neighborhood Search (VNS) and log the process.
        :param start: Index of the starting city.
        :param iteration_max: Maximum number of outer VNS iterations.
        :param k_max: Maximum neighborhood depth for the shaking phase.
        :return: (tour, total_distance, total_exploration_time, total_exploitation_time)
        """
        method_key = self.method.lower() if isinstance(self.method, str) else "random"
        self.best_known_hit_iteration = None

        if method_key in {"greedy", "nearest_neighbour", "nearest_neighbor", "nrnbr"}:
            seed = start if start is not None else random.randint(0, self.n_cities - 1)
            tour = nearest_neighbour_tour(self.graph, start=seed)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial {method_key} tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
            self._record_best_known_hit(float(total_distance), 0)
        elif method_key in {"q_learning", "qlearning", "q-learn"}:
            tour, full_distance_matrix = q_learning_tour(
                self.graph,
                start=start,
                cfg=self.q_learning_cfg,
            )
            total_distance = self.tour_distance(tour, full_distance_matrix)
            self.logger.info(
                f"Initial q_learning tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
            self._record_best_known_hit(float(total_distance), 0)
        elif method_key == "random":
            tour = np.arange(self.n_cities)
            np.random.shuffle(tour)
            tour = np.append(tour, tour[0])
            tour = np.array(tour)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial random tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
            self._record_best_known_hit(float(total_distance), 0)
        elif method_key == "rcl":
            tour = rcl_nearest_neighbour_tour(self.graph, start=start, alpha=0.3)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial RCL tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
            self._record_best_known_hit(float(total_distance), 0)
        elif method_key == "marl":
            params = self.marl_params or {}
            rng = random.Random(params.get("seed"))
            population_size = int(params.get("population_size", max(10, self.n_cities)))
            population = marl_initial_population(
                graph=self.graph,
                rng=rng,
                population_size=population_size,
                marl_agents=int(params.get("marl_agents", 6)),
                marl_iterations=int(params.get("marl_iterations", 40)),
                marl_candidate_ratio=float(params.get("marl_candidate_ratio", 1.5)),
                marl_two_opt_passes=int(params.get("marl_two_opt_passes", 1)),
                marl_top_k=params.get("marl_top_k"),
                marl_reward=float(params.get("marl_reward", 1.0)),
                marl_learning_rate=float(params.get("marl_learning_rate", 0.4)),
                marl_softmax_beta=float(params.get("marl_softmax_beta", 2.0)),
                marl_epsilon=float(params.get("marl_epsilon", 0.15)),
            )
            if not population:
                raise ValueError("MARL initialisation produced no candidates.")

            # Pick the best tour among generated candidates since VNS is single-solution.
            best_core = min(
                population,
                key=lambda t: self.tour_distance(np.append(t, t[0]), self.distance_matrix)
                if t[0] != t[-1]
                else self.tour_distance(t, self.distance_matrix),
            )
            if best_core[0] != best_core[-1]:
                best_core = np.append(best_core, best_core[0])
            tour = best_core
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                "Initial MARL tour (best of %d): %s, distance: %.2f",
                len(population),
                tour.tolist(),
                total_distance,
            )
            self._record_best_known_hit(float(total_distance), 0)
        else:
            raise ValueError(
                f"Unknown initialisation method '{self.method}'. "
                "Supported values: 'greedy', 'nearest_neighbour', 'random', 'q_learning', 'marl'."
            )


        total_distance = float(total_distance)
        self.initial_solution = total_distance
        total_exploration_time = 0
        total_exploitation_time = 0
        iteration = 1
        last_completed_iteration = 0
        iteration_limit = max(1, int(iteration_max)) if iteration_max is not None else 1
        max_neighbourhood = max(1, int(k_max))
        consecutive_non_improving_iterations = 0
        non_improve_limit = max(1, int(max_non_improving_iterations))
        best_known_reached = self.best_known_hit_iteration is not None

        try:
            while iteration <= iteration_limit and not best_known_reached:
                self.logger.info("=== Iteration %d ===", iteration)
                k = 1
                improved_in_iteration = False

                while k <= max_neighbourhood:
                    self.logger.info("Iteration %d - neighbourhood k=%d", iteration, k)
                    time_start = time.time()
                    tour_from_shaking = self.shaking(tour, k)
                    time_end = time.time()
                    exploration_time = time_end - time_start
                    total_exploration_time += exploration_time
                    shaken_distance = self.tour_distance(tour_from_shaking, self.distance_matrix)
                    self.logger.info(
                        "Shaking phase completed in %.4f seconds (distance %.2f)",
                        exploration_time,
                        shaken_distance,
                    )

                    time_start = time.time()
                    new_tour, new_distance = self.local_search(tour_from_shaking, shaken_distance)
                    time_end = time.time()
                    exploitation_time = time_end - time_start
                    total_exploitation_time += exploitation_time
                    self.logger.info(
                        "Local search phase completed in %.4f seconds (distance %.2f)",
                        exploitation_time,
                        new_distance,
                    )

                    old_distance = self.tour_distance(tour, self.distance_matrix)
                    self.logger.info(
                        "Old distance: %.2f, New distance: %.2f", old_distance, new_distance
                    )

                    if new_distance < old_distance:
                        self.logger.info(
                            "Improvement found at k=%d. Restarting neighbourhood search from k=1.",
                            k,
                        )
                        tour = new_tour
                        total_distance = new_distance
                        self._record_best_known_hit(float(total_distance), iteration)
                        improved_in_iteration = True
                        if self.best_known_hit_iteration is not None:
                            best_known_reached = True
                            last_completed_iteration = iteration
                            break
                            print("******** reached best known ********")
                        k = 1
                        continue
                    else:
                        self.logger.info(
                            "No improvement at k=%d. Moving to the next neighbourhood.", k
                        )
                        k += 1

                if improved_in_iteration:
                    consecutive_non_improving_iterations = 0
                else:
                    consecutive_non_improving_iterations += 1
                    self.logger.info(
                        "No improvement in iteration %d (%d/%d without improvement).",
                        iteration,
                        consecutive_non_improving_iterations,
                        non_improve_limit,
                    )
                    if (
                        consecutive_non_improving_iterations
                        >= non_improve_limit
                    ):
                        self.logger.info(
                            "Terminating search after %d consecutive non-improving iterations.",
                            consecutive_non_improving_iterations,
                        )
                        last_completed_iteration = iteration
                        break

                if best_known_reached:
                    break

                last_completed_iteration = iteration
                iteration += 1

            if not self.constraint.is_valid_tour(tour):
                self.logger.warning("Solution does not satisfy TSP constraints.")
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self._record_best_known_hit(float(total_distance), last_completed_iteration)
            self.logger.info(
                f"Final tour: {tour.tolist()}, total distance: {total_distance:.2f}"
            )
            total_distance = float(total_distance)
            total_exploration_time = float(total_exploration_time)
            total_exploitation_time = float(total_exploitation_time)
            self.iterations_executed = int(last_completed_iteration)
            return (
                tour.tolist(),
                total_distance,
                total_exploration_time,
                total_exploitation_time,
            )
        finally:
            self._stop_logging()
