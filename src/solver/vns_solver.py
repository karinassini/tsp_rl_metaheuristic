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

from src.solver.local_search_operators import LocalSearchOperatorsMixin

class ShakeOp:
    def __init__(self, func, name):
        self.func = func
        self.name = name
    def __call__(self, tour):
        return self.func(tour)
    

class VNS_Solver(LocalSearchOperatorsMixin):
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
        segment_len_val: int | None = None,
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
        self.segment_len_val = segment_len_val
        self.restricted_two_opt_max_span = max(2, restricted_two_opt_max_span)
        self.max_flip_subsequence_length = max(2, max_flip_subsequence_length)
        self.max_inversion_segment_length = max(2, max_inversion_segment_length)
        self.q_learning_cfg = q_learning_cfg
        self.marl_params = marl_params or {}
        self.best_known_distance = best_known_distance
        self.best_known_hit_iteration: int | None = None
        self.iterations_executed: int = 0
        self.initial_solution_time: float | None = None
        self.timestamp = (
            timestamp if timestamp is not None else time.strftime("%Y%m%d_%H%M%S")
        )

        if self.save_dir:
            logs_dir = os.path.join(self.save_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(
                logs_dir,
                f"vns_steps_{timestamp}_{self.method}_{os.getpid()}_{id(self)}.log",
            )
        else:
            log_path = (
                f"vns_steps_{self.timestamp}_{self.method}_{os.getpid()}_{id(self)}.log"
            )
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
            "segment_len_val": self.segment_len_val,
            "best_known_distance": self.best_known_distance,
            "q_learning_cfg": (
                asdict(self.q_learning_cfg) if self.q_learning_cfg else None
            ),
            "marl_params": self.marl_params,
        }

    def _stop_logging(self) -> None:
        """Flush and stop the asynchronous logging listener."""
        if getattr(self, "_queue_listener_stopped", True):
            return
        try:
            self._queue_listener.stop()
        finally:
            for handler in getattr(
                self._queue_listener, "handlers", ()
            ):  # pragma: no branch
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

    def local_search(
        self, tour: np.ndarray, current_distance: float, operators=None
    ) -> np.ndarray:
        """Variable Neighborhood Descent over 2-opt, 2-exchange, and double bridge moves."""
        if operators is None:
            operators = [
                self._two_opt_first_improvement,
                self._one_move_insertion_improvement,
                self._three_opt_first_improvement,
                #self._two_exchange_first_improvement,
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

   
    def shaking(self, tour: np.ndarray, k: int) -> np.ndarray:
        """
        VNS shaking stratified by strength (k) and adapted to instance size.
        k = 1: light | k = 2: medium | k = 3: strong | k >= 4: very strong.
        Avoids operators duplicated in the local search set.
        """
        core, is_closed = self._extract_core(tour)
        n = len(core)

        # Adaptive segment sizes (robust across different n)
        seg_len_mild_adapt   = max(4, min(8,  n // 20 or 4))
        seg_len_medium_adapt = max(5, min(12, n // 16 or 5))
        seg_len_strong_adapt = max(6, min(18, n // 12 or 6))

        # If self.segment_len_val is set, it overrides the adaptive values
        if self.segment_len_val is not None:
            seg_len_mild   = self.segment_len_val
            seg_len_medium = self.segment_len_val
            seg_len_strong = self.segment_len_val
        else:
            seg_len_mild, seg_len_medium, seg_len_strong = (
                seg_len_mild_adapt, seg_len_medium_adapt, seg_len_strong_adapt
            )

        k_small  = max(3, min(4, n // 25 or 3))
        k_medium = max(4, min(6, n // 18 or 4))
        k_large  = max(5, min(8, n // 12 or 5))

        # Operator sets by strength (without duplicating LS operators)
        if k <= 1:
            ops = [
                ShakeOp(lambda t: self._shake_oropt_block(t, block_len=2), "_shake_oropt_block_2"),
                ShakeOp(lambda t: self._shake_oropt_block(t, block_len=3), "_shake_oropt_block_3"),
                ShakeOp(lambda t: self._shake_shuffle_segment(t, seg_len=seg_len_mild), "_shake_shuffle_segment_mild"),
            ]
        elif k == 2:
            ops = [
                ShakeOp(lambda t: self._shake_k_exchange(t, k_nodes=k_small), "_shake_k_exchange_small"),
                ShakeOp(lambda t: self._shake_cross_exchange(t, len1=seg_len_mild // 2 + 1, len2=seg_len_mild // 2 + 1), "_shake_cross_exchange_mild"),
                ShakeOp(lambda t: self._shake_shuffle_segment(t, seg_len=seg_len_medium), "_shake_shuffle_segment_medium"),
            ]
        elif k == 3:
            ops = [
                ShakeOp(lambda t: self._shake_double_bridge(t), "_shake_double_bridge"),
                ShakeOp(lambda t: self._shake_k_exchange(t, k_nodes=k_medium), "_shake_k_exchange_medium"),
                ShakeOp(lambda t: self._shake_cross_exchange(t, len1=seg_len_medium // 2 + 2, len2=seg_len_medium // 2 + 2), "_shake_cross_exchange_medium"),
                ShakeOp(lambda t: self._shake_block_recombine(t, block_size=max(6, n // 10)), "_shake_block_recombine_medium"),
            ]
        else:  # k >= 4
            ops = [
                ShakeOp(lambda t: self._shake_double_bridge(t), "_shake_double_bridge"),
                ShakeOp(lambda t: self._shake_k_exchange(t, k_nodes=k_large), "_shake_k_exchange_large"),
                ShakeOp(lambda t: self._shake_cross_exchange(t, len1=seg_len_strong // 2 + 2, len2=seg_len_strong // 2 + 2), "_shake_cross_exchange_strong"),
                ShakeOp(lambda t: self._shake_block_recombine(t, block_size=max(8, n // 8)), "_shake_block_recombine_strong"),
                ShakeOp(lambda t: self._shake_shuffle_segment(t, seg_len=seg_len_strong), "_shake_shuffle_segment_strong"),
            ]

        # Try a few times to avoid occasional no-ops
        tries = min(8, 2 + len(ops))
        new_tour = tour
        for _ in range(tries):
            op = random.choice(ops)
            self.logger.info("Shaking with operator: %s", op.name)
            candidate = op(tour)
            if candidate is not None and candidate.shape == tour.shape and not np.array_equal(candidate, tour):
                new_tour = candidate
                break
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

        init_start = time.time()

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
                log=self.logger,
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
                marl_agents=int(params.get("marl_agents", 5)),
                marl_iterations=int(params.get("marl_iterations", 40)),
                marl_candidate_ratio=float(params.get("marl_candidate_ratio", 1.5)),
                marl_two_opt_passes=int(params.get("marl_two_opt_passes", 1)),
                marl_top_k=params.get("marl_top_k"),
                marl_reward=float(params.get("marl_reward", 1.0)),
                marl_learning_rate=float(params.get("marl_learning_rate", 0.4)),
                marl_softmax_beta=float(params.get("marl_softmax_beta", 2.0)),
                marl_epsilon=float(params.get("marl_epsilon", 0.15)),
                log=self.logger,
            )
            if not population:
                raise ValueError("MARL initialisation produced no candidates.")

            # Pick the best tour among generated candidates since VNS is single-solution.
            best_core = min(
                population,
                key=lambda t: (
                    self.tour_distance(np.append(t, t[0]), self.distance_matrix)
                    if t[0] != t[-1]
                    else self.tour_distance(t, self.distance_matrix)
                ),
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

        self.initial_solution_time = time.time() - init_start
        self.logger.info(
            "Initial solution built in %.4f seconds (method=%s)",
            self.initial_solution_time,
            method_key,
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
                    shaken_distance = self.tour_distance(
                        tour_from_shaking, self.distance_matrix
                    )
                    self.logger.info(
                        "Shaking phase completed in %.4f seconds (distance %.2f)",
                        exploration_time,
                        shaken_distance,
                    )

                    time_start = time.time()
                    new_tour, new_distance = self.local_search(
                        tour_from_shaking, shaken_distance
                    )
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
                        "Old distance: %.2f, New distance: %.2f",
                        old_distance,
                        new_distance,
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
                            "No improvement at k=%d. Moving to the next neighbourhood.",
                            k,
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
                    if consecutive_non_improving_iterations >= non_improve_limit:
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





