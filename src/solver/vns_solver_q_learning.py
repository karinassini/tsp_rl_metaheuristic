import random
import time

import numpy as np

from src.structures.graph import Graph
from src.solver.initial_solution import (
    QLearningConfig,
    nearest_neighbour_tour,
    q_learning_tour,
    rcl_nearest_neighbour_tour,
    marl_initial_population,
)
from src.solver.vns_solver import VNS_Solver


class VNS_Solver_Q_Learnings(VNS_Solver):
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
        segment_len_val: int | None = None,
        rl_alpha: float = 0.3,
        rl_gamma: float = 0.5,
        rl_epsilon: float = 0.5,
        rl_epsilon_min: float = 0.05,
        rl_epsilon_decay: float = 0.98,
        max_local_search_iterations: int | None = 80,
        negative_reward_scale: float = 1.5,
        operator_failure_limit: int = 10,
        q_learning_cfg: QLearningConfig | None = None,
        rl_local_search_cfg: object | None = None,
        marl_params: dict | None = None,
        best_known_distance: float | None = None,
        timestamp: str | None = None,
        verbose_route_log: bool = False,
    ):
        """
        Initialize the VNS solver with a Graph object.
        :param graph: Graph instance representing the TSP.
        :param max_double_bridge_checks: Maximum sampled double-bridge moves evaluated per local-search call.
        """
        super().__init__(
            graph=graph,
            save_dir=save_dir,
            method=method,
            max_double_bridge_checks=max_double_bridge_checks,
            restricted_two_opt_max_span=restricted_two_opt_max_span,
            max_flip_subsequence_length=max_flip_subsequence_length,
            max_inversion_segment_length=max_inversion_segment_length,
            segment_len_val=segment_len_val,
            q_learning_cfg=q_learning_cfg,
            marl_params=marl_params,
            best_known_distance=best_known_distance,
            timestamp=timestamp,
            verbose_route_log=verbose_route_log,
        )
        self.rl_alpha = rl_alpha
        self.rl_gamma = rl_gamma
        self.rl_epsilon = rl_epsilon
        self.rl_epsilon_min = rl_epsilon_min
        self.rl_epsilon_decay = rl_epsilon_decay
        self.max_local_search_iterations = max_local_search_iterations
        self.negative_reward_scale = max(1.0, negative_reward_scale)
        self.operator_failure_limit = max(1, operator_failure_limit)
        self.rl_local_search_cfg = rl_local_search_cfg
        # Track time to build the initial solution
        self.initial_solution_time: float | None = None

        if rl_local_search_cfg is not None:

            def _value(name, default):
                if hasattr(rl_local_search_cfg, name):
                    return getattr(rl_local_search_cfg, name)
                if (
                    isinstance(rl_local_search_cfg, dict)
                    and name in rl_local_search_cfg
                ):
                    return rl_local_search_cfg[name]
                return default

            self.rl_alpha = _value("rl_alpha", self.rl_alpha)
            self.rl_gamma = _value("rl_gamma", self.rl_gamma)
            self.rl_epsilon = _value("rl_epsilon", self.rl_epsilon)
            self.rl_epsilon_min = _value("rl_epsilon_min", self.rl_epsilon_min)
            self.rl_epsilon_decay = _value("rl_epsilon_decay", self.rl_epsilon_decay)
            self.max_local_search_iterations = _value(
                "max_local_search_iterations",
                self.max_local_search_iterations,
            )
            self.negative_reward_scale = max(
                1.0,
                _value("negative_reward_scale", self.negative_reward_scale),
            )
            self.operator_failure_limit = max(
                1,
                int(_value("operator_failure_limit", self.operator_failure_limit)),
            )
        self._rl_q_table: np.ndarray | None = None
        self._rl_operator_index: dict[str, int] | None = None

    def export_parameters(self) -> dict[str, object]:
        params = super().export_parameters()
        params.update(
            {
                "rl_alpha": self.rl_alpha,
                "rl_gamma": self.rl_gamma,
                "rl_epsilon": self.rl_epsilon,
                "rl_epsilon_min": self.rl_epsilon_min,
                "rl_epsilon_decay": self.rl_epsilon_decay,
                "max_local_search_iterations": self.max_local_search_iterations,
                "negative_reward_scale": self.negative_reward_scale,
                "operator_failure_limit": self.operator_failure_limit,
            }
        )
        return params

    def _initialize_rl_table(self, operator_names: list[str]) -> None:
        if (
            self._rl_operator_index is None
            or set(self._rl_operator_index) != set(operator_names)
            or len(self._rl_operator_index) != len(operator_names)
        ):
            self._rl_operator_index = {
                name: idx for idx, name in enumerate(operator_names)
            }
            size = len(operator_names)
            self._rl_q_table = np.zeros((size, size), dtype=float)
        elif self._rl_q_table is None or self._rl_q_table.shape[0] != len(
            operator_names
        ):
            size = len(operator_names)
            self._rl_q_table = np.zeros((size, size), dtype=float)
        # self.logger.info(
        #     "Initialized RL Q-table with operators: %s", self._rl_q_table
        # )

    def _select_search(self, state_idx: int | None, available: list[int]) -> int:
        if not available:
            raise ValueError("No local-search operators available for selection.")
        if self._rl_q_table is None or random.random() < self.rl_epsilon:
            return random.choice(available)

        if state_idx is None:
            preferences = self._rl_q_table.max(axis=0)
        else:
            preferences = self._rl_q_table[state_idx]

        best_idx = max(available, key=lambda idx: preferences[idx])
        return best_idx

    def _update_q_value(
        self, state_idx: int | None, action_idx: int, reward: float
    ) -> None:
        if state_idx is None or self._rl_q_table is None:
            return
        current = self._rl_q_table[state_idx, action_idx]
        future_max = (
            self._rl_q_table[action_idx].max() if self._rl_q_table.size else 0.0
        )
        updated = (1 - self.rl_alpha) * current + self.rl_alpha * (
            reward + self.rl_gamma * future_max
        )
        self._rl_q_table[state_idx, action_idx] = updated

    @staticmethod
    def _positive_reward(previous_best: float, candidate_distance: float) -> float:
        if not np.isfinite(previous_best):
            return 1.0
        delta = previous_best - candidate_distance
        if delta <= 0:
            return 0.0
        return delta / max(previous_best, 1e-9)

    def _negative_reward(
        self, previous_best: float, candidate_distance: float
    ) -> float:
        if not np.isfinite(previous_best):
            return -0.01 * self.negative_reward_scale
        delta = candidate_distance - previous_best
        if delta <= 0:
            return -0.01 * self.negative_reward_scale
        penalty = delta / max(previous_best, 1e-9)
        return -self.negative_reward_scale * penalty

    def _decay_epsilon(self) -> None:
        if self.rl_epsilon > self.rl_epsilon_min:
            self.rl_epsilon = max(
                self.rl_epsilon_min, self.rl_epsilon * self.rl_epsilon_decay
            )

    _OP_SHORT = {
        '_two_opt_first_improvement':        '2opt',
        '_one_move_insertion_improvement':   '1move',
        '_three_opt_first_improvement':      '3opt',
        '_two_exchange_first_improvement':   '2exch',
        '_double_bridge_first_improvement':  'dbl-br',
    }

    def _log_q_table(self) -> None:
        """Log the current Q-table in a human-readable format."""
        if self._rl_q_table is None or self._rl_operator_index is None:
            return
        names = [name for name, _ in sorted(self._rl_operator_index.items(), key=lambda x: x[1])]
        short = [self._OP_SHORT.get(n, n[:6]) for n in names]
        col_w = 8
        header_row = ' ' * col_w + ''.join(s.rjust(col_w) for s in short)
        rows = ['Q-table (row=prev → col=next):', header_row]
        for i, row_name in enumerate(short):
            vals = ''.join(f'{self._rl_q_table[i, j]:>{col_w}.4f}' for j in range(len(short)))
            rows.append(row_name.rjust(col_w) + vals)
        self.logger.info('%s', '\n'.join(rows))

    def local_search(
        self, tour: np.ndarray, current_distance: float, operators=None, max_iterations_override: int | None = None
    ) -> tuple[np.ndarray, float]:
        """Reinforcement-learning driven selection of neighbourhood operators."""
        if operators is None:
            operators = [
                self._two_opt_first_improvement,
                self._one_move_insertion_improvement,
                self._three_opt_first_improvement,
                self._two_exchange_first_improvement,
            ]

        named_ops = [(op.__name__, op) for op in operators]
        operator_names = [name for name, _ in named_ops]

        # Initialize RL Q-table if needed -> the RL table depends on the available operators
        self._initialize_rl_table(operator_names)

        # If consecutive failures exceed limit, ban operator for remainder of local search
        failure_streaks = [0 for _ in named_ops]
        banned_ops: set[int] = set()

        def _fresh_available() -> set[int]:
            return {idx for idx in range(len(named_ops)) if idx not in banned_ops}

        available = _fresh_available()
        self.logger.info(
            "Available operators: %s", [named_ops[idx][0] for idx in sorted(available)]
        )

        if not available:
            return tour.copy(), current_distance

        state_idx = self._select_search(None, list(available))
        available.discard(state_idx)

        best_tour = tour.copy()
        best_distance = current_distance
        tolerance = 1e-9

        state_name, state_fn = named_ops[state_idx]

        candidate, improved, candidate_distance = state_fn(best_tour, best_distance)
        if improved and candidate_distance + tolerance < best_distance:
            best_tour = candidate
            best_distance = candidate_distance
            failure_streaks[state_idx] = 0
            self.logger.info(
                "Operator %s improved distance to %.2f",
                state_name,
                best_distance,
            )
            self.logger.debug(
                "route: %s, cost: %.2f",
                best_tour.tolist(),
                best_distance,
            )
        else:
            failure_streaks[state_idx] += 1
            if failure_streaks[state_idx] >= self.operator_failure_limit:
                banned_ops.add(state_idx)
                self.logger.info(
                    "Operator %s banned for remainder of local search after %d initial failures",
                    state_name,
                    failure_streaks[state_idx],
                )

        max_iterations = (
            max_iterations_override
            if max_iterations_override is not None
            else (
                self.max_local_search_iterations
                if self.max_local_search_iterations is not None
                else len(named_ops) * max(5, self.n_cities // 10)
            )
        )

        self.logger.info(
            "RL local search initial operator: %s (ε=%.3f) with max iterations %d",
            state_name,
            self.rl_epsilon,
            max_iterations,
        )

        iteration = 0
        while available and iteration < max_iterations:
            action_idx = self._select_search(state_idx, list(available))
            action_name, action_fn = named_ops[action_idx]
            previous_name = named_ops[state_idx][0]
            self.logger.info(
                "RL iteration %d (ε=%.3f)",
                iteration,
                self.rl_epsilon,
            )
            candidate, improved, candidate_distance = action_fn(
                best_tour, best_distance
            )
            iteration += 1
            if improved and candidate_distance + tolerance < best_distance:
                reward = self._positive_reward(best_distance, candidate_distance)
                self._update_q_value(state_idx, action_idx, reward)
                best_tour = candidate
                best_distance = candidate_distance
                state_idx = action_idx
                failure_streaks[action_idx] = 0
                available = _fresh_available()
                available.discard(state_idx)
                self.logger.info(
                    "Improvement with %s -> distance %.2f (reward %.4f)",
                    action_name,
                    best_distance,
                    reward,
                )
                self.logger.debug(
                    "route: %s, cost: %.2f",
                    best_tour.tolist(),
                    best_distance,
                )
                self._log_q_table()
            else:
                penalty = self._negative_reward(best_distance, candidate_distance)
                self._update_q_value(state_idx, action_idx, penalty)
                available.discard(action_idx)
                failure_streaks[action_idx] += 1
                if failure_streaks[action_idx] >= self.operator_failure_limit:
                    banned_ops.add(action_idx)
                    available.discard(action_idx)
                    self.logger.info(
                        "Operator %s banned for remainder of local search after %d failures",
                        action_name,
                        failure_streaks[action_idx],
                    )
                # to avoid hammering the same neighbourhood repeatedly within one local-search sweep
                self.logger.info(
                    "No improvement with %s (penalty %.4f). Remaining: %s",
                    action_name,
                    penalty,
                    [named_ops[idx][0] for idx in sorted(available)],
                )
                self.logger.debug(
                    "route: %s, cost: %.2f",
                    best_tour.tolist(),
                    best_distance,
                )
                if not available:
                    break

        self.logger.info(
            "RL local search finished after %d iterations. Best distance %.2f",
            iteration,
            best_distance,
        )
        self._decay_epsilon()
        return best_tour, best_distance

    def vns_solve(
        self,
        iteration_max: int = 500,
        k_max: int = 2,
        max_non_improving_iterations: int = 10,
        start: int | None = None,
    ):
        """
        Solve the TSP using Variable Neighborhood Search (VNS) and log the process.
        Mirrors the base solver's iteration bookkeeping while using RL-driven local search.
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
            if self.verbose_route_log:
                self.logger.debug("route: %s, cost: %.2f", tour.tolist(), float(total_distance))
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
            if self.verbose_route_log:
                self.logger.debug("route: %s, cost: %.2f", tour.tolist(), float(total_distance))
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
            if self.verbose_route_log:
                self.logger.debug("route: %s, cost: %.2f", tour.tolist(), float(total_distance))
            self._record_best_known_hit(float(total_distance), 0)
        elif method_key == "rcl":
            tour = rcl_nearest_neighbour_tour(self.graph, start=start, alpha=0.3)
            total_distance = self.tour_distance(tour, self.distance_matrix)
            self.logger.info(
                f"Initial RCL tour: {tour.tolist()}, distance: {total_distance:.2f}"
            )
            if self.verbose_route_log:
                self.logger.debug("route: %s, cost: %.2f", tour.tolist(), float(total_distance))
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
                verbose_route_log=self.verbose_route_log,
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
            if self.verbose_route_log:
                self.logger.debug("route: %s, cost: %.2f", tour.tolist(), float(total_distance))
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
        total_exploration_time = 0.0
        total_exploitation_time = 0.0
        iteration = 1
        last_completed_iteration = 0
        iteration_limit = max(1, int(iteration_max)) if iteration_max is not None else 1
        max_neighbourhood = max(1, int(k_max))
        consecutive_non_improving_iterations = 0
        non_improve_limit = max(1, int(max_non_improving_iterations))
        best_known_reached = self.best_known_hit_iteration is not None

        # Initialise Q-table once before the search loop so it accumulates
        # knowledge across all local-search calls instead of being lazily
        # reset on the first call inside local_search.
        default_operators = [
            self._two_opt_first_improvement,
            self._one_move_insertion_improvement,
            self._three_opt_first_improvement,
            self._two_exchange_first_improvement,
        ]
        self._initialize_rl_table([op.__name__ for op in default_operators])

        try:
            while iteration <= iteration_limit and not best_known_reached:
                self.logger.info("=== Iteration %d ===", iteration)
                k = 1
                improved_in_iteration = False

                while k <= max_neighbourhood:
                    self.logger.info("Iteration %d - neighbourhood k=%d", iteration, k)
                    if self.verbose_route_log:
                        self.logger.debug(
                            "route: %s, cost: %.2f",
                            tour.tolist(),
                            self.tour_distance(tour, self.distance_matrix),
                        )
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
                    if self.verbose_route_log:
                        self.logger.debug(
                            "route: %s, cost: %.2f",
                            tour_from_shaking.tolist(),
                            shaken_distance,
                        )

                    time_start = time.time()

                    # Dynamic iteration budget based on k: larger perturbations need more recovery
                    iterations_for_k = self.max_local_search_iterations
                    if k == 2 and self.max_local_search_iterations is not None:
                        iterations_for_k = int(self.max_local_search_iterations * 1.25)  # 25% more for k=2
                    elif k >= 3 and self.max_local_search_iterations is not None:
                        iterations_for_k = int(self.max_local_search_iterations * 1.5)   # 50% more for k>=3

                    new_tour, new_distance = self.local_search(
                        tour_from_shaking, shaken_distance, max_iterations_override=iterations_for_k
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
                "Final tour: %s, total distance: %.2f",
                tour.tolist(),
                total_distance,
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
