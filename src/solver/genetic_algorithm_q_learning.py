from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from logging.handlers import QueueHandler, QueueListener

from src.structures.graph import Graph
from src.solver.genetic_algorithm import GeneticAlgorithmConfig, GeneticTSPSolver


logger = logging.getLogger(__name__)


@dataclass
class GeneticAlgorithmQLearningConfig(GeneticAlgorithmConfig):
    rl_alpha: float = 0.3
    rl_gamma: float = 0.5
    rl_epsilon: float = 0.5
    rl_epsilon_min: float = 0.05
    rl_epsilon_decay: float = 0.98
    negative_reward_scale: float = 1.5


class GeneticTSPSolverQLearning(GeneticTSPSolver):
    """GA variant that learns which crossover operator to use via Q-learning."""

    def __init__(
        self,
        graph: Graph,
        save_dir,
        config: Optional[GeneticAlgorithmQLearningConfig] = None,
        best_known_distance: Optional[float] = None,
    ):
        self.config: GeneticAlgorithmQLearningConfig = config or GeneticAlgorithmQLearningConfig()
        self.rl_alpha = self.config.rl_alpha
        self.rl_gamma = self.config.rl_gamma
        self.rl_epsilon = self.config.rl_epsilon
        self.rl_epsilon_min = self.config.rl_epsilon_min
        self.rl_epsilon_decay = self.config.rl_epsilon_decay
        self.negative_reward_scale = max(1.0, self.config.negative_reward_scale)

        self._rl_q_table: np.ndarray | None = None
        self._rl_operator_index: dict[str, int] | None = None
        self._last_action_idx: int | None = None

        super().__init__(
            graph=graph,
            save_dir=save_dir,
            config=self.config,
            best_known_distance=best_known_distance,
        )

    # --- RL helpers ----------------------------------------------------
    def _initialize_rl_table(self, operator_names: list[str]) -> None:
        if (
            self._rl_operator_index is None
            or set(self._rl_operator_index) != set(operator_names)
            or len(self._rl_operator_index) != len(operator_names)
        ):
            self._rl_operator_index = {name: idx for idx, name in enumerate(operator_names)}
            size = len(operator_names)
            self._rl_q_table = np.zeros((size, size), dtype=float)
        elif self._rl_q_table is None or self._rl_q_table.shape[0] != len(operator_names):
            size = len(operator_names)
            self._rl_q_table = np.zeros((size, size), dtype=float)

    def _select_operator(self, state_idx: int | None, available: list[int]) -> int:
        if not available:
            raise ValueError("No crossover operators available for selection.")
        if self._rl_q_table is None or random.random() < self.rl_epsilon:
            return random.choice(available)

        if state_idx is None:
            preferences = self._rl_q_table.max(axis=0)
        else:
            preferences = self._rl_q_table[state_idx]

        return max(available, key=lambda idx: preferences[idx])

    def _update_q_value(self, state_idx: int | None, action_idx: int, reward: float) -> None:
        if state_idx is None or self._rl_q_table is None:
            return
        current = self._rl_q_table[state_idx, action_idx]
        future_max = self._rl_q_table[action_idx].max() if self._rl_q_table.size else 0.0
        updated = (1 - self.rl_alpha) * current + self.rl_alpha * (reward + self.rl_gamma * future_max)
        self._rl_q_table[state_idx, action_idx] = updated

    @staticmethod
    def _positive_reward(previous_best: float, candidate_distance: float) -> float:
        if not np.isfinite(previous_best):
            return 1.0
        delta = previous_best - candidate_distance
        if delta <= 0:
            return 0.0
        return delta / max(previous_best, 1e-9)

    def _negative_reward(self, previous_best: float, candidate_distance: float) -> float:
        if not np.isfinite(previous_best):
            return -0.01 * self.negative_reward_scale
        delta = candidate_distance - previous_best
        if delta <= 0:
            return -0.01 * self.negative_reward_scale
        penalty = delta / max(previous_best, 1e-9)
        return -self.negative_reward_scale * penalty

    def _decay_epsilon(self) -> None:
        if self.rl_epsilon > self.rl_epsilon_min:
            self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon * self.rl_epsilon_decay)

    # --- Public API -----------------------------------------------------
    def run(
        self, time_limit_seconds: Optional[float] = None
    ) -> Tuple[List[int], float, List[dict], float]:
        """Execute the GA using Q-learning to select crossover operators."""
        run_start = time.perf_counter()
        try:
            init_start = time.perf_counter()
            population = self._create_initial_population()
            elapsed = time.perf_counter() - init_start
            self.logger.info("Initial population generated in %.4f seconds", elapsed)

            fitness = [self._tour_distance(individual) for individual in population]
            self.logger.info(
                "Initial population created with %d individuals, avg fitness=%.4f",
                len(population),
                np.mean(fitness),
            )

            self.logger.info("Fitness values: %s", fitness)
            best_idx = int(np.argmin(fitness))
            best_tour = population[best_idx]
            best_distance = fitness[best_idx]
            history: List[dict] = [self._generation_stats(0, fitness, best_distance)]
            self._record_best_known_hit(best_distance, 0)

            stagnation_counter = 0
            generation = 0
            last_generation = 0
            max_generations = max(1, self.config.max_generations)
            population_limit = max(1, self.config.population_size)
            deadline = (
                run_start + max(0.0, float(time_limit_seconds))
                if time_limit_seconds is not None
                else None
            )

            crossover_ops: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
                ("smx", self._smx_crossover),
                ("ordered", lambda p1, p2: self._ordered_crossover(p1, p2)[0]),
                ("er", self._edge_recombination_crossover),
            ]
            op_names = [name for name, _ in crossover_ops]
            self._initialize_rl_table(op_names)

            if self.best_known_hit_generation is None:
                while generation < max_generations:
                    if deadline is not None and time.perf_counter() >= deadline:
                        self.logger.info(
                            "Time limit reached before starting generation %d.",
                            generation + 1,
                        )
                        break

                    generation += 1
                    last_generation = generation

                    state_idx = self._last_action_idx
                    action_idx = self._select_operator(state_idx, list(range(len(crossover_ops))))
                    action_name, action_fn = crossover_ops[action_idx]
                    self.logger.info("Selected crossover operator: %s (ε=%.3f)", action_name, self.rl_epsilon)

                    parent1 = self._selection_operator(population, fitness)
                    parent2 = self._selection_operator(population, fitness)

                    previous_best = best_distance
                    child = action_fn(parent1, parent2)
                    self._crossovers += 1

                    if self.random.random() < self.config.mutation_rate:
                        self._swap_mutation(child)

                    child_distance = self._tour_distance(child)

                    survivor_method = (self.config.survivor_selection_method or "tournament").lower()

                    if survivor_method == "tournament":
                        population.append(child)
                        fitness.append(child_distance)
                        population, fitness = self._tournament_survivor_selection(
                            population,
                            fitness,
                            population_limit,
                        )
                    elif survivor_method == "best_improves":
                        population, fitness = self._best_improves_survivor(
                            population,
                            fitness,
                            child,
                            child_distance,
                            population_limit,
                        )
                    else:
                        raise ValueError(
                            "Unsupported survivor_selection_method. Use 'tournament' or 'best_improves'."
                        )

                    current_best_idx = int(np.argmin(fitness))
                    current_best_distance = fitness[current_best_idx]
                    hit_best = False
                    if current_best_distance + 1e-9 < best_distance:
                        best_distance = current_best_distance
                        best_tour = population[current_best_idx].copy()
                        stagnation_counter = 0
                        hit_best = self._record_best_known_hit(best_distance, generation)
                    else:
                        stagnation_counter += 1

                    if best_distance + 1e-9 < previous_best:
                        reward = self._positive_reward(previous_best, best_distance)
                    else:
                        reward = self._negative_reward(previous_best, child_distance)
                    self._update_q_value(state_idx, action_idx, reward)
                    self._last_action_idx = action_idx
                    self._decay_epsilon()

                    history.append(
                        self._generation_stats(
                            generation,
                            fitness,
                            best_distance,
                        )
                    )

                    if deadline is not None and time.perf_counter() >= deadline:
                        self.logger.info(
                            "Time limit reached after generation %d.", generation
                        )
                        break

                    if hit_best:
                        break

                    if self.config.stagnation_limit is not None and stagnation_counter >= self.config.stagnation_limit:
                        break

            optimized_best = self._two_opt_improve(
                best_tour.copy(),
                max(1, int(self.config.marl_two_opt_passes or 1)),
            )
            optimized_distance = self._tour_distance(optimized_best)
            if optimized_distance + 1e-9 < best_distance:
                best_distance = optimized_distance
                best_tour = optimized_best
                population.append(best_tour.copy())
                fitness.append(best_distance)
                if len(population) > population_limit:
                    worst_idx = int(np.argmax(fitness))
                    del population[worst_idx]
                    del fitness[worst_idx]
                history.append(
                    self._generation_stats(
                        last_generation + 1,
                        fitness,
                        best_distance,
                    )
                )

            closed_tour = self._close_tour(best_tour)
            elapsed_run = time.perf_counter() - run_start
            self.logger.info("Quantity of crossovers performed: %d", self._crossovers)
            return closed_tour.tolist(), float(best_distance), history, elapsed_run
        finally:
            self._stop_logging()
