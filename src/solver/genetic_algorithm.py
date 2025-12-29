"""Genetic algorithm solver for the Traveling Salesman Problem.

The implementation follows a standard steady-generation approach with
ordered crossover, swap mutation, and tournament selection. It allows seeding
the initial population with a nearest-neighbour tour to speed convergence while
keeping the remaining individuals randomised.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Sequence, Tuple

import numpy as np
from logging.handlers import QueueHandler, QueueListener

from src.structures.graph import Graph
from src.solver.initial_solution import nearest_neighbour_tour


logger = logging.getLogger(__name__)


@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02
    tournament_size: int = 3
    elitism: bool = True
    seed: Optional[int] = None
    max_generations: int = 300
    stagnation_limit: Optional[int] = None
    selection_method: str = "roulette"
    rank_selection_pressure: float = 1.7
    truncation_ratio: float = 0.3
    initialization_method: str = "marl"  # Options: "nearest_random", "marl"
    marl_agents: int = 6
    marl_iterations: int = 40
    marl_epsilon_mix: float = 0.5  # probability of epsilon-greedy vs softmax
    marl_epsilon: float = 0.15
    marl_softmax_beta: float = 2.0
    marl_learning_rate: float = 0.4
    marl_discount: float = 0.6
    marl_candidate_ratio: float = 1.5  # how many MARL tours relative to population size
    marl_two_opt_passes: int = 1
    log_dir: Optional[str] = None


class GeneticTSPSolver:
    """Metaheuristic solver based on a Genetic Algorithm."""

    def __init__(self, graph: Graph, save_dir, config: Optional[GeneticAlgorithmConfig] = None):
        self.graph = graph
        self.distance_matrix = np.asarray(graph.get_distance_matrix())
        self.n_cities = graph.n_nodes
        self.config = config or GeneticAlgorithmConfig()
        self.random = random.Random(self.config.seed)
        self.save_dir = save_dir

        if self.n_cities < 3:
            raise ValueError("Genetic algorithm requires at least 3 cities.")
        if self.config.population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if self.config.tournament_size > self.config.population_size:
            raise ValueError("Tournament size cannot exceed population size.")
        if not 0.0 < self.config.truncation_ratio <= 1.0:
            raise ValueError("truncation_ratio must be in (0, 1].")

        self._sus_pointer_start: Optional[float] = None
        self._sus_offsets_used: int = 0
        self._selection_operator = self._resolve_selection_operator(
            self.config.selection_method
        )

        # Configure asynchronous file logger, mirroring the VNS solver approach.
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.save_dir:
            logs_dir = os.path.join(self.save_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(
                logs_dir, f"ga_steps_{self.timestamp}_{self.config.initialization_method}_{os.getpid()}_{id(self)}.log"
            )

        log_dir = self.config.log_dir
        filename = f"ga_steps_{self.timestamp}_{self.config.initialization_method}_{os.getpid()}_{id(self)}.log"
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, filename)


        self.logger = logging.getLogger(f"ga_solver.{self.timestamp}.{id(self)}")
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

    # --- Public API -----------------------------------------------------
    def run(self) -> Tuple[List[int], float, List[dict]]:
        """Execute the genetic algorithm and return the best tour found."""
        try:
            population = self._create_initial_population()
            fitness = [self._tour_distance(individual) for individual in population]

            self.logger.info("Initial population created with %d individuals, avg fitness=%.4f", len(population), np.mean(fitness))

            best_idx = int(np.argmin(fitness))
            best_tour = population[best_idx]
            best_distance = fitness[best_idx]
            history: List[dict] = [self._generation_stats(0, fitness, best_distance)]

            stagnation_counter = 0

            for generation in range(1, self.config.max_generations + 1):
                population, fitness = self._next_generation(population, fitness)

                current_idx = int(np.argmin(fitness))
                current_distance = fitness[current_idx]
                if current_distance + 1e-9 < best_distance:
                    best_distance = current_distance
                    best_tour = population[current_idx]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                history.append(self._generation_stats(generation, fitness, best_distance))

                if (
                    self.config.stagnation_limit is not None
                    and stagnation_counter >= self.config.stagnation_limit
                ):
                    break

            closed_tour = self._close_tour(best_tour)
            return closed_tour.tolist(), float(best_distance), history
        finally:
            self._stop_logging()

    # --- Genetic operators ----------------------------------------------
    def _create_initial_population(self) -> List[np.ndarray]:
        method = (self.config.initialization_method or "nearest_random").lower()
        if method == "marl":
            self.logger.info("Initial population method: MARL")
            population = self._marl_initial_population()
        else:
            self.logger.info("Initial population method: nearest-random")
            population = self._nearest_random_population()

        while len(population) < self.config.population_size:
            population.append(self._nearest_random_individual())

        self.logger.info("Initial population size after padding: %d", len(population))
        return population

    def _nearest_random_population(self) -> List[np.ndarray]:
        return [self._nearest_random_individual()]

    def _nearest_random_individual(self) -> np.ndarray:
        seed_tour = nearest_neighbour_tour(self.graph)
        return np.asarray(seed_tour[:-1], dtype=int)

    def _marl_initial_population(self) -> List[np.ndarray]:
        cfg = self.config
        self.logger.info("Starting MARL-based initial population generation with %d agents for up to %d populations", cfg.marl_agents, cfg.population_size)
        q_table = np.zeros((self.n_cities, self.n_cities), dtype=float)
        candidate_set: List[tuple[np.ndarray, float]] = []
        best_distance = float("inf")
        target = max(1, int(round(cfg.marl_candidate_ratio * cfg.population_size)))

        for _ in range(max(1, cfg.marl_iterations)):
            iteration_tours: List[tuple[np.ndarray, float]] = []
            for _agent in range(max(1, cfg.marl_agents)):
                tour = self._marl_construct_tour(q_table)
                distance = self._tour_distance(tour)
                iteration_tours.append((tour, distance))

            iteration_tours.sort(key=lambda item: item[1])
            if not iteration_tours:
                continue

            iteration_best_tour, iteration_best_distance = iteration_tours[0]
            self._marl_update_q_table(q_table, iteration_best_tour, iteration_best_distance)

            if iteration_best_distance + 1e-9 < best_distance:
                best_distance = iteration_best_distance

            for tour, distance in iteration_tours:
                if distance  <= best_distance + 1e-9:
                    candidate_set.append((tour.copy(), distance))
                if len(candidate_set) >= target:
                    break
            if len(candidate_set) >= target:
                break

        if not candidate_set:
            self.logger.warning(
                "MARL generation yielded no improving candidates; falling back to nearest-random"
            )
            return self._nearest_random_population()

        candidate_set.sort(key=lambda item: item[1])
        unique_population: List[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for tour, _distance in candidate_set:
            key = tuple(tour.tolist())
            if key in seen:
                continue
            improved = self._two_opt_improve(tour.copy(), self.config.marl_two_opt_passes)
            unique_population.append(improved)
            seen.add(key)
            if len(unique_population) >= self.config.population_size:
                break

        self.logger.info(
            "Generated %d MARL tours from %d candidates",
            len(unique_population),
            len(candidate_set),
        )
        return unique_population

    def _marl_construct_tour(self, q_table: np.ndarray) -> np.ndarray:
        start = self.random.randrange(self.n_cities)
        unvisited = set(range(self.n_cities))
        unvisited.remove(start)
        tour = [start]

        while unvisited:
            current = tour[-1]
            action = self._marl_select_action(current, list(unvisited), q_table)
            tour.append(action)
            unvisited.remove(action)

        return np.asarray(tour, dtype=int)

    def _marl_select_action(self, state: int, available: List[int], q_table: np.ndarray) -> int:
        if not available:
            return state
        mix = max(0.0, min(1.0, self.config.marl_epsilon_mix))
        if self.random.random() < mix:
            if self.random.random() < self.config.marl_epsilon:
                return self.random.choice(available)
            return max(available, key=lambda idx: q_table[state, idx])

        weights = self._marl_softmax([q_table[state, idx] for idx in available])
        threshold = self.random.random()
        cumulative = 0.0
        for idx, weight in zip(available, weights):
            cumulative += weight
            if cumulative >= threshold:
                return idx
        return available[-1]

    def _marl_softmax(self, values: List[float]) -> List[float]:
        beta = max(0.1, self.config.marl_softmax_beta)
        shifted = np.asarray(values, dtype=float)
        shifted = shifted - float(np.max(shifted))
        exp_values = np.exp(beta * shifted)
        total = float(np.sum(exp_values))
        if total <= 0:
            return [1.0 / len(values)] * len(values)
        return (exp_values / total).tolist()

    def _marl_update_q_table(self, q_table: np.ndarray, tour: np.ndarray, tour_distance: float) -> None:
        reward = 1.0 / max(tour_distance, 1e-9)
        lr = self.config.marl_learning_rate
        gamma = self.config.marl_discount
        cycle = self._close_tour(tour)
        for current, nxt in zip(cycle, np.roll(cycle, -1)):
            old_value = q_table[current, nxt]
            future = q_table[nxt].max() if q_table.shape[0] else 0.0
            q_table[current, nxt] = (1 - lr) * old_value + lr * (reward + gamma * future)

    def _two_opt_improve(self, tour: np.ndarray, passes: int) -> np.ndarray:
        best = tour
        best_distance = self._tour_distance(best)
        for _ in range(max(1, passes)):
            improved = False
            length = len(best)
            for i in range(1, length - 2):
                for j in range(i + 1, length - 1):
                    if j - i == 1:
                        continue
                    candidate = best.copy()
                    candidate[i:j] = candidate[i:j][::-1]
                    candidate_distance = self._tour_distance(candidate)
                    if candidate_distance + 1e-9 < best_distance:
                        best = candidate
                        best_distance = candidate_distance
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def _next_generation(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> Tuple[List[np.ndarray], List[float]]:
        new_population: List[np.ndarray] = []

        elites = []
        if self.config.elitism:
            best_idx = int(np.argmin(fitness))
            elites.append(population[best_idx])
            new_population.extend(elites)

        while len(new_population) < self.config.population_size:
            parent1 = self._selection_operator(population, fitness)
            parent2 = self._selection_operator(population, fitness)

            if self.random.random() < self.config.crossover_rate:
                child1, child2 = self._ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if self.random.random() < self.config.mutation_rate:
                self._swap_mutation(child1)
            if self.random.random() < self.config.mutation_rate:
                self._swap_mutation(child2)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        new_fitness = [self._tour_distance(individual) for individual in new_population]
        return new_population, new_fitness

    def _resolve_selection_operator(self, method: Optional[str]):
        lookup = {
            "tournament": self._select_tournament,
            "roulette": self._select_roulette_wheel,
            "stochastic_universal": self._select_stochastic_universal,
            "rank": self._select_linear_rank,
            "truncation": self._select_truncation,
        }
        key = (method or "tournament").lower()
        if key not in lookup:
            raise ValueError(
                f"Unknown selection_method '{method}'. Supported: {', '.join(lookup)}."
            )
        return lookup[key]

    def _select_tournament(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        competitors = self.random.sample(
            range(len(population)), self.config.tournament_size
        )
        best_idx = min(competitors, key=lambda idx: fitness[idx])
        return population[best_idx].copy()

    def _select_roulette_wheel(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        weights = self._fitness_to_weights(fitness)
        idx = self._choose_index_by_prob(weights)
        return population[idx].copy()

    def _select_stochastic_universal(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        weights = self._fitness_to_weights(fitness)
        total = float(np.sum(weights))
        if total <= 0:
            return population[self.random.randrange(len(population))].copy()
        cdf = np.cumsum(weights) / total
        size = len(population)
        if self._sus_pointer_start is None or self._sus_offsets_used >= size:
            self._sus_pointer_start = self.random.random() / size
            self._sus_offsets_used = 0
        pointer = (self._sus_pointer_start + (self._sus_offsets_used / size)) % 1.0
        self._sus_offsets_used += 1
        idx = int(np.searchsorted(cdf, pointer, side="right"))
        idx = min(idx, size - 1)
        return population[idx].copy()

    def _select_linear_rank(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        n = len(population)
        if n == 1:
            return population[0].copy()
        sp = max(1.0, min(2.0, self.config.rank_selection_pressure))
        sorted_indices = sorted(range(n), key=lambda idx: fitness[idx])
        denom = n - 1 if n > 1 else 1
        probabilities = [
            (1.0 / n) * (sp - (2 * sp - 2) * (rank / denom))
            for rank in range(n)
        ]
        choice = self._choose_index_by_prob(probabilities)
        return population[sorted_indices[choice]].copy()

    def _select_truncation(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        n = len(population)
        if n == 1:
            return population[0].copy()
        sorted_indices = sorted(range(n), key=lambda idx: fitness[idx])
        top_k = max(1, min(n, int(round(self.config.truncation_ratio * n))))
        chosen = self.random.choice(sorted_indices[:top_k])
        return population[chosen].copy()

    def _fitness_to_weights(self, fitness: Sequence[float]) -> np.ndarray:
        values = np.asarray(fitness, dtype=float)
        weights = 1.0 / (values + 1e-12)
        weights[~np.isfinite(weights)] = 0.0
        if np.all(weights <= 0):
            return np.ones_like(weights)
        return weights

    def _choose_index_by_prob(self, probabilities: Sequence[float]) -> int:
        total = float(sum(probabilities))
        if total <= 0:
            return self.random.randrange(len(probabilities))
        threshold = self.random.random() * total
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if cumulative >= threshold:
                return idx
        return len(probabilities) - 1

    def _ordered_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = len(parent1)
        a, b = sorted(self.random.sample(range(size), 2))

        def make_child(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            child = -np.ones(size, dtype=int)
            child[a:b] = first[a:b]
            fill_pointer = b
            for gene in second:
                if gene in child:
                    continue
                if fill_pointer >= size:
                    fill_pointer = 0
                child[fill_pointer] = gene
                fill_pointer += 1
            return child

        return make_child(parent1, parent2), make_child(parent2, parent1)

    def _swap_mutation(self, individual: np.ndarray) -> None:
        i, j = sorted(self.random.sample(range(len(individual)), 2))
        individual[i], individual[j] = individual[j], individual[i]

    # --- Helpers ---------------------------------------------------------
    def _tour_distance(self, tour: np.ndarray) -> float:
        cycle = self._close_tour(tour)
        rolled = np.roll(cycle, -1)
        lower = np.minimum(cycle, rolled)
        upper = np.maximum(cycle, rolled)
        return float(np.sum(self.distance_matrix[lower, upper]))

    @staticmethod
    def _close_tour(tour: np.ndarray) -> np.ndarray:
        if tour[0] == tour[-1]:
            return tour
        return np.append(tour, tour[0])

    def _generation_stats(
        self, generation: int, fitness: Sequence[float], best_distance: float
    ) -> dict:
        return {
            "generation": generation,
            "best_distance": float(best_distance),
            "mean_distance": float(np.mean(fitness)),
            "std_distance": float(np.std(fitness)) if len(fitness) > 1 else 0.0,
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
                except Exception:
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

    def __del__(self):  # pragma: no cover
        try:
            self._stop_logging()
        except Exception:
            pass
