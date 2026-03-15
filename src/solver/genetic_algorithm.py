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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from logging.handlers import QueueHandler, QueueListener

from src.structures.graph import Graph
from src.solver.initial_solution import nearest_neighbour_tour, marl_initial_population


logger = logging.getLogger(__name__)


@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism: bool = True
    elite_fraction: float = 0.05
    seed: Optional[int] = None
    max_generations: int = 300
    stagnation_limit: Optional[int] = None
    selection_method: str = "roulette"
    crossover_method: str = "smx"  # options: "smx", "ordered", "er"
    rank_selection_pressure: float = 1.7
    truncation_ratio: float = 0.3
    initialization_method: str = "marl"  # Options: "nearest_random", "marl"
    marl_agents: int = 6
    marl_iterations: int = 40
    marl_epsilon: float = 0.15
    marl_softmax_beta: float = 2.0
    marl_learning_rate: float = 0.4
    marl_discount: float = 0.6
    marl_reward: float = 1.0
    marl_candidate_ratio: float = 1.5  # how many MARL tours relative to population size
    marl_two_opt_passes: int = 1
    marl_top_k: Optional[int] = 5
    log_dir: Optional[str] = None
    smx_max_segment_length: Optional[int] = None
    survivor_selection_method: str = (
        "tournament"  # options: "tournament", "best_improves"
    )


class GeneticTSPSolver:
    """Metaheuristic solver based on a Genetic Algorithm."""

    def __init__(
        self,
        graph: Graph,
        save_dir,
        config: Optional[GeneticAlgorithmConfig] = None,
        best_known_distance: Optional[float] = None,
    ):
        self.graph = graph
        self.distance_matrix = np.asarray(graph.build_adj_matrix_full(), dtype=float)
        self.n_cities = graph.n_nodes
        self.config = config or GeneticAlgorithmConfig()
        self.random = random.Random(self.config.seed)
        self.save_dir = save_dir
        self._crossovers = 0
        self.best_known_distance = best_known_distance
        self.best_known_hit_generation: Optional[int] = None

        if self.n_cities < 3:
            raise ValueError("Genetic algorithm requires at least 3 cities.")
        if self.config.population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if self.config.tournament_size > self.config.population_size:
            raise ValueError("Tournament size cannot exceed population size.")
        if not 0.0 < self.config.truncation_ratio <= 1.0:
            raise ValueError("truncation_ratio must be in (0, 1].")
        if not 0.0 < self.config.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1].")

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
                logs_dir,
                f"ga_steps_{self.timestamp}_{self.config.initialization_method}_{os.getpid()}_{id(self)}.log",
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
    def run(
        self, time_limit_seconds: Optional[float] = None
    ) -> Tuple[List[int], float, List[dict], float]:
        """Execute the genetic algorithm with an optional wall-clock limit."""
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

                    parent1 = self._selection_operator(population, fitness)
                    parent2 = self._selection_operator(population, fitness)

                    child = self._crossover(parent1, parent2)
                    self._crossovers += 1

                    if self.random.random() < self.config.mutation_rate:
                        self._swap_mutation(child)

                    child_distance = self._tour_distance(child)

                    survivor_method = (
                        self.config.survivor_selection_method or "tournament"
                    ).lower()

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
                        hit_best = self._record_best_known_hit(
                            best_distance,
                            generation,
                        )
                    else:
                        stagnation_counter += 1

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

                    if (
                        self.config.stagnation_limit is not None
                        and stagnation_counter >= self.config.stagnation_limit
                    ):
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

    def _record_best_known_hit(self, distance: float, generation: int) -> bool:
        """Track when the best-known tour length has been matched or beaten."""
        if self.best_known_distance is None:
            return False
        if self.best_known_hit_generation is not None:
            return False
        tolerance = 1e-6
        if distance <= self.best_known_distance + tolerance:
            self.best_known_hit_generation = generation
            self.logger.info(
                "Best-known distance %.4f reached at generation %d (target %.4f)",
                distance,
                generation,
                self.best_known_distance,
            )
            return True
        return False

    # --- Genetic operators ----------------------------------------------
    def _create_initial_population(self) -> List[np.ndarray]:
        method = (self.config.initialization_method or "nearest_random").lower()
        if method == "marl":
            self.logger.info("Initial population method: MARL")
            population = marl_initial_population(
                graph=self.graph,
                rng=self.random,
                population_size=self.config.population_size,
                marl_agents=self.config.marl_agents,
                marl_iterations=self.config.marl_iterations,
                marl_candidate_ratio=self.config.marl_candidate_ratio,
                marl_two_opt_passes=self.config.marl_two_opt_passes,
                marl_top_k=self.config.marl_top_k,
                marl_reward=self.config.marl_reward,
                marl_learning_rate=self.config.marl_learning_rate,
                marl_softmax_beta=self.config.marl_softmax_beta,
                marl_epsilon=self.config.marl_epsilon,
                log=self.logger,
            )
            fitness = [self._tour_distance(individual) for individual in population]
            self.logger.info(
                "Initial population created with %d individuals, avg fitness=%.4f",
                len(population),
                np.mean(fitness),
            )
        else:
            self.logger.info("Initial population method: nearest-random")
            population = self._nearest_random_population()

        increase_pop = False
        if increase_pop:
            population_method = population.copy()
            seen: set[tuple[int, ...]] = set()
            unique_population: List[np.ndarray] = []
            for individual in population:
                self._append_unique_candidate(unique_population, seen, individual)
            population = unique_population

            assert len(population_method) == len(
                unique_population
            ), "Initial population uniqueness check failed."

            attempts = 0
            max_attempts = max(10 * self.config.population_size, 100)
            while (
                len(population) < self.config.population_size
                and attempts < max_attempts
            ):
                attempts += 1
                candidate = self._nearest_random_individual()
                before = len(population)
                self._append_unique_candidate(population, seen, candidate)
                if len(population) == before:
                    continue

            if population:
                fitness_snapshot = [
                    self._tour_distance(individual) for individual in population
                ]
                mean_distance = float(np.mean(fitness_snapshot))
                std_distance = (
                    float(np.std(fitness_snapshot))
                    if len(fitness_snapshot) > 1
                    else 0.0
                )
                self.logger.info(
                    "Initial population mean distance: %.4f (std=%.4f)",
                    mean_distance,
                    std_distance,
                )

            self.logger.info(
                "Initial population size after padding: %d", len(population)
            )
        else:
            self.config.population_size = len(population)
            self.logger.info("Initial population size: %d", len(population))

        return population

    def _nearest_random_population(self) -> List[np.ndarray]:
        return [self._nearest_random_individual()]

    def _nearest_random_individual(self) -> np.ndarray:
        seed_tour = nearest_neighbour_tour(self.graph)
        return np.asarray(seed_tour[:-1], dtype=int)

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

    def _append_unique_candidate(
        self,
        population: List[np.ndarray],
        seen: set[tuple[int, ...]],
        candidate: np.ndarray,
    ) -> None:
        key = tuple(candidate.tolist())
        if key in seen:
            return
        population.append(candidate)
        seen.add(key)

    def _next_generation(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> Tuple[List[np.ndarray], List[float], int]:
        new_population: List[np.ndarray] = []
        new_fitness: List[float] = []
        fitness_cache: Dict[tuple[int, ...], float] = {
            tuple(individual.tolist()): float(value)
            for individual, value in zip(population, fitness)
        }

        if self.config.elitism:
            best_idx = int(np.argmin(fitness))
            elite = population[best_idx].copy()
            new_population.append(elite)
            new_fitness.append(self._fitness_with_cache(elite, fitness_cache))

        while len(new_population) < self.config.population_size:
            parent1 = self._selection_operator(population, fitness)
            parent2 = self._selection_operator(population, fitness)

            if self.random.random() < self.config.crossover_rate:
                self._crossovers += 1
                child1, child2 = self._ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if self.random.random() < self.config.mutation_rate:
                self._swap_mutation(child1)
            if self.random.random() < self.config.mutation_rate:
                self._swap_mutation(child2)

            new_population.append(child1)
            new_fitness.append(self._fitness_with_cache(child1, fitness_cache))

            if len(new_population) < self.config.population_size:
                new_population.append(child2)
                new_fitness.append(self._fitness_with_cache(child2, fitness_cache))

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
            (1.0 / n) * (sp - (2 * sp - 2) * (rank / denom)) for rank in range(n)
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

    def _tournament_survivor_selection(
        self,
        population: Sequence[np.ndarray],
        fitness: Sequence[float],
        population_limit: int,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Select survivors via tournaments over the combined parent/offspring pool."""

        if len(population) <= population_limit:
            return list(population), list(map(float, fitness))

        survivors: List[np.ndarray] = []
        survivor_fitness: List[float] = []
        available_indices = list(range(len(population)))
        tournament_size = max(1, self.config.tournament_size)

        while len(survivors) < population_limit and available_indices:
            competitors = self.random.sample(
                available_indices,
                min(tournament_size, len(available_indices)),
            )
            best_idx = min(competitors, key=lambda idx: fitness[idx])
            survivors.append(population[best_idx].copy())
            survivor_fitness.append(float(fitness[best_idx]))
            available_indices.remove(best_idx)

        if len(survivors) < population_limit and available_indices:
            remaining = sorted(available_indices, key=lambda idx: fitness[idx])
            for idx in remaining:
                if len(survivors) >= population_limit:
                    break
                survivors.append(population[idx].copy())
                survivor_fitness.append(float(fitness[idx]))

        return survivors, survivor_fitness

    def _best_improves_survivor(
        self,
        population: Sequence[np.ndarray],
        fitness: Sequence[float],
        child: np.ndarray,
        child_distance: float,
        population_limit: int,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Accept the child only if it improves the current best tour.

        If accepted and the pool exceeds the limit, remove the worst individual.
        """

        pop_list: List[np.ndarray] = [individual.copy() for individual in population]
        fitness_list: List[float] = [float(value) for value in fitness]

        if not pop_list:
            pop_list.append(child.copy())
            fitness_list.append(float(child_distance))
            return pop_list, fitness_list

        best_current = min(fitness_list)
        tolerance = 1e-9
        if child_distance + tolerance < best_current:
            pop_list.append(child.copy())
            fitness_list.append(float(child_distance))

            if len(pop_list) > population_limit:
                worst_idx = int(np.argmax(fitness_list))
                del pop_list[worst_idx]
                del fitness_list[worst_idx]

        return pop_list, fitness_list

    def _fitness_to_weights(self, fitness: Sequence[float]) -> np.ndarray:
        values = np.asarray(fitness, dtype=float)
        weights = 1.0 / (values + 1e-12)
        weights[~np.isfinite(weights)] = 0.0
        if np.all(weights <= 0):
            return np.ones_like(weights)
        return weights

    def _fitness_with_cache(
        self,
        individual: np.ndarray,
        cache: Dict[tuple[int, ...], float],
    ) -> float:
        key = tuple(individual.tolist())
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = self._tour_distance(individual)
        cache[key] = value
        return value

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

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Dispatch crossover based on `crossover_method` config.

        Returns a single offspring to fit the current steady-state loop.
        Supported methods:
        - "smx": sequential mixing crossover (default)
        - "ordered": ordered crossover (first child)
        - "er": edge recombination crossover
        """

        method = (self.config.crossover_method or "smx").lower()
        if method == "smx":
            return self._smx_crossover(parent1, parent2)
        if method == "ordered":
            child, _ = self._ordered_crossover(parent1, parent2)
            return child
        if method == "er":
            return self._edge_recombination_crossover(parent1, parent2)
        raise ValueError(
            f"Unknown crossover_method '{self.config.crossover_method}'. Supported: smx, ordered, er."
        )

    def _smx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Sequential mixing crossover following the reference SMX flow chart."""

        size = len(parent1)
        offspring = -np.ones(size, dtype=int)
        parents = (parent1, parent2)
        parent_positions = [0, 0]  # how far we have read from each parent
        parent_index = 0  # which parent we currently pull from
        offspring_pos = 0  # next write position in the child
        seen: set[int] = set()  # genes already placed

        configured_segment = self.config.smx_max_segment_length
        if configured_segment is None or configured_segment <= 0:
            configured_segment = max(1, size // 4)
        max_segment = min(size, configured_segment)

        while offspring_pos < size:
            # If the active parent is exhausted, try the other; stop if both are done.
            if parent_positions[parent_index] >= size:
                other_index = 1 - parent_index
                if parent_positions[other_index] >= size:
                    break
                parent_index = other_index
                continue

            # Choose how many consecutive genes to copy from the current parent.
            segment_len = self.random.randint(1, max_segment)
            inserted_this_round = 0
            parent = parents[parent_index]
            pos = parent_positions[parent_index]

            # Stream genes from the current parent, skipping duplicates, until the segment ends.
            while (
                pos < size
                and inserted_this_round < segment_len
                and offspring_pos < size
            ):
                gene = int(parent[pos])
                pos += 1
                if gene in seen:
                    continue  # skip already placed genes
                offspring[offspring_pos] = gene
                offspring_pos += 1
                seen.add(gene)
                inserted_this_round += 1

            parent_positions[parent_index] = pos
            parent_index = 1 - parent_index  # alternate parents for the next segment

            # If nothing was inserted and both parents are spent, leave the loop.
            if (
                inserted_this_round == 0
                and parent_positions[0] >= size
                and parent_positions[1] >= size
            ):
                break

        # Fill any remaining slots with unseen genes in parent order (fallback completion).
        if offspring_pos < size:
            for parent in parents:
                for gene in parent:
                    gene = int(gene)
                    if gene in seen:
                        continue
                    offspring[offspring_pos] = gene
                    offspring_pos += 1
                    seen.add(gene)
                    if offspring_pos == size:
                        break
                if offspring_pos == size:
                    break

        if offspring_pos != size:
            raise ValueError("SMX crossover failed to construct a valid child tour")

        return offspring

    def _edge_recombination_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """Edge Recombination crossover (ER) preserving parent edges when possible."""

        size = len(parent1)
        edge_map: dict[int, set[int]] = {int(city): set() for city in parent1}

        def add_edge(a: int, b: int) -> None:
            edge_map.setdefault(a, set()).add(b)
            edge_map.setdefault(b, set()).add(a)

        for parent in (parent1, parent2):
            for idx, city in enumerate(parent):
                c = int(city)
                left = int(parent[(idx - 1) % size])
                right = int(parent[(idx + 1) % size])
                add_edge(c, left)
                add_edge(c, right)

        start_city = self.random.choice([int(parent1[0]), int(parent2[0])])
        child: list[int] = []
        unvisited: set[int] = set(int(c) for c in parent1)
        current = start_city

        while len(child) < size:
            child.append(current)
            unvisited.discard(current)

            for neighbors in edge_map.values():
                neighbors.discard(current)

            if not unvisited:
                break

            neighbors = [n for n in edge_map.get(current, set()) if n in unvisited]
            if neighbors:
                min_degree = None
                candidates: list[int] = []
                for n in neighbors:
                    deg = len(edge_map.get(n, set()))
                    if (min_degree is None) or (deg < min_degree):
                        min_degree = deg
                        candidates = [n]
                    elif deg == min_degree:
                        candidates.append(n)
                current = self.random.choice(candidates)
            else:
                current = self.random.choice(list(unvisited))

        if len(child) != size:
            raise ValueError("ER crossover failed to construct a valid child tour")

        return np.asarray(child, dtype=int)

    def _swap_mutation(self, individual: np.ndarray) -> None:
        i, j = sorted(self.random.sample(range(len(individual)), 2))
        individual[i], individual[j] = individual[j], individual[i]

    # --- Helpers ---------------------------------------------------------
    def _distance(self, i: int, j: int) -> float:
        lower = min(i, j)
        upper = max(i, j)
        return float(self.distance_matrix[lower, upper])

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
        self,
        generation: int,
        fitness: Sequence[float],
        best_distance: float,
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
            for handler in getattr(
                self._queue_listener, "handlers", ()
            ):  # pragma: no branch
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
