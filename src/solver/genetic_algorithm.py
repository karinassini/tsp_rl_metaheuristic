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
from src.solver.initial_solution import nearest_neighbour_tour


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
    crossover_method: str = "smx"
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
    survivor_selection_method: str = "tournament"  # options: "tournament", "best_improves"


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
    def run(self) -> Tuple[List[int], float, List[dict], float]:
        """Execute the genetic algorithm and return the best tour found."""
        try:
            
            start_time = time.perf_counter()
            population = self._create_initial_population()
            elapsed = time.perf_counter() - start_time
            self.logger.info("Initial population generated in %.4f seconds", elapsed)

            fitness = [self._tour_distance(individual) for individual in population]
            self.logger.info("Initial population created with %d individuals, avg fitness=%.4f", len(population), np.mean(fitness))
            
            self.logger.info("Fitness values: %s", fitness)
            best_idx = int(np.argmin(fitness))
            best_tour = population[best_idx]
            best_distance = fitness[best_idx]
            history: List[dict] = [self._generation_stats(0, fitness, best_distance)]
            self._record_best_known_hit(best_distance, 0)

            stagnation_counter = 0
            generation = 0
            start_time = time.perf_counter()
            last_generation = 0
            max_generations = max(1, self.config.max_generations)
            population_limit = max(1, self.config.population_size)

            if self.best_known_hit_generation is None:
                while generation < max_generations:
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
            elapsed = time.perf_counter() - start_time
            self.logger.info("Quantity of crossovers performed: %d", self._crossovers)
            return closed_tour.tolist(), float(best_distance), history, elapsed
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
            population = self._marl_initial_population()
            fitness = [self._tour_distance(individual) for individual in population]
            self.logger.info("Initial population created with %d individuals, avg fitness=%.4f", len(population), np.mean(fitness))
            
            #self.config.population_size = len(population) # Test
            #self.logger.info("Initial population size from MARL reset: %d", self.config.population_size)
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

            assert len(population_method) == len(unique_population), "Initial population uniqueness check failed."

            attempts = 0
            max_attempts = max(10 * self.config.population_size, 100)
            while len(population) < self.config.population_size and attempts < max_attempts:
                attempts += 1
                candidate = self._nearest_random_individual()
                before = len(population)
                self._append_unique_candidate(population, seen, candidate)
                if len(population) == before:
                    continue

            if population:
                fitness_snapshot = [self._tour_distance(individual) for individual in population]
                mean_distance = float(np.mean(fitness_snapshot))
                std_distance = float(np.std(fitness_snapshot)) if len(fitness_snapshot) > 1 else 0.0
                self.logger.info(
                    "Initial population mean distance: %.4f (std=%.4f)",
                    mean_distance,
                    std_distance,
                )

            self.logger.info("Initial population size after padding: %d", len(population))
        else:
            self.config.population_size = len(population)
            self.logger.info("Initial population size: %d", len(population))

        return population

    def _nearest_random_population(self) -> List[np.ndarray]:
        return [self._nearest_random_individual()]

    def _nearest_random_individual(self) -> np.ndarray:
        seed_tour = nearest_neighbour_tour(self.graph)
        return np.asarray(seed_tour[:-1], dtype=int)

    def _marl_candidate_list_limit(self) -> Optional[int]:
        if self.n_cities <= 200:
            return max(1, self.n_cities // 4)
        return 50
    
    def _distance_ranked_actions(
        self,
        state: int,
        available: Sequence[int]
        ) -> List[int]:
        limit = self.config.marl_top_k
        candidates = list(available)
        if not candidates:
            return []
        candidates.sort(key=lambda idx: self._distance(state, idx))
        if limit is None or limit <= 0 or len(candidates) <= limit:
            return candidates
        return candidates[:limit]
    
    def _marl_initial_population(self) -> List[np.ndarray]:
        cfg = self.config
        self.logger.info("Starting MARL-based initial population generation with %d agents for up to %d populations", cfg.marl_agents, cfg.population_size)
        q_table = np.ones((self.n_cities, self.n_cities), dtype=float)
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

            if iteration_best_distance + 1e-9 < best_distance:
                best_distance = iteration_best_distance
                self._marl_update_q_table(q_table, iteration_best_tour, iteration_best_distance)

            for tour, distance in iteration_tours:
                if distance  <= best_distance + 1e-9:
                    candidate_set.append((tour.copy(), distance))

        candidate_set.sort(key=lambda item: item[1])
        unique_population: List[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()

        for tour, _distance in candidate_set:
            opt_refined = self._two_opt_junction_links(tour.copy(), 5)
            self._append_unique_candidate(unique_population, seen, opt_refined)
            if len(unique_population) >= self.config.population_size:
                break

            # nich_refined = self._nich_local_search(opt_refined.copy())
            # self._append_unique_candidate(unique_population, seen, nich_refined)
            # if len(unique_population) >= self.config.population_size:
            #     break

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
            candidate_actions = self._distance_ranked_actions(current, unvisited)
            
            # Next action will take in consideration ranked distance candidates
            action = self._marl_select_action(current, list(candidate_actions), q_table)
            tour.append(action)
            unvisited.remove(action)

        return np.asarray(tour, dtype=int)

    def _marl_candidate_actions(
        self,
        state: int,
        available: Sequence[int],
        q_table: np.ndarray,
    ) -> List[int]:
        candidates = list(available)
        limit = self._marl_candidate_list_limit()
        if limit is None or limit <= 0 or len(candidates) <= limit:
            return candidates
        weights = self._marl_softmax(state, candidates, q_table)
        ranked = sorted(
            zip(candidates, weights),
            key=lambda item: item[1],
            reverse=True,
        )
        return [idx for idx, _weight in ranked[:limit]]

    def _marl_select_action(self, state: int, available: List[int], q_table: np.ndarray) -> int:
        if not available:
            return state

        candidate_actions = self._marl_candidate_actions(state, available, q_table)

        if self.random.random() < self.config.marl_epsilon:
            return candidate_actions[0]
            #return random.choice(candidate_actions)

        weights = self._marl_softmax(state, candidate_actions, q_table)
        threshold = self.random.random()
        cumulative = 0.0
        for idx, weight in zip(candidate_actions, weights):
            cumulative += weight
            if cumulative >= threshold:
                return idx
        return candidate_actions[-1]

    def _marl_softmax(self, state: int, available: List[int], q_table: np.ndarray) -> List[float]:
        beta = max(0.1, self.config.marl_softmax_beta)
        q_values = np.asarray([q_table[state, idx] for idx in available], dtype=float)
        inverse_distances = np.asarray(
            [1.0 / max(self._distance(state, idx), 1e-9) for idx in available],
            dtype=float,
        )
        energy = q_values * np.power(inverse_distances, beta)
        energy -= float(np.max(energy))
        exp_values = np.exp(energy)
        total = float(np.sum(exp_values))
        if total <= 0:
            return [1.0 / len(available)] * len(available)
        return (exp_values / total).tolist()


    def _marl_update_q_table(self, q_table: np.ndarray, tour: np.ndarray, tour_distance: float) -> None:
        #reward = 1.0 / max(tour_distance, 1e-9) # the paper implements something fixed
        reward = self.config.marl_reward
        lr = self.config.marl_learning_rate
        cycle = self._close_tour(tour)
        for current, nxt in zip(cycle, np.roll(cycle, -1)):
            old_value = q_table[current, nxt]
            # Replace the constant‑reward update with standard Q‑learning using your marl_discount
            q_table[current, nxt] = old_value + lr * reward 


    def _two_opt_junction_links(
        self,
        tour: np.ndarray,
        max_junction_swaps: Optional[int] = None,
    ) -> np.ndarray:
        """Run 2-opt swaps only on intersecting edges (junction links) per MARL paper."""

        coords = getattr(self.graph, "coords", None)
        fallback_passes = max(
            1,
            int(max_junction_swaps or self.config.marl_two_opt_passes or 1),
        )
        if not coords:
            return self._two_opt_improve(tour, fallback_passes)

        if any(coords[int(node)] is None for node in tour):
            return self._two_opt_improve(tour, fallback_passes)

        improved = tour.copy()
        n = len(improved)
        if n < 4:
            return improved

        limit = max(1, fallback_passes)

        def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def edges_cross(a_idx: int, b_idx: int, c_idx: int, d_idx: int) -> bool:
            pa = coords[a_idx]
            pb = coords[b_idx]
            pc = coords[c_idx]
            pd = coords[d_idx]
            if pa is None or pb is None or pc is None or pd is None:
                return False
            o1 = orientation(pa, pb, pc)
            o2 = orientation(pa, pb, pd)
            o3 = orientation(pc, pd, pa)
            o4 = orientation(pc, pd, pb)
            eps = 1e-9
            return (o1 * o2 < -eps) and (o3 * o4 < -eps)

        swaps = 0
        while swaps < limit:
            improved_this_pass = False
            for i in range(n):
                a_idx = int(improved[i])
                b_idx = int(improved[(i + 1) % n])
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c_idx = int(improved[j])
                    d_idx = int(improved[(j + 1) % n])
                    if len({a_idx, b_idx, c_idx, d_idx}) < 4:
                        continue
                    if edges_cross(a_idx, b_idx, c_idx, d_idx):
                        improved[i + 1 : j + 1] = improved[i + 1 : j + 1][::-1]
                        swaps += 1
                        improved_this_pass = True
                        break
                if improved_this_pass or swaps >= limit:
                    break
            if not improved_this_pass:
                break
        return improved
    
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

    def _nich_local_search(self, tour: np.ndarray) -> np.ndarray:
        hull = self._convex_hull_indices()
        if not hull:
            return tour

        hull_cycle = hull.copy()
        hull_set = set(hull_cycle)
        nich_tour = hull_cycle.copy()
        remaining = [int(node) for node in tour.tolist() if int(node) not in hull_set]
        if not remaining:
            return np.asarray(nich_tour, dtype=int)

        for node in remaining:
            best_pos: Optional[int] = None
            best_delta = float("inf")
            cycle_len = len(nich_tour)

            if cycle_len == 0:
                nich_tour.append(node)
                continue

            for i in range(cycle_len):
                a = nich_tour[i]
                b = nich_tour[(i + 1) % cycle_len] if cycle_len > 1 else nich_tour[0]
                delta = self._distance(a, node) + self._distance(node, b)
                if cycle_len > 1:
                    delta -= self._distance(a, b)
                if delta < best_delta:
                    best_delta = delta
                    best_pos = i + 1

            insert_at = best_pos if best_pos is not None else len(nich_tour)
            nich_tour.insert(insert_at % (len(nich_tour) + 1), node)

        return np.asarray(nich_tour, dtype=int)

    def _convex_hull_indices(self) -> List[int]:
        coords = getattr(self.graph, "coords", None)
        if not coords:
            return []

        points: List[tuple[float, float, int]] = []
        for idx, coord in enumerate(coords):
            if coord is None:
                continue
            x, y = coord
            points.append((float(x), float(y), idx))

        if len(points) <= 1:
            return [idx for *_ignored, idx in points]

        points.sort(key=lambda item: (item[0], item[1], item[2]))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def build_half(seq):
            half: List[tuple[float, float, int]] = []
            for pt in seq:
                while len(half) >= 2 and cross(half[-2], half[-1], pt) <= 0:
                    half.pop()
                half.append(pt)
            return half

        lower = build_half(points)
        upper = build_half(reversed(points))
        hull = lower[:-1] + upper[:-1]

        seen: set[int] = set()
        ordered_indices: List[int] = []
        for _, _, idx in hull:
            if idx in seen:
                continue
            ordered_indices.append(idx)
            seen.add(idx)
        if not ordered_indices and points:
            ordered_indices = [points[0][2]]
        return ordered_indices

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
                new_fitness.append(
                    self._fitness_with_cache(child2, fitness_cache)
                )

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
        """

        method = (self.config.crossover_method or "smx").lower()
        if method == "smx":
            return self._smx_crossover(parent1, parent2)
        if method == "ordered":
            child, _ = self._ordered_crossover(parent1, parent2)
            return child
        raise ValueError(
            f"Unknown crossover_method '{self.config.crossover_method}'. Supported: smx, ordered."
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
            if inserted_this_round == 0 and parent_positions[0] >= size and parent_positions[1] >= size:
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
