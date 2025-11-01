"""Genetic algorithm solver for the Traveling Salesman Problem.

The implementation follows a standard steady-generation approach with
ordered crossover, swap mutation, and tournament selection. It allows seeding
the initial population with a nearest-neighbour tour to speed convergence while
keeping the remaining individuals randomised.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.structures.graph import Graph
from src.solver.initial_solution import nearest_neighbour_tour


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


class GeneticTSPSolver:
    """Metaheuristic solver based on a Genetic Algorithm."""

    def __init__(self, graph: Graph, config: Optional[GeneticAlgorithmConfig] = None):
        self.graph = graph
        self.distance_matrix = np.asarray(graph.get_distance_matrix())
        self.n_cities = graph.n_nodes
        self.config = config or GeneticAlgorithmConfig()
        self.random = random.Random(self.config.seed)

        if self.n_cities < 3:
            raise ValueError("Genetic algorithm requires at least 3 cities.")
        if self.config.population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if self.config.tournament_size > self.config.population_size:
            raise ValueError("Tournament size cannot exceed population size.")

    # --- Public API -----------------------------------------------------
    def run(self) -> Tuple[List[int], float, List[dict]]:
        """Execute the genetic algorithm and return the best tour found."""

        population = self._create_initial_population()
        fitness = [self._tour_distance(individual) for individual in population]

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

    # --- Genetic operators ----------------------------------------------
    def _create_initial_population(self) -> List[np.ndarray]:
        population: List[np.ndarray] = []
        seed_tour = nearest_neighbour_tour(self.graph)
        population.append(seed_tour[:-1])

        while len(population) < self.config.population_size:
            tour = list(range(self.n_cities))
            self.random.shuffle(tour)
            population.append(np.asarray(tour, dtype=int))

        return population

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
            parent1 = self._tournament_select(population, fitness)
            parent2 = self._tournament_select(population, fitness)

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

    def _tournament_select(
        self, population: Sequence[np.ndarray], fitness: Sequence[float]
    ) -> np.ndarray:
        competitors = self.random.sample(
            range(len(population)), self.config.tournament_size
        )
        best_idx = min(competitors, key=lambda idx: fitness[idx])
        return population[best_idx].copy()

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
        return float(np.sum(self.distance_matrix[cycle, rolled]))

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
