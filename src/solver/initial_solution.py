"""Initial solution builders for TSP heuristics.

This module groups reusable constructors that create feasible tours before a
metaheuristic refines them. The nearest-neighbour strategy implemented here is
the same approach referenced in Aarts & Lenstra (2003) and subsequent work,
where a partial tour is extended by repeatedly appending the closest
unvisited city – effectively a Prim-like growth of a Hamiltonian cycle.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.structures.graph import Graph


@dataclass
class QLearningConfig:
    alpha: float = 0.4
    gamma: float = 0.8
    epsilon: float = 0.4
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 3000
    cache_dir: Optional[Path] = None


def _reward_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Build the reward matrix r(s,a) = M_i / d_ij as described in the paper."""
    with np.errstate(divide="ignore"):
        mean_dist = distance_matrix.mean(axis=1)
        rewards = np.divide(
            mean_dist[:, None], distance_matrix, where=distance_matrix > 0
        )
    rewards[np.isinf(rewards)] = 0.0
    rewards[np.isnan(rewards)] = 0.0
    return rewards


def _epsilon_greedy_action(
    q_values: np.ndarray,
    available_actions: List[int],
    epsilon: float,
) -> int:
    if not available_actions:
        raise ValueError("No available actions to choose from.")
    if random.random() < epsilon:
        return random.choice(available_actions)
    best_action = max(available_actions, key=lambda action: q_values[action])
    return best_action


def _q_learning(
    distance_matrix: np.ndarray,
    cfg: QLearningConfig,
) -> np.ndarray:
    n = distance_matrix.shape[0]
    rewards = _reward_matrix(distance_matrix)
    q_table = np.zeros_like(distance_matrix)

    epsilon = cfg.epsilon
    for _ in range(cfg.episodes):
        start = random.randrange(n)
        current = start
        unvisited = set(range(n))
        unvisited.remove(current)
        while unvisited:
            actions = list(unvisited)
            action = _epsilon_greedy_action(q_table[current], actions, epsilon)
            reward = rewards[current, action]
            unvisited.remove(action)
            next_state = action
            max_future = (
                q_table[next_state, list(unvisited)].max() if unvisited else 0.0
            )
            q_table[current, action] = (1 - cfg.alpha) * q_table[
                current, action
            ] + cfg.alpha * (reward + cfg.gamma * max_future)
            current = next_state

        # close the tour by returning to start
        return_reward = rewards[current, start]
        q_table[current, start] = (1 - cfg.alpha) * q_table[
            current, start
        ] + cfg.alpha * (return_reward)

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    return q_table


def _cache_key(distance_matrix: np.ndarray, cfg: QLearningConfig) -> str:
    payload: Dict[str, Tuple] = {
        "shape": distance_matrix.shape,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "epsilon": cfg.epsilon,
        "epsilon_min": cfg.epsilon_min,
        "epsilon_decay": cfg.epsilon_decay,
        "episodes": cfg.episodes,
        "hash": hash(distance_matrix.tobytes()),
    }
    return json.dumps(payload, sort_keys=True)


def _load_q_table(
    distance_matrix: np.ndarray, cfg: QLearningConfig
) -> Optional[np.ndarray]:
    if cfg.cache_dir is None:
        return None
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(distance_matrix, cfg)
    cache_file = cfg.cache_dir / f"q_table_{abs(hash(key))}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    return None


def _save_q_table(
    distance_matrix: np.ndarray, q_table: np.ndarray, cfg: QLearningConfig
) -> None:
    if cfg.cache_dir is None:
        return
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(distance_matrix, cfg)
    cache_file = cfg.cache_dir / f"q_table_{abs(hash(key))}.npy"
    np.save(cache_file, q_table)


def q_learning_tour(
    graph: Graph,
    *,
    start: Optional[int] = None,
    cfg: Optional[QLearningConfig] = None,
) -> np.ndarray:
    """Generate a tour using a Q-learning policy trained on the TSP instance."""

    cfg = cfg or QLearningConfig()
    distance_matrix = np.asarray(graph.build_adj_matrix_full(), dtype=float)

    if start is not None and start not in graph.nodes:
        raise ValueError("Provided start city is not part of the graph.")

    q_table = _load_q_table(distance_matrix, cfg)
    if q_table is None:
        q_table = _q_learning(distance_matrix, cfg)
        _save_q_table(distance_matrix, q_table, cfg)

    nodes = list(graph.nodes)
    start_city = start if start is not None else random.choice(nodes)
    tour = [start_city]
    unvisited = set(nodes)
    unvisited.remove(start_city)
    current = start_city

    while unvisited:
        actions = list(unvisited)
        next_city = max(actions, key=lambda city: q_table[current][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    tour.append(start_city)
    return np.asarray(tour, dtype=int), distance_matrix


def nearest_neighbour_tour(graph: Graph, start: Optional[int] = None) -> np.ndarray:
    """Construct a TSP tour using the Nearest Neighbour heuristic.

    The procedure mirrors the "NrNbr" strategy cited in the literature: begin
    with a (possibly random) seed city, then repeatedly append the closest
    vertex that has not been visited yet. The final tour closes the cycle by
    returning to the starting city.

    Parameters
    ----------
    graph:
            The problem graph that exposes an adjacency matrix through
            ``get_distance_matrix``.
    start:
            Optional index of the starting city. When ``None`` a random city is
            selected.

    Returns
    -------
    numpy.ndarray
            A sequence of node indices representing a full Hamiltonian cycle where
            the first and last entries coincide.
    """

    nodes: List[int] = list(graph.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes – cannot build an initial tour.")

    distance_matrix = np.asarray(graph.get_distance_matrix())

    start_city = start if start is not None else random.choice(nodes)
    if start_city not in nodes:
        raise ValueError("Provided start city is not part of the graph.")

    unvisited = set(nodes)
    unvisited.remove(start_city)
    tour = [start_city]
    current = start_city

    while unvisited:
        # Find the nearest unvisited neighbour to the current city.
        next_city = min(
            unvisited,
            key=lambda candidate: distance_matrix[current][candidate],
        )
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    tour.append(start_city)
    return np.asarray(tour, dtype=int)


def available_initialisation_methods() -> Iterable[str]:
    """Expose the supported initial-solution strategies by name."""

    return ("nearest_neighbour", "random", "greedy", "q_learning")
