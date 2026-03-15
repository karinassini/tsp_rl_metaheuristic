"""Initial solution builders for TSP heuristics.

This module groups reusable constructors that create feasible tours before a
metaheuristic refines them. The nearest-neighbour strategy implemented here is
the same approach referenced in Aarts & Lenstra (2003) and subsequent work,
where a partial tour is extended by repeatedly appending the closest
unvisited city – effectively a Prim-like growth of a Hamiltonian cycle.
"""

from __future__ import annotations

import json
import logging
import random
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
from config import QLearningConfig
from src.structures.graph import Graph

logger = logging.getLogger(__name__)


# MARL helpers for GA initialisation
def _distance(distance_matrix: np.ndarray, i: int, j: int) -> float:
    lower = min(i, j)
    upper = max(i, j)
    return float(distance_matrix[lower, upper])


def _tour_distance(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    cycle = tour if tour[0] == tour[-1] else np.append(tour, tour[0])
    rolled = np.roll(cycle, -1)
    lower = np.minimum(cycle, rolled)
    upper = np.maximum(cycle, rolled)
    return float(np.sum(distance_matrix[lower, upper]))


def _append_unique_candidate(
    population: List[np.ndarray],
    seen: Set[Tuple[int, ...]],
    candidate: np.ndarray,
) -> None:
    key = tuple(candidate.tolist())
    if key in seen:
        return
    population.append(candidate)
    seen.add(key)


def _two_opt_improve(
    distance_matrix: np.ndarray, tour: np.ndarray, passes: int
) -> np.ndarray:
    best = tour
    best_distance = _tour_distance(best, distance_matrix)
    for _ in range(max(1, passes)):
        improved = False
        length = len(best)
        for i in range(1, length - 2):
            for j in range(i + 1, length - 1):
                if j - i == 1:
                    continue
                candidate = best.copy()
                candidate[i:j] = candidate[i:j][::-1]
                candidate_distance = _tour_distance(candidate, distance_matrix)
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


def _two_opt_junction_links(
    graph: Graph,
    distance_matrix: np.ndarray,
    tour: np.ndarray,
    max_junction_swaps: Optional[int] = None,
    two_opt_passes: int = 1,
) -> np.ndarray:
    coords = getattr(graph, "coords", None)
    fallback_passes = max(1, int(max_junction_swaps or two_opt_passes or 1))
    if not coords:
        return _two_opt_improve(distance_matrix, tour, fallback_passes)

    if any(coords[int(node)] is None for node in tour):
        return _two_opt_improve(distance_matrix, tour, fallback_passes)

    improved = tour.copy()
    n = len(improved)
    if n < 4:
        return improved

    limit = max(1, fallback_passes)

    def orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
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


def _convex_hull_indices(graph: Graph) -> List[int]:
    coords = getattr(graph, "coords", None)
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

    seen: Set[int] = set()
    ordered_indices: List[int] = []
    for _, _, idx in hull:
        if idx in seen:
            continue
        ordered_indices.append(idx)
        seen.add(idx)
    if not ordered_indices and points:
        ordered_indices = [points[0][2]]
    return ordered_indices


def _nich_local_search(
    graph: Graph, distance_matrix: np.ndarray, tour: np.ndarray
) -> np.ndarray:
    hull = _convex_hull_indices(graph)
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
            delta = _distance(distance_matrix, a, node) + _distance(
                distance_matrix, node, b
            )
            if cycle_len > 1:
                delta -= _distance(distance_matrix, a, b)
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1

        insert_at = best_pos if best_pos is not None else len(nich_tour)
        nich_tour.insert(insert_at % (len(nich_tour) + 1), node)

    return np.asarray(nich_tour, dtype=int)


def _marl_candidate_list_limit(n_cities: int) -> int:
    if n_cities <= 200:
        # Paper: MCL size = n/4 (use ceil to avoid zero for very small n)
        return max(1, int(math.ceil(n_cities / 4)))
    return 50


def _distance_ranked_actions(
    distance_matrix: np.ndarray,
    state: int,
    available: Iterable[int],
    limit: Optional[int],
) -> List[int]:
    candidates = list(available)
    if not candidates:
        return []
    candidates.sort(key=lambda idx: _distance(distance_matrix, state, idx))
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return candidates
    return candidates[:limit]


def _marl_candidate_lists(
    distance_matrix: np.ndarray,
    state: int,
    available: Iterable[int],
    marl_top_k: Optional[int],
) -> tuple[list[int], list[int]]:
    """Build the candidate list (CL) and mutation candidate list (MCL).

    CL = closest ``marl_top_k`` cities (or all if ``marl_top_k`` is falsy).
    MCL = broader set limited by ``_marl_candidate_list_limit`` as in the paper.
    Both lists are ordered by distance so sampling respects nearest-first ties.
    """

    mutation_limit = _marl_candidate_list_limit(len(distance_matrix))
    mutation_list = _distance_ranked_actions(
        distance_matrix,
        state,
        available,
        mutation_limit,
    )

    if marl_top_k is None or marl_top_k <= 0:
        candidate_list = mutation_list
    else:
        candidate_list = mutation_list[:marl_top_k]

    return candidate_list, mutation_list


def _marl_softmax(
    distance_matrix: np.ndarray,
    state: int,
    available: List[int],
    q_table: np.ndarray,
    beta: float,
) -> List[float]:
    
    beta = max(0.1, beta)
    q_values = np.asarray([q_table[state, idx] for idx in available], dtype=float)
    inverse_distances = np.asarray(
        [1.0 / max(_distance(distance_matrix, state, idx), 1e-9) for idx in available],
        dtype=float,
    )
    energy = q_values * np.power(inverse_distances, beta)
    energy -= float(np.max(energy))
    exp_values = np.exp(energy)
    total = float(np.sum(exp_values))

    # Paper form: P(a|s_c) = exp_i / sum_{j!=i} exp_j. For sampling we
    # renormalise these ratios so they form a proper distribution.
    if total <= 0:
        return [1.0 / len(available)] * len(available)
    if len(available) == 1:
        return [1.0]

    raw = []
    for val in exp_values:
        denom = total - val # Exclude the current action's energy from the denominator as per the paper's formulation.
        ratio = val / denom if denom > 1e-12 else 1.0 
        raw.append(ratio)

    raw_sum = float(np.sum(raw))
    if raw_sum <= 0:
        return [1.0 / len(raw)] * len(raw)
    return (np.asarray(raw, dtype=float) / raw_sum).tolist()


def _marl_select_action(
    distance_matrix: np.ndarray,
    state: int,
    available: List[int],
    q_table: np.ndarray,
    rng: random.Random,
    epsilon: float,
    beta: float,
    marl_top_k: Optional[int],
    policy_epsilon_greedy_prob: float = 1.0,
) -> int:
    if not available:
        return state

    candidate_list, mutation_list = _marl_candidate_lists(
        distance_matrix,
        state,
        available,
        marl_top_k,
    )

    if not candidate_list:
        # Fall back to any remaining city if CL is empty.
        return rng.choice(mutation_list) if mutation_list else state

    weights = _marl_softmax(distance_matrix, state, candidate_list, q_table, beta)

    # Choose which policy to apply for this decision: with probability p use epsilon-greedy, otherwise softmax.
    use_eps = rng.random() < max(0.0, min(1.0, policy_epsilon_greedy_prob))

    if use_eps:
        # With small probability pick a random city from the larger MCL (exploration).
        if rng.random() < epsilon and mutation_list:
            return rng.choice(mutation_list)
        # Otherwise choose the argmax of P(a|s_c) within the CL (exploitation).
        best_idx = int(np.argmax(weights))
        return candidate_list[best_idx]

    # Softmax policy: sample inside the CL proportionally to P(a|s_c).
    threshold = rng.random()
    cumulative = 0.0
    for idx, weight in zip(candidate_list, weights):
        cumulative += weight
        if cumulative >= threshold:
            return idx
    return candidate_list[-1]


def _marl_construct_tour(
    distance_matrix: np.ndarray,
    q_table: np.ndarray,
    rng: random.Random,
    marl_top_k: Optional[int],
    epsilon: float,
    beta: float,
    policy_epsilon_greedy_prob: float = 1.0,
) -> np.ndarray:
    

    n_cities = distance_matrix.shape[0]
    start = rng.randrange(n_cities) # starts randomly from any city, as per the paper. This also ensures we get some diversity in the initial population when multiple MARL tours are generated.
    unvisited = set(range(n_cities))
    unvisited.remove(start) # disable action
    tour = [start]

    while unvisited: # for n times until we have a complete tour, the agent selects the next city to visit using the _marl_select_action function, which implements the action selection strategy based on the current Q-table and distance matrix. The selected city is then appended to the tour, and removed from the set of unvisited cities. This process continues until all cities have been visited, resulting in a complete tour.
        current = tour[-1]
        action = _marl_select_action(
            distance_matrix,
            current,
            list(unvisited),
            q_table,
            rng,
            epsilon,
            beta,
            marl_top_k,
            policy_epsilon_greedy_prob,
        )
        tour.append(action)
        unvisited.remove(action)

    return np.asarray(tour, dtype=int)


def _marl_update_q_table(
    distance_matrix: np.ndarray,
    q_table: np.ndarray,
    tour: np.ndarray,
    tour_distance: float,
    learning_rate: float,
    reward: float,
) -> None:
    lr = learning_rate
    cycle = tour if tour[0] == tour[-1] else np.append(tour, tour[0])
    for current, nxt in zip(cycle, np.roll(cycle, -1)):
        old_value = q_table[current, nxt]
        q_table[current, nxt] = old_value + lr * reward


def marl_initial_population(
    graph: Graph,
    rng: random.Random,
    *,
    population_size: int,
    marl_agents: int,
    marl_iterations: int,
    marl_candidate_ratio: float,
    marl_two_opt_passes: int,
    marl_top_k: Optional[int],
    marl_reward: float,
    marl_learning_rate: float,
    marl_softmax_beta: float,
    marl_epsilon: float,
    marl_policy_epsilon_greedy_prob: float = 1.0,
    log: Optional[logging.Logger] = None,
) -> List[np.ndarray]:
    """Generate a MARL-seeded population for the GA."""

    logger_obj = log

    distance_matrix = np.asarray(graph.build_adj_matrix_full(), dtype=float)
    n_cities = graph.n_nodes
    q_table = np.ones((n_cities, n_cities), dtype=float) # first step is to build a Q-table with all 1s, meaning the agent has no prior preference for any action in any state. This allows the learning process to start from a neutral baseline, where the agent can explore the environment and learn which actions lead to better outcomes based on the rewards received.
    candidate_set: List[tuple[np.ndarray, float]] = []
    best_distance = float("inf")
    target = max(1, int(round(marl_candidate_ratio * population_size)))


    for _ in range(max(1, marl_iterations)):
        iteration_tours: List[tuple[np.ndarray, float]] = []
        for _agent in range(max(1, marl_agents)):
            tour = _marl_construct_tour(
                distance_matrix,
                q_table,
                rng,
                marl_top_k,
                marl_epsilon,
                marl_softmax_beta,
                marl_policy_epsilon_greedy_prob,
            )
            distance = _tour_distance(tour, distance_matrix)
            iteration_tours.append((tour, distance))

        iteration_tours.sort(key=lambda item: item[1])
        if not iteration_tours:
            continue

        iteration_best_tour, iteration_best_distance = iteration_tours[0]

        if iteration_best_distance + 1e-9 < best_distance:
            if logger_obj is not None:
                logger_obj.info(
                    "New best MARL tour found with distance %.2f at iteration %d/%d",
                    iteration_best_distance,
                    _ + 1,
                    marl_iterations,
                )
            best_distance = iteration_best_distance
            _marl_update_q_table(
                distance_matrix,
                q_table,
                iteration_best_tour,
                iteration_best_distance,
                marl_learning_rate,
                marl_reward,
            )

        for tour, distance in iteration_tours:
            if distance <= best_distance + 1e-9 and len(candidate_set) < target:
                candidate_set.append((tour.copy(), distance))

    candidate_set.sort(key=lambda item: item[1])
    unique_population: List[np.ndarray] = []
    seen: Set[Tuple[int, ...]] = set()

    for tour, _distance_val in candidate_set:
        opt_refined = _two_opt_junction_links(
            graph,
            distance_matrix,
            tour.copy(),
            5,
            marl_two_opt_passes,
        )
        _append_unique_candidate(unique_population, seen, opt_refined)
        if len(unique_population) >= population_size:
            break

        nich_refined = _nich_local_search(graph, distance_matrix, opt_refined.copy())
        _append_unique_candidate(unique_population, seen, nich_refined)
        if len(unique_population) >= population_size:
            break

    return unique_population


def _reward_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Build the reward matrix r(s,a) = M_i / d_ij as described in the paper."""
    with np.errstate(divide="ignore"):
        mean_dist = distance_matrix.mean(axis=1)
        rewards = np.divide(
            mean_dist[:, None], distance_matrix, where=distance_matrix > 0
        )

    # Replace invalid entries before scaling.
    rewards[np.isinf(rewards)] = 0.0
    rewards[np.isnan(rewards)] = 0.0

    # Clamp extreme ratios (close cities lead to huge rewards).
    finite_values = rewards[np.isfinite(rewards)]
    if finite_values.size:
        upper_clip = np.percentile(finite_values, 99)
        rewards = np.clip(rewards, 0.0, upper_clip)

    # Log-scale to compress the dynamic range and normalise to [0, 1].
    rewards = np.log1p(rewards)
    max_val = rewards.max()
    if max_val > 0:
        rewards = rewards / max_val

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
    # best action = argmax_a Q(s,a)
    best_action = max(available_actions, key=lambda action: q_values[action])
    return best_action


def _deterministic_tour_cost(
    q_table: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    start_city: int = 0,
) -> Tuple[float, List[int]]:
    """Roll out the greedy policy encoded in ``q_table`` and return its cost."""

    n = q_table.shape[0]
    if n == 0:
        return 0.0, []
    start = start_city % n
    current = start
    unvisited = set(range(n))
    unvisited.remove(current)
    tour = [current]

    while unvisited:
        next_city = max(unvisited, key=lambda city: q_table[current, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    tour.append(start)
    total = 0.0
    for u, v in zip(tour, tour[1:]):
        total += float(distance_matrix[u, v])
    return total, tour


def _q_learning(
    distance_matrix: np.ndarray,
    cfg: QLearningConfig,
    log: Optional[logging.Logger] = None,
) -> np.ndarray:
    logger_obj = log
    n = distance_matrix.shape[0]
    rewards = _reward_matrix(distance_matrix)
    q_table = np.zeros_like(distance_matrix)
    # Q-Table: For environments with a finite number of states and actions, Q-values are often stored in a
    # table called the Q-table. Each cell in the table corresponds to a state-action pair \((s,a)\) and stores
    # its associated Q-value.

    epsilon = cfg.epsilon
    previous_snapshot: Optional[np.ndarray] = None
    monitor_interval = (
        cfg.monitor_interval
        if cfg.monitor_interval and cfg.monitor_interval > 0
        else None
    )
    best_monitored_cost = float("inf")
    best_q_snapshot: Optional[np.ndarray] = None
    reset_interval = (
        cfg.epsilon_reset_interval
        if cfg.epsilon_reset_interval and cfg.epsilon_reset_interval > 0
        else None
    )
    reset_value = (
        cfg.epsilon_reset_value if cfg.epsilon_reset_value is not None else cfg.epsilon
    )

    for episode in range(cfg.episodes):  # how many episodes
        start = random.randrange(n)
        current = start
        unvisited = set(range(n))
        unvisited.remove(current)
        while unvisited:
            actions = list(unvisited)
            action = _epsilon_greedy_action(q_table[current], actions, epsilon)
            reward = rewards[current, action]  # next city to visit
            unvisited.remove(action)
            next_state = action
            max_future = (
                q_table[next_state, list(unvisited)].max() if unvisited else 0.0
            )
            q_table[current, action] = (1 - cfg.alpha) * q_table[
                current, action
            ] + cfg.alpha * (reward + cfg.gamma * max_future)

            # Q(s, a) ← Q(s, a) + α [R + γ max(Q(s', a')) - Q(s, a)]

            current = next_state

        # close the tour by returning to start
        return_reward = rewards[current, start]
        q_table[current, start] = (1 - cfg.alpha) * q_table[
            current, start
        ] + cfg.alpha * (return_reward)

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        if reset_interval and (episode + 1) % reset_interval == 0:
            # Periodic reset keeps exploration alive once epsilon reaches epsilon_min.
            epsilon = max(cfg.epsilon_min, reset_value)
            if logger_obj is not None:
                logger_obj.info(
                    "Episode %d: epsilon reset to %.4f after decay plateau",
                    episode + 1,
                    epsilon,
                )

        if monitor_interval and (episode + 1) % monitor_interval == 0:
            current_snapshot = q_table.copy()
            if previous_snapshot is None:
                delta_fro = float("inf")
                delta_max = float("inf")
            else:
                diff = current_snapshot - previous_snapshot
                delta_fro = float(np.linalg.norm(diff))
                delta_max = float(np.max(np.abs(diff)))
            greedy_cost, tour = _deterministic_tour_cost(
                current_snapshot,
                distance_matrix,
            )
            is_best = " (new best)" if greedy_cost < best_monitored_cost else ""
            if greedy_cost < best_monitored_cost:
                best_monitored_cost = greedy_cost
                best_q_snapshot = current_snapshot.copy()
            if logger_obj is not None:
                logger_obj.info(
                    "Episode %d: Q-table Δ_fro=%.6f, Δ_max=%.6f, T %s greedy_cost=%.2f%s",
                    episode + 1,
                    delta_fro,
                    delta_max,
                    tour,
                    greedy_cost,
                    is_best,
                )
            previous_snapshot = current_snapshot

    if best_q_snapshot is not None:
        return best_q_snapshot
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
    log: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Generate a tour using a Q-learning policy trained on the TSP instance."""

    logger_obj = log

    cfg = cfg or QLearningConfig()
    distance_matrix = np.asarray(graph.build_adj_matrix_full(), dtype=float)

    if start is not None and start not in graph.nodes:
        raise ValueError("Provided start city is not part of the graph.")

    q_table = _load_q_table(distance_matrix, cfg)
    if q_table is None:
        q_table = _q_learning(distance_matrix, cfg, log=logger_obj)
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


def rcl_nearest_neighbour_tour(
    graph: Graph,
    *,
    start: int | None = None,
    rcl_size: int = 3,
    distance_threshold: float | None = None,
    alpha: float | None = None,
) -> np.ndarray:
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes.")
    distance_matrix = np.asarray(graph.get_distance_matrix())
    start_city = start if start is not None else random.choice(nodes)
    if start_city not in nodes:
        raise ValueError("Provided start city is not part of the graph.")

    unvisited = set(nodes)
    unvisited.remove(start_city)
    tour = [start_city]
    current = start_city

    while unvisited:
        candidates = []
        best_cost = float("inf")
        worst_cost = float("-inf")
        for city in unvisited:
            cost = distance_matrix[current][city]
            if not np.isfinite(cost):
                continue
            best_cost = min(best_cost, cost)
            worst_cost = max(worst_cost, cost)
            candidates.append((city, cost))

        if not candidates:
            # Fall back to any unvisited city when all distances are invalid.
            next_city = random.choice(tuple(unvisited))
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            continue

        # keep either a fixed number of closest cities or all cities whose
        # cost is within a factor of the best (if threshold is provided)
        candidates.sort(key=lambda pair: pair[1])
        if distance_threshold is not None:
            rcl = [
                city
                for city, cost in candidates
                if cost <= best_cost * distance_threshold
            ]
        elif alpha is not None:
            bounded_alpha = max(0.0, min(1.0, alpha))
            cutoff = best_cost + bounded_alpha * (worst_cost - best_cost)
            rcl = [city for city, cost in candidates if cost <= cutoff]
        else:
            rcl = [
                city for city, _ in candidates[: max(1, min(rcl_size, len(candidates)))]
            ]

            rcl = [
                city for city, _ in candidates[: max(1, min(rcl_size, len(candidates)))]
            ]
            # Ensure we always have at least one candidate to choose from.
            rcl = [candidates[0][0]]

        next_city = random.choice(rcl)
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    tour.append(start_city)
    return np.asarray(tour, dtype=int)


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
