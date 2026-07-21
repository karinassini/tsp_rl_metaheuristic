import itertools
import random
import numpy as np
from typing import Tuple
import math

class LocalSearchOperatorsMixin:
    """
    Encapsulates local-search operators for TSP tours.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        *,
        max_double_bridge_checks: int = 250,
        restricted_two_opt_max_span: int = 8,
        max_flip_subsequence_length: int = 5,
        max_inversion_segment_length: int = 5,
    ) -> None:
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.max_double_bridge_checks = max(0, int(max_double_bridge_checks))
        self.restricted_two_opt_max_span = max(2, int(restricted_two_opt_max_span))
        self.max_flip_subsequence_length = max(2, int(max_flip_subsequence_length))
        self.max_inversion_segment_length = max(2, int(max_inversion_segment_length))

    # Primitive tour transformers --------------------------------------------------
    @staticmethod
    def two_opt(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Reverse segment [i, j] to perform a 2-opt move."""
        return np.concatenate((tour[:i], tour[i : j + 1][::-1], tour[j + 1 :]))

    @staticmethod
    def three_opt(tour: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
        """Swap two middle segments to realize a 3-opt rearrangement."""
        return np.concatenate((tour[:i], tour[j:k], tour[i:j], tour[k:]))

    @staticmethod
    def two_exchange(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Swap the positions of two vertices in the tour."""
        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        new_core = core.copy()
        new_core[i], new_core[j] = new_core[j], new_core[i]
        if is_closed:
            return np.concatenate((new_core, [new_core[0]]))
        return new_core

    @staticmethod
    def one_insertion(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """Remove vertex at position i and insert it before position j."""
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
        """Apply the classic double-bridge perturbation defined by four cuts."""
        if len(tour) < 6:
            return tour.copy()
        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        if len(core) < 6:
            return tour.copy()
        s1, s2, s3, s4, s5 = (
            core[:a],
            core[a:b],
            core[b:c],
            core[c:d],
            core[d:],
        )
        new_core = np.concatenate((s1, s3, s2, s4, s5))
        if is_closed:
            new_core = np.concatenate((new_core, [new_core[0]]))
        return new_core

    @classmethod
    def random_double_bridge(cls, tour: np.ndarray) -> np.ndarray:
        """Sample random cut points and perform a double-bridge move."""
        is_closed = tour[0] == tour[-1]
        core_len = len(tour) - 1 if is_closed else len(tour)
        if core_len < 6:
            return tour.copy()
        a, b, c, d = sorted(random.sample(range(1, core_len), 4))
        return cls.double_bridge_move(tour, a, b, c, d)

    def _sample_inversion_insertion(self, tour: np.ndarray) -> np.ndarray:
        """Reverse a random segment and reinsert it at a random position."""
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
        new_core = np.concatenate(
            (remainder[:insert_idx], segment, remainder[insert_idx:])
        )
        if tour[0] == tour[-1]:
            new_core = np.concatenate((new_core, [new_core[0]]))
        return new_core


    def _one_move_insertion_improvement(
    self,
    tour: np.ndarray,
    current_distance: float,
    *,
    verify: bool = False,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:

        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 3:
            return tour, False, current_distance

        def edge_cost(u: int, v: int) -> float:
            lower, upper = (u, v) if u <= v else (v, u)
            return float(self.distance_matrix[lower, upper])

        core = tour[:-1] if is_closed else tour

        # i = position of the node to move (keep core[0] as anchor)
        for i in range(1, length):
            city = core[i]
            prev_i = core[i - 1]
            next_i = core[(i + 1) % length] if (is_closed or i + 1 < length) else None

            # Δ for removing 'city'
            removed_rem = edge_cost(prev_i, city)
            if next_i is not None:
                removed_rem += edge_cost(city, next_i)
            added_rem = edge_cost(prev_i, next_i) if next_i is not None else 0.0
            base_delta = added_rem - removed_rem

            # vector without 'city'
            core_removed = np.delete(core, i)
            m = len(core_removed)

            # j = insertion position (insert before core_removed[j])
            for j in range(1, m + 1):
                # Note: no need to skip j==i/i+1; if it is a no-op, delta = 0.
                if is_closed:
                    # correct wrap-around for closed tours
                    prev_j = core_removed[(j - 1) % m]
                    next_j = core_removed[j % m]
                else:
                    prev_j = core_removed[j - 1] if j - 1 >= 0 else None  # j starts at 1, so prev_j always exists
                    next_j = core_removed[j] if j < m else None            # if j==m, insert at end (no next_j)

                # Δ for inserting between (prev_j, next_j) → (prev_j, city) + (city, next_j)
                removed_ins = edge_cost(prev_j, next_j) if (prev_j is not None and next_j is not None) else 0.0
                added_ins = 0.0
                if prev_j is not None:
                    added_ins += edge_cost(prev_j, city)
                if next_j is not None:
                    added_ins += edge_cost(city, next_j)

                delta = base_delta + (added_ins - removed_ins)

                if delta < -tol:
                    # Rebuild the candidate consistently with delta
                    new_core = np.concatenate([core_removed[:j], [city], core_removed[j:]])
                    candidate = (
                        np.concatenate([new_core, new_core[:1]]) if is_closed else new_core
                    )
                    new_distance = current_distance + delta

                    if verify:
                        # Validate via full recomputation (useful for debugging)
                        true_len = self.tour_distance(candidate, self.distance_matrix)
                        if abs(true_len - new_distance) > 1e-6:
                            # incremental delta mismatch — discard and keep searching
                            continue
                        return candidate, True, true_len
                    else:
                        return candidate, True, new_distance

        return tour, False, current_distance
    
    # First-improvement neighborhoods -------------------------------------------
    def _two_opt_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        """Return the first 2-opt move that improves the objective."""
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
        """Return the first improving swap of two vertices."""
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
        """Return the first improving single-vertex reinsertion."""
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

    def _one_move_insertion_improvement(
    self,
    tour: np.ndarray,
    current_distance: float,
    *,
    verify: bool = False,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:

        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 3:
            return tour, False, current_distance

        def edge_cost(u: int, v: int) -> float:
            lower, upper = (u, v) if u <= v else (v, u)
            return float(self.distance_matrix[lower, upper])

        core = tour[:-1] if is_closed else tour

        # i = position of the node to move (keep core[0] as anchor)
        for i in range(1, length):
            city = core[i]
            prev_i = core[i - 1]
            next_i = core[(i + 1) % length] if (is_closed or i + 1 < length) else None

            # Δ for removing 'city'
            removed_rem = edge_cost(prev_i, city)
            if next_i is not None:
                removed_rem += edge_cost(city, next_i)
            added_rem = edge_cost(prev_i, next_i) if next_i is not None else 0.0
            base_delta = added_rem - removed_rem

            # vector without 'city'
            core_removed = np.delete(core, i)
            m = len(core_removed)

            # j = insertion position (insert before core_removed[j])
            for j in range(1, m + 1):
                # Note: no need to skip j==i/i+1 here; if it is a no-op, delta = 0.
                if is_closed:
                    # correct wrap-around for closed tours
                    prev_j = core_removed[(j - 1) % m]
                    next_j = core_removed[j % m]
                else:
                    prev_j = core_removed[j - 1] if j - 1 >= 0 else None  # j starts at 1, so prev_j always exists
                    next_j = core_removed[j] if j < m else None            # if j==m, insert at the end (no next_j)

                # Δ for inserting between (prev_j, next_j) → (prev_j, city) + (city, next_j)
                removed_ins = edge_cost(prev_j, next_j) if (prev_j is not None and next_j is not None) else 0.0
                added_ins = 0.0
                if prev_j is not None:
                    added_ins += edge_cost(prev_j, city)
                if next_j is not None:
                    added_ins += edge_cost(city, next_j)

                delta = base_delta + (added_ins - removed_ins)

                if delta < -tol:
                    # Rebuild the candidate consistently with delta
                    new_core = np.concatenate([core_removed[:j], [city], core_removed[j:]])
                    candidate = (
                        np.concatenate([new_core, new_core[:1]]) if is_closed else new_core
                    )
                    new_distance = current_distance + delta

                    if verify:
                        # Validate via full recomputation (useful for debugging)
                        true_len = self.tour_distance(candidate, self.distance_matrix)
                        if abs(true_len - new_distance) > 1e-6:
                            # incremental delta mismatch — discard and continue searching
                            continue
                        return candidate, True, true_len
                    else:
                        return candidate, True, new_distance

        return tour, False, current_distance
    

    def _double_bridge_first_improvement(
        self, tour: np.ndarray, current_distance: float
    ) -> tuple[np.ndarray, bool, float]:
        """Sample double-bridge candidates and return the first improvement."""
        is_closed = tour[0] == tour[-1]
        length = len(tour) - 1 if is_closed else len(tour)
        if length < 6:
            return tour, False, current_distance

        def edge_cost(u: int | None, v: int | None) -> float:
            if u is None or v is None:
                return 0.0
            lower, upper = (u, v) if u <= v else (v, u)
            return self.distance_matrix[lower, upper]

        max_checks = self.max_double_bridge_checks
        if max_checks == 0:
            return tour, False, current_distance

        core = tour[:-1] if is_closed else tour
        max_unique_combinations = math.comb(length - 1, 4)
        target_checks = min(max_checks, max_unique_combinations)
        seen_combinations: set[tuple[int, int, int, int]] = set()

        while len(seen_combinations) < target_checks:
            a, b, c, d = sorted(random.sample(range(1, length), 4))
            combination = (a, b, c, d)
            if combination in seen_combinations:
                continue
            seen_combinations.add(combination)
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
        """First-improvement 2-opt limited to a short span window."""
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
        """First-improvement flip of short subsequences."""
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
                next_city = core[(end + 1) % length] if (is_closed or end + 1 < length) else None
                removed = edge_cost(prev_city, first_city) + edge_cost(last_city, next_city)
                added = edge_cost(prev_city, last_city) + edge_cost(first_city, next_city)
                delta = added - removed
                if delta < 0:
                    candidate = self.two_opt(tour, start, end)
                    new_distance = current_distance + delta
                    return candidate, True, new_distance
        return tour, False, current_distance


    def _three_opt_first_improvement(
        self,
        tour: np.ndarray,
        current_distance: float,
        *,
        include_2opt_cases: bool = True,
        verify: bool = False,
        tol: float = 1e-12,
        k_neighbors: int = 12,   # narrower candidate list for speed
        max_i_checks: int | None = 40,  # cap outer loop work
        max_j_checks: int | None = 80,  # cap inner loop work per i
    ) -> Tuple[np.ndarray, bool, float]:
        """
        First-improvement 3-opt with candidate-list pruning and O(1) delta eval.
        Requires self._dist: np.ndarray[n, n] distances.
        Optional: self._nn nearest neighbors list (n x k); if absent, it will be built or we fallback.
        """
        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        n = len(core)
        if n < 6:
            return tour, False, current_distance

        dist = self.distance_matrix  # np.ndarray[n, n]

        # Build/ensure nearest neighbors (nn[u] is array of neighbor nodes for u)
        nn = getattr(self, "_nn", None)
        if nn is None or (len(nn) != n) or (len(nn[0]) < k_neighbors):
            nn = self._build_candidate_lists(dist, k=k_neighbors)
            self._nn = nn

        # Helper to compute distance quickly with symmetry and NaN guard
        def D(u: int, v: int) -> float:
            val = dist[u, v] if u <= v else dist[v, u]
            return val if np.isfinite(val) else np.inf

        # Apply a selected 3-opt reconnection to the core tour
        def apply_3opt(core, i, j, k, variant_tag):
            # A = core[:i], B = core[i:j], C = core[j:k], D = core[k:]
            # Variants:
            #   2-opt equivalents (optional):
            #   "B^R C", "B C^R", "B^R C^R"
            #   True 3-opt:
            #   "C B", "C^R B", "C B^R", "C^R B^R"
            A = core[:i]
            B = core[i:j]
            C = core[j:k]
            D = core[k:]

            if variant_tag == "BrC":
                new_core = np.concatenate([A, B[::-1], C, D])
            elif variant_tag == "BCr":
                new_core = np.concatenate([A, B, C[::-1], D])
            elif variant_tag == "BrCr":
                new_core = np.concatenate([A, B[::-1], C[::-1], D])
            elif variant_tag == "CB":
                new_core = np.concatenate([A, C, B, D])
            elif variant_tag == "CrB":
                new_core = np.concatenate([A, C[::-1], B, D])
            elif variant_tag == "CBr":
                new_core = np.concatenate([A, C, B[::-1], D])
            elif variant_tag == "CrBr":
                new_core = np.concatenate([A, C[::-1], B[::-1], D])
            else:
                raise ValueError("Unknown variant")

            return (np.concatenate([new_core, new_core[:1]]) if is_closed else new_core)

        # ---- Main search with pruning by candidate lists ----
        # We iterate 'i', then try to pick 'j' and 'k' based on neighbor relationships
        # Heuristic: choose j so that core[j] (or core[j-1]) is in neighbors of core[i] (or core[i-1])
        # and choose k similarly with respect to core[j], to reduce search volume drastically.

        i_checked = 0
        for i in range(1, n - 2):
            if max_i_checks is not None and i_checked >= max_i_checks:
                break
            a = core[i - 1]
            b = core[i]

            i_checked += 1

            # Consider j such that c=d's edge touches nearest neighbors of a or b
            # We will iterate j but skip most cases unless either core[j] or core[j-1]
            # is in the neighbor set of a or b (or vice-versa).
            a_neighbors = nn[a]
            b_neighbors = nn[b]

            j_checked = 0
            for j in range(i + 1, n - 1):
                if max_j_checks is not None and j_checked >= max_j_checks:
                    break
                c = core[j - 1]
                d = core[j]

                j_checked += 1

                # Quick neighbor gating
                if (d not in a_neighbors) and (d not in b_neighbors) and (c not in a_neighbors) and (c not in b_neighbors):
                    continue

                for k in range(j + 1, n):
                    e = core[k - 1]
                    f = core[k]

                    # Another neighbor gate w.r.t. j edge or the next edge (k)
                    c_neighbors = nn[c]
                    d_neighbors = nn[d]
                    if (e not in c_neighbors) and (e not in d_neighbors) and (f not in c_neighbors) and (f not in d_neighbors):
                        continue

                    # Ensure B, C, D non-empty (they are by loop ranges)
                    # Removed sum:
                    removed = D(a, b) + D(c, d) + D(e, f)

                    # Prebind A_end and D_start
                    A_end = a
                    D_start = f

                    # Evaluate variants by delta only, no array construction
                    # We'll map each variant to the first and last of X and Y:
                    # X, Y chosen from B or B^R and C or C^R
                    B_first, B_last = b, c
                    C_first, C_last = d, e

                    # Helper to compute Δ for endpoints only
                    def delta_for(x0, x1, y0, y1):
                        a1 = D(A_end, x0)
                        a2 = D(x1, y0)
                        a3 = D(y1, D_start)
                        if not np.isfinite(a1) or not np.isfinite(a2) or not np.isfinite(a3):
                            return np.inf
                        return (a1 + a2 + a3) - removed

                    # Collect candidates (ordered to find good moves sooner)
                    # 2-opt-equivalents (optional)
                    if include_2opt_cases:
                        # "B^R C"  => X=B^R (x0=c, x1=b), Y=C (y0=d, y1=e)
                        d1 = delta_for(B_last, B_first, C_first, C_last)
                        if d1 < -tol:
                            new_tour = apply_3opt(core, i, j, k, "BrC")
                            return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d1)

                        # "B C^R" => X=B (x0=b, x1=c), Y=C^R (y0=e, y1=d)
                        d2 = delta_for(B_first, B_last, C_last, C_first)
                        if d2 < -tol:
                            new_tour = apply_3opt(core, i, j, k, "BCr")
                            return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d2)

                        # "B^R C^R" => X=B^R (c,b), Y=C^R (e,d)
                        d3 = delta_for(B_last, B_first, C_last, C_first)
                        if d3 < -tol:
                            new_tour = apply_3opt(core, i, j, k, "BrCr")
                            return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d3)

                    # True 3‑opt cases (four)
                    # "C B" => X=C (d,e), Y=B (b,c)
                    d4 = delta_for(C_first, C_last, B_first, B_last)
                    if d4 < -tol:
                        new_tour = apply_3opt(core, i, j, k, "CB")
                        return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d4)

                    # "C^R B" => X=C^R (e,d), Y=B (b,c)
                    d5 = delta_for(C_last, C_first, B_first, B_last)
                    if d5 < -tol:
                        new_tour = apply_3opt(core, i, j, k, "CrB")
                        return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d5)

                    # "C B^R" => X=C (d,e), Y=B^R (c,b)
                    d6 = delta_for(C_first, C_last, B_last, B_first)
                    if d6 < -tol:
                        new_tour = apply_3opt(core, i, j, k, "CBr")
                        return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d6)

                    # "C^R B^R" => X=C^R (e,d), Y=B^R (c,b)
                    d7 = delta_for(C_last, C_first, B_last, B_first)
                    if d7 < -tol:
                        new_tour = apply_3opt(core, i, j, k, "CrBr")
                        return new_tour, True, (self._tour_length(new_tour) if verify else current_distance + d7)

        # No improvement
        return tour, False, current_distance


    def _build_candidate_lists(self, dist: np.ndarray, k: int = 20):
        n = dist.shape[0]
        # argsort per row once; skip the 0 self-distance
        nn = []
        for u in range(n):
            # Sort neighbors by distance; take the first k excluding itself
            order = np.argsort(dist[u])
            order = order[order != u][:k]
            nn.append(order)
        return nn
    
    # Shaking / perturbation operators -------------------------------------------
    def _shake_oropt_block(self, tour: np.ndarray, block_len: int = 2) -> np.ndarray:
        core, is_closed = self._extract_core(tour)
        n = len(core)
        if n < block_len + 3:
            return tour.copy()

        start = random.randrange(1, n - block_len + 1)
        block = core[start:start + block_len].copy()
        remainder = np.concatenate([core[:start], core[start + block_len:]])

        # Insert far from the original start to make the shaking stronger
        far_min = max(1, start + block_len + max(2, n // 10))
        far_candidates = list(range(1, n - block_len + 1))  # [1..n-block_len]
        # Remove window near start to avoid no-ops
        far_candidates = [pos for pos in far_candidates if (pos < start - 1 or pos > far_min)]
        if not far_candidates:
            insert_pos = random.randrange(1, len(remainder) + 1)
        else:
            insert_pos = random.choice(far_candidates)
            if insert_pos > start:
                insert_pos -= block_len

        new_core = np.concatenate([remainder[:insert_pos], block, remainder[insert_pos:]])
        return self._reclose(new_core, is_closed)

    def _shake_k_exchange(self, tour: np.ndarray, k_nodes: int = 4) -> np.ndarray:
        core, is_closed = self._extract_core(tour)
        n = len(core)
        k_nodes = max(2, min(k_nodes, max(2, n - 3)))
        idxs = sorted(random.sample(range(1, n), k_nodes))  # avoid anchor position 0
        removed_nodes = [core[i] for i in idxs]
        remainder = np.delete(core, idxs)

        # Reinsert as a shuffled block (more disruptive than light scatter)
        random.shuffle(removed_nodes)
        insert_pos = random.randrange(1, len(remainder) + 1)
        new_core = np.concatenate([remainder[:insert_pos], removed_nodes, remainder[insert_pos:]])
        
        return self._reclose(new_core, is_closed)
        
    def _shake_cross_exchange(self, tour: np.ndarray, len1: int = 4, len2: int = 4) -> np.ndarray:
        core, is_closed = self._extract_core(tour)
        n = len(core)
        len1 = max(2, min(len1, n // 6 if n >= 12 else 3))
        len2 = max(2, min(len2, n // 6 if n >= 12 else 3))
        if n < (len1 + len2 + 5):
            return tour.copy()

        # Sample until valid disjoint segments are found
        for _ in range(20):
            i = random.randrange(1, n - len1)
            j = i + len1
            k = random.randrange(1, n - len2)
            l = k + len2
            if (j <= k or l <= i):  # non-overlapping
                s1 = core[i:j].copy()
                s2 = core[k:l].copy()
                new_core = core.copy()
                # Rebuild the middle slice to avoid size conflicts
                left = min(i, k)
                right = max(j, l)
                mid = core[left:right].copy()

                # Build the middle slice by swapping s1 and s2 in place
                if i < k:
                    mid = np.concatenate([core[left:i], s2, core[j:k], s1, core[l:right]])
                else:
                    mid = np.concatenate([core[left:k], s1, core[l:i], s2, core[j:right]])

                new_core = np.concatenate([core[:left], mid, core[right:]])
                return self._reclose(new_core, is_closed)

        return tour.copy()
    
    def _shake_shuffle_segment(self, tour: np.ndarray, seg_len: int = 6) -> np.ndarray:
        core, is_closed = self._extract_core(tour)
        n = len(core)
        if n < seg_len + 3:
            return tour.copy()

        start = random.randrange(1, n - seg_len)
        segment = core[start:start + seg_len].copy()
        random.shuffle(segment)
        new_core = np.concatenate([core[:start], segment, core[start + seg_len:]])
        return self._reclose(new_core, is_closed)

    def _shake_block_recombine(self, tour: np.ndarray, block_size: int = 10) -> np.ndarray:
        core, is_closed = self._extract_core(tour)
        n = len(core)
        block_size = max(3, min(block_size, max(3, n // 8)))
        if n < block_size + 4:
            return tour.copy()

        blocks = [core[i:i + block_size] for i in range(1, n, block_size)]
        # Ensure the first node (position 0) stays as a stable anchor
        head = core[:1]
        random.shuffle(blocks)
        new_core = np.concatenate([head] + blocks)
        return self._reclose(new_core, is_closed)

    # --- Double-bridge (keep core implementation) ---
    def _shake_double_bridge(self, tour: np.ndarray) -> np.ndarray:
        return self.random_double_bridge(tour)

    # --- Helpers---
    def _extract_core(self, tour: np.ndarray) -> tuple[np.ndarray, bool]:
        is_closed = tour[0] == tour[-1]
        core = tour[:-1] if is_closed else tour
        return core, is_closed

    def _reclose(self, core: np.ndarray, is_closed: bool) -> np.ndarray:
        return np.concatenate([core, [core[0]]]) if is_closed else core
    
    def _edge_cost(self, u: int, v: int) -> float:
        if u <= v:
            return self.distance_matrix[u, v]
        else:
            return self.distance_matrix[v, u]

    def _tour_length(self, tour: np.ndarray) -> float:
        """Works for open and explicitly closed tours (last == first)."""
        if len(tour) < 2:
            return 0.0
        total = 0.0
        if tour[0] == tour[-1]:
            # closed: sum edges between consecutive unique nodes
            for t in range(len(tour) - 1):
                total += self._edge_cost(tour[t], tour[t + 1])
        else:
            # open: sum consecutive edges, no closure
            for t in range(len(tour) - 1):
                total += self._edge_cost(tour[t], tour[t + 1])
        return total