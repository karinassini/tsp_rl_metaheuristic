"""
Statistical comparison of RL-GVNS algorithmic configurations.

ANALYSIS A — Friedman test across all 10 paper instances (N=10), using
             mean tour costs directly from Table 4. No simulation required.

ANALYSIS B — Friedman test restricted to the 5 hard instances (non-trivial
             inter-configuration variation).

ANALYSIS C — Per-instance pairwise Mann-Whitney U tests on all 5 hard
             instances, with Holm–Bonferroni correction per instance.
             Run distributions are imputed from Table 4 as a two-point
             (BKS / reported-max) distribution preserving the BKSH count.

Configurations compared:
  (1) RCL  + GVNS
  (2) MARL + GVNS
  (3) RCL  + RL-GVNS
  (4) MARL + RL-GVNS

References:
  Demšar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. JMLR, 7, 1–30.
  Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scand. J. Statist., 6(2), 65–70.
  Vargha, A. & Delaney, H. D. (2000). A critique and improvement of the CL
    common language effect size statistic. J. Educ. Behav. Statist., 25(2).
"""

import itertools
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata, wilcoxon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGS = [
    "(1) RCL+GVNS",
    "(2) MARL+GVNS",
    "(3) RCL+RL-GVNS",
    "(4) MARL+RL-GVNS",
]

PAPER_INSTANCES = [
    "bays29", "swiss42", "gr48", "berlin52", "kroA100",
    "kroB100", "ch150", "si175", "pr226", "pcb442",
]

HARD_INSTANCES = ["kroB100", "ch150", "si175", "pr226", "pcb442"]

N_RUNS = 30
ALPHA  = 0.05

# ---------------------------------------------------------------------------
# Table 4 summary statistics — (best_cost, max_cost, mean_cost, bksh_count)
# ---------------------------------------------------------------------------
# All values taken directly from Table 4 of the paper.
# Trivial instances (gap=0, all configs identical) are included with zeros.

_T = {  # shorthand
    "(1) RCL+GVNS":    0,
    "(2) MARL+GVNS":   1,
    "(3) RCL+RL-GVNS": 2,
    "(4) MARL+RL-GVNS":3,
}

TABLE_STATS: dict[str, dict[str, tuple]] = {
    # trivial instances — all configs reach BKS in every run
    "bays29":  {c: (2020,  2020,  2020.00, 30) for c in CONFIGS},
    "swiss42": {c: (1273,  1273,  1273.00, 30) for c in CONFIGS},
    "gr48":    {c: (5046,  5046,  5046.00, 30) for c in CONFIGS},
    "berlin52":{c: (7542,  7542,  7542.00, 30) for c in CONFIGS},
    "kroA100": {c: (21282, 21282, 21282.00, 30) for c in CONFIGS},
    # hard instances — non-zero inter-configuration variance
    "kroB100": {
        "(1) RCL+GVNS":     (22141, 22199, 22162.27, 19),
        "(2) MARL+GVNS":    (22141, 22199, 22150.67, 25),
        "(3) RCL+RL-GVNS":  (22141, 22199, 22144.87, 28),
        "(4) MARL+RL-GVNS": (22141, 22141, 22141.00, 30),
    },
    "ch150": {
        "(1) RCL+GVNS":     (6528, 6564, 6532.83, 19),
        "(2) MARL+GVNS":    (6528, 6564, 6534.70, 21),
        "(3) RCL+RL-GVNS":  (6528, 6566, 6535.17, 20),
        "(4) MARL+RL-GVNS": (6528, 6549, 6530.77, 26),
    },
    "si175": {
        "(1) RCL+GVNS":     (21407, 21421, 21409.57, 15),
        "(2) MARL+GVNS":    (21407, 21420, 21408.00, 17),
        "(3) RCL+RL-GVNS":  (21407, 21427, 21410.43, 17),
        "(4) MARL+RL-GVNS": (21407, 21419, 21410.89, 20),
    },
    "pr226": {
        "(1) RCL+GVNS":     (80369, 80442, 80381.00, 27),
        "(2) MARL+GVNS":    (80369, 80442, 80421.00, 28),
        "(3) RCL+RL-GVNS":  (80369, 80440, 80371.37, 29),
        "(4) MARL+RL-GVNS": (80369, 80373, 80369.13, 29),
    },
    "pcb442": {
        "(1) RCL+GVNS":     (50778, 51210, 50950.00, 18),
        "(2) MARL+GVNS":    (50778, 51150, 50910.00, 21),
        "(3) RCL+RL-GVNS":  (50778, 51180, 50930.00, 23),
        "(4) MARL+RL-GVNS": (50778, 50980, 50850.00, 27),
    },
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_bimodal(best: int, max_val: int, bksh: int,
                     n: int = N_RUNS) -> list[float]:
    """
    Impute n run costs as a two-point distribution:
      bksh values at 'best' (BKS hit) and (n - bksh) at 'max_val'.
    Preserves the BKSH count from Table 4 exactly.
    """
    if bksh >= n or max_val <= best:
        return [float(best)] * n
    return [float(best)] * bksh + [float(max_val)] * (n - bksh)


def get_simulated_data() -> dict[str, dict[str, list[float]]]:
    """Return imputed run vectors for all hard instances."""
    data: dict[str, dict[str, list[float]]] = {}
    for inst in HARD_INSTANCES:
        data[inst] = {}
        for cfg in CONFIGS:
            best, max_val, _, bksh = TABLE_STATS[inst][cfg]
            data[inst][cfg] = simulate_bimodal(best, max_val, bksh)
    return data

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def vargha_delaney_a12(x: list[float], y: list[float]) -> float:
    """
    Vargha–Delaney A12 effect size.
    A12 < 0.5 → x tends to be smaller → x is better for minimisation.
    Thresholds (|A12 – 0.5|): ≥0.21 large, ≥0.14 medium, ≥0.06 small.
    """
    m = len(x)
    r = rankdata(list(x) + list(y))
    return (r[:m].sum() - m * (m + 1) / 2) / (m * len(y))


def effect_label(a12: float) -> str:
    d = abs(a12 - 0.5)
    return ("large"      if d >= 0.21 else
            "medium"     if d >= 0.14 else
            "small"      if d >= 0.06 else
            "negligible")


def holm_bonferroni(p_values: list[float]) -> list[float]:
    k = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(k)
    prev = 0.0
    for ri, oi in enumerate(order):
        adj = max(float(p_values[oi]) * (k - ri), prev)
        adjusted[oi] = min(adj, 1.0)
        prev = adjusted[oi]
    return adjusted.tolist()

# ---------------------------------------------------------------------------
# Analysis A & B — Friedman tests using Table 4 means
# ---------------------------------------------------------------------------

def run_friedman_table(instances: list[str], label: str) -> None:
    N, k = len(instances), len(CONFIGS)
    print(f"\n{'='*72}")
    print(f"FRIEDMAN TEST — {label}")
    print(f"{'='*72}")
    print(f"N = {N} instances: {', '.join(instances)}")
    print(f"k = {k} configurations; performance measure: mean tour cost (Table 4)\n")

    # Build mean-cost matrix [N × k]
    mean_matrix = np.array(
        [[TABLE_STATS[inst][cfg][2] for cfg in CONFIGS] for inst in instances]
    )

    stat, p_value = friedmanchisquare(*[mean_matrix[:, j] for j in range(k)])
    rank_matrix   = np.array([rankdata(row) for row in mean_matrix])
    avg_ranks     = rank_matrix.mean(axis=0)

    print(f"Friedman χ²  = {stat:.4f}")
    print(f"p-value      = {p_value:.6f}")
    print(f"Result       : {'SIGNIFICANT' if p_value < ALPHA else 'not significant'} at α = {ALPHA}")
    print("\nAverage ranks (1 = best / lowest cost):")
    for cfg, ar in zip(CONFIGS, avg_ranks):
        print(f"  {cfg:<24} {ar:.3f}")

    # Exploratory post-hoc Wilcoxon
    print(f"\n{'  (exploratory post-hoc)' if p_value >= ALPHA else ''}")
    pairs = list(itertools.combinations(range(k), 2))
    raw_p, w_stats, a12s = [], [], []
    for i, j in pairs:
        x, y = mean_matrix[:, i], mean_matrix[:, j]
        try:
            ws, wp = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided")
        except ValueError:
            ws, wp = 0.0, 1.0
        raw_p.append(wp); w_stats.append(ws)
        a12s.append(vargha_delaney_a12(list(x), list(y)))
    adj_p = holm_bonferroni(raw_p)

    print(f"\n{'Pair':<46} {'W':>6} {'p-raw':>8} {'p-adj':>8} sig  {'A12':>5}  Effect")
    print("-" * 80)
    for idx, (i, j) in enumerate(pairs):
        ci, cj = CONFIGS[i], CONFIGS[j]
        a12     = a12s[idx]
        winner  = ci if a12 < 0.499 else (cj if a12 > 0.501 else "tie")
        sig     = "*" if adj_p[idx] < ALPHA else ""
        print(f"{ci} vs {cj:<16} "
              f"{w_stats[idx]:>6.1f} {raw_p[idx]:>8.4f} {adj_p[idx]:>8.4f} "
              f"{'':>2}{sig:<2}  {a12:>5.3f}  {effect_label(a12)} ({winner})")

    print("\nPer-instance mean costs and ranks:")
    print(f"{'Instance':<12}" + "".join(f"{c:>26}" for c in CONFIGS))
    for i, inst in enumerate(instances):
        row = f"{inst:<12}"
        for j in range(k):
            row += f"  {mean_matrix[i,j]:>11.2f} (r{rank_matrix[i,j]:.0f})"
        print(row)


# ---------------------------------------------------------------------------
# Analysis C — Per-instance pairwise Mann-Whitney U (hard instances)
# ---------------------------------------------------------------------------

def run_per_instance_mwu(data: dict[str, dict[str, list[float]]]) -> None:
    print(f"\n\n{'='*72}")
    print("ANALYSIS C — PER-INSTANCE PAIRWISE MANN-WHITNEY U TESTS")
    print("  All runs imputed from Table 4 (two-point BKS/max distribution).")
    print(f"{'='*72}")

    # Collect all results for a combined LaTeX table at the end
    latex_rows: list[str] = []
    all_sig_results: dict[str, list] = {}

    for inst in HARD_INSTANCES:
        inst_data = data[inst]
        pairs = list(itertools.combinations(CONFIGS, 2))
        raw_p, u_stats, a12s = [], [], []

        for ci, cj in pairs:
            u, p = mannwhitneyu(inst_data[ci], inst_data[cj], alternative="two-sided")
            raw_p.append(p)
            u_stats.append(u)
            a12s.append(vargha_delaney_a12(inst_data[ci], inst_data[cj]))

        adj_p = holm_bonferroni(raw_p)
        all_sig_results[inst] = list(zip(pairs, u_stats, raw_p, adj_p, a12s))

        n_per = {c: len(v) for c, v in inst_data.items()}
        print(f"\n--- {inst}  (n=30 per config, imputed) ---")
        print(f"  {'Pair':<44} {'U':>8} {'p-raw':>8} {'p-adj':>8} sig  {'A12':>5}  Effect")
        print(f"  {'-'*85}")
        for idx, (ci, cj) in enumerate(pairs):
            a12    = a12s[idx]
            winner = ci if a12 < 0.499 else (cj if a12 > 0.501 else "tie")
            sig    = "*" if adj_p[idx] < ALPHA else ""
            print(f"  {ci} vs {cj:<16} "
                  f"{u_stats[idx]:>8.0f} "
                  f"{raw_p[idx]:>8.4f} {adj_p[idx]:>8.4f} "
                  f"{'':>2}{sig:<2} {a12:>5.3f}  "
                  f"{effect_label(a12)} ({winner})")

        print(f"\n  Descriptive (imputed distribution):")
        for cfg in CONFIGS:
            arr = inst_data[cfg]
            best, max_val, table_mean, bksh = TABLE_STATS[inst][cfg]
            print(f"    {cfg:<24}  bksh={bksh}/30  imputed_mean={np.mean(arr):.2f}"
                  f"  table_mean={table_mean:.2f}")

    # Print LaTeX table block for significant results
    print(f"\n\n{'='*72}")
    print("LATEX TABLE — All pairwise MWU results for hard instances")
    print(f"{'='*72}")
    print(r"\begin{tabular}{llrrrrl r}")
    print(r"\toprule")
    print(r"\textbf{Instance} & \textbf{Config A} & \textbf{Config B} & "
          r"\textbf{U} & \textbf{$p$-raw} & \textbf{$p$-adj} & \textbf{Sig.} & $\hat{A}_{12}$ \\")
    print(r"\midrule")
    for inst, results in all_sig_results.items():
        first = True
        for (ci, cj), u, pr, pa, a12 in results:
            sig = "**" if pa < 0.01 else ("*" if pa < ALPHA else "")
            inst_label = f"\\texttt{{{inst}}}" if first else ""
            print(f"{inst_label} & {ci} & {cj} & "
                  f"{u:.0f} & {pr:.4f} & {pa:.4f} & {sig} & {a12:.3f} \\\\")
            first = False
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Statistical Analysis — RL-GVNS Configuration Comparison")
    print("Reference: Demšar (2006) framework")
    print("=" * 72)

    # ---- Analyses A & B: Friedman using Table 4 means ----
    run_friedman_table(PAPER_INSTANCES, "ALL 10 PAPER INSTANCES (Table 4 means)")
    run_friedman_table(HARD_INSTANCES,  "5 HARD INSTANCES ONLY (non-trivial variation)")

    # ---- Analysis C: per-instance MWU (imputed data) ----
    sim_data = get_simulated_data()
    run_per_instance_mwu(sim_data)

    print("\n\nDone.\n")
