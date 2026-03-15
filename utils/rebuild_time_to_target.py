from pathlib import Path
import json

# ajuste aqui para o run desejado
run_dir = Path(
    "/Users/karinaassiniandreatta/Documents/01 phd/codes/tsp_rl_metaheuristic/outputs/solutions/ch150/VNS_Solver_Q_Learnings/rcl/20260303_203249"
)

instance = "ch150"  # nome da instância sem .tsp
method = "rcl"  # método usado
iteration_max = 800  # o iteration_max usado no run
solver = "VNS_Solver_Q_Learnings"  # nome do solver na pasta


# Carrega todos os *_solution_*.json (padrão salvo pelo main_vns)
solutions = []
for f in sorted(run_dir.glob("*_solution_*.json")):
    with f.open() as fh:
        solutions.append(json.load(fh))

if not solutions:
    raise SystemExit("Nenhum *_solution.json encontrado")

# Descobre o best_known: tenta solver_params.best_known_distance, depois best_total_distance, depois melhor total_distance
def _best_known_from_solution(sol: dict) -> float | None:
    solver_params = sol.get("solver_params") or {}
    if "best_known_distance" in solver_params:
        return solver_params.get("best_known_distance")
    if "best_total_distance" in sol:
        return sol.get("best_total_distance")
    return None

best_known = next((bk for bk in (_best_known_from_solution(s) for s in solutions) if bk is not None), None)
if best_known is None:
    best_known = min(sol["total_distance"] for sol in solutions)

# Filtra sucessos (atingiram best_known)
tolerance = 1e-6
times = []
for sol in solutions:
    params = sol.get("params") or {}
    exploration = params.get("exploration_time") or sol.get("exploration_time", 0.0)
    exploitation = params.get("exploitation_time") or sol.get("exploitation_time", 0.0)
    total_time = float(exploration) + float(exploitation)
    if abs(sol["total_distance"] - best_known) <= tolerance:
        times.append(total_time)
times.sort()

total_runs = len(solutions)
success_count = len(times)
failures = total_runs - success_count

probabilities = [(i + 0.5) / total_runs for i in range(success_count)] if success_count else []

# Label depende do start_city do run
start_city = solutions[0].get("params", {}).get("start_city_initialization")
start_label = "null" if start_city is None else str(start_city)
store_label = f"{method}|start_{start_label}"

key = f"{instance}|{method}|{iteration_max}|{solver}"
payload = {
    key: {
        store_label: {
            "times": times,
            "probabilities": probabilities,
            "mean_time": (sum(times) / success_count) if success_count else None,
            "total_runs": total_runs,
            "success_count": success_count,
            "failures": failures,
            "style": {
                "color": "forestgreen",
                "marker": "x",
                "label": store_label,
                "linewidth": 2,
            },
            "best_known_slack": None,
        }
    }
}

out_path = run_dir / "time_to_target_data.json"
with out_path.open("w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(f"gerado {out_path}")

# Gera plot diretamente deste run (salva em run_dir/plots/...); se faltar matplotlib/numpy, apenas avisa
try:
    from plot_time_to_target_all import plot_time_to_target_collection

    plot_time_to_target_collection(
        [run_dir],
        output_dir=run_dir / "plots",
        start_filters={start_label},
        instance_filters={instance},
        method_filters={method},
        solver_filters={solver},
        group_by_method=False,
    )
except ModuleNotFoundError as exc:
    print(f"Plotting skipped (missing dependency: {exc.name}). JSON foi gerado.")