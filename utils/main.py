"""Helper script to organize results and produce key plots."""

from __future__ import annotations

from pathlib import Path

from organize_time_to_target_data import organize_time_to_target_files
from plot_time_to_target_all import plot_time_to_target_collection
from solution_csv_aggregator import concatenate_solution_csvs


def _run_plots(start_values: list[str | None], *, group_by_method: bool = False, group_all: bool = False) -> None:
	repo_root = Path(__file__).resolve().parents[1]
	sources = [repo_root / "outputs" / "plots" / "solutions"]
	for start in start_values:
		if start is None:
			start_filters = None
			label = "all starts"
		else:
			start_filters = {start}
			label = f"start={start}"
		if group_all:
			mode = "all combined"
			subdir = "all_combined"
		elif group_by_method:
			mode = "method comparison"
			subdir = "grouped_by_method"
		else:
			mode = "per solver"
			subdir = "per_solver"
		print(f"Plotting time-to-target curves for {label} ({mode})...")
		plot_time_to_target_collection(
			sources,
			output_dir=repo_root / "outputs" / "plots" / "time_to_target" / subdir,
			start_filters=start_filters,
			group_by_method=group_by_method,
			group_all=group_all,
		)


def main() -> None:
	print("Organizing time-to-target JSON files...")
	organize_time_to_target_files()
	repo_root = Path(__file__).resolve().parents[1]
	solutions_root = repo_root / "outputs" /  "solutions"
	output_csv = repo_root / "outputs" / "results" / "solutions_combined.csv"
	print("Aggregating solution CSV files...")
	concatenate_solution_csvs(solutions_root=solutions_root, output_csv=output_csv)
	#_run_plots(["null", "0"])
	_run_plots(["null", "0"], group_by_method=False)
	_run_plots(["null", "0"], group_by_method=True)
	_run_plots(["null", "0"], group_all=True)


if __name__ == "__main__":
	main()
