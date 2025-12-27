"""Copy time-to-target JSON files into a structured plots folder.

The script scans every instance inside ``outputs/solutions`` and copies each
``time_to_target_data.json`` produced by a VNS run into the plots hierarchy:

``outputs/plots/solutions/<method>/<instance>/start_<value>/``

Where ``<value>`` is the ``start_city_initialization`` captured in the solution
JSON ("null" when the field is ``None``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Set

RUN_FILE_SUFFIX = "_solution.json"
TIME_TO_TARGET_NAME = "time_to_target_data.json"

def _find_start_city(run_dir: Path) -> Optional[int]:
    for solution_path in sorted(run_dir.glob(f"*{RUN_FILE_SUFFIX}")):
        try:
            with solution_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        params = payload.get("params") or {}
        if "start_city_initialization" in params:
            return params["start_city_initialization"]
    return None


def _extract_start_labels(json_path: Path) -> Set[str]:
    labels: Set[str] = set()
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"all"}

    for series_map in payload.values():
        if not isinstance(series_map, dict):
            continue
        for series_name in series_map.keys():
            label = "all"
            if isinstance(series_name, str) and "|start_" in series_name:
                label = series_name.split("|start_", 1)[-1] or "all"
            labels.add(label)

    return labels or {"all"}

def organize_time_to_target_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    solutions_root = repo_root / "outputs" / "solutions"
    plots_root = repo_root / "outputs" / "plots" / "solutions"
    if not solutions_root.exists():
        raise FileNotFoundError(f"Missing source folder: {solutions_root}")

    copied = 0
    for instance_dir in sorted(solutions_root.iterdir()):
        if not instance_dir.is_dir():
            continue
        instance_name = instance_dir.name
        for solver_dir in sorted(instance_dir.iterdir()):
            if not solver_dir.is_dir():
                continue
            solver_name = solver_dir.name
            for method_dir in sorted(solver_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method_name = method_dir.name
                for run_dir in sorted(method_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    source_file = run_dir / TIME_TO_TARGET_NAME
                    if not source_file.exists():
                        continue
                    json_start_labels = _extract_start_labels(source_file)
                    if not json_start_labels:
                        json_start_labels = {"all"}

                    for start_label in json_start_labels:
                        dest_dir = (
                            plots_root
                            / solver_name
                            / instance_name
                            / f"start_{start_label}"
                        )
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_dir / f"time_to_target_data_{run_dir.name}.json"
                        shutil.copy2(source_file, dest_file)
                        copied += 1

    print(f"Copied {copied} time_to_target_data.json files into {plots_root}")


def main() -> None:
    organize_time_to_target_files()


if __name__ == "__main__":
    main()
