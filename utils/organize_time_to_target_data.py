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
from typing import Optional

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
            method_name = solver_dir.name
            for init_dir in sorted(solver_dir.iterdir()):
                if not init_dir.is_dir():
                    continue
                for run_dir in sorted(init_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    source_file = run_dir / TIME_TO_TARGET_NAME
                    if not source_file.exists():
                        continue
                    start_city = _find_start_city(run_dir)
                    start_label = "null" if start_city is None else str(start_city)
                    dest_dir = (
                        plots_root
                        / method_name
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
