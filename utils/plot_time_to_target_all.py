"""Utility to plot aggregated time-to-target curves from JSON data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

def _smooth_curve(times: np.ndarray, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(times) <= 1:
        return times, probabilities
    point_count = max(len(times) * 10, 200)
    smooth_times = np.linspace(times[0], times[-1], point_count)
    smooth_prob = np.interp(smooth_times, times, probabilities)
    return smooth_times, smooth_prob


def _parse_key(key: str) -> Tuple[str, str, str, str]:
    instance, method, iteration_max, solver = key.split("|")
    return instance, method, iteration_max, solver


def _iter_json_files(
    sources: Iterable[Path | str],
    *,
    start_filters: set[str] | None = None,
) -> Iterator[Path]:
    for source in sources:
        path = Path(source)
        if path.is_dir():
            pattern = "time_to_target*.json"
            for candidate in sorted(
                candidate
                for candidate in path.rglob(pattern)
                if candidate.is_file()
            ):
                if _match_start_filter(candidate, start_filters):
                    yield candidate
        elif (
            path.suffix.lower() == ".json"
            and path.exists()
            and path.name.startswith("time_to_target")
        ):
            if _match_start_filter(path, start_filters):
                yield path


def _compose_label(base_label: str, counters: Dict[str, int]) -> str:
    label = base_label or "series"
    if base_label:
        return label

    count = counters.get(label, 0)
    counters[label] = count + 1
    return label if count == 0 else f"{label}-{count}"


def _match_start_filter(path: Path, filters: set[str] | None) -> bool:
    if not filters:
        return True
    start_part = next((part for part in path.parts if part.startswith("start_")), None)
    if start_part is None:
        return False
    suffix = start_part.split("start_", 1)[-1]
    return suffix in filters or start_part in filters


def _extract_start_label(path: Path) -> str:
    start_part = next((part for part in path.parts if part.startswith("start_")), None)
    if not start_part:
        return "all"
    suffix = start_part.split("start_", 1)[-1]
    return suffix or "all"


def plot_time_to_target_collection(
    json_sources: Iterable[Path | str],
    *,
    output_dir: Path | str | None = None,
    start_filters: set[str] | None = None,
    instance_filters: set[str] | None = None,
    method_filters: set[str] | None = None,
    solver_filters: set[str] | None = None,
    group_by_method: bool = False,
) -> None:
    sources = [Path(src) for src in json_sources]
    if not sources:
        raise ValueError("No JSON sources provided")
    if output_dir:
        destination = Path(output_dir)
    else:
        first_source = sources[0]
        destination = first_source if first_source.is_dir() else first_source.parent
    destination.mkdir(parents=True, exist_ok=True)

    per_group_series: Dict[tuple[str, str, str], list[Dict[str, object]]] = defaultdict(list)
    label_counters: Dict[str, int] = {}
    palette_colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Stable colors per solver/method label to keep plots consistent across instances.
    fixed_colors = {
        "VNS_Solver": "#1f77b4",
        "VNS_Solver_Q_Learnings": "#d62728",
    }

    for json_path in _iter_json_files(sources, start_filters=start_filters):
        with json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        start_label = _extract_start_label(json_path)
        for key, series_map in payload.items():
            instance, method, iteration_max, solver = _parse_key(key)
            if instance_filters and instance not in instance_filters:
                continue
            if method_filters and method not in method_filters:
                continue
            if solver_filters and solver not in solver_filters:
                continue
            default_label = f"{method} | iter={iteration_max} | {solver}"
            for series_name, series in series_map.items():
                times = np.asarray(series.get("times", []), dtype=float)
                probabilities = np.asarray(series.get("probabilities", []), dtype=float)
                incoming_style = dict(series.get("style", {}))
                base_label = incoming_style.get("label") or series_name or default_label
                label = _compose_label(base_label, label_counters)
                custom_color = incoming_style.get("color")
                if custom_color in {"", "forestgreen"}:
                    custom_color = None
                style = {k: v for k, v in incoming_style.items() if k != "color"}
                style["label"] = label

                key = (instance, method, start_label) if group_by_method else (solver, instance, start_label)
                per_group_series[key].append(
                    {
                        "times": times,
                        "probabilities": probabilities,
                        "total_runs": int(series.get("total_runs", len(times))),
                        "success_count": int(series.get("success_count", len(times))),
                        "failures": int(series.get("failures", 0)),
                        "mean_time": series.get("mean_time"),
                        "method": method,
                        "solver": solver,
                        "custom_color": custom_color,
                        "style": style,
                    }
                )

    for group_key, series_list in per_group_series.items():
        if not series_list:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        min_time: float | None = None
        color_cycle = cycle(palette_colors)
        color_map: Dict[str, str] = {}

        if group_by_method:
            instance_name, method_name, start_label = group_key
        else:
            solver_name, instance_name, start_label = group_key

        for idx, series in enumerate(series_list):
            raw_times = np.asarray(series.get("times", []), dtype=float)
            if raw_times.size == 0:
                continue

            order = np.argsort(raw_times)
            times = raw_times[order]
            total_runs = int(series.get("total_runs") or times.size)
            if total_runs <= 0:
                total_runs = times.size
            count = int(series.get("success_count") or times.size)
            if count <= 0 or count > times.size:
                count = times.size
            probabilities = (np.arange(1, count + 1) - 0.5) / total_runs
            times = times[:count]
            style = series.get("style") or {}
            series_method = series.get("method") or "method"
            series_solver = series.get("solver") or "solver"
            custom_color = series.get("custom_color")
            if custom_color:
                color = custom_color
            else:
                color_key = series_solver if group_by_method else series_method
                color = fixed_colors.get(color_key)
                if color is None:
                    if color_key not in color_map:
                        color_map[color_key] = next(color_cycle)
                    color = color_map[color_key]
            linewidth = style.get("linewidth", 2)
            marker = style.get("marker", "x")
            # Legend should show only the initialization method (e.g., marl or rcl), without start_* suffixes
            label = series_method

            first_time = float(times[0])
            min_time = first_time if min_time is None else min(min_time, first_time)

            smooth_times, smooth_prob = _smooth_curve(times, probabilities)
            mean_time = series.get("mean_time")
            legend_label = label
            if group_by_method:
                legend_label = f"{series_solver} - {legend_label}" if legend_label else series_solver
            if mean_time is not None:
                legend_label = f"{legend_label} (μ={mean_time:.2f}s)" if legend_label else f"μ={mean_time:.2f}s"
            ax.plot(smooth_times, smooth_prob, color=color, linewidth=linewidth, label=legend_label)
            if marker:
                ax.scatter(times, probabilities, color=color, marker=marker)

            if mean_time is not None:
                baseline_y = min(0.15, 0.025 + 0.02 * idx)
                ax.scatter(
                    [mean_time],
                    [baseline_y],
                    color=color,
                    marker="|",
                    s=160,
                    linewidths=2,
                    zorder=5,
                )

        if min_time is not None:
            ax.set_xlim(left=min_time)

        title_suffix = "start null" if start_label == "null" else f"start {start_label}" if start_label != "all" else "all starts"
        if group_by_method:
            ax.set_title(f"Time to Target - {instance_name}")# (method {method_name}, {title_suffix})")
        else:
            ax.set_title(f"Time to Target - {instance_name}") #({solver_name}, {title_suffix})")
        ax.set_xlabel("Time to target (s)")
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(loc="lower right")

        suffix = start_label if start_filters and len(start_filters) == 1 else start_label
        if group_by_method:
            instance_dir = destination / "method_compare" / method_name / instance_name / f"start_{suffix}"
            out_path = instance_dir / f"time_to_target_{instance_name}_{method_name}_start_{suffix}.png"
        else:
            instance_dir = destination / solver_name / instance_name / f"start_{suffix}"
            out_path = instance_dir / f"time_to_target_{instance_name}_{solver_name}_start_{suffix}.png"
        instance_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time-to-target curves")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Directories or files containing time_to_target_data*.json",
    )
    parser.add_argument(
        "--start",
        action="append",
        dest="start_filters",
        default=None,
        help="Restrict to runs whose folder contains start_<value> (e.g. --start 0)",
    )
    parser.add_argument(
        "--instance",
        action="append",
        dest="instance_filters",
        default=None,
        help="Restrict plotting to specific instance names (e.g. --instance swiss42)",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="method_filters",
        default=None,
        help="Restrict plotting to initialization methods (e.g. --method random)",
    )
    parser.add_argument(
        "--solver",
        action="append",
        dest="solver_filters",
        default=None,
        help="Restrict plotting to solver types (values stored in JSON keys, e.g. --solver VNS_Solver)",
    )
    parser.add_argument(
        "--group-by-method",
        action="store_true",
        dest="group_by_method",
        help="Plot one chart per method comparing every solver present",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "outputs" / "plots" / "solutions"
    sources = [Path(p) for p in args.paths] if args.paths else [default_dir]

    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Provided path does not exist: {source}")

    start_filters: set[str] | None = None
    if args.start_filters:
        start_filters = set()
        for raw in args.start_filters:
            if not raw:
                continue
            value = raw.replace("start_", "", 1) if raw.startswith("start_") else raw
            start_filters.add(value)
    instance_filters: set[str] | None = None
    if args.instance_filters:
        instance_filters = {name for name in args.instance_filters if name}
    method_filters: set[str] | None = None
    if args.method_filters:
        method_filters = {name for name in args.method_filters if name}
    solver_filters: set[str] | None = None
    if args.solver_filters:
        solver_filters = {name for name in args.solver_filters if name}

    plot_time_to_target_collection(
        sources,
        start_filters=start_filters,
        instance_filters=instance_filters,
        method_filters=method_filters,
        solver_filters=solver_filters,
        group_by_method=args.group_by_method,
    )


if __name__ == "__main__":
    main()

