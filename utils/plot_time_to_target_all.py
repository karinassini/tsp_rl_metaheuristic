"""Utility to plot aggregated time-to-target curves from JSON data."""

from __future__ import annotations

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


def _iter_json_files(sources: Iterable[Path | str]) -> Iterator[Path]:
    for source in sources:
        path = Path(source)
        if path.is_dir():
            yield from sorted(candidate for candidate in path.glob("*.json") if candidate.is_file())
        elif path.suffix.lower() == ".json" and path.exists():
            yield path


def _compose_label(base_label: str, counters: Dict[str, int]) -> str:
    label = base_label or "series"
    count = counters.get(label, 0)
    counters[label] = count + 1
    return label if count == 0 else f"{label}-{count}"


def plot_time_to_target_collection(
    json_sources: Iterable[Path | str],
    *,
    output_dir: Path | str | None = None,
) -> None:
    destination = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "data"
    destination.mkdir(parents=True, exist_ok=True)

    per_instance: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    label_counters: Dict[str, int] = {}
    color_cycle = cycle([
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
    ])
    file_colors: Dict[str, str] = {}

    for json_path in _iter_json_files(json_sources):
        file_stem = json_path.stem
        with json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        color = file_colors.setdefault(file_stem or "series", next(color_cycle))

        for key, series_map in payload.items():
            instance, method, iteration_max, solver = _parse_key(key)
            default_label = f"{method} | iter={iteration_max} | {solver}"
            for series_name, series in series_map.items():
                times = np.asarray(series.get("times", []), dtype=float)
                probabilities = np.asarray(series.get("probabilities", []), dtype=float)
                style = dict(series.get("style", {}))
                base_label = style.get("label") or series_name or default_label
                style["label"] = _compose_label(base_label, label_counters)
                style["color"] = color

                per_instance[instance].append(
                    {
                        "times": times,
                        "probabilities": probabilities,
                        "total_runs": int(series.get("total_runs", len(times))),
                        "success_count": int(series.get("success_count", len(times))),
                        "failures": int(series.get("failures", 0)),
                        "mean_time": series.get("mean_time"),
                        "style": style,
                    }
                )

    for instance_name, series_list in per_instance.items():
        if not series_list:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        min_time: float | None = None

        for series in series_list:
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
            color = style.get("color", "forestgreen")
            linewidth = style.get("linewidth", 2)
            marker = style.get("marker", "x")
            label = style.get("label")

            first_time = float(times[0])
            min_time = first_time if min_time is None else min(min_time, first_time)

            smooth_times, smooth_prob = _smooth_curve(times, probabilities)
            ax.plot(smooth_times, smooth_prob, color=color, linewidth=linewidth, label=label)
            if marker:
                ax.scatter(times, probabilities, color=color, marker=marker)

            mean_time = series.get("mean_time")
            if mean_time is not None:
                ax.axvline(
                    mean_time,
                    color=color,
                    linestyle="--",
                    linewidth=max(linewidth * 0.75, 1.0),
                    alpha=0.7,
                    label="_nolegend_",
                )
                ax.text(
                    mean_time,
                    0.03,
                    f"{mean_time:.2f}s",
                    color=color,
                    rotation=90,
                    rotation_mode="anchor",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                )

        if min_time is not None:
            ax.set_xlim(left=min_time)

        ax.set_title(f"Time to Target - {instance_name}")
        ax.set_xlabel("Time to target (s)")
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(loc="lower right")

        out_path = destination / f"time_to_target_{instance_name}_combined.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    plot_time_to_target_collection([data_dir])


if __name__ == "__main__":
    main()

