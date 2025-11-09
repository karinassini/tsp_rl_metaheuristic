import csv
import datetime
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class SolutionSaver:
    """
    Class to save TSP solutions (route and total distance) to a file.
    """

    def __init__(self, save_dir="solutions", instance_name=None):
        self.save_dir = save_dir
        self.instance_name = instance_name

        if os.path.exists("./src/solver/best_known.json"):
            with open("./src/solver/best_known.json", "r") as f:
                best_data = json.load(f)
            self.best_total_distance = best_data.get(self.instance_name, None)
            self.best_total_distance = self.best_total_distance["value"]
        else:
            self.best_total_distance = None
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, route, total_distance, **params):
        data = {
            "route": route,
            "total_distance": total_distance,
            "best_total_distance": self.best_total_distance,
            "params": params if params else None,
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.save_dir}/{timestamp}_solution.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Solution saved to {file_path}")

    def save_txt(self, route, total_distance, filename="solution.txt"):
        file_path = f"{self.save_dir}/{filename}"
        with open(file_path, "w") as f:
            f.write(f"Route: {route}\n")
            f.write(f"Total distance: {total_distance}\n")
            f.write(
                f"Best known distance: {self.best_total_distance}\n"
            )  # Placeholder for best known
        print(f"Solution saved to {file_path}")

    def plot_solution_graph(self, route, adj_matrix, filename="solution_graph.png"):
        """
        Plots the TSP route as a graph using matplotlib and networkx. The graph is built from an adjacency matrix,
        and the TSP route is highlighted with edge labels indicating the order of the tour.

        Parameters:
            route (list): Ordered list of nodes.
            adj_matrix (list of lists): Adjacency matrix representing graph weights. A non-zero entry at [i][j]
                                        indicates an edge between node i and node j.
            filename (str): The name of the file to save the graph image.
        """
        import matplotlib.pyplot as plt

        # Create the graph from the adjacency matrix
        num_nodes = len(adj_matrix)
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] != 0:  # treat non-zero as an edge
                    G.add_edge(i, j, weight=adj_matrix[i][j])

        # Generate positions for all nodes using a layout algorithm
        pos = nx.spring_layout(G, seed=42)

        # Draw the entire graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=500,
        )

        # Build and draw the TSP route edges, highlighting the cycle
        route_edges = []
        for i in range(len(route) - 1):
            route_edges.append((route[i], route[i + 1]))
        # Complete the cycle by returning to the starting node
        route_edges.append((route[-1], route[0]))

        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color="red", width=2)

        # Label each edge with its order in the tour
        edge_labels = {}
        for idx, edge in enumerate(route_edges):
            edge_labels[edge] = str(idx + 1)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

        plt.title("TSP Solution Route")
        plt.savefig(filename)
        plt.close()
        print(f"Solution graph saved to {filename}")


class VNSSummaryTracker:
    """Track repeated VNS runs and export consolidated metrics and plots."""

    def __init__(self) -> None:
        self._history: Dict[
            Tuple[str, str, int, str], List[Dict[str, Any]]
        ] = defaultdict(list)
        self._summary_cache: Dict[
            str, Dict[Tuple[str, str, int, str], Dict[str, Any]]
        ] = defaultdict(dict)
        self._cdf_cache: Dict[
            str, Dict[Tuple[str, str, int, str], Dict[str, Dict[str, Any]]]
        ] = defaultdict(dict)

    @staticmethod
    def _cdf_key_to_str(key: Tuple[str, str, int, str]) -> str:
        instance, method, iteration_max, solver = key
        return f"{instance}|{method}|{iteration_max}|{solver}"

    @staticmethod
    def _cdf_str_to_key(key_str: str) -> Tuple[str, str, int, str]:
        instance, method, iteration_max, solver = key_str.split("|")
        return instance, method, int(iteration_max), solver

    def _persist_cdf_series(self, save_dir: str) -> None:
        cache_for_dir = self._cdf_cache.get(save_dir)
        if not cache_for_dir:
            return

        serializable: Dict[str, Dict[str, Any]] = {}
        for key, series_by_label in cache_for_dir.items():
            key_str = self._cdf_key_to_str(key)
            serializable[key_str] = series_by_label

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "time_to_target_data.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2)

    def _load_cdf_series(self, save_dir: str) -> None:
        path = os.path.join(save_dir, "time_to_target_data.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        cache_for_dir = self._cdf_cache.setdefault(save_dir, {})
        for key_str, series_map in data.items():
            key = self._cdf_str_to_key(key_str)
            normalised: Dict[str, Dict[str, Any]] = {}
            for label, payload in series_map.items():
                if "probabilities" not in payload and "percentages" in payload:
                    probs = [val / 100 for val in payload["percentages"]]
                    payload["probabilities"] = probs
                    payload.pop("percentages", None)
                if "style" not in payload:
                    payload["style"] = {
                        "color": "forestgreen",
                        "marker": "x",
                        "label": label,
                        "linewidth": 2,
                    }
                normalised[label] = payload
            cache_for_dir[key] = normalised

    @staticmethod
    def _format_route(route: List[int]) -> str:
        return "->".join(map(str, route))

    @staticmethod
    def _compute_summary(
        history: List[Dict[str, Any]],
        best_known: Optional[float],
    ) -> Dict[str, Any]:
        distances = [entry["total_distance"] for entry in history]
        exploration_times = [entry["exploration_time"] for entry in history]
        exploitation_times = [entry["exploitation_time"] for entry in history]

        mean_total = mean(distances)
        std_total = stdev(distances) if len(distances) > 1 else 0.0

        best_run = min(history, key=lambda entry: entry["total_distance"])
        best_total = best_run["total_distance"]
        mean_distance_to_best = mean_total - best_total
        mean_distance_to_best_pct = (
            (mean_distance_to_best / best_total) * 100 if best_total else 0.0
        )

        mean_distance_to_optimal: Optional[float] = None
        mean_distance_to_optimal_pct: Optional[float] = None
        best_known_hits: Optional[int] = None
        if best_known is not None:
            mean_distance_to_optimal = mean_total - best_known
            mean_distance_to_optimal_pct = (
                (mean_distance_to_optimal / best_known) * 100 if best_known else 0.0
            )
            tolerance = 1e-6
            best_known_hits = sum(
                1
                for entry in history
                if abs(entry["total_distance"] - best_known) <= tolerance
            )

        return {
            "mean_total_distance": mean_total,
            "std_total_distance": std_total,
            "best_total_distance": best_total,
            "mean_distance_to_best": mean_distance_to_best,
            "mean_distance_to_best_pct": mean_distance_to_best_pct,
            "mean_distance_to_optimal": mean_distance_to_optimal,
            "mean_distance_to_optimal_pct": mean_distance_to_optimal_pct,
            "best_known_total_distance": best_known,
            "best_known_hits": best_known_hits,
            "best_route": best_run["route"],
            "best_exploration_time": best_run["exploration_time"],
            "best_exploitation_time": best_run["exploitation_time"],
            "mean_exploration_time": mean(exploration_times),
            "mean_exploitation_time": mean(exploitation_times),
        }

    def _write_summary_csv(
        self, save_dir: str, instance_name: str, method: str
    ) -> None:
        if save_dir not in self._summary_cache:
            return

        fieldnames = [
            "instance",
            "method",
            "runs",
            "iteration_max",
            "mean_total_distance",
            "std_total_distance",
            "best_total_distance",
            "mean_distance_to_best",
            "mean_distance_to_best_pct",
            "best_known_total_distance",
            "best_known_hits",
            "mean_distance_to_optimal",
            "mean_distance_to_optimal_pct",
            "mean_exploration_time",
            "mean_exploitation_time",
            "best_route",
            "best_exploration_time",
            "best_exploitation_time",
            "solver"
        ]

        csv_filename = f"{instance_name}_{method}_summary.csv"
        csv_path = os.path.join(save_dir, csv_filename)

        rows = []
        for summary in self._summary_cache[save_dir].values():
            row = summary.copy()
            row["best_route"] = self._format_route(row["best_route"])
            for key in (
                "mean_total_distance",
                "std_total_distance",
                "best_total_distance",
                "mean_distance_to_best",
                "mean_distance_to_best_pct",
                "best_known_total_distance",
                "mean_distance_to_optimal",
                "mean_distance_to_optimal_pct",
                "mean_exploration_time",
                "mean_exploitation_time",
                "best_exploration_time",
                "best_exploitation_time",
            ):
                if row.get(key) is None:
                    row[key] = ""
                elif isinstance(row[key], float):
                    row[key] = f"{row[key]:.4f}"
            rows.append(row)

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def record_run(
        self,
        *,
        instance_name: str,
        method: str,
        iteration_max: int,
        solver_name: str,
        total_distance: float,
        exploration_time: float,
        exploitation_time: float,
        route: List[int],
        save_dir: str,
        best_known: Optional[float],
    ) -> Dict[str, Any]:
        key = (instance_name, method, iteration_max, solver_name)
        self._history[key].append(
            {
                "total_distance": total_distance,
                "exploration_time": exploration_time,
                "exploitation_time": exploitation_time,
                "route": route,
            }
        )

        summary = self._compute_summary(self._history[key], best_known)
        summary.update(
            {
                "instance": instance_name,
                "method": method,
                "solver": solver_name,
                "runs": len(self._history[key]),
                "iteration_max": iteration_max,
            }
        )

        self._summary_cache[save_dir][key] = summary
        self._cdf_cache[save_dir].pop(key, None)
        self._write_summary_csv(save_dir, instance_name, method)
        return summary

    def plot_objective_trend(
        self,
        *,
        instance_name: str,
        method: str,
        iteration_max: int,
        save_dir: str,
    solver_name: str,
        normalize: bool = True,
        filename: str = "objective_trend.png",
    ) -> str:
        key = (instance_name, method, iteration_max, solver_name)
        if key not in self._history:
            raise ValueError("No history available for the requested configuration.")

        history = self._history[key]
        distances = [entry["total_distance"] for entry in history]

        summary = self._summary_cache.get(save_dir, {}).get(key, {})
        best_known = summary.get("best_known_total_distance")

        if normalize and best_known:
            values = [distance / best_known for distance in distances]
            ylabel = "Objective / Best Known"
        else:
            values = distances
            ylabel = "Objective Function Value"

        running_std: List[float] = []
        for idx in range(1, len(values) + 1):
            if idx == 1:
                running_std.append(0.0)
            else:
                running_std.append(stdev(values[:idx]))

        runs = list(range(1, len(values) + 1))

        plt.figure(figsize=(6, 4))
        plt.plot(
            runs, values, marker="o", label="Objective Function", color="royalblue"
        )
        plt.plot(runs, running_std, label="Standard Deviation", color="dimgray")
        plt.title(f"{instance_name} Instance")
        plt.xlabel("Number of Runs")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_time_to_target(
        self,
        *,
        instance_name: str,
        method: str,
        iteration_max: int,
        save_dir: str,
        solver_name: str,
        filename: str = "time_to_target.png",
        tolerance: float = 1e-6,
        store_key: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Plot cumulative probability of reaching a target solution over time."""
        key = (instance_name, method, iteration_max, solver_name)
        if key not in self._history:
            raise ValueError("No history available for the requested configuration.")

        summary = self._summary_cache.get(save_dir, {}).get(key, {})
        best_known = summary.get("best_known_total_distance")
        if best_known is None:
            raise ValueError(
                "Best known solution is unavailable; cannot compute time to target."
            )

        history = self._history[key]
        total_runs = len(history)
        successful_times = sorted(
            entry["exploration_time"] + entry["exploitation_time"]
            for entry in history
            if abs(entry["total_distance"] - best_known) <= tolerance
        )

        plt.figure(figsize=(6, 4))
        plot_style = {
            "color": "forestgreen",
            "marker": "x",
            "label": method,
            "linewidth": 2,
        }
        if style:
            plot_style.update(style)
        plot_style.setdefault("label", method)

        dir_cache = self._cdf_cache[save_dir]

        if successful_times:
            unique_times, counts = np.unique(successful_times, return_counts=True)
            cumulative_counts = np.cumsum(counts)
            cumulative_prob = cumulative_counts / total_runs
            if len(unique_times) > 1:
                point_count = max(len(unique_times) * 10, 200)
                smooth_times = np.linspace(unique_times[0], unique_times[-1], point_count)
                smooth_prob = np.interp(smooth_times, unique_times, cumulative_prob)
            else:
                smooth_times = unique_times
                smooth_prob = cumulative_prob

            ax = plt.gca()
            ax.plot(
                smooth_times,
                smooth_prob,
                color=plot_style["color"],
                linewidth=plot_style.get("linewidth", 2),
                label=plot_style["label"],
            )
            marker_symbol = plot_style.get("marker")
            if marker_symbol:
                ax.scatter(
                    unique_times,
                    cumulative_prob,
                    color=plot_style["color"],
                    marker=marker_symbol,
                )
            avg_time = mean(successful_times)
            ax.set_xlim(left=unique_times[0])
            ax.axvline(
                avg_time,
                color=plot_style["color"],
                linestyle="--",
                linewidth=max(plot_style.get("linewidth", 2) * 0.75, 1.0),
                alpha=0.7,
                label="_nolegend_",
            )
            ax.text(
                avg_time,
                0.03,
                f"{avg_time:.2f}s",
                color=plot_style["color"],
                rotation=90,
                rotation_mode="anchor",
                ha="left",
                va="bottom",
                fontsize=9,
            )
            if store_key is not None:
                series_map = dir_cache.setdefault(key, {})
                series_map[store_key] = {
                    "times": unique_times.tolist(),
                    "probabilities": cumulative_prob.tolist(),
                    "mean_time": float(avg_time),
                    "total_runs": total_runs,
                    "style": plot_style,
                }
                self._persist_cdf_series(save_dir)
        else:
            plt.text(
                0.5,
                0.5,
                "Target not reached",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                color="dimgray",
            )
            plt.ylim(0, 1)
            if store_key is not None:
                series_map = dir_cache.setdefault(key, {})
                series_map[store_key] = {
                    "times": [],
                    "probabilities": [],
                    "mean_time": None,
                    "total_runs": total_runs,
                    "style": plot_style,
                }
                self._persist_cdf_series(save_dir)

        plt.title(f"Time to Target - {instance_name}")
        plt.xlabel("Time to target (s)")
        plt.ylabel("Cumulative probability")
        plt.ylim(0, 1)
        plt.grid(True, linestyle=":", linewidth=0.5)
        if successful_times:
            plt.legend(loc="lower right")

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def get_time_to_target_series(
        self,
        *,
        instance_name: str,
        method: str,
        iteration_max: int,
        save_dir: str,
        solver_name: str,
        key_name: str,
    ) -> Dict[str, Any]:
        key = (instance_name, method, iteration_max, solver_name)
        cached = self._cdf_cache.get(save_dir, {}).get(key)
        if not cached or key_name not in cached:
            self._load_cdf_series(save_dir)
            cached = self._cdf_cache.get(save_dir, {}).get(key)
        if not cached or key_name not in cached:
            raise ValueError(
                "Requested time-to-target series is not cached. Run plot_time_to_target with store_key first."
            )
        return cached[key_name]
