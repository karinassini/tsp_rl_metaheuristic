import csv
import datetime
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

class SolutionSaver:
    """
    Class to save TSP solutions (route and total distance) to a file.
    """
    def __init__(self, save_dir='solutions', instance_name=None):
        self.save_dir = save_dir
        self.instance_name = instance_name
        
        if os.path.exists('./src/solver/best_known.json'):
            with open('./src/solver/best_known.json', 'r') as f:
                best_data = json.load(f)
            self.best_total_distance = best_data.get(self.instance_name, None)
            self.best_total_distance = self.best_total_distance["value"]
        else:
            self.best_total_distance = None
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, route, total_distance, **params):
        data = {
            'route': route,
            'total_distance': total_distance,
            'best_total_distance': self.best_total_distance,
            'params': params if params else None
        }

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{self.save_dir}/{timestamp}_solution.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Solution saved to {file_path}")

    def save_txt(self, route, total_distance, filename='solution.txt'):
        file_path = f"{self.save_dir}/{filename}"
        with open(file_path, 'w') as f:
            f.write(f"Route: {route}\n")
            f.write(f"Total distance: {total_distance}\n")
            f.write(f"Best known distance: {self.best_total_distance}\n")  # Placeholder for best known
        print(f"Solution saved to {file_path}")
        
    def plot_solution_graph(self, route, adj_matrix, filename='solution_graph.png'):
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
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
            
            # Build and draw the TSP route edges, highlighting the cycle
            route_edges = []
            for i in range(len(route) - 1):
                route_edges.append((route[i], route[i + 1]))
            # Complete the cycle by returning to the starting node
            route_edges.append((route[-1], route[0]))
            
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2)
            
            # Label each edge with its order in the tour
            edge_labels = {}
            for idx, edge in enumerate(route_edges):
                edge_labels[edge] = str(idx + 1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
            
            plt.title("TSP Solution Route")
            plt.savefig(filename)
            plt.close()
            print(f"Solution graph saved to {filename}")


class VNSSummaryTracker:
    """Track repeated VNS runs and export consolidated metrics and plots."""

    def __init__(self) -> None:
        self._history: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        self._summary_cache: Dict[str, Dict[Tuple[str, str, int], Dict[str, Any]]] = defaultdict(dict)

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

    def _write_summary_csv(self, save_dir: str, instance_name:str, method:str) -> None:
        if save_dir not in self._summary_cache:
            return

        fieldnames = [
            "instance",
            "method",
            "runs",
            "k_max",
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
        k_max: int,
        total_distance: float,
        exploration_time: float,
        exploitation_time: float,
        route: List[int],
        save_dir: str,
        best_known: Optional[float],
    ) -> Dict[str, Any]:
        key = (instance_name, method, k_max)
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
                "runs": len(self._history[key]),
                "k_max": k_max,
            }
        )

        self._summary_cache[save_dir][key] = summary
        self._write_summary_csv(save_dir, instance_name, method)
        return summary

    def plot_objective_trend(
        self,
        *,
        instance_name: str,
        method: str,
        k_max: int,
        save_dir: str,
        normalize: bool = True,
        filename: str = "objective_trend.png",
    ) -> str:
        key = (instance_name, method, k_max)
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
        plt.plot(runs, values, marker="o", label="Objective Function", color="royalblue")
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
        k_max: int,
        save_dir: str,
        filename: str = "time_to_target.png",
        tolerance: float = 1e-6,
    ) -> str:
        """Plot the time required to reach the best-known solution for each run."""
        key = (instance_name, method, k_max)
        if key not in self._history:
            raise ValueError("No history available for the requested configuration.")

        summary = self._summary_cache.get(save_dir, {}).get(key, {})
        best_known = summary.get("best_known_total_distance")
        if best_known is None:
            raise ValueError("Best known solution is unavailable; cannot compute time to target.")

        history = self._history[key]
        successful_runs: List[int] = []
        successful_times: List[float] = []
        unsuccessful_runs: List[int] = []
        unsuccessful_times: List[float] = []

        for idx, entry in enumerate(history, start=1):
            total_time = entry["exploration_time"] + entry["exploitation_time"]
            if abs(entry["total_distance"] - best_known) <= tolerance:
                successful_runs.append(idx)
                successful_times.append(total_time)
            else:
                unsuccessful_runs.append(idx)
                unsuccessful_times.append(total_time)

        total_runs = len(history)

        fig, (ax_counts, ax_percentage) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        if successful_times:
            ordered_times = sorted(successful_times)
            cumulative_counts = list(range(1, len(ordered_times) + 1))
            cumulative_percentage = [count / total_runs * 100 for count in cumulative_counts]

            ax_counts.plot(
                cumulative_counts,
                ordered_times,
                marker="o",
                color="forestgreen",
                label="Reached target",
            )
            mean_time = mean(ordered_times)
            ax_counts.axhline(
                mean_time,
                color="royalblue",
                linestyle="--",
                label=f"Mean TTT {mean_time:.2f}s",
            )

            ax_percentage.plot(
                cumulative_percentage,
                ordered_times,
                marker="o",
                color="forestgreen",
            )
            ax_percentage.axvline(
                cumulative_percentage[-1],
                color="dimgray",
                linestyle="--",
                label="Final % of runs",
            )
        else:
            ax_counts.text(
                0.5,
                0.5,
                "Target not reached",
                ha="center",
                va="center",
                transform=ax_counts.transAxes,
                color="dimgray",
            )
            ax_percentage.text(
                0.5,
                0.5,
                "Target not reached",
                ha="center",
                va="center",
                transform=ax_percentage.transAxes,
                color="dimgray",
            )

        if unsuccessful_times:
            ax_counts.scatter(
                unsuccessful_runs,
                unsuccessful_times,
                color="salmon",
                marker="x",
                label="Not reached",
            )
            ax_percentage.scatter(
                [0] * len(unsuccessful_times),
                unsuccessful_times,
                color="salmon",
                marker="x",
            )

        ax_counts.set_title("Cumulative runs")
        ax_counts.set_xlabel("Runs reaching target")
        ax_counts.set_ylabel("Time to target (s)")
        ax_counts.grid(True, linestyle=":", linewidth=0.5)
        handles, labels = ax_counts.get_legend_handles_labels()
        if handles:
            ax_counts.legend()

        ax_percentage.set_title("Cumulative runs (%)")
        ax_percentage.set_xlabel("Runs reaching target (%)")
        ax_percentage.set_ylabel("Time to target (s)")
        ax_percentage.set_xlim(left=0, right=100)
        ax_percentage.grid(True, linestyle=":", linewidth=0.5)
        pct_handles, pct_labels = ax_percentage.get_legend_handles_labels()
        if pct_handles:
            ax_percentage.legend(loc="lower right")

        fig.suptitle(f"Time to Target - {instance_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, filename)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return plot_path
