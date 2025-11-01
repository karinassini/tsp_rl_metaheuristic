import os
import glob
import json
import pandas as pd
import plotly.graph_objects as go


class SolutionComparisonPlotter:
    def __init__(self, directory):
        self.directory = directory
        self.data = []

    def read_solutions(self):
        pattern = os.path.join(self.directory, "*.json")
        files = glob.glob(pattern)
        for file_path in files:
            with open(file_path, "r") as f:
                try:
                    solution = json.load(f)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            filename = os.path.basename(file_path)
            name = os.path.splitext(filename)[0]
            total_distance = solution.get("total_distance")
            best_total_distance = solution.get("best_total_distance")
            method = solution.get("params", {}).get("method", "unknown")
            self.data.append(
                {
                    "solution_name": name,
                    "total_distance": total_distance,
                    "best_total_distance": best_total_distance,
                    "method": method,
                }
            )

    def plot_comparison(self):
        if not self.data:
            print("No data to plot. Did you run read_solutions()?")
            return

        df = pd.DataFrame(self.data)
        x = df["solution_name"]

        # Choose bar colors based on the method field:
        # If method is "greed", use blue; otherwise use green.
        colors = ["blue" if m == "greedy" else "green" for m in df["method"]]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x,
                y=df["total_distance"],
                name="Total Distance",
                marker_color=colors,
                opacity=0.7,
            )
        )

        # Plot a horizontal line for best_total_distance using the first value
        best_value = df["best_total_distance"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[best_value] * len(x),
                mode="lines",
                name="Best Total Distance",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title="Solution Comparison: Total vs Best Distance",
            xaxis_title="Solution Name",
            yaxis_title="Distance",
            legend_title="Metric",
            template="plotly_white",
        )

        fig.show()

    def run(self):
        self.read_solutions()
        self.plot_comparison()
