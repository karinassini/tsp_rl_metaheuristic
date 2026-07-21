"""Parse VNS log files and render each route snapshot as an image using NetworkX."""

from __future__ import annotations
""""
# Example:
# python -m utils.plot_routes_from_log_circle \
--log "outputs/solutions/burma14/VNS_Solver/rcl/20260502_220433/logs/vns_steps_20260502_220433_rcl_96688_4566621488.log" \
--out "outputs/plots/burma14_rcl_20260502" \
--only-improvements
"""


import argparse
import ast
import math
import re
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.structures.graph import Graph


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_ROUTE_RE = re.compile(r"route:\s*(\[[\d,\s]+\]),\s*cost:\s*([\d.]+)")
_CONTEXT_RE = re.compile(r"INFO\s*-\s*(.+)")


def _parse_context(line: str) -> Optional[str]:
    m = _CONTEXT_RE.search(line)
    if not m:
        return None
    text = m.group(1).strip()
    if len(text) > 120:
        text = text[:117] + "..."
    return text


def parse_routes_from_log(log_path: str | Path) -> List[dict]:
    log_path = Path(log_path)
    entries: list[dict] = []
    prev_context: Optional[str] = None

    with open(log_path, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip()
            m = _ROUTE_RE.search(line)
            if m:
                route = ast.literal_eval(m.group(1))
                cost = float(m.group(2))
                label = prev_context or f"step {len(entries) + 1}"
                entries.append({"route": route, "cost": cost, "label": label, "line_number": lineno})
                prev_context = None
            else:
                ctx = _parse_context(line)
                if ctx is not None:
                    prev_context = ctx
    return entries


def filter_improvements(entries: List[dict]) -> List[dict]:
    filtered: list[dict] = []
    best = float("inf")
    for e in entries:
        if e["cost"] < best - 1e-9:
            best = e["cost"]
            filtered.append(e)
    return filtered


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _node_positions(graph: Optional[Graph]) -> Optional[dict[int, Tuple[float, float]]]:
    if graph is None:
        return None
    coords = getattr(graph, "coords", None)
    if coords is None:
        return None
    pos: dict[int, Tuple[float, float]] = {}
    for idx, c in enumerate(coords):
        if c is not None:
            pos[idx] = (float(c[0]), float(c[1]))
    return pos if pos else None


def _circular_layout(nodes: List[int], radius: float = 1.0) -> dict[int, Tuple[float, float]]:
    """Place nodes evenly on a circle, sorted by node id."""
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)
    pos: dict[int, Tuple[float, float]] = {}
    for i, nid in enumerate(sorted_nodes):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[nid] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _edge_set(route: List[int]) -> set[Tuple[int, int]]:
    """Return the set of undirected edges for a closed route."""
    closed = route if route[0] == route[-1] else route + [route[0]]
    return {(min(closed[i], closed[i + 1]), max(closed[i], closed[i + 1]))
            for i in range(len(closed) - 1)}


# ---------------------------------------------------------------------------
# NetworkX rendering
# ---------------------------------------------------------------------------

def render_route(
    route: List[int],
    cost: float,
    label: str,
    output_path: str | Path,
    positions: Optional[dict[int, Tuple[float, float]]] = None,
    prev_route: Optional[List[int]] = None,
    fmt: str = "png",
    figsize: Tuple[float, float] = (16, 14),
    dpi: int = 150,
) -> Path:
    """Render a single TSP route using NetworkX.

    Edge colouring:
    - Light gray  : all possible edges (background, inside the circle)
    - Blue        : edges in current route unchanged from previous (or all if no prev)
    - Red         : edges added in current route vs previous
    - Yellow      : edges removed from previous route (not in current)
    """
    output_path = Path(output_path)

    # Unique nodes (strip repeated closing node if present)
    seen: set[int] = set()
    unique_nodes: list[int] = []
    for nid in route:
        if nid not in seen:
            seen.add(nid)
            unique_nodes.append(nid)

    # All nodes (current + previous) so removed edges can be drawn
    all_nodes_set: set[int] = set(unique_nodes)
    if prev_route is not None:
        for nid in prev_route:
            all_nodes_set.add(nid)
    all_nodes: list[int] = sorted(all_nodes_set)

    # Compute edge sets
    current_edges: set[Tuple[int, int]] = _edge_set(route)
    prev_edges: set[Tuple[int, int]] = _edge_set(prev_route) if prev_route is not None else set()

    # added = in current but NOT in previous
    added_edges: set[Tuple[int, int]] = current_edges - prev_edges
    # removed = in previous but NOT in current
    removed_edges: set[Tuple[int, int]] = prev_edges - current_edges
    # unchanged = in both
    unchanged_edges: set[Tuple[int, int]] = current_edges & prev_edges

    # First route: all edges are blue, nothing red/yellow
    if prev_route is None:
        unchanged_edges = current_edges
        added_edges = set()
        removed_edges = set()

    # All possible background edges (every pair of nodes)
    all_bg_edges = [
        (u, v) for u, v in combinations(all_nodes, 2)
    ]

    # Node positions: circular layout sorted by node id
    pos = _circular_layout(all_nodes, radius=1.0)

    # Build graph for node drawing
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    def _draw_edges(edges, color, alpha, linewidth, linestyle="solid", zorder=2):
        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")

    # 1. All possible edges — very light gray background
    _draw_edges(all_bg_edges, "#DDDDDD", alpha=0.25, linewidth=0.5, zorder=1)

    # 2. Removed edges (yellow) — were in prev route, not in current
    _draw_edges(removed_edges, "#EAB308", alpha=1.0, linewidth=2.8,
                linestyle="dashed", zorder=3)

    # 3. Unchanged current route edges (blue)
    _draw_edges(unchanged_edges, "#2563EB", alpha=0.9, linewidth=2.5, zorder=4)

    # 4. Added edges (red) — new in current route vs previous
    _draw_edges(added_edges, "#DC2626", alpha=1.0, linewidth=2.8, zorder=5)

    # 5. Nodes — large white circles with pink border, red for start node
    node_colors = ["#DC2626" if nid == unique_nodes[0] else "#FFFFFF"
                   for nid in all_nodes]
    border_colors = ["#7F1D1D" if nid == unique_nodes[0] else "#EC4899"
                     for nid in all_nodes]

    nodes_collection = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=all_nodes,
        node_color=node_colors,
        node_size=3200,
        edgecolors=border_colors,
        linewidths=3.0,
    )
    if nodes_collection is not None:
        nodes_collection.set_zorder(8)

    # 6. Node labels — large and bold
    for nid in all_nodes:
        if nid not in pos:
            continue
        x, y = pos[nid]
        color = "white" if nid == unique_nodes[0] else "#1E293B"
        ax.text(x, y, str(nid), fontsize=14, ha="center", va="center",
                fontweight="bold", color=color, zorder=12)

    short_label = label if len(label) <= 90 else label[:87] + "..."
    ax.set_title(
        f"{short_label}\ncost: {cost:.2f}  |  nodes: {len(unique_nodes)}",
        fontsize=13, fontweight="bold", pad=18,
    )

    legend_handles = [
        mpatches.Patch(color="#2563EB", label="Current route edges (unchanged)"),
        mpatches.Patch(color="#DC2626", label="New edges added vs previous route"),
        mpatches.Patch(color="#EAB308", label="Edges removed vs previous route"),
        mpatches.Patch(color="#DDDDDD", label="All possible edges"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10, framealpha=0.9)

    ax.axis("off")
    plt.tight_layout()

    dest = output_path.with_suffix(f".{fmt}")
    fig.savefig(str(dest), format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------------

def render_routes_from_log(
    log_path: str | Path,
    output_dir: str | Path,
    tsp_path: Optional[str | Path] = None,
    fmt: str = "png",
    only_improvements: bool = False,
    step: int = 1,
) -> List[Path]:
    entries = parse_routes_from_log(log_path)
    if not entries:
        print("No route entries found in the log.")
        return []

    if only_improvements:
        entries = filter_improvements(entries)

    if step > 1:
        entries = entries[::step]

    graph: Optional[Graph] = None
    positions: Optional[dict] = None
    if tsp_path is not None:
        graph = Graph.from_tsplib(str(tsp_path))
        positions = _node_positions(graph)
        # only use tsp positions if explicitly requested via --tsp flag
        # for circular layout we ignore them (set to None)
        positions = None  # force circular layout always

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    pad = len(str(len(entries)))
    prev_route: Optional[List[int]] = None

    for idx, entry in enumerate(entries, start=1):
        filename = f"route_{idx:0{pad}d}_cost_{entry['cost']:.0f}"
        out_path = output_dir / filename
        img = render_route(
            route=entry["route"],
            cost=entry["cost"],
            label=entry["label"],
            output_path=out_path,
            positions=positions,
            prev_route=prev_route,
            fmt=fmt,
        )
        rendered.append(img)
        prev_route = entry["route"]
        print(f"  [{idx}/{len(entries)}] {img.name}  (cost {entry['cost']:.2f})")

    print(f"\n{len(rendered)} images saved to {output_dir}/")
    return rendered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render TSP route snapshots from a VNS log file.",
    )
    parser.add_argument("--log", required=True, help="Path to the VNS .log file.")
    parser.add_argument("--tsp", default=None, help="Path to the TSPLIB .tsp file.")
    parser.add_argument("--out", default=None, help="Output directory for images.")
    parser.add_argument("--fmt", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--only-improvements", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    out_dir = args.out or str(Path(args.log).parent / "route_plots")
    render_routes_from_log(
        log_path=args.log,
        output_dir=out_dir,
        tsp_path=args.tsp,
        fmt=args.fmt,
        only_improvements=args.only_improvements,
        step=args.step,
    )


if __name__ == "__main__":
    main()