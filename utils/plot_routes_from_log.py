"""Parse VNS log files and render each route snapshot as an image using NetworkX."""

from __future__ import annotations

""" Example usage:
python -m utils.plot_routes_from_log \
--log "outputs/solutions/burma14/VNS_Solver/rcl/20260502_220433/logs/vns_steps_20260502_220433_rcl_96688_4566621488.log" \
--tsp "instances/tsplib/burma14.tsp" \
--out "outputs/plots/burma14_rcl_20260502_220433" \
--only-improvements
"""

import argparse
import ast
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


def _grid_layout(nodes: List[int]) -> dict[int, Tuple[float, float]]:
    """Arrange nodes in a rectangle grid layout."""
    import math
    n = len(nodes)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pos: dict[int, Tuple[float, float]] = {}
    for i, nid in enumerate(sorted(nodes)):
        row = i // cols
        col = i % cols
        # y inverted so row 0 is at top
        pos[nid] = (float(col), float(rows - 1 - row))
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
    figsize: Tuple[float, float] = (16, 13),
    dpi: int = 150,
) -> Path:
    """Render a single TSP route using NetworkX.

    Edge colouring:
    - Light gray  : all possible edges not in current route or changed edges
    - Blue        : edges in current route that are unchanged from previous
    - Red         : edges that are NEW in current route (not in previous)
    - Yellow      : edges that were in previous route but REMOVED in current
    """
    output_path = Path(output_path)

    # Unique nodes (strip repeated closing node if present)
    seen: set[int] = set()
    unique_nodes: list[int] = []
    for nid in route:
        if nid not in seen:
            seen.add(nid)
            unique_nodes.append(nid)

    # Compute edge sets
    current_edges: set[Tuple[int, int]] = _edge_set(route)
    prev_edges: set[Tuple[int, int]] = _edge_set(prev_route) if prev_route is not None else set()

    # added = in current but NOT in previous
    added_edges: set[Tuple[int, int]] = current_edges - prev_edges
    # removed = in previous but NOT in current
    removed_edges: set[Tuple[int, int]] = prev_edges - current_edges
    # unchanged = in both current and previous
    unchanged_edges: set[Tuple[int, int]] = current_edges & prev_edges

    # If no previous route, all current edges are "unchanged" (draw blue, no red/yellow)
    if prev_route is None:
        unchanged_edges = current_edges
        added_edges = set()
        removed_edges = set()

    # Background edges: all pairs NOT involved in current route or removed edges
    active_display = current_edges | removed_edges
    all_bg_edges = [
        (u, v) for u, v in combinations(sorted(unique_nodes), 2)
        if (min(u, v), max(u, v)) not in active_display
    ]

    # Node positions
    if positions is not None:
        pos = {n: positions[n] for n in unique_nodes if n in positions}
        missing = [n for n in unique_nodes if n not in pos]
        if missing:
            extra = _grid_layout(missing)
            pos.update(extra)
    else:
        pos = _grid_layout(unique_nodes)

    # Build graph (only for node drawing)
    G = nx.Graph()
    G.add_nodes_from(unique_nodes)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def _draw_edges(edges, color, alpha, linewidth, linestyle="solid", zorder=2):
        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")

    # 1. All possible background edges (very light gray)
    _draw_edges(all_bg_edges, "#CCCCCC", alpha=0.25, linewidth=0.6, zorder=1)

    # 2. Removed edges (yellow) — were in prev route, not in current
    _draw_edges(removed_edges, "#EAB308", alpha=1.0, linewidth=3.0,
                linestyle="dashed", zorder=3)

    # 3. Unchanged current route edges (blue)
    _draw_edges(unchanged_edges, "#2563EB", alpha=0.9, linewidth=2.5, zorder=4)

    # 4. Added edges (red) — new in current route vs previous
    _draw_edges(added_edges, "#DC2626", alpha=1.0, linewidth=3.0, zorder=5)

    # 5. Nodes — large circles, white fill with pink border
    node_colors = ["#DC2626" if nid == unique_nodes[0] else "#FFFFFF"
                   for nid in unique_nodes]
    border_colors = ["#7F1D1D" if nid == unique_nodes[0] else "#EC4899"
                     for nid in unique_nodes]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=unique_nodes,
        node_color=node_colors,
        node_size=1800,
        edgecolors=border_colors,
        linewidths=3.0,
    )

    # 6. Node labels — large and bold
    for nid in unique_nodes:
        if nid not in pos:
            continue
        x, y = pos[nid]
        color = "white" if nid == unique_nodes[0] else "#1E293B"
        ax.text(x, y, str(nid), fontsize=13, ha="center", va="center",
                fontweight="bold", color=color, zorder=10)

    short_label = label if len(label) <= 90 else label[:87] + "..."
    ax.set_title(
        f"{short_label}\ncost: {cost:.2f}  |  nodes: {len(unique_nodes)}",
        fontsize=13, fontweight="bold", pad=18,
    )

    legend_handles = [
        mpatches.Patch(color="#2563EB", label="Current route edges (unchanged)"),
        mpatches.Patch(color="#DC2626", label="New edges added vs previous route"),
        mpatches.Patch(color="#EAB308", label="Edges removed vs previous route"),
        mpatches.Patch(color="#CCCCCC", label="All other possible edges"),
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
    parser.add_argument("--tsp", default=None, help="Path to the TSPLIB .tsp file (for geo coordinates).")
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