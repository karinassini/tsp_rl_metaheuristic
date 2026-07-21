import networkx as nx
import os
import matplotlib.pyplot as plt
import math
import numpy as np


class Graph:
    """
    Represents a graph for the TSP problem, where nodes are cities and edges are distances.
    """

    def __init__(self, nodes, edges=None, coords=None):
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.edges = edges if edges is not None else {}
        self.coords = coords if coords is not None else None
        self.adj_matrix = self._build_adj_matrix()

    def from_tsplib(filepath):

        with open(filepath, "r") as f:
            lines = f.readlines()

        dimension = None
        matrix_lines = []
        node_coords = []
        in_matrix = False
        in_coords = False
        edge_weight_format = "FULL_MATRIX"
        edge_weight_type = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            upper_line = line.upper()
            if upper_line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
                continue
            if upper_line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip().upper()
                continue
            if upper_line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = line.split(":")[1].strip().upper()
                continue
            if upper_line.startswith("EDGE_WEIGHT_SECTION"):
                in_matrix = True
                in_coords = False
                continue
            if upper_line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                in_matrix = False
                continue
            if upper_line.startswith("EOF"):
                break

            if in_matrix:
                if upper_line in {
                    "DISPLAY_DATA_SECTION",
                    "NODE_COORD_SECTION",
                    "DEPOT_SECTION",
                    "DEMAND_SECTION",
                    "TOUR_SECTION",
                    "EDGE_DATA_SECTION",
                }:
                    in_matrix = False
                    continue
                matrix_lines.extend(line.split())
            elif in_coords:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0]) - 1
                    x, y = float(parts[1]), float(parts[2])
                    node_coords.append((node_id, x, y))

        if dimension is None:
            raise ValueError("DIMENSION not found in TSPLIB file.")

        if edge_weight_type is None:
            if matrix_lines:
                edge_weight_type = "EXPLICIT"
            elif node_coords:
                edge_weight_type = "EUC_2D"
            else:
                raise ValueError("Unable to infer EDGE_WEIGHT_TYPE for TSPLIB file.")

        nodes = list(range(dimension))
        edges = {}
        coord_list = None

        if edge_weight_type == "EXPLICIT":
            weights = np.array(list(map(float, matrix_lines)))
            matrix = np.zeros((dimension, dimension), dtype=float)
            fmt = edge_weight_format

            if fmt in {"FULL_MATRIX", "FUNCTION"}:
                if weights.size != dimension * dimension:
                    raise ValueError(
                        f"Expected {dimension * dimension} weights for FULL_MATRIX, got {weights.size}."
                    )
                matrix = weights.reshape((dimension, dimension))
            elif fmt in {"LOWER_DIAG_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "UPPER_ROW"}:
                idx = 0
                include_diag = "DIAG" in fmt
                is_lower = fmt.startswith("LOWER")
                for i in range(dimension):
                    if is_lower:
                        j_range = range(i + 1) if include_diag else range(i)
                    else:
                        j_range = (
                            range(i, dimension)
                            if include_diag
                            else range(i + 1, dimension)
                        )
                    for j in j_range:
                        if idx >= weights.size:
                            raise ValueError(
                                f"Not enough weights supplied for {fmt}; stopped at index {idx}."
                            )
                        value = weights[idx]
                        idx += 1
                        row, col = (i, j) if i <= j else (j, i)
                        matrix[row, col] = value
                if idx != weights.size:
                    excess = weights.size - idx
                    if excess > 0:
                        raise ValueError(
                            f"Received {excess} extra weights for {fmt}; input may be malformed."
                        )
            else:
                raise ValueError(
                    f"Unsupported EDGE_WEIGHT_FORMAT '{edge_weight_format}'."
                )

        elif edge_weight_type in {"EUC_2D", "CEIL_2D", "ATT"}:
            if len(node_coords) != dimension:
                raise ValueError(
                    f"Expected {dimension} node coordinates, got {len(node_coords)}."
                )
            coords = [None] * dimension
            for node_id, x, y in node_coords:
                if not (0 <= node_id < dimension):
                    raise ValueError(
                        f"Node id {node_id + 1} out of expected range 1..{dimension}."
                    )
                coords[node_id] = (x, y)
            if any(c is None for c in coords):
                missing = [i + 1 for i, c in enumerate(coords) if c is None]
                raise ValueError(f"Missing coordinates for nodes: {missing}")

            matrix = np.zeros((dimension, dimension), dtype=float)
            for i in range(dimension):
                xi, yi = coords[i]
                for j in range(i + 1, dimension):
                    xj, yj = coords[j]
                    dx = xi - xj
                    dy = yi - yj
                    dist = math.sqrt(dx * dx + dy * dy)

                    if edge_weight_type == "EUC_2D":
                        value = int(round(dist))
                    elif edge_weight_type == "CEIL_2D":
                        value = math.ceil(dist)
                    else:  # ATT
                        r = math.sqrt((dx * dx + dy * dy) / 10.0)
                        t = int(round(r))
                        if t < r:
                            t += 1
                        value = t
                    row, col = (i, j) if i <= j else (j, i)
                    matrix[row, col] = value
            coord_list = coords

        elif edge_weight_type == "GEO":
            if len(node_coords) != dimension:
                raise ValueError(
                    f"Expected {dimension} node coordinates, got {len(node_coords)}."
                )
            coords = [None] * dimension
            for node_id, x, y in node_coords:
                if not (0 <= node_id < dimension):
                    raise ValueError(
                        f"Node id {node_id + 1} out of expected range 1..{dimension}."
                    )
                coords[node_id] = (x, y)
            if any(c is None for c in coords):
                missing = [i + 1 for i, c in enumerate(coords) if c is None]
                raise ValueError(f"Missing coordinates for nodes: {missing}")

            def _geo_dist(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> int:
                """TSPLIB GEO distance (great-circle, RRR=6378.388 km)."""
                RRR = 6378.388

                def to_rad(deg_float):
                    deg = int(deg_float)
                    minute = deg_float - deg
                    return math.pi * (deg + 5.0 * minute / 3.0) / 180.0

                lat1 = to_rad(lat1_deg)
                lon1 = to_rad(lon1_deg)
                lat2 = to_rad(lat2_deg)
                lon2 = to_rad(lon2_deg)
                q1 = math.cos(lon1 - lon2)
                q2 = math.cos(lat1 - lat2)
                q3 = math.cos(lat1 + lat2)
                return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

            matrix = np.zeros((dimension, dimension), dtype=float)
            for i in range(dimension):
                xi, yi = coords[i]
                for j in range(i + 1, dimension):
                    xj, yj = coords[j]
                    matrix[i, j] = _geo_dist(xi, yi, xj, yj)
            coord_list = coords

        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE '{edge_weight_type}'.")

        for i in range(dimension):
            for j in range(i + 1, dimension):
                edges[(i, j)] = matrix[i][j]
        return Graph(nodes, edges, coord_list)

    def _build_adj_matrix(self):
        matrix = [[float("inf")] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            matrix[i][i] = 0
        for (i, j), dist in self.edges.items():
            row, col = (i, j) if i <= j else (j, i)
            matrix[row][col] = dist
        return matrix

    def build_adj_matrix_full(self):
        matrix = [[float("inf")] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            matrix[i][i] = 0
        for (i, j), dist in self.edges.items():
            row, col = (i, j) if i <= j else (j, i)
            matrix[row][col] = dist
            matrix[col][row] = dist
        return matrix

    def add_edge(self, i, j, dist):
        row, col = (i, j) if i <= j else (j, i)
        self.edges[(row, col)] = dist
        self.adj_matrix[row][col] = dist

    def get_distance_matrix(self):
        return self.adj_matrix

    def __repr__(self):
        return f"Graph(nodes={self.nodes}, edges={len(self.edges)})"

    def visualize(self, save_dir=None):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        added = set()
        for (i, j), dist in self.edges.items():
            if (j, i) not in added:
                G.add_edge(i, j, weight=round(dist, 2))
                added.add((i, j))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Graph Visualization")
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, "graph.png")
            plt.savefig(file_path)
            print(f"Plot saved to {file_path}")
        else:
            plt.show()
        plt.clf()