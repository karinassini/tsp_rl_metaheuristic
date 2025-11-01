import networkx as nx
import os
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np


class Graph:
    """
    Represents a graph for the TSP problem, where nodes are cities and edges are distances.
    """

    def __init__(self, nodes, edges=None):
        """
        Initialize the graph.
        :param nodes: List of node identifiers (e.g., city names or indices).
        :param edges: Optional dictionary {(node1, node2): distance}.
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.edges = edges if edges is not None else {}
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
                        # The logic below builds per-row index ranges over the flattened weights so each
                        # iteration can read exactly the entries corresponding to that row of the triangular matrix.
                    for j in j_range:
                        if idx >= weights.size:
                            raise ValueError(
                                f"Not enough weights supplied for {fmt}; stopped at index {idx}."
                            )
                        value = weights[idx]
                        idx += 1
                        matrix[i, j] = value
                        # matrix[j, i] = value # Symmetric
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

        elif edge_weight_type in {"EUC_2D", "CEIL_2D"}:
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
                    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if edge_weight_type == "EUC_2D":
                        value = int(round(dist))
                    else:  # CEIL_2D
                        value = math.ceil(dist)
                    matrix[i, j] = value
                    # matrix[j, i] = value
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE '{edge_weight_type}'.")

        for i in range(dimension):
            for j in range(i + 1, dimension):
                edges[(i, j)] = matrix[i][j]
        return Graph(nodes, edges)

    def _build_adj_matrix(self):
        matrix = [[float("inf")] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            matrix[i][i] = 0
        for (i, j), dist in self.edges.items():
            matrix[i][j] = dist
            # matrix[j][i] = dist  # Assuming undirected graph
        return matrix

    def add_edge(self, i, j, dist):
        self.edges[(i, j)] = dist
        self.edges[(j, i)] = dist
        self.adj_matrix[i][j] = dist
        # self.adj_matrix[j][i] = dist

    def get_distance_matrix(self):
        return self.adj_matrix

    def __repr__(self):
        return f"Graph(nodes={self.nodes}, edges={len(self.edges)})"

    def visualize(self, save_dir=None):
        """
        Visualize the graph. If save_dir is provided, the plot will be saved in that directory
        instead of being displayed.
        """
        # Create a NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        # Add edges only once (for i < j)
        added = set()
        for (i, j), dist in self.edges.items():
            if (j, i) not in added:
                G.add_edge(i, j, weight=round(dist, 2))
                added.add((i, j))

        # Compute layout
        pos = nx.spring_layout(G)
        # Draw nodes and edges with labels
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
