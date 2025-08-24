import networkx as nx
import os
import matplotlib.pyplot as plt
import math
import networkx as nx

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

    @staticmethod
    def from_tsplib(filename):
        nodes = []
        coords = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            start = False
            for line in lines:
                if line.strip() == "NODE_COORD_SECTION":
                    start = True
                    continue
                if start:
                    if line.strip() == "EOF" or not line.strip():
                        break
                    parts = line.strip().split()
                    node_id = int(parts[0]) - 1  # zero-based index
                    x, y = float(parts[1]), float(parts[2])
                    nodes.append(node_id)
                    coords.append((x, y))
        n = len(nodes)
        edges = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                edges[(i, j)] = dist
                edges[(j, i)] = dist
        return Graph(nodes, edges)

    def _build_adj_matrix(self):
        matrix = [[float('inf')] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            matrix[i][i] = 0
        for (i, j), dist in self.edges.items():
            matrix[i][j] = dist
            matrix[j][i] = dist  # Assuming undirected graph
        return matrix
    

    def build_distance_dict(self):
        dist_dict = {}
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                dist_dict[(i, j)] = self.adj_matrix[i][j]
        return dist_dict

    def add_edge(self, i, j, dist):
        self.edges[(i, j)] = dist
        self.edges[(j, i)] = dist
        self.adj_matrix[i][j] = dist
        self.adj_matrix[j][i] = dist

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
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        edge_labels = nx.get_edge_attributes(G, 'weight')
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