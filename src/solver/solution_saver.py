import json
import os
import networkx as nx

class SolutionSaver:
    """
    Class to save TSP solutions (route and total distance) to a file.
    """
    def __init__(self, save_dir='solutions'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, route, total_distance, filename='solution.json'):
        data = {
            'route': route,
            'total_distance': total_distance
        }
        file_path = f"{self.save_dir}/{filename}"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Solution saved to {file_path}")

    def save_txt(self, route, total_distance, filename='solution.txt'):
        file_path = f"{self.save_dir}/{filename}"
        with open(file_path, 'w') as f:
            f.write(f"Route: {route}\n")
            f.write(f"Total distance: {total_distance}\n")
        print(f"Solution saved to {file_path}")
        
    def plot_solution_graph(self, route, adj_matrix, filename='solution_graph.png'):
            """
            Plots the TSP route as a graph using matplotlib and networkx. The graph is built from an adjacency matrix,
            and the TSP route is highlighted.

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
            
            plt.title("TSP Solution Route")
            plt.savefig(filename)
            plt.close()
            print(f"Solution graph saved to {filename}")