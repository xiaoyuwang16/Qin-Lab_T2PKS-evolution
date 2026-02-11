import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

class KNNGraphBuilder:
    def __init__(self, embeddings: np.ndarray, k: int, threshold: float):
        """
        Initialize KNN graph builder.
        
        Args:
            embeddings: Input embeddings matrix
            k: Maximum number of neighbors
            threshold: Distance threshold
        """
        self.embeddings = embeddings
        self.k = k
        self.threshold = threshold
        self.k_min = 5
        self.adj_list = None

    def build_graph(self) -> Dict[int, Set[int]]:
        """Build KNN graph using adaptive k values."""
        nbrs = NearestNeighbors(n_neighbors=self.k, metric='euclidean').fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        self.adj_list = {}
        for i in range(len(self.embeddings)):
            adaptive_k = self.k_min
            for j in range(self.k_min, self.k):
                if distances[i][j] > self.threshold:
                    break
                adaptive_k = j + 1
            self.adj_list[i] = set(indices[i][1:adaptive_k])

        return self.adj_list

    def extract_edges(self) -> List[Tuple[int, int]]:
        """Extract edges from adjacency list."""
        if not self.adj_list:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        G = nx.Graph(self.adj_list)
        return list(G.edges())

    def save_edges(self, output_file: str):
        """Save edges to file."""
        edges = self.extract_edges()
        with open(output_file, "w") as f:
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")

class GraphVisualizer:
    def __init__(self, color_scheme: Dict[str, str]):
        """
        Initialize graph visualizer.
        
        Args:
            color_scheme: Mapping of orders to colors
        """
        self.color_scheme = color_scheme

    def read_order_file(self, filename: str) -> Dict[int, str]:
        """Read sequence orders from file."""
        sequence_orders = {}
        with open(filename, 'r') as f:
            for line in f:
                try:
                    if 'Sequence' in line and 'Order:' in line:
                        seq_part = line.split('Sequence')[1].split(':')[0].strip()
                        seq_num = int(seq_part) - 1
                        order = line.split('Order:')[1].strip().rstrip(')')
                        sequence_orders[seq_num] = order
                except Exception as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
        return sequence_orders

    def visualize(self, adj_list: Dict[int, Set[int]], sequence_orders: Dict[int, str]):
        """Visualize KNN graph."""
        G = nx.Graph(adj_list)
        node_colors = [self.color_scheme.get(sequence_orders.get(node, 'Unknown'), '#CCCCCC') 
                      for node in G.nodes()]

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=order,
                                    markerfacecolor=color, markersize=16)
                         for order, color in self.color_scheme.items()]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.35, 1), fontsize=16)

        plt.axis('off')
        plt.tight_layout()
        plt.show()