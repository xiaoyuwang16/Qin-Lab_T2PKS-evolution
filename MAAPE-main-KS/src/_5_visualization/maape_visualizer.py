import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Set
from sklearn.neighbors import NearestNeighbors

class MAAPEVisualizer:
    def __init__(self, color_scheme: Dict[str, str]):
        """
        Initialize MAAPE visualizer.
        
        Args:
            color_scheme: Mapping of orders to colors
        """
        self.color_scheme = color_scheme
        self.min_node_size = 200
        self.max_node_size = 1000
        self.cmap = plt.cm.viridis

    def build_knn_graph(self, embeddings: np.ndarray, k: int) -> Dict[int, Set[int]]:
        """Build KNN graph from embeddings."""
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        return {i: set(indices[i][1:]) for i in range(len(embeddings))}

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
        return sequence_orders

    def _rank_scale_weights(self, weights: Dict[Tuple[int, int], float], 
                          min_display: float = 0.1, max_display: float = 1.0) -> Dict[Tuple[int, int], float]:
        """Scale weights using rank-based normalization."""
        values = np.array(list(weights.values()))
        ranks = np.argsort(np.argsort(values))
        scaled_values = min_display + (max_display - min_display) * ranks / (len(ranks) - 1)
        return dict(zip(weights.keys(), scaled_values))

    def _setup_figure(self) -> Tuple:
        """Setup figure and axes for visualization."""
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(1, 20)
        ax_main = fig.add_subplot(gs[0, :16])
        ax_legend = fig.add_subplot(gs[0, 16:])
        ax_legend.axis('off')
        return fig, ax_main

    def _create_graph(self, adj_list: Dict[int, Set[int]], sequence_orders: Dict[int, str]) -> Tuple:
        """Create and setup networkx graph."""
        G = nx.DiGraph(adj_list)
        pos = nx.spring_layout(G, k=0.2, seed=42)
        
        node_colors = [self.color_scheme.get(sequence_orders.get(node, 'Unknown'), '#CCCCCC') 
                      for node in G.nodes()]
        node_sizes = [max(self.min_node_size, 
                         min(self.max_node_size, 15 * (1 + 0.5 * G.degree(node)))) 
                     for node in G.nodes()]
        
        return G, pos, node_colors, node_sizes

    def _draw_edges(self, G: nx.DiGraph, pos: Dict, valid_edges: List[Tuple[int, int]], 
                   edge_colors: List[float], ax_main: plt.Axes):
        """Draw network edges with colors based on weights."""
        norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        for edge, color in zip(valid_edges, edge_colors):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=0.5, arrows=True, 
                                 arrowsize=15, alpha=0.7, 
                                 edge_color=[self.cmap(norm(color))], 
                                 edge_cmap=self.cmap, ax=ax_main)
        return norm

    def _add_colorbar(self, fig: plt.Figure, norm: mcolors.Normalize):
        """Add colorbar to figure."""
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.3])
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Normalized Edge Weight Difference\n(Absolute Value)',
                      fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)

    def _add_legend(self, ax_main: plt.Axes):
        """Add legend to main axes."""
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, label=group_name, 
                                    markersize=15)
                         for group_name, color in self.color_scheme.items()]
        
        ax_main.legend(handles=legend_elements,
                      title='Group Legend',
                      title_fontsize=14,
                      loc='center left',
                      bbox_to_anchor=(1.02, 0.5),
                      fontsize=14)


    def visualize(self, adj_list: Dict[int, Set[int]], sequence_orders: Dict[int, str],
                 directed_edges: List[Tuple[int, int]], edge_weights: Dict[Tuple[int, int], float]):
        """Main visualization method."""

        fig, ax_main = self._setup_figure()
        G, pos, node_colors, node_sizes = self._create_graph(adj_list, sequence_orders)
        

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color=node_colors, alpha=0.7, ax=ax_main)
        

        valid_edges = [(u, v) for u, v in directed_edges if u in G.nodes and v in G.nodes]
        valid_edge_weights = {edge: edge_weights[edge] for edge in valid_edges}
        edge_weights_dict = self._rank_scale_weights(valid_edge_weights)
        edge_colors = [edge_weights_dict[edge] for edge in valid_edges]

        norm = self._draw_edges(G, pos, valid_edges, edge_colors, ax_main)
        self._add_colorbar(fig, norm)
        self._add_legend(ax_main)
        
        ax_main.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        

        
        plt.show()

