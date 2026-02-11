import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def visualize_aggregated_graph(cluster_graph, color_scheme, root_node, arrow_style='->', arrow_size=15):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(cluster_graph, k=0.3, iterations=50)
        label_pos = {node: (coord[0], coord[1] - 0.05) for node, coord in pos.items()}
        node_colors = [color_scheme[node.split('_')[0]] for node in cluster_graph.nodes()]

        nx.draw_networkx_nodes(cluster_graph, pos, node_size=500, node_color=node_colors, alpha=1)

        if root_node:
            nx.draw_networkx_nodes(cluster_graph, pos, nodelist=[root_node], 
                                 node_size=500, node_color='none', 
                                 node_shape='o', linewidths=3, edgecolors='red')

        edge_weights = [d['weight'] for (u, v, d) in cluster_graph.edges(data=True)]
        log_weights = np.log1p(edge_weights)
        max_log_weight = np.max(log_weights) if len(log_weights) > 0 else 1
        normalized_weights = log_weights / max_log_weight if max_log_weight != 0 else log_weights

        min_width, max_width = 1, 5
        cmap = plt.cm.viridis

        for (u, v, d), weight in zip(cluster_graph.edges(data=True), normalized_weights):
            nx.draw_networkx_edges(cluster_graph, pos,
                                 edgelist=[(u, v)],
                                 width=min_width + weight * (max_width - min_width),
                                 edge_color=[cmap(weight)],
                                 arrows=True,
                                 arrowsize=arrow_size,
                                 arrowstyle=arrow_style,
                                 connectionstyle='arc3,rad=0.2',
                                 alpha=0.8)

        labels = {node: node.split('_')[0] for node in cluster_graph.nodes()}
        nx.draw_networkx_labels(cluster_graph, label_pos, labels, font_size=22, font_weight='bold')

        plt.title("Aggregated Cluster Graph with Predicted Root Node", fontsize=20, pad=20)
        plt.axis('off')
        plt.margins(0.2)
        plt.tight_layout()
        plt.show()