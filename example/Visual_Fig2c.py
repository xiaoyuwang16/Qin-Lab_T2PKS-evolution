import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import constants 

def load_embeddings(file_path):
    return np.load(file_path)

def build_knn_graph(embeddings, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    adj_list = {i: set(indices[i][1:]) for i in range(len(embeddings))}
    return adj_list

def classify_nodes(num_nodes):
    classifications = []
    for node in range(num_nodes):
        if 0 <= node <= 166:
            classifications.append('KS')
        elif 167 <= node <= 333:
            classifications.append('CLF')
        else:
            classifications.append('KS')
    return classifications

def process_bidirectional_edges(cluster_graph):
    edges_to_remove = []
    edges_to_keep = []

    for edge in cluster_graph.edges(data=True):
        source, target, data = edge
        if cluster_graph.has_edge(target, source):
            weight_forward = data['weight']
            weight_backward = cluster_graph[target][source]['weight']

            diff = abs(weight_forward - weight_backward)
            max_weight = max(weight_forward, weight_backward)

            if diff > 0.5 * max_weight:
                if weight_forward > weight_backward:
                    edges_to_remove.append((target, source))
                    edges_to_keep.append((source, target, weight_forward))
                else:
                    edges_to_remove.append((source, target))
                    edges_to_keep.append((target, source, weight_backward))

    for edge in edges_to_remove:
        if cluster_graph.has_edge(*edge):
            cluster_graph.remove_edge(*edge)

    return cluster_graph, edges_to_keep

def predict_root_node(cluster_graph):
    node_scores = {}
    for node in cluster_graph.nodes():
        in_degree = cluster_graph.in_degree(node)
        out_degree = cluster_graph.out_degree(node)
        score = out_degree / (in_degree + 1)
        node_scores[node] = score
    if not node_scores:
        return None
    return max(node_scores, key=node_scores.get)

def assign_levels(G, root_node):
    levels = {node: float('inf') for node in G.nodes()}
    if root_node in levels:
        levels[root_node] = 0
        queue = [root_node]
        while queue:
            current = queue.pop(0)
            for neighbor in G.neighbors(current):
                if levels[neighbor] > levels[current] + 1:
                    levels[neighbor] = levels[current] + 1
                    queue.append(neighbor)
    return levels

def hierarchical_layout(G, levels, width=1, height=1):
    pos = {}
    nodes_by_level = defaultdict(list)
    for node, level in levels.items():
        nodes_by_level[level].append(node)

    valid_levels = [l for l in levels.values() if l != float('inf')]
    max_level = max(valid_levels) if valid_levels else 0
    
    for level, nodes in nodes_by_level.items():
        
        if level == float('inf'):
            y = 0 
        else:
            y = (max_level - level) / (max_level or 1) * height
            
        for i, node in enumerate(nodes):
            x = i / (len(nodes) - 1 or 1) * width
            pos[node] = (x, y)
    return pos

def visualize_aggregated_graph(cluster_graph, color_scheme, cluster_results, root_node, arrow_style='->', arrow_size=15):
    plt.figure(figsize=(6, 6))

    levels = assign_levels(cluster_graph, root_node)
    pos = hierarchical_layout(cluster_graph, levels)

    node_colors = [color_scheme.get(node.split('_')[0], 'gray') for node in cluster_graph.nodes()]

    nx.draw_networkx_nodes(cluster_graph, pos,
                         node_size=600,
                         node_color=node_colors,
                         alpha=1,
                         edgecolors='white',
                         linewidths=2)

    if root_node:
        nx.draw_networkx_nodes(cluster_graph, pos,
                             nodelist=[root_node],
                             node_size=450,
                             node_color='none',
                             node_shape='o',
                             linewidths=3,
                             edgecolors='red')

    edge_weights = [d['weight'] for (u, v, d) in cluster_graph.edges(data=True)]
    log_weights = np.log1p(edge_weights)
    max_log_weight = np.max(log_weights) if len(log_weights) > 0 else 1
    normalized_weights = log_weights / max_log_weight if max_log_weight != 0 else log_weights

    min_width = 1
    max_width = 5

    colors = ['#FFDD00', '#7FEE00', '#00B5FF'] 
    bright_colormap = LinearSegmentedColormap.from_list('bright_yellow_green_blue', colors)

    norm = mcolors.Normalize(vmin=0, vmax=1)

    for (u, v, d), weight in zip(cluster_graph.edges(data=True), normalized_weights):
        edge_color = bright_colormap(norm(weight))
        nx.draw_networkx_edges(cluster_graph, pos,
                               edgelist=[(u, v)],
                               width=min_width + weight * (max_width - min_width),
                               edge_color=[edge_color],
                               arrows=True,
                               arrowsize=arrow_size,
                               arrowstyle=arrow_style,
                               connectionstyle='arc3,rad=0',
                               alpha=0.8)

    labels = {node: node.split('_')[-1] for node in cluster_graph.nodes()}
    nx.draw_networkx_labels(cluster_graph, pos, labels, font_size=20, font_weight='bold')

    plt.title("Hierarchical Cluster Graph with Predicted Root Node", fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=group,
                  markerfacecolor=color, markersize=10,
                  markeredgecolor='white', markeredgewidth=1.5)
        for group, color in color_scheme.items() if group != 'FAB'
    ]

    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='red', label='Predicted Root',
                  markerfacecolor='none', markersize=15, markeredgewidth=3)
    )

    plt.legend(handles=legend_elements,
              loc='upper left',
              title="Group Legend",
              bbox_to_anchor=(1, 1),
              fontsize=20,
              title_fontsize=20)
    
    plt.show()

def cluster_nodes(embeddings, classifications):
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(embeddings)

    threshold = 0.3 * max(clustering.distances_)
    sub_clustering = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None)
    labels = sub_clustering.fit_predict(embeddings)

    sub_clusters = defaultdict(list)
    for i, (label, classification) in enumerate(zip(labels, classifications)):
        cluster_name = f"{classification}_{label}"
        sub_clusters[cluster_name].append(i)

    return dict(sub_clusters)

def aggregate_edges(G, clusters, edge_weights):
    cluster_graph = nx.DiGraph()

    node_to_cluster = {}
    for cluster, nodes in clusters.items():
        for node in nodes:
            node_to_cluster[node] = cluster

    for u, v in G.edges():
        cluster_u = node_to_cluster[u]
        cluster_v = node_to_cluster[v]

        if cluster_u != cluster_v:
            weight = edge_weights.get((u, v), 1)

            if cluster_graph.has_edge(cluster_u, cluster_v):
                cluster_graph[cluster_u][cluster_v]['weight'] += weight
            else:
                cluster_graph.add_edge(cluster_u, cluster_v, weight=weight)

    return cluster_graph

if __name__ == "__main__":
    data_dir = constants.OUTPUT_DIR
    
    embeddings_file = data_dir / "normalized_pca_embeddings.npy"
    new_edge_weights_file = data_dir / "new_edge_weights_pca.pkl"

    embeddings = load_embeddings(embeddings_file)

    classifications = classify_nodes(len(embeddings))

    clusters = cluster_nodes(embeddings, classifications)

    with open(new_edge_weights_file, "rb") as f:
        edge_weights = pickle.load(f)

    k = 15
    knn_graph = build_knn_graph(embeddings, k)

    G = nx.DiGraph(knn_graph)

    cluster_graph = aggregate_edges(G, clusters, edge_weights)

    cluster_graph, kept_edges = process_bidirectional_edges(cluster_graph)

    root_node = predict_root_node(cluster_graph)


    color_scheme = {
        'FAB': '#FFA500',  
        'KS': '#00DFA2',   
        'CLF': '#FFB433',  
    }

    visualize_aggregated_graph(cluster_graph, color_scheme, clusters, root_node)
