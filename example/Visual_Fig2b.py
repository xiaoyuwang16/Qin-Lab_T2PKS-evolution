import sys
import os
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import constants 

def load_embeddings(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {file_path}")
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
    return max(node_scores, key=node_scores.get)

def cluster_nodes(embeddings, classifications):

    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(embeddings)

    threshold = 0.3 * max(clustering.distances_)
    print(f"Clustering threshold: {threshold}")
    sub_clustering = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None)
    labels = sub_clustering.fit_predict(embeddings)

    unique_labels = np.unique(labels)
    print(f"Number of clusters: {len(unique_labels)}")

    sub_clusters = defaultdict(list)
    for i, (label, classification) in enumerate(zip(labels, classifications)):
        cluster_name = f"{classification}_{label}"
        sub_clusters[cluster_name].append(i)

    print("\nDetailed cluster information:")
    for cluster_name, cluster_nodes in sub_clusters.items():
        print(f"Cluster {cluster_name}:")
        print(f"  Number of nodes: {len(cluster_nodes)}")
        print(f"  Nodes: {cluster_nodes}")
        print()

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

def visualize_aggregated_graph(cluster_graph, color_scheme, cluster_results, root_node, output_dir=None, arrow_style='->', arrow_size=15):

    plt.figure(figsize=(8, 6)) 
    
    pos = nx.spring_layout(cluster_graph, k=0.99, iterations=100)


    node_colors = [color_scheme.get(node.split('_')[0], '#808080') for node in cluster_graph.nodes()]

    nx.draw_networkx_nodes(cluster_graph, pos, node_size=600, node_color=node_colors, alpha=1)


    if root_node:
        nx.draw_networkx_nodes(cluster_graph, pos, nodelist=[root_node], node_size=600, node_color='none', node_shape='o', linewidths=3, edgecolors='red')

    edge_weights = [d['weight'] for (u, v, d) in cluster_graph.edges(data=True)]
    log_weights = np.log1p(edge_weights)
    max_log_weight = np.max(log_weights) if len(log_weights) > 0 else 1
    normalized_weights = log_weights / max_log_weight if max_log_weight != 0 else log_weights

    min_width = 2
    max_width = 5

    for (u, v, d), weight in zip(cluster_graph.edges(data=True), normalized_weights):
        nx.draw_networkx_edges(cluster_graph, pos,
                               edgelist=[(u, v)],
                               width=min_width + weight * (max_width - min_width),
                               edge_color='#D0D8E0',
                               arrows=True,
                               arrowsize=arrow_size,
                               arrowstyle=arrow_style,
                               connectionstyle='arc3,rad=0.2',
                               alpha=0.6)

    labels = {node: node.split('_')[-1] for node in cluster_graph.nodes()}
    nx.draw_networkx_labels(cluster_graph, pos, labels, font_size=20, font_weight='bold')

    plt.title("Aggregated Cluster Graph with Predicted Root Node", fontsize=20)
    plt.axis('off')


    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=group,
                                  markerfacecolor=color, markersize=10)
                       for group, color in color_scheme.items() if group != 'FAB']
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', label='Predicted Root',
                                      markerfacecolor='none', markersize=15, markeredgewidth=3))
    

    plt.legend(handles=legend_elements, 
               title="Group Legend",
               bbox_to_anchor=(1.05, 1), 
               loc='upper left', 
               borderaxespad=0.)

    plt.tight_layout()


    if output_dir:
        save_path = output_dir / "aggregated_cluster_graph.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph image saved to: {save_path}")

    plt.show()


if __name__ == "__main__":

    data_dir = constants.OUTPUT_DIR
    
    embeddings_file = data_dir / "normalized_pca_embeddings.npy"
    new_edge_weights_file = data_dir / "new_edge_weights_pca.pkl"


    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found at {embeddings_file}")
        sys.exit(1)
    
    print(f"Processing data from directory: {data_dir}")


    embeddings = load_embeddings(embeddings_file)
    print(f"Total number of embeddings: {len(embeddings)}")


    classifications = classify_nodes(len(embeddings))
    class_counts = defaultdict(int)
    for c in classifications:
        class_counts[c] += 1
    print("\nNode counts for each class:")
    for c, count in class_counts.items():
        print(f"{c}: {count}")

    clusters = cluster_nodes(embeddings, classifications)
    print("\nNodes in each cluster:")
    for cluster, nodes in clusters.items():
        print(f"Cluster {cluster}: {len(nodes)} nodes")

    print(f"Loading edge weights from: {new_edge_weights_file}")
    with open(new_edge_weights_file, "rb") as f:
        edge_weights = pickle.load(f)


    k = 15
    print(f"Building KNN graph with k={k}...")
    knn_graph = build_knn_graph(embeddings, k)
    G = nx.DiGraph(knn_graph)

    cluster_graph = aggregate_edges(G, clusters, edge_weights)
    cluster_graph, kept_edges = process_bidirectional_edges(cluster_graph)

    root_node = predict_root_node(cluster_graph)

    custom_color_scheme = {
        'FAB': '#FFA500',
        'KS': '#00DFA2',
        'CLF': '#FFB433',
    }

    visualize_aggregated_graph(
        cluster_graph, 
        custom_color_scheme, 
        clusters, 
        root_node,
        output_dir=data_dir 
    )

    print(f"\nNumber of nodes in cluster graph: {cluster_graph.number_of_nodes()}")
    print(f"Number of edges in cluster graph: {cluster_graph.number_of_edges()}")
    print(f"Predicted root node: {root_node}")
    print("\nAll edge weights:")
    for (u, v, d) in cluster_graph.edges(data=True):
        print(f"Edge from {u} to {v}: weight = {d['weight']}")
