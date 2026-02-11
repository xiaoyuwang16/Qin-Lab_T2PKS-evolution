import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict

class GraphBuilder:
    @staticmethod
    def build_knn_graph(embeddings, k):
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        adj_list = {i: set(indices[i][1:]) for i in range(len(embeddings))}
        return adj_list

    @staticmethod
    def cluster_nodes(embeddings, sequence_orders):
        order_groups = defaultdict(list)
        for i, order in sequence_orders.items():
            order_groups[order].append(i)

        sub_clusters = {}
        for order, nodes in order_groups.items():
            cluster_name = f"{order}_0"
            sub_clusters[cluster_name] = nodes

        unknown_nodes = []
        for i in range(len(embeddings)):
            if i not in sequence_orders:
                unknown_nodes.append(i)
        if unknown_nodes:
            sub_clusters['Unknown_0'] = unknown_nodes

        return dict(sub_clusters)

    @staticmethod
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