import networkx as nx

class GraphProcessor:
    @staticmethod
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

    @staticmethod
    def predict_root_node(cluster_graph):
        node_scores = {}
        for node in cluster_graph.nodes():
            in_degree = cluster_graph.in_degree(node)
            out_degree = cluster_graph.out_degree(node)
            score = out_degree / (in_degree + 1)
            node_scores[node] = score
        return max(node_scores, key=node_scores.get)