import numpy as np
import pickle
from collections import defaultdict

class Utils:
    @staticmethod
    def read_order_file(filename):
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

    @staticmethod
    def load_embeddings(file_path):
        return np.load(file_path)

    @staticmethod
    def load_edge_weights(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def print_statistics(embeddings, sequence_orders, cluster_graph):
        print(f"Total number of embeddings: {len(embeddings)}")
        
        order_counts = defaultdict(int)
        for order in sequence_orders.values():
            order_counts[order] += 1

        print("\nNode counts for each order:")
        for order, count in order_counts.items():
            print(f"{order}: {count}")

        print(f"\nNumber of nodes in cluster graph: {cluster_graph.number_of_nodes()}")
        print(f"Number of edges in cluster graph: {cluster_graph.number_of_edges()}")
        
        print("\nAll edge weights:")
        for (u, v, d) in cluster_graph.edges(data=True):
            print(f"Edge from {u} to {v}: weight = {d['weight']}")