import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np

class WeightExtractor:
    def __init__(self, knn_edges_file: str, weights_file: str):
        """
        Initialize weight extractor.
        
        Args:
            knn_edges_file: Path to KNN edges file
            weights_file: Path to weights file
        """
        self.knn_edges_file = knn_edges_file
        self.weights_file = weights_file
        self.new_edge_weights = {}
        self.not_found_edges = []
        self.found_edges_with_weights = []

    def load_edges(self) -> List[Tuple[int, int]]:
        """Load edges from KNN graph file."""
        edges = []
        with open(self.knn_edges_file, "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                edges.append((node1, node2))
        return edges

    def load_selected_edges(self) -> Dict:
        """Load selected edges with weights."""
        with open(self.weights_file, "rb") as f:
            return pickle.load(f)

    def extract_weights(self) -> Dict[Tuple[int, int], float]:
        """Extract weights for KNN edges."""
        edges = self.load_edges()
        selected_edges = self.load_selected_edges()

        for edge in edges:
            node1, node2 = edge
            edge_key = (node1, node2)
            rev_edge_key = (node2, node1)

            if edge_key in selected_edges:
                weights = selected_edges[edge_key]
                total_weight = weights['total_weight1'] + weights['total_weight2']
                self.new_edge_weights[edge_key] = total_weight
                self.found_edges_with_weights.append((edge_key, total_weight))
            elif rev_edge_key in selected_edges:
                weights = selected_edges[rev_edge_key]
                total_weight = weights['total_weight1'] + weights['total_weight2']
                self.new_edge_weights[edge_key] = total_weight
                self.found_edges_with_weights.append((edge_key, total_weight))
            else:
                self.new_edge_weights[edge_key] = 0
                self.not_found_edges.append(edge_key)

        return self.new_edge_weights

    def save_weights(self, output_file: str):
        """Save extracted weights to file."""
        with open(output_file, "wb") as f:
            pickle.dump(self.new_edge_weights, f)

    def print_statistics(self):
        """Print statistics about extracted weights."""
        print("\nNew edge weights (first 5 lines):")
        for edge, weight in list(self.new_edge_weights.items())[:5]:
            print(f"({edge[0]}, {edge[1]}): {weight}")

        non_zero_weights = {k: v for k, v in self.new_edge_weights.items() if v > 0}
        if non_zero_weights:
            max_weight_edge = max(non_zero_weights.items(), key=lambda x: x[1])
            min_weight_edge = min(non_zero_weights.items(), key=lambda x: x[1])

            print("\nEdge with the highest weight:")
            print(f"({max_weight_edge[0][0]}, {max_weight_edge[0][1]}): {max_weight_edge[1]}")
            print("Edge with the lowest non-zero weight:")
            print(f"({min_weight_edge[0][0]}, {min_weight_edge[0][1]}): {min_weight_edge[1]}")

    def visualize_weights(self):
        """Visualize edge weights distribution."""
        weights = list(self.new_edge_weights.values())
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(weights)), weights, s=10)
        plt.xlabel('Edge Index')
        plt.ylabel('Edge Weight')
        plt.title('Edge Weights Scatter Plot')
        plt.show()