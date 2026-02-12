import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

class WeightCalculator:
    def __init__(self, processed_paths: List[Dict], window_sizes: List[int], 
                 weights: np.ndarray):
        self.processed_paths = processed_paths
        self.window_sizes = window_sizes
        self.weights = weights
        self.co_occurrence_matrices = {
            window_size: defaultdict(lambda: defaultdict(int)) 
            for window_size in window_sizes
        }
        self.total_co_occurrences = {window_size: 0 for window_size in window_sizes}
        self.total_matrix = {}
        self.all_edges = []
        self.all_edges_data = {}

    def calculate_co_occurrences(self):
        """Calculate co-occurrence matrices."""
        for item in tqdm(self.processed_paths, desc="Processing paths"):
            query_indices = [vector[1] for vector in item['query_vectors']]
            result_indices = [vector[1] for vector in item['result_vectors']]
            window_size = item['query_vectors'][0][0]

            for query_index in query_indices:
                for result_index in result_indices:
                    self.co_occurrence_matrices[window_size][query_index][result_index] += 1
                    self.total_co_occurrences[window_size] += 1

    def calculate_total_weights(self):
        """Calculate total weight matrix."""
        for window_size, co_matrix in tqdm(self.co_occurrence_matrices.items(), 
                                         desc="Calculating total weight matrix"):
            weight = self.weights[self.window_sizes.index(window_size)]
            
            for query_index, result_dict in co_matrix.items():
                if query_index not in self.total_matrix:
                    self.total_matrix[query_index] = {}
                    
                for result_index, count in result_dict.items():
                    weight1 = count * weight
                    weight2 = co_matrix[result_index].get(query_index, 0) * weight
                    
                    if result_index not in self.total_matrix[query_index]:
                        self.total_matrix[query_index][result_index] = 0
                    self.total_matrix[query_index][result_index] += weight1
                    
                    if result_index not in self.total_matrix:
                        self.total_matrix[result_index] = {}
                    if query_index not in self.total_matrix[result_index]:
                        self.total_matrix[result_index][query_index] = 0
                    self.total_matrix[result_index][query_index] += weight2

    def generate_edge_data(self):
        """Generate edge data."""
        for query_index in tqdm(self.total_matrix, desc="Generating edge data"):
            for result_index in self.total_matrix[query_index]:
                if query_index < result_index:
                    total_weight1 = self.total_matrix[query_index][result_index]
                    total_weight2 = self.total_matrix[result_index][query_index]
                    
                    self.all_edges.append((query_index, result_index, 
                                         total_weight1, total_weight2))
                    
                    edge_info = self._create_edge_info(query_index, result_index, 
                                                     total_weight1, total_weight2)
                    self.all_edges_data[(query_index, result_index)] = edge_info

    def _create_edge_info(self, query_index, result_index, weight1, weight2):
        """Create edge information dictionary."""
        return {
            'total_weight1': weight1,
            'total_weight2': weight2,
            'window_data': [
                {
                    'window_size': window_size,
                    'count1': self.co_occurrence_matrices[window_size][query_index][result_index],
                    'count2': self.co_occurrence_matrices[window_size][result_index][query_index],
                    'weight': self.weights[self.window_sizes.index(window_size)],
                    'total_count': self.total_co_occurrences[window_size]
                }
                for window_size in self.window_sizes
            ]
        }

    def calculate_weights(self):
        """Execute complete weight calculation process."""
        self.calculate_co_occurrences()
        self.calculate_total_weights()
        self.generate_edge_data()
        return self.all_edges, self.all_edges_data

    def print_statistics(self):
        """Print calculation statistics."""
        print(f"Total number of edges: {len(self.all_edges)}")
        
        print("\nTotal co-occurrences for each window size:")
        for window_size, total_count in self.total_co_occurrences.items():
            print(f"Window size: {window_size}, Total co-occurrences: {total_count}")
        
        if self.all_edges_data:
            max_edge = max(self.all_edges_data.items(), 
                         key=lambda x: max(x[1]['total_weight1'], x[1]['total_weight2']))
            min_edge = min(self.all_edges_data.items(), 
                         key=lambda x: min(x[1]['total_weight1'], x[1]['total_weight2']))
            
            print("\nEdge with maximum weight:")
            self._print_edge_info(max_edge)
            
            print("\nEdge with minimum weight:")
            self._print_edge_info(min_edge)

    def _print_edge_info(self, edge):
        """Print detailed edge information."""
        print(f"Edge: {edge[0]}")
        print(f"  Total weight1: {edge[1]['total_weight1']:.6f}")
        print(f"  Total weight2: {edge[1]['total_weight2']:.6f}")
        print("  Calculation data:")
        for window_data in edge[1]['window_data']:
            print(f"    Window size: {window_data['window_size']}, "
                  f"Count1: {window_data['count1']}, "
                  f"Count2: {window_data['count2']}, "
                  f"Weight: {window_data['weight']}, "
                  f"Total co-occurrences: {window_data['total_count']}")