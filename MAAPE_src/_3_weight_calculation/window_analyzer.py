import numpy as np
from typing import List, Dict, Tuple
from pprint import pprint

class WindowAnalyzer:
    def __init__(self, vectors: np.ndarray, window_sizes: List[int], stride: int = 1):
        self.vectors = vectors
        self.window_sizes = window_sizes
        self.stride = stride
        self.window_info = {}
        self.normalized_weights = None

    def generate_windows(self, vector: np.ndarray, window_size: int) -> Tuple[List, List]:
        """Generate sliding windows from a vector."""
        vector_length = len(vector)
        windows = []
        indices = []
        for i in range(0, vector_length - window_size + 1, self.stride):
            window = vector[i:i+window_size]
            windows.append(window)
            indices.append(i)
        return windows, indices

    def count_sliced_vectors(self) -> Dict[int, int]:
        """Count total sliced vectors for each window size."""
        sliced_vectors_counts = {}
        for window_size in self.window_sizes:
            total_sliced_vectors = 0
            for vector in self.vectors:
                sliced_vectors, _ = self.generate_windows(vector, window_size)
                total_sliced_vectors += len(sliced_vectors)
            sliced_vectors_counts[window_size] = total_sliced_vectors
        return sliced_vectors_counts

    def count_results_by_window_size(self, search_results: Dict) -> Dict[int, int]:
        """Count filtered vectors for each window size."""
        return {window_size: len(vectors) 
                for window_size, vectors in search_results.items()}

    def analyze_windows(self, search_results: Dict) -> Dict:
        """Analyze windows and calculate normalized weights."""
        sliced_vectors_counts = self.count_sliced_vectors()
        result_counts = self.count_results_by_window_size(search_results)

        self.window_info = {}
        for window_size in self.window_sizes:
            self.window_info[window_size] = {
                'filtered_vectors': result_counts.get(window_size, 0),
                'total_sliced_vectors': sliced_vectors_counts[window_size]
            }

        probabilities = {}
        weight_coefficients = {}
        normalized_coefficients = {}

        for window_size, info in self.window_info.items():
            filtered_vectors = info['filtered_vectors']
            total_sliced_vectors = info['total_sliced_vectors']

            probability = (filtered_vectors / total_sliced_vectors 
                         if total_sliced_vectors != 0 else 0.0)
            probabilities[window_size] = probability

            weight_coefficient = 1 / probability if probability != 0 else 0.0
            weight_coefficients[window_size] = weight_coefficient

        total_weight = sum(weight_coefficients.values())
        for window_size, coefficient in weight_coefficients.items():
            normalized_coefficients[window_size] = (
                coefficient / total_weight if total_weight != 0 else 0.0)

        self.normalized_weights = [normalized_coefficients[window_size] 
                                 for window_size in sorted(self.window_info.keys())]

        return self.window_info

    def print_statistics(self):
        """Print analysis results."""
        print("Window size information:")
        pprint(self.window_info, width=100)
        print("\nNormalized Weight Coefficients:")
        print(self.normalized_weights)

    def get_normalized_weights(self) -> List[float]:
        """Return normalized weights list."""
        return self.normalized_weights