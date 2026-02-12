import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from .utils import split_vectors
import pickle

class PathGenerator:
    def __init__(self, search_results: Dict, thresholds: np.ndarray):
        self.search_results = search_results
        self.thresholds = thresholds
        self.unique_vectors_by_window = {}
        self.split_vectors_by_window = {}
        
    def _prepare_vectors(self):
        """Prepare vectors for path generation."""
        for window_size, unique_vectors in self.search_results.items():
            self.unique_vectors_by_window[window_size] = unique_vectors
            
        window_sizes = sorted(self.unique_vectors_by_window.keys())
        for i in range(1, len(window_sizes)):
            current_window_size = window_sizes[i]
            prev_window_size = window_sizes[i-1]
            split_vectors_current = []
            
            for vector, index, similar_indices in self.unique_vectors_by_window[current_window_size]:
                split_vectors_current.extend(
                    split_vectors([(vector, index, similar_indices)], 
                                prev_window_size, 1))
            
            self.split_vectors_by_window[current_window_size] = split_vectors_current
    
    def generate_paths(self) -> List:
        """Generate paths from vectors."""
        self._prepare_vectors()
        print("Searching for vector generation path...")
        
        window_sizes = sorted(self.unique_vectors_by_window.keys())
        all_paths = []
        
        for i in tqdm(range(len(window_sizes) - 1), desc="Window Size"):
            current_window_size = window_sizes[i]
            next_window_size = window_sizes[i+1]
            current_vectors = self.unique_vectors_by_window[current_window_size]
            next_split_vectors = self.split_vectors_by_window[next_window_size]
            paths = [[] for _ in range(len(current_vectors))]
            threshold = self.thresholds[i]
            
            for vector_index, (vector, index, similar_indices) in enumerate(current_vectors):
                query_info = (vector, index, similar_indices)
                
                for split_vector, next_index, next_similar_indices in next_split_vectors:
                    if index[1:] == next_index[1:]:
                        distance = np.linalg.norm(vector - split_vector)
                        if distance <= threshold:
                            path_info = (split_vector, next_index, next_similar_indices)
                            paths[vector_index].append((query_info, path_info))
                
                if (vector_index + 1) % 1000 == 0:
                    progress = (vector_index + 1) / len(current_vectors)
                    print(f"\rProcessing Window Size {current_window_size}: "
                          f"[{int(progress * 50) * '='}>{(50 - int(progress * 50)) * ' '}] "
                          f"{int(progress * 100)}%", end="", flush=True)
            
            all_paths.extend(paths)
            print(f"\nCurrent Window Size: {current_window_size}")
            print(f"Next Window Size: {next_window_size}")
            print(f"Current Vectors: {len(current_vectors)}")
            print(f"Next Split Vectors: {len(next_split_vectors)}")
            print(f"Found Paths: {sum(len(path) > 0 for path in paths)}\n")
        
        return all_paths

class PathProcessor:
    def __init__(self, paths: List[Any]):
        self.paths = paths
        self.processed_paths = []
    
    def process(self) -> List[Dict]:
        """Process paths and extract relevant information."""
        for path in self.paths:
            if len(path) > 0:
                for item in path:
                    query_vectors = item[0][2]
                    result_vectors = set()
                    
                    for subitem in item[1:]:
                        result_vectors.update(tuple(vector) for vector in subitem[2])
                    
                    self.processed_paths.append({
                        'query_vectors': query_vectors,
                        'result_vectors': list(result_vectors)
                    })
        
        return self.processed_paths
    
    @staticmethod
    def save_paths(paths: List[Dict], output_path: str):
        """Save processed paths to file."""
        with open(output_path, 'wb') as f:
            pickle.dump(paths, f)
    
    @staticmethod
    def load_paths(input_path: str) -> List[Dict]:
        """Load paths from file."""
        with open(input_path, 'rb') as f:
            return pickle.load(f)
