
import numpy as np
from tqdm import tqdm
import pickle

class PathFinder:
    def __init__(self, thresholds, window_sizes):  
        """
        Initialize PathFinder
        
        Args:
            thresholds: numpy array of thresholds for different window sizes
            window_sizes: list of window sizes to use
        """
        self.window_sizes = window_sizes 
        self.thresholds = thresholds
        
        if len(self.thresholds) != len(self.window_sizes):
            raise ValueError(f"Expected {len(self.window_sizes)} thresholds, got {len(self.thresholds)}")

    @staticmethod
    def split_vectors(vectors, window_size, step_size):
        split_vectors = []
        for vector, index, similar_indices in vectors:
            for start_index in range(0, len(vector) - window_size + 1, step_size):
                split_vector = vector[start_index:start_index+window_size]
                split_index = (index[0], index[1], index[2] + start_index)
                split_similar_indices = [(si[0], si[1], si[2] + start_index) 
                                       for si in similar_indices]
                split_vectors.append((split_vector, split_index, split_similar_indices))
        return split_vectors

    def find_generation_path(self, search_results_dict):
        """
        generating paths
        
        Args:
            search_results_dict: A dictionary containing the search results, with the key being the window size
        
        Returns:
            list: path list
        """
        print(f"Window sizes: {self.window_sizes}")
        print(f"Thresholds shape: {self.thresholds.shape}")
        
        split_vectors_by_window = {}
        print("\nGenerating split vectors...")
        for i in range(1, len(self.window_sizes)):
            current_window_size = self.window_sizes[i]
            prev_window_size = self.window_sizes[i-1]
            
            if current_window_size not in search_results_dict:
                continue
                
            current_vectors = search_results_dict[current_window_size]
            print(f"\nProcessing window size {current_window_size}")
            print(f"Number of vectors: {len(current_vectors)}")
            
            split_vectors_current = []
            for entry in current_vectors:
                split_vectors_current.extend(self.split_vectors(
                    [entry], prev_window_size, 1))
            split_vectors_by_window[current_window_size] = split_vectors_current
            
            print(f"Generated {len(split_vectors_current)} split vectors")

        print("\nSearching for vector generation paths...")
        all_paths = []

        for i in range(len(self.window_sizes) - 1):
            current_window_size = self.window_sizes[i]
            next_window_size = self.window_sizes[i+1]
            current_threshold = self.thresholds[i]
            
            if current_window_size not in search_results_dict or \
               next_window_size not in split_vectors_by_window:
                continue
            
            current_vectors = search_results_dict[current_window_size]
            next_split_vectors = split_vectors_by_window[next_window_size]
            
            print(f"\nProcessing window sizes {current_window_size} -> {next_window_size}")
            print(f"Current threshold: {current_threshold}")
            print(f"Current vectors: {len(current_vectors)}")
            print(f"Next split vectors: {len(next_split_vectors)}")
            
            paths = [[] for _ in range(len(current_vectors))]

            for vector_index, (vector, index, similar_indices) in enumerate(
                tqdm(current_vectors, desc=f"Processing vectors")):
                
                query_info = (vector, index, similar_indices)
                
                for split_vector, next_index, next_similar_indices in next_split_vectors:
                    if index[1:] == next_index[1:]:
                        distance = np.linalg.norm(vector - split_vector)
                        if distance <= current_threshold:
                            paths[vector_index].append((query_info, 
                                                      (split_vector, next_index, next_similar_indices)))

            valid_paths = [p for p in paths if p]
            all_paths.extend(valid_paths)
            print(f"\nFound {len(valid_paths)} valid paths for window size {current_window_size}")

        return all_paths