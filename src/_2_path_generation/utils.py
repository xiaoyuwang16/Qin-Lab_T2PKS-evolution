import numpy as np
from typing import List, Tuple, Any

def split_vectors(vectors: List[Tuple], window_size: int, step_size: int) -> List[Tuple]:
    """Split vectors according to window size and step size."""
    split_vectors = []
    
    for vector, index, similar_indices in vectors:
        for start_index in range(0, len(vector) - window_size + 1, step_size):
            split_vector = vector[start_index:start_index+window_size]
            split_index = (index[0], index[1], index[2] + start_index)
            split_similar_indices = [(si[0], si[1], si[2] + start_index) 
                                   for si in similar_indices]
            split_vectors.append((split_vector, split_index, split_similar_indices))
    
    return split_vectors