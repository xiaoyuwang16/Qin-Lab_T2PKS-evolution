import numpy as np
import faiss
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class VectorSearcher:
    def __init__(self):
        pass

    @staticmethod
    def generate_windows(vector, window_size, stride):
        vector_length = len(vector)
        windows = []
        indices = []
        for i in range(0, vector_length - window_size + 1, stride):
            window = vector[i:i+window_size]
            windows.append(window)
            indices.append(i)
        return windows, indices

    @staticmethod
    def create_sliced_vectors(input_vectors, window_size, stride):
        sliced_vectors = []
        sliced_indices = []
        input_vector_indices = []

        for index, input_vector in enumerate(input_vectors):
            windows, start_indices = VectorSearcher.generate_windows(input_vector, window_size, stride)
            sliced_vectors.extend(windows)
            for start_index in start_indices:
                sliced_indices.append(start_index)
                input_vector_indices.append((index, start_index))

        return np.array(sliced_vectors), sliced_indices, input_vector_indices

    @staticmethod
    def find_similar_vectors(sliced_vectors, sliced_indices, input_vector_indices, threshold, window_size):
        index = faiss.IndexFlatL2(sliced_vectors.shape[1])
        index.add(sliced_vectors)

        vector_groups = []
        processed_vectors = set()

        for i in tqdm(range(len(sliced_vectors)), desc="Finding similar vectors"):
            query_vector = sliced_vectors[i]
            query_vector_hash = hash(query_vector.tobytes())

            if query_vector_hash in processed_vectors:
                continue

            D, I = index.search(np.array([query_vector]), k=100)
            similar_indices = I[0][D[0] <= threshold]
            similar_indices = similar_indices[similar_indices != i]

            if len(similar_indices) > 0:
                orig_idx, start_idx = input_vector_indices[i]
                sub_vector_idx = start_idx // 1 
                
                similar_vector_indices = []
                for sim_idx in similar_indices:
                    if sim_idx < len(input_vector_indices):
                        sim_orig_idx, sim_start_idx = input_vector_indices[sim_idx]
                        sim_sub_idx = sim_start_idx // 1  
                        similar_vector_indices.append((window_size, sim_orig_idx, sim_sub_idx))

                current_index = (window_size, orig_idx, sub_vector_idx)
                similar_vector_indices.insert(0, current_index)  

                vector_groups.append((
                    query_vector,  
                    current_index, 
                    similar_vector_indices  
                ))
                
                processed_vectors.add(query_vector_hash)

        return vector_groups

    @staticmethod
    def search_window_size(args):
        window_size, input_vectors, stride, threshold = args
        print(f"Searching with window size: {window_size}, threshold: {threshold}")

        sliced_vectors, sliced_indices, input_vector_indices = VectorSearcher.create_sliced_vectors(
            input_vectors, window_size, stride)
        vector_groups = VectorSearcher.find_similar_vectors(
            sliced_vectors, sliced_indices, input_vector_indices, threshold, window_size)

        print(f"Found {len(vector_groups)} vectors for window size {window_size}")
        return window_size, vector_groups

def run_vector_search(input_vectors, window_sizes, thresholds, output_path):
    searcher = VectorSearcher()
    search_results = {}

    with ProcessPoolExecutor() as executor:
        search_args = [(window_size, input_vectors, 1, thresholds[i]) 
                      for i, window_size in enumerate(window_sizes)]
        
        results = list(executor.map(searcher.search_window_size, search_args))
        
        for window_size, vector_groups in results:
            search_results[window_size] = vector_groups

    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(search_results, f)

    print("Search results saved.")
    return search_results