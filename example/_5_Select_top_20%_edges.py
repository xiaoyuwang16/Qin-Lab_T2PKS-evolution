import sys
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

def main():

    input_file = constants.OUTPUT_DIR / "processed_paths.pkl"
    output_data_file = constants.OUTPUT_DIR / "selected_edges_data.pkl"
    output_list_file = constants.OUTPUT_DIR / "selected_edges.pkl"


    if not input_file.exists():
        print(f"Error_File not found: {input_file}")
        return


    print(f"Loading: {input_file} ...")
    with open(input_file, 'rb') as f:
        processed_paths = pickle.load(f)

    window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 150, 200]
    weights = np.array([0.0003068619382584253, 0.0006688031587033521, 0.002031285700003598, 
                        0.0058194892351399, 0.013077220229932147, 0.021224308168892324, 
                        0.02910983497127332, 0.03517029217975047, 0.040707564772526135, 
                        0.04581361386423672, 0.05644219436996148, 0.0710346635144513, 
                        0.08334518906678039, 0.10224993273840496, 0.12870288080271006, 
                        0.36429586528897545, 0.0, 0.0, 0.0])
    
    weight_map = dict(zip(window_sizes, weights))

    co_occurrence_matrices = {window_size: defaultdict(lambda: defaultdict(int)) for window_size in window_sizes}

    total_co_occurrences = {window_size: 0 for window_size in window_sizes}

    for item in tqdm(processed_paths, desc="Processing paths"):

        query_indices = [vector[1] for vector in item['query_vectors']]
        result_indices = [vector[1] for vector in item['result_vectors']]
        
        if not item['query_vectors']: 
            continue
        window_size = item['query_vectors'][0][0] 

        for query_index in query_indices:
            for result_index in result_indices:
                co_occurrence_matrices[window_size][query_index][result_index] += 1
                total_co_occurrences[window_size] += 1

    total_co_occurrence_matrix = defaultdict(lambda: defaultdict(float))

    for window_size, co_occurrence_matrix in tqdm(co_occurrence_matrices.items(), desc="Calculating total weight matrix"):
        current_weight = weight_map.get(window_size, 0)
        
        if current_weight == 0:
            continue

        for query_index, targets in co_occurrence_matrix.items():
            for result_index, count in targets.items():
                weight1 = count * current_weight
                total_co_occurrence_matrix[query_index][result_index] += weight1
                
                
                count_reverse = co_occurrence_matrix[result_index].get(query_index, 0)
                weight2 = count_reverse * current_weight
                total_co_occurrence_matrix[result_index][query_index] += weight2

    edge_weights = []
    final_matrix = {k: dict(v) for k, v in total_co_occurrence_matrix.items()}

    for query_index in tqdm(final_matrix, desc="Selecting top 20% edges"):
        for result_index in final_matrix[query_index]:
            if query_index < result_index:  
                total_weight1 = final_matrix[query_index].get(result_index, 0)
                total_weight2 = final_matrix[result_index].get(query_index, 0)
                edge_weights.append((query_index, result_index, total_weight1, total_weight2))

    edge_weights.sort(key=lambda x: max(x[2], x[3]), reverse=True)
    
    num_edges = len(edge_weights)
    num_selected_edges = int(num_edges * 0.2)
    selected_edges = edge_weights[:num_selected_edges]

    print(f"Total number of edges: {num_edges}")
    print(f"Number of selected edges (top 20%): {num_selected_edges}")

    print("Total co-occurrences for each window size:")
    for window_size, total_count in total_co_occurrences.items():
        print(f"Window size: {window_size}, Total co-occurrences: {total_count}")
    print()

    selected_edges_data = {}
    

    for query_index, result_index, total_weight1, total_weight2 in tqdm(selected_edges, desc="Building detailed data"):
        edge_key = (query_index, result_index)
        edge_info = {
            'total_weight1': total_weight1,
            'total_weight2': total_weight2,
            'window_data': []
        }
        for window_size in window_sizes:
            count1 = co_occurrence_matrices[window_size][query_index][result_index]
            count2 = co_occurrence_matrices[window_size][result_index][query_index]
            weight = weight_map[window_size]
            total_count = total_co_occurrences[window_size]
            
            edge_info['window_data'].append({
                'window_size': window_size,
                'count1': count1,
                'count2': count2,
                'weight': weight,
                'total_count': total_count
            })
        selected_edges_data[edge_key] = edge_info

    if selected_edges_data:
        max_weight_edge = max(selected_edges_data.items(), key=lambda x: max(x[1]['total_weight1'], x[1]['total_weight2']))
        min_weight_edge = min(selected_edges_data.items(), key=lambda x: min(x[1]['total_weight1'], x[1]['total_weight2']))

        print("Edge with maximum weight:")
        print(f"Edge: {max_weight_edge[0]}")
        print(f"  Total weight1: {max_weight_edge[1]['total_weight1']:.6f}")
        print(f"  Total weight2: {max_weight_edge[1]['total_weight2']:.6f}")
        
        print("\nEdge with minimum weight:")
        print(f"Edge: {min_weight_edge[0]}")
        print(f"  Total weight1: {min_weight_edge[1]['total_weight1']:.6f}")
        print(f"  Total weight2: {min_weight_edge[1]['total_weight2']:.6f}")


    print(f"\nSaving data to: {output_data_file}")
    with open(output_data_file, 'wb') as file:
        pickle.dump(selected_edges_data, file)

    print(f"Saving list to: {output_list_file}")
    with open(output_list_file, 'wb') as f:
        pickle.dump(selected_edges, f)

    print("Completed")

if __name__ == "__main__":
    main()