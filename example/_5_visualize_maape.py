import numpy as np
import pickle
from pathlib import Path
from src._5_visualization.maape_visualizer import MAAPEVisualizer
from constants import KNN_K, COLOR_SCHEME, BASE_DIR, OUTPUT_DIR

def main():
    """
    Visualize the MAAPE graph with embeddings and weights.
    """

    embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    order_index_path = BASE_DIR / "order_index.txt"
    weights_path = OUTPUT_DIR / "new_edge_weights_pca.pkl"
    

    print("Configuration:")
    print(f"- KNN K: {KNN_K}")
    print(f"- Color classes: {len(COLOR_SCHEME)}")
   
    try:
        print("Initializing visualizer...")
        visualizer = MAAPEVisualizer(COLOR_SCHEME)

        print("\nLoading data...")
        embeddings = np.load(embeddings_path)
        sequence_orders = visualizer.read_order_file(str(order_index_path))
        
        with open(weights_path, 'rb') as f:
            edge_weights = pickle.load(f)

        print(f"Loaded:")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Sequence orders: {len(sequence_orders)}")
        print(f"  - Edge weights: {len(edge_weights)}")

        print("\nBuilding KNN graph...")
        knn_graph = visualizer.build_knn_graph(embeddings, KNN_K)
        
        directed_edges = [(edge[0], edge[1]) for edge in edge_weights.keys()]

        print("\nGenerating visualizations...")
        print("\n1. Graph without node labels:")
        visualizer.visualize(knn_graph, sequence_orders, directed_edges, edge_weights)

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()