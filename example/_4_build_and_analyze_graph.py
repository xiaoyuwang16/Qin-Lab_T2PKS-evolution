import numpy as np
from pathlib import Path
from src._4_graph_construction.knn_graph_builder import KNNGraphBuilder, GraphVisualizer
from src._4_graph_construction.weight_extractor import WeightExtractor
from constants import KNN_K, KNN_THRESHOLD, COLOR_SCHEME, BASE_DIR, OUTPUT_DIR

def main():
    """
    Construct and visualize KNN graph, and extract weights.
    """
    embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    order_index_path = BASE_DIR / "order_index.txt"
    all_edges_path = OUTPUT_DIR / "all_edges_data.pkl"
    knn_edges_path = OUTPUT_DIR / "knn_graph_edges.txt"
    new_weights_path = OUTPUT_DIR / "new_edge_weights_pca.pkl"
    
    print("Configuration:")
    print(f"KNN parameters:")
    print(f"- K: {KNN_K}")
    print(f"- Threshold: {KNN_THRESHOLD}")
    print(f"- Color scheme: {len(COLOR_SCHEME)} classes")
    
    print("\nFile paths:")
    print(f"Input files:")
    print(f"- Embeddings: {embeddings_path}")
    print(f"- Order index: {order_index_path}")
    print(f"- All edges: {all_edges_path}")
    print(f"\nOutput files:")
    print(f"- KNN edges: {knn_edges_path}")
    print(f"- New weights: {new_weights_path}")
    
    try:

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print("\nStep 1: Loading embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings shape: {embeddings.shape}")
        
        print("\nStep 2: Building KNN graph...")
        graph_builder = KNNGraphBuilder(
            embeddings, 
            k=KNN_K, 
            threshold=KNN_THRESHOLD
        )
        adj_list = graph_builder.build_graph()
        graph_builder.save_edges(str(knn_edges_path))
        print(f"KNN graph constructed and saved to: {knn_edges_path}")
        
        print("\nStep 3: Visualizing graph...")
        visualizer = GraphVisualizer(COLOR_SCHEME)
        sequence_orders = visualizer.read_order_file(str(order_index_path))
        visualizer.visualize(adj_list, sequence_orders)
        print("Graph visualization completed")
        
        print("\nStep 4: Extracting weights...")
        weight_extractor = WeightExtractor(
            knn_edges_file=str(knn_edges_path),
            weights_file=str(all_edges_path)  
        )
        new_weights = weight_extractor.extract_weights()
        weight_extractor.save_weights(str(new_weights_path))
        
        print("\nWeight Analysis:")
        weight_extractor.print_statistics()
        
        print("\nGenerating weight visualization...")
        weight_extractor.visualize_weights()
        
        print(f"\nResults Summary:")
        print(f"- KNN graph edges saved to: {knn_edges_path}")
        print(f"- New weights saved to: {new_weights_path}")
        print(f"- All output files in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    
    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()