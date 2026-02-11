import networkx as nx
from pathlib import Path
from src._6_aggregated_visualization.graph_builder import GraphBuilder
from src._6_aggregated_visualization.graph_processor import GraphProcessor
from src._6_aggregated_visualization.visualizer import Visualizer
from src._6_aggregated_visualization.utils import Utils
from constants import (
    WINDOW_SIZES, COLOR_SCHEME, KNN_K, KNN_THRESHOLD,
    PCA_COMPONENTS, BASE_DIR, OUTPUT_DIR
)

def main():
    """
    Main function to process and visualize the MAAPE graph.
    """

    embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    order_index_path = BASE_DIR / "order_index.txt"
    edge_weights_path = OUTPUT_DIR / "new_edge_weights_pca.pkl"

    print("Configuration:")
    print(f"Window sizes: {WINDOW_SIZES}")
    print(f"KNN parameters:")
    print(f"- K: {KNN_K}")
    print(f"- Threshold: {KNN_THRESHOLD}")
    print(f"PCA components: {PCA_COMPONENTS}")
    
    print("\nFile paths:")
    print(f"Input files:")
    print(f"- Embeddings: {embeddings_path}")
    print(f"- Order index: {order_index_path}")
    print(f"- Edge weights: {edge_weights_path}")

    try:
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("\nStep 1: Loading data...")
        embeddings = Utils.load_embeddings(embeddings_path)
        sequence_orders = Utils.read_order_file(order_index_path)
        edge_weights = Utils.load_edge_weights(edge_weights_path)
        print(f"Loaded embeddings shape: {embeddings.shape}")

        print("\nStep 2: Building KNN graph...")
        knn_graph = GraphBuilder.build_knn_graph(
            embeddings=embeddings,
            k=KNN_K
        )
        G = nx.DiGraph(knn_graph)
        print(f"Built graph with {len(knn_graph)} nodes")

        print("\nStep 3: Clustering nodes...")
        clusters = GraphBuilder.cluster_nodes(embeddings, sequence_orders)
        print("Clusters created:")
        for cluster, nodes in clusters.items():
            print(f"- {cluster}: {len(nodes)} nodes")

        print("\nStep 4: Aggregating edges...")
        cluster_graph = GraphBuilder.aggregate_edges(G, clusters, edge_weights)
        print(f"Aggregated graph has {cluster_graph.number_of_edges()} edges")

        print("\nStep 5: Processing graph...")
        cluster_graph, kept_edges = GraphProcessor.process_bidirectional_edges(cluster_graph)
        root_node = GraphProcessor.predict_root_node(cluster_graph)
        print(f"Identified root node: {root_node}")

        print("\nGraph Statistics:")
        Utils.print_statistics(embeddings, sequence_orders, cluster_graph)

        print("\nStep 6: Generating visualization...")
        Visualizer.visualize_aggregated_graph(
            cluster_graph=cluster_graph,
            color_scheme=COLOR_SCHEME,
            root_node=root_node
        )

        print("\nAll processes completed successfully.")

    except FileNotFoundError as e:
        print(f"\nError: Required file not found: {e}")
        raise
    except Exception as e:
        print(f"\nError occurred during processing: {e}")
        raise

if __name__ == "__main__":
    main()