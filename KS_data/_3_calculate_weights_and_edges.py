import numpy as np
import pickle
from pathlib import Path
from src._3_weight_calculation.window_analyzer import WindowAnalyzer
from src._3_weight_calculation.weight_calculator import WeightCalculator
from constants import WINDOW_SIZES, BASE_DIR, OUTPUT_DIR

def main():
    """
    Calculate weights from processed paths and embeddings.
    """
    embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    processed_paths_path = OUTPUT_DIR / "processed_paths.pkl"
    search_results_path = OUTPUT_DIR / "search_results.pkl"
    window_info_path = OUTPUT_DIR / "window_info.pkl"
    edges_path = OUTPUT_DIR / "all_edges.pkl"
    edges_data_path = OUTPUT_DIR / "all_edges_data.pkl"
    
    print("Configuration:")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Window sizes: {WINDOW_SIZES}")
    
    try:
        print("\nLoading input files...")
        vectors = np.load(embeddings_path)
        with open(processed_paths_path, 'rb') as f:
            processed_paths = pickle.load(f)
        with open(search_results_path, 'rb') as f:
            search_results = pickle.load(f)
            
        print("\nLoaded data statistics:")
        print(f"- Embeddings shape: {vectors.shape}")
        print(f"- Processed paths: {len(processed_paths)}")
        print(f"- Search results window sizes: {len(search_results)}")
        
    except Exception as e:
        print(f"Error loading input files: {str(e)}")
        raise
    
    try:

        print("\nStep 1: Analyzing Windows...")
        window_analyzer = WindowAnalyzer(vectors, WINDOW_SIZES)
        window_info = window_analyzer.analyze_windows(search_results)
        normalized_weights = window_analyzer.get_normalized_weights()
        

        with open(window_info_path, 'wb') as f:
            pickle.dump({
                'window_info': window_info,
                'normalized_weights': normalized_weights
            }, f)
        print(f"Window analysis results saved to: {window_info_path}")
            
        print("\nWindow Analysis Results:")
        window_analyzer.print_statistics()
        

        print("\nStep 2: Calculating Weights...")
        weight_calculator = WeightCalculator(
            processed_paths=processed_paths,
            window_sizes=WINDOW_SIZES,
            weights=np.array(normalized_weights)
        )
        
        all_edges, all_edges_data = weight_calculator.calculate_weights()
        

        print("\nSaving weight calculation results...")
        with open(edges_path, 'wb') as f:
            pickle.dump(all_edges, f)
        with open(edges_data_path, 'wb') as f:
            pickle.dump(all_edges_data, f)
            
        print("\nWeight Calculation Results:")
        weight_calculator.print_statistics()

        print("\nOutput files:")
        print(f"- Window info: {window_info_path}")
        print(f"- Edges: {edges_path}")
        print(f"- Edge data: {edges_data_path}")
        
        print("\nFinal Summary:")
        print(f"- Processed Paths: {len(processed_paths)}")
        print(f"- Window Sizes: {len(WINDOW_SIZES)}")
        print(f"- Total Edges: {len(all_edges)}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    
    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()