import numpy as np
import pickle
from pathlib import Path
from src._2_path_generation.vector_search import run_vector_search
from src._2_path_generation.path_finder import PathFinder
from src._2_path_generation.path_processor import PathGenerator, PathProcessor
from constants import WINDOW_SIZES, BASE_DIR, OUTPUT_DIR

def main():
    """
    Generate paths from protein embeddings through vector search and path generation.
    """
    print("Configuration:")
    print(f"Window sizes: {WINDOW_SIZES}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    thresholds_path = BASE_DIR / "converted_thresholds_pca.npy"
    search_results_path = OUTPUT_DIR / "search_results.pkl"
    initial_paths_path = OUTPUT_DIR / "initial_paths.pkl"
    advanced_paths_path = OUTPUT_DIR / "advanced_paths.pkl"
    processed_paths_path = OUTPUT_DIR / "processed_paths.pkl"
    
    print(f"\nInput files:")
    print(f"Embeddings: {embeddings_path}")
    print(f"Thresholds: {thresholds_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        vectors = np.load(embeddings_path)
        all_thresholds = np.load(thresholds_path)
        print(f"\nLoaded data:")
        print(f"Embeddings shape: {vectors.shape}")
        print(f"Thresholds shape: {all_thresholds.shape}")

        window_sizes = list(WINDOW_SIZES)
        thresholds = all_thresholds[:len(window_sizes)]
        print(f"Using {len(thresholds)} thresholds for {len(window_sizes)} window sizes")
        
    except Exception as e:
        print(f"Error loading input files: {e}")
        raise
    
    # Step 1: Vector Search
    print("\nStep 1: Running Vector Search...")
    try:
        search_results = run_vector_search(
            input_vectors=vectors,
            window_sizes=window_sizes,
            thresholds=thresholds,
            output_path=str(search_results_path)
        )
        print(f"Vector search results saved to: {search_results_path}")
    except Exception as e:
        print(f"Error during vector search: {e}")
        raise
    
    # Step 2: Path Generation
    print("\nStep 2: Generating Initial Paths...")
    try:
        path_finder = PathFinder(thresholds=thresholds, window_sizes=window_sizes)
        
        if isinstance(search_results, dict):
            search_results_dict = search_results
        else:
            print("Loading search results from file...")
            with open(search_results_path, 'rb') as f:
                search_results_dict = pickle.load(f)
        
        print(f"Search results contain {len(search_results_dict)} window sizes")
        
        paths = path_finder.find_generation_path(search_results_dict)
        print(f"Found {len(paths)} initial paths")
        
        with open(initial_paths_path, 'wb') as f:
            pickle.dump(paths, f)
        print(f"Initial paths saved to: {initial_paths_path}")
        
    except Exception as e:
        print(f"Error during initial path generation: {e}")
        raise
    
    # Step 3: Advanced Path Generation
    print("\nStep 3: Generating Advanced Paths...")
    try:
        path_generator = PathGenerator(search_results_dict, thresholds)
        advanced_paths = path_generator.generate_paths()
        print(f"Generated {len(advanced_paths)} advanced paths")
        
        with open(advanced_paths_path, 'wb') as f:
            pickle.dump(advanced_paths, f)
        print(f"Advanced paths saved to: {advanced_paths_path}")
        
    except Exception as e:
        print(f"Error during advanced path generation: {e}")
        raise
    
    # Step 4: Path Processing
    print("\nStep 4: Processing Paths...")
    try:
        path_processor = PathProcessor(advanced_paths)
        processed_paths = path_processor.process()
        PathProcessor.save_paths(processed_paths, str(processed_paths_path))
        
        print(f"\nFinal Summary:")
        print(f"- Initial Paths: {len(paths)}")
        print(f"- Advanced Paths: {len(advanced_paths)}")
        print(f"- Processed Paths: {len(processed_paths)}")
        print(f"\nAll results saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during path processing: {e}")
        raise
    
    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()