import numpy as np
from typing import List, Tuple, Dict

def save_results(filepath: str, data: any) -> None:
    """Save results to pickle file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_results(filepath: str) -> any:
    """Load results from pickle file."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(vector1 - vector2)

def validate_inputs(window_sizes: List[int], weights: np.ndarray) -> bool:
    """Validate input parameters."""
    if len(window_sizes) != len(weights):
        raise ValueError("Length of window_sizes must match length of weights")
    if not all(w >= 0 for w in window_sizes):
        raise ValueError("All window sizes must be positive")
    if not all(w >= 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    return True

def format_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Format progress bar string."""
    progress = current / total
    filled = int(width * progress)
    bar = f"[{'=' * filled}>{' ' * (width - filled)}]"
    percentage = int(progress * 100)
    return f"{bar} {percentage}%"

def get_edge_statistics(edges_data: Dict) -> Dict:
    """Calculate edge statistics."""
    weights = []
    for edge_info in edges_data.values():
        weights.extend([edge_info['total_weight1'], edge_info['total_weight2']])
    
    return {
        'min_weight': min(weights) if weights else 0,
        'max_weight': max(weights) if weights else 0,
        'mean_weight': np.mean(weights) if weights else 0,
        'median_weight': np.median(weights) if weights else 0,
        'total_edges': len(edges_data)
    }