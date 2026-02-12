import numpy as np

def save_embeddings(embeddings, file_path):
    """
    Save embeddings to a file.
    
    Args:
        embeddings (np.ndarray): Embeddings to save
        file_path (str): Path to save the embeddings
    """
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """
    Load embeddings from a file.
    
    Args:
        file_path (str): Path to the embeddings file
        
    Returns:
        np.ndarray: Loaded embeddings
    """
    return np.load(file_path)