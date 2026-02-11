
import numpy as np
from sklearn.decomposition import PCA



class EmbeddingNormalizer:
    def __init__(self, pca_components):
        """
        Initialize the normalizer.
        
        Args:
            pca_components (int): Number of components for PCA
        """
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)

    def l2_normalize(self, vectors):
        """
        Perform L2 normalization on vectors.
        
        Args:
            vectors (np.ndarray): Input vectors
            
        Returns:
            np.ndarray: L2 normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1)
        return vectors / norms[:, np.newaxis]

    def pca_transform(self, vectors):
        """
        Perform PCA transformation on vectors.
        
        Args:
            vectors (np.ndarray): Input vectors
            
        Returns:
            np.ndarray: PCA transformed vectors
        """
        low_dim_embeddings = self.pca.fit_transform(vectors)
        print(f"Shape of the low-dimensional embeddings: {low_dim_embeddings.shape}")
        return low_dim_embeddings