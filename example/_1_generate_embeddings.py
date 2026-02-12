from pathlib import Path
from src._1_embedding.esm_embedder import ESMEmbedder
from src._1_embedding.normalizer import EmbeddingNormalizer
from src._1_embedding.utils import save_embeddings
from constants import PCA_COMPONENTS, BASE_DIR, OUTPUT_DIR

def main():
    """Generate, normalize and transform sequence embeddings."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fasta_path = BASE_DIR / "KS_all&outgroup.fasta"
    raw_embeddings_path = OUTPUT_DIR / "raw_embeddings.npy"
    normalized_embeddings_path = OUTPUT_DIR / "normalized_embeddings.npy"
    pca_embeddings_path = OUTPUT_DIR / "normalized_pca_embeddings.npy"
    
    embedder = ESMEmbedder()
    
    print("Loading sequences...")
    sequences = embedder.load_sequences(str(fasta_path))

    print("Generating embeddings...")
    embeddings = embedder.generate_embeddings(sequences)
    save_embeddings(embeddings, str(raw_embeddings_path))
    print(f"Raw embeddings saved to: {raw_embeddings_path}")
 
    normalizer = EmbeddingNormalizer(pca_components=PCA_COMPONENTS)

    print("\nNormalizing embeddings...")
    normalized_embeddings = normalizer.l2_normalize(embeddings)
    save_embeddings(normalized_embeddings, str(normalized_embeddings_path))
    print(f"Normalized embeddings saved to: {normalized_embeddings_path}")
    
    print("\nPerforming PCA transformation...")
    pca_embeddings = normalizer.pca_transform(normalized_embeddings)
    save_embeddings(pca_embeddings, str(pca_embeddings_path))
    print(f"PCA embeddings saved to: {pca_embeddings_path}")
    
    print("\nAll processing completed successfully.")

if __name__ == "__main__":
    main()
