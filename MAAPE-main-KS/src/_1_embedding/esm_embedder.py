import torch
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import time

class ESMEmbedder:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", batch_size=16):
        """
        Initialize the ESM embedder.
        
        Args:
            model_name (str): Name of the ESM model to use
            batch_size (int): Batch size for processing sequences
        """
        self.model = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def load_sequences(self, file_path):
        """
        Load sequences from a FASTA file.
        
        Args:
            file_path (str): Path to the FASTA file
            
        Returns:
            list: List of sequences
        """
        return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

    def generate_embeddings(self, sequences):
        """
        Generate embeddings for the given sequences.
        
        Args:
            sequences (list): List of protein sequences
            
        Returns:
            np.ndarray: Generated embeddings
        """
        embeddings = []
        start_time = time.time()

        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Embedding Progress"):
            batch_sequences = sequences[i:i + self.batch_size]
            inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(batch_embeddings)

            self._print_progress(start_time, i, len(sequences))

        embeddings = np.concatenate(embeddings, axis=0)
        print(f"Embedding completed. Total time: {time.time() - start_time:.2f} seconds")
        
        return embeddings

    def _print_progress(self, start_time, current_position, total_length):
        """Print progress information."""
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (current_position + self.batch_size) * total_length
        remaining_time = estimated_total_time - elapsed_time
        print(f"Estimated remaining time: {remaining_time:.2f} seconds")