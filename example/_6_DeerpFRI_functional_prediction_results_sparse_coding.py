import pandas as pd
import numpy as np
import os
from pathlib import Path
import csv  
import constants

def main():

    input_file = constants.BASE_DIR / "DeepFRI function predictions.csv"
    output_file = constants.OUTPUT_DIR / "protein_go_vectors.csv"


    if not input_file.exists():
        print(f"Error_File not found: {input_file}")
        return

    print(f"Loading: {input_file} ...")

    try:
        df = pd.read_csv(input_file, quotechar='"', quoting=1)
        
        required_columns = ['Protein', 'GO_term/EC_number', 'Score']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: essential columns missing. Needs: {required_columns}")
            return

        unique_go_terms = df['GO_term/EC_number'].unique()
        go_term_index = {term: i for i, term in enumerate(unique_go_terms)}
        
        print(f"Found {len(unique_go_terms)} unique GO Terms，Generating function vectors...")

        protein_vectors = {}

        for protein, group in df.groupby('Protein'):
            vector = np.zeros(len(unique_go_terms))
            
            for _, row in group.iterrows():
                go_term = row['GO_term/EC_number']
                score = row['Score']
                if go_term in go_term_index:
                    vector[go_term_index[go_term]] = score
            
            protein_vectors[protein] = vector


        print("Transforming data format...")
        result_df = pd.DataFrame({
            'Protein': list(protein_vectors.keys()),
            'Vector': [v.tolist() for v in protein_vectors.values()]
        })

        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)


        result_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

        print(f"Finished!")
        print(f"Processed {len(protein_vectors)} proteins.")
        print(f"Results saved to: {output_file}")
        
        print("\nSample data format:")
        print(result_df.head(1)['Vector'].values[0][:5], "... ")

    except Exception as e:
        print(f"Unkonwn error occured: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()