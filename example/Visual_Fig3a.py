import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import ast  
from pathlib import Path
import constants  

def calculate_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def calculate_cluster_similarity(cluster1_indices, cluster2_indices, index_to_vector):
    similarities = []
    for idx1, idx2 in itertools.product(cluster1_indices, cluster2_indices):
        if idx1 in index_to_vector and idx2 in index_to_vector:
            sim = calculate_similarity(index_to_vector[idx1], index_to_vector[idx2])
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0

def calculate_intra_cluster_similarity(cluster_indices, index_to_vector):
    similarities = []
    for idx1, idx2 in itertools.combinations(cluster_indices, 2):
        if idx1 in index_to_vector and idx2 in index_to_vector:
            sim = calculate_similarity(index_to_vector[idx1], index_to_vector[idx2])
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0


def main():
    seq_index_file = constants.BASE_DIR / 'sequence_index.txt'
    vectors_file = constants.OUTPUT_DIR / 'protein_go_vectors.csv'
    
    heatmap_output = constants.OUTPUT_DIR / 'cluster_similarity_heatmap.png'
    boxplot_output = constants.OUTPUT_DIR / 'similarity_boxplot.png'

    print(f"Reading sequence index: {seq_index_file} ...")
    protein_to_index = {}
    index_to_protein = {}
    
    if not seq_index_file.exists():
        print(f"File not found {seq_index_file}")
        return

    with open(seq_index_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                idx_str, protein = parts[0], parts[1]
                index = int(idx_str)
                protein_to_index[protein] = index
                index_to_protein[index] = protein

    print(f"Reading vectors: {vectors_file} ...")
    if not vectors_file.exists():
        print(f"File not found {vectors_file}")
        return

    df = pd.read_csv(vectors_file)
    
    print("Analysing vector format...")
    df['Vector'] = df['Vector'].apply(ast.literal_eval)

    index_to_vector = {}
    missing_proteins = []
    
    for protein, vector in zip(df['Protein'], df['Vector']):
        if protein in protein_to_index:
            index = protein_to_index[protein]
            index_to_vector[index] = np.array(vector)
        else:
            missing_proteins.append(protein)

    print(f"Total proteins: {len(df)}")
    print(f"Total protens in index file: {len(protein_to_index)}")
    print(f"Successful mapping: {len(index_to_vector)}")
    print(f"Unsuccessful mapping: {len(missing_proteins)}")
    if missing_proteins:
        print("Examples of Unmatched Proteins:", missing_proteins[:5])


    clusters = {
        "KS_3": [0, 24, 47, 78, 114, 132, 142, 145],
        "KS_2": [1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48,
                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124,
                125, 126, 127, 128, 129, 131, 133, 134, 135, 136, 137, 139, 140, 141, 143, 144, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                162, 163, 164, 165, 166],
        "KS_0": [2, 10, 98, 99, 122, 138, 148],
        "KS_4": [11, 34, 35, 44, 161, 130],
        "CLF_5":  [167, 168, 169, 174, 175, 176, 178, 186, 195, 199, 202, 205, 211, 212, 214, 219, 220, 226, 229, 233, 234, 235, 237, 238, 241, 243, 245, 248, 249, 271,
                  272, 277, 279, 281, 283, 288, 289, 290, 297, 298, 302, 303, 305, 306, 310, 313, 317, 319, 325, 328, 330, 331, 332],
        "CLF_1":  [170, 171, 172, 173, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 203, 204, 206, 207, 208, 213, 215, 216,
                  217, 218, 221, 222, 223, 224, 225, 228, 230, 231, 232, 236, 239, 240, 242, 244, 246, 247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
                  262, 263, 264, 265, 266, 267, 268, 269, 270, 273, 274, 278, 280, 282, 284, 285, 286, 287, 291, 292, 294, 299, 300, 301, 307, 308, 309, 311, 312, 314,
                  315, 318, 320, 321, 322, 323, 324, 326, 329],
        "CLF_4":  [177, 200, 201, 210, 327, 296],
        "CLF_3":  [209, 227, 275, 316],
        "CLF_0":  [293, 295, 304]
    }
    
    if not clusters:
        print("\nWaring: 'clusters' is empty!")
        return 

    all_cluster_indices = []
    for indices in clusters.values():
        all_cluster_indices.extend(indices)
        
    missing_indices = set(all_cluster_indices) - set(index_to_vector.keys())
    print(f"Indexes present in the cluster but missing from the vector library: {len(missing_indices)}")

    for cluster_name in clusters:
        original_count = len(clusters[cluster_name])
        clusters[cluster_name] = [idx for idx in clusters[cluster_name] if idx in index_to_vector]
        if len(clusters[cluster_name]) < original_count:
            print(f"  Cluster {cluster_name}: removed {original_count - len(clusters[cluster_name])} missing index")

    print("Calculating similarities...")
    results = {}
    sorted_cluster_names = sorted(clusters.keys(), key=lambda x: (not x.startswith('KS'), x))
    
    for cluster1, cluster2 in itertools.combinations(sorted_cluster_names, 2):
        key = f"{cluster1} vs {cluster2}"
        similarity = calculate_cluster_similarity(clusters[cluster1], clusters[cluster2], index_to_vector)
        results[key] = similarity

    print("Calculating similarities within clusters...")
    intra_cluster_similarities = {}
    for cluster_name in sorted_cluster_names:
        cluster_indices = clusters[cluster_name]
        intra_similarity = calculate_intra_cluster_similarity(cluster_indices, index_to_vector)
        intra_cluster_similarities[cluster_name] = intra_similarity

    heatmap_data = np.zeros((len(sorted_cluster_names), len(sorted_cluster_names)))
    
    for i, cluster1 in enumerate(sorted_cluster_names):
        for j, cluster2 in enumerate(sorted_cluster_names):
            if i == j:
                heatmap_data[i, j] = intra_cluster_similarities[cluster1]
            else:
                key1 = f"{cluster1} vs {cluster2}"
                key2 = f"{cluster2} vs {cluster1}"
                if key1 in results:
                    heatmap_data[i, j] = results[key1]
                elif key2 in results:
                    heatmap_data[i, j] = results[key2]
                else:
                    heatmap_data[i, j] = 0.0

    plt.figure(figsize=(12, 10))

    ks_clusters = [name for name in sorted_cluster_names if name.startswith('KS')]
    division_point = len(ks_clusters)

    ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='GnBu', cbar_kws={'shrink': .4},
                     xticklabels=sorted_cluster_names, yticklabels=sorted_cluster_names)

    if division_point > 0 and division_point < len(sorted_cluster_names):
        ax.hlines([division_point], *ax.get_xlim(), colors="white", linewidths=3)
        ax.vlines([division_point], *ax.get_ylim(), colors="white", linewidths=3)

    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title('Cluster Similarity Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_output}")
    plt.show()

    ks_similarities = []
    clf_similarities = []
    ks_clf_similarities = []

    for key, value in results.items():
        clusters_involved = key.split(' vs ')
        if len(clusters_involved) != 2: continue
        
        c1, c2 = clusters_involved[0], clusters_involved[1]
        
        if c1.startswith('KS') and c2.startswith('KS'):
            ks_similarities.append(value)
        elif c1.startswith('CLF') and c2.startswith('CLF'):
            clf_similarities.append(value)
        else:
            ks_clf_similarities.append(value)

    if ks_similarities and clf_similarities:
        t_stat, p_value = stats.ttest_ind(ks_similarities, clf_similarities)
        print(f"\n(KS vs KS) vs (CLF vs CLF):")
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
    else:
        print("\nNot enough data for T-test")

if __name__ == "__main__":
    main()