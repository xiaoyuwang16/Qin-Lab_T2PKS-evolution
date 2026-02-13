import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import re
import os
from pathlib import Path
import ast  
import constants

plt.rcParams.update({'font.size': 14})

CLUSTERS = {
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

SPECIAL_NAMES = ['CLF_Dendrubin', 'CLF_Pyxidicycline', 'KS_Dendrubin', 'KS_Pyxidicycline']

def calculate_similarity(vec1, vec2):
    """calculate_similarity"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def clean_name(name):
    """clean_protein_name"""
    for special in SPECIAL_NAMES:
        if name == special or name.startswith(special + '_'):
            return re.sub(r'_(X|[0-9]+unknown|\d+)$', '', name)

    name = re.sub(r'_\d+unknown$', '', name)
    name = re.sub(r'_Bu_\d+unknown$', '', name)
    name = re.sub(r'_[A-Za-z]\'?$', '', name)
    name = re.sub(r'_\d+$', '', name)
    return name

def parse_vector_string(s):

    try:
        s = str(s).strip()
        
        if s.startswith('[') and ',' in s and 'array' not in s:
            try:
                return np.array(ast.literal_eval(s))
            except:
                pass

        if 'array' in s:
            start = s.find('(')
            end = s.rfind(')')
            if start != -1 and end != -1:
                s = s[start+1:end]
        
        s = s.replace('[', '').replace(']', '').strip()
        
        if not s:
            return np.array([])
            
        if ',' in s:
            return np.fromstring(s, sep=',')
        else:
            return np.fromstring(s, sep=' ')
            
    except Exception as e:
        print(f"error: {s[:50]}... mistake: {e}")
        return np.array([])

def main():
    sequence_index_path = constants.BASE_DIR / 'sequence_index.txt'
    vectors_path = constants.OUTPUT_DIR / 'protein_go_vectors.csv'
    if not vectors_path.exists():
        vectors_path = constants.BASE_DIR / 'protein_go_vectors.csv'
    output_image_path = constants.OUTPUT_DIR / 'gnbu_protein_similarity_heatmap_with_special_names.png'

    if not sequence_index_path.exists():
        print(f"File not found {sequence_index_path}")
        return
    if not vectors_path.exists():
        print(f"File not found {vectors_path}")
        return

    print("Loading...")

    protein_to_index = {}
    index_to_protein = {}
    with open(sequence_index_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                index = int(parts[0])
                protein = parts[1]
                protein_to_index[protein] = index
                index_to_protein[index] = protein

    df = pd.read_csv(vectors_path)
    
    print("Analysing vectors...")
    df['Vector'] = df['Vector'].apply(parse_vector_string)
    
    if len(df) > 0 and df['Vector'].iloc[0].size == 0:
         print("Warning: First vector is empty, might cause error")

    index_to_vector = {protein_to_index[protein]: vector
                       for protein, vector in zip(df['Protein'], df['Vector'])
                       if protein in protein_to_index}

    selected_clusters = ['CLF_0', 'KS_0', 'KS_3', 'CLF_4', 'KS_4']

    cluster_proteins = {cluster: [idx for idx in CLUSTERS[cluster] if idx in index_to_vector] 
                        for cluster in selected_clusters}
    all_indices = [idx for cluster in selected_clusters for idx in cluster_proteins[cluster]]

    if not all_indices:
        print("Warning: no valid protein found.")
        return

    protein_names = []
    for idx in all_indices:
        name = index_to_protein[idx]
        name = name.replace("ksa", "KS").replace("ksb", "CLF")
        name = clean_name(name)
        protein_names.append(name)

    print("Calculating similarity matrix...")
    similarity_matrix = np.zeros((len(all_indices), len(all_indices)))
    for i, idx1 in enumerate(all_indices):
        for j, idx2 in enumerate(all_indices):
            similarity_matrix[i, j] = calculate_similarity(index_to_vector[idx1], index_to_vector[idx2])

    print("Generating heatmap...")
    plt.figure(figsize=(14, 12))
    ax = plt.subplot(111)

    sns_heatmap = sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap='GnBu',
        cbar_kws={'shrink': .5, 'label': 'Similarity Score', 'pad': 0.1},
        xticklabels=protein_names,
        yticklabels=protein_names,
        ax=ax
    )

    plt.subplots_adjust(top=0.85, right=0.85)
    plt.title('Protein Similarity Heatmap for Selected Clusters', fontsize=24, pad=50)
    plt.xticks(rotation=45, fontsize=14, ha='right')
    plt.yticks(rotation=0, fontsize=13)

    cbar = sns_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Similarity Score', fontsize=18, labelpad=15)

    current_pos = 0
    for cluster in selected_clusters:
        cluster_size = len(cluster_proteins[cluster])
        if current_pos > 0:
            ax.axhline(y=current_pos, color='white', linewidth=4)
            ax.axvline(x=current_pos, color='white', linewidth=4)

        display_cluster = cluster.split('_')[0]

        plt.text(current_pos + cluster_size/2, -1, display_cluster,
                 verticalalignment='top', horizontalalignment='center',
                 fontsize=16, fontweight='bold', color='#00496B')

        current_pos += cluster_size

    current_pos = 0
    for cluster in selected_clusters:
        cluster_size = len(cluster_proteins[cluster])
        display_cluster = cluster.split('_')[0]
        plt.text(len(all_indices) + 0.8, current_pos + cluster_size/2, display_cluster,
                 verticalalignment='center', horizontalalignment='left', rotation=0,
                 fontsize=16, fontweight='bold', color='#00496B')
        current_pos += cluster_size

    ax.set_yticks(np.arange(similarity_matrix.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(similarity_matrix.shape[1])+0.5, minor=False)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)

    if not output_image_path.parent.exists():
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=[0, 0, 0.85, 0.85])
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Fig saved to: {output_image_path}")
    print(f"Total protein number: {len(all_indices)}")
    for cluster in selected_clusters:
        print(f"{cluster} has: {len(cluster_proteins[cluster])}")

if __name__ == "__main__":
    main()