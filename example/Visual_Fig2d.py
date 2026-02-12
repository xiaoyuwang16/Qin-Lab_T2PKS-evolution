import sys
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from pathlib import Path

def load_embeddings(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not find: {file_path}")
    embeddings = np.load(file_path)
    return embeddings

def calculate_ancestor_node(embeddings, ancestor_nodes):
    return np.mean([embeddings[i] for i in ancestor_nodes], axis=0)

def calculate_distances(embeddings, ancestor_node, ks_nodes, clf_nodes):
    distances = {}
    for ks, clf in zip(ks_nodes, clf_nodes):
        ks_dist = euclidean(ancestor_node, embeddings[ks])
        clf_dist = euclidean(ancestor_node, embeddings[clf])
        distances[(ks, clf)] = (ks_dist, clf_dist)
    return distances

def analyze_distances(distances):
    ratios = []
    ks_indices = []
    clf_indices = []
    for (ks, clf), (ks_dist, clf_dist) in distances.items():
        ratio = ks_dist / clf_dist
        ratios.append(ratio)
        ks_indices.append(ks)
        clf_indices.append(clf)
    return pd.DataFrame({'KS': ks_indices, 'CLF': clf_indices, 'Ratio': ratios})

def identify_outliers(df, threshold=0.2):
   
    mean_ratio = df['Ratio'].mean()
    outliers = df[abs(df['Ratio'] - mean_ratio) > threshold]
    return outliers

def plot_analysis(df, outliers, custom_palette):
    
    plt.figure(figsize=(8, 6))

    sns.regplot(
        x='KS',
        y='Ratio',
        data=df,
        scatter_kws={'alpha': 0.7, 'color': custom_palette[0], 's': 70},
        line_kws={'color': custom_palette[2], 'linewidth': 2}
    )

    plt.title('KS/CLF Distance to Ancestor Node Ratio', fontsize=18, pad=15)
    plt.xlabel('KS Index', fontsize=18)
    plt.ylabel('Ratio (KS distance / CLF distance)', fontsize=18)

    plt.scatter(outliers['KS'], outliers['Ratio'], color=custom_palette[3], s=100, zorder=5, edgecolor='white', linewidth=1)
    
    for idx, row in outliers.iterrows():
        ks_idx, clf_idx = int(row['KS']), int(row['CLF'])
        
        if ks_idx == 150:
            xytext = (-40, 5)
            ha = 'right'
        elif ks_idx == 129:
            xytext = (0, -25)
            ha = 'center'
        else:
            xytext = (5, 5)
            ha = 'left'

        plt.annotate(
            f"KS{ks_idx}/CLF{clf_idx}",
            (row['KS'], row['Ratio']),
            xytext=xytext,
            textcoords='offset points',
            fontsize=12,
            color=custom_palette[4],
            fontweight='bold',
            ha=ha
        )

    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(left=False, bottom=False, right=False, top=False)

    mean_ratio = df['Ratio'].mean()
    plt.axhline(y=mean_ratio, color=custom_palette[1], linestyle='--', alpha=0.8)
    plt.text(
        df['KS'].max() * 0.02,
        mean_ratio + 0.03,
        f'Mean: {mean_ratio:.4f}',
        fontsize=14,
        color=custom_palette[1]
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk")
    custom_palette = ["#B2EC5D", "#7FBC41", "#4D9221", "#FDE725", "#FFAA00"]

    embeddings_file = constants.OUTPUT_DIR / "normalized_pca_embeddings.npy"
    
    print(f"Loading file: {embeddings_file}")
    
    try:
        embeddings = load_embeddings(embeddings_file)
        
        ancestor_nodes = [0, 2, 10, 24, 47, 78, 98, 99, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 293, 295, 304, 316]

        ancestor_node = calculate_ancestor_node(embeddings, ancestor_nodes)
        
        all_nodes = set(range(1, 333))
        ancestor_set = set(ancestor_nodes)
        remaining_nodes = all_nodes - ancestor_set
        ks_nodes = [node for node in remaining_nodes if 1 <= node <= 166]
        clf_nodes = [node + 166 for node in ks_nodes if node + 166 <= 333]
        ks_nodes = ks_nodes[:len(clf_nodes)]

        distances = calculate_distances(embeddings, ancestor_node, ks_nodes, clf_nodes)
        df = analyze_distances(distances)
        outliers = identify_outliers(df)

        plot_analysis(df, outliers, custom_palette)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the prense of embeddings_normalized_pca.npy in constants.py: OUTPUT_DIR.")