import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.patheffects as PathEffects
import re
import sys
from pathlib import Path

def load_embeddings(file_path):
    if not file_path.exists():
        print(f"ERROR_File not find: {file_path}")
        sys.exit(1)
    embeddings = np.load(file_path)
    print(f"Loaded: {file_path}, Shape: {embeddings.shape}")
    return embeddings

def build_knn_graph(embeddings, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    adj_list = {i: set(indices[i][1:]) for i in range(len(embeddings))}
    return adj_list

def visualize_knn_graph_with_ellipses(adj_list, node_categories, cluster_info, show_nodes=True, show_labels=True, selected_nodes=None):
    G = nx.DiGraph(adj_list)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    pos = nx.spring_layout(G, k=0.25, seed=42)

    if 'KS_2' in cluster_info:
        ks2_nodes = cluster_info['KS_2']['nodes']
        shift_right = 0.3
        for node in ks2_nodes:
            if node in pos: pos[node] = (pos[node][0] + shift_right, pos[node][1])

    if 'KS_4' in cluster_info:
        ks4_nodes = cluster_info['KS_4']['nodes']
        shift_up = 0.1
        for node in ks4_nodes:
            if node in pos: pos[node] = (pos[node][0], pos[node][1] + shift_up)

    if 'CLF_4' in cluster_info:
        clf4_nodes = cluster_info['CLF_4']['nodes']
        shift_right = 0.1
        shift_up = 0.1
        for node in clf4_nodes:
            if node in pos: pos[node] = (pos[node][0] + shift_right, pos[node][1] + shift_up)

    if 'CLF_0' in cluster_info:
        clf0_nodes = cluster_info['CLF_0']['nodes']
        shift_right = 0.15
        shift_down = 0.1
        for node in clf0_nodes:
            if node in pos: pos[node] = (pos[node][0] + shift_right, pos[node][1] - shift_down)

    cluster_colors = {
        'KS_3': '#4ED7F1',  
        'KS_2': '#00DFA2',  
        'KS_0': '#00CAFF',  
        'KS_4': '#D2FF72',  
        'CLF_5': '#FFBE0B', 
        'CLF_1': '#FF885B', 
        'CLF_4': '#FFEB00', 
        'CLF_3': '#FE5D26', 
        'CLF_0': '#FF748B', 
        'CLF_2': '#F16767'  
    }

    node_color_map = {}
    for cluster_name, cluster_data in cluster_info.items():
        for node in cluster_data['nodes']:
            node_color_map[node] = cluster_colors.get(cluster_name, '#D3D3D3')

    if show_nodes:
        edges = []
        for node in adj_list:
            for neighbor in adj_list[node]:
                edges.append((node, neighbor))

        nx.draw_networkx_edges(G, pos,
                             edge_color='gray',
                             width=0.5,
                             alpha=0.4,
                             arrows=False,
                             ax=ax)

    for cluster_name, cluster_data in cluster_info.items():
        cluster_nodes = [n for n in cluster_data['nodes'] if n in pos]
        if len(cluster_nodes) > 1: 
            cluster_x = [pos[i][0] for i in cluster_nodes]
            cluster_y = [pos[i][1] for i in cluster_nodes]

            if cluster_name == 'CLF_0':
                min_x = min(cluster_x) - 0.05  
                max_x = max(cluster_x) + 0.05
                min_y = min(cluster_y) - 0.05
                max_y = max(cluster_y) + 0.05

                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                width = max_x - min_x + 0.05  
                height = max_y - min_y + 0.05

                ellipse = Ellipse((center_x, center_y), width, height,
                                 angle=45,  
                                 color=cluster_colors[cluster_name],
                                 alpha=0.2)
                ax.add_patch(ellipse)
                continue  

            center_x = np.mean(cluster_x)
            center_y = np.mean(cluster_y)

            if cluster_name == 'KS_2':
                center_x += 0.1
                center_y += 0.05  
            elif cluster_name == 'KS_4':
                center_y += 0.05  
            elif cluster_name == 'CLF_4':
                center_x += 0.05  
                center_y += 0.05
            elif cluster_name == 'CLF_1':
                center_x -= 0.05  
                center_y -= 0.05
            elif cluster_name == 'CLF_5':
                center_x += 0.05  
                center_y -= 0.05

            if cluster_name == 'KS_3':
                width_factor = 1.05
                height_factor = 1.05
            elif cluster_name == 'CLF_1':
                width_factor = 1.1
                height_factor = 1.1
            elif cluster_name == 'CLF_5':
                width_factor = 1.1
                height_factor = 1.1
            elif cluster_name == 'KS_4':
                width_factor = 1.3
                height_factor = 1.3
            elif cluster_name == 'CLF_4':
                width_factor = 1.4
                height_factor = 1.4
            else:
                width_factor = 1.2
                height_factor = 1.2

            width = max(np.max(cluster_x) - np.min(cluster_x), 0.1) * width_factor
            height = max(np.max(cluster_y) - np.min(cluster_y), 0.1) * height_factor

            if cluster_name == 'CLF_2':
                width = 0.1
                height = 0.1
            elif len(cluster_nodes) < 5 and cluster_name not in ['KS_4', 'CLF_4', 'KS_3']:
                width *= 0.9
                height *= 0.9

            if len(cluster_nodes) > 2:
                coords = np.vstack((cluster_x, cluster_y)).T
                coords_mean = coords.mean(axis=0)
                coords_centered = coords - coords_mean
                cov = np.cov(coords_centered.T)
                eigvals, eigvecs = np.linalg.eig(cov)

                idx = eigvals.argsort()[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                angle = np.degrees(angle)

                if cluster_name == 'KS_4':
                    angle = angle - 15  
                elif cluster_name == 'CLF_4':
                    angle = angle + 10
            else:
                angle = 0

            ellipse = Ellipse((center_x, center_y), width, height,
                             angle=angle,
                             color=cluster_colors[cluster_name],
                             alpha=0.2)
            ax.add_patch(ellipse)

    if show_nodes:
        node_colors_list = [node_color_map.get(i, '#D3D3D3') for i in range(len(node_categories))]

        nx.draw_networkx_nodes(G, pos,
                             node_size=300,
                             node_color=node_colors_list,
                             alpha=0.9,
                             ax=ax,
                             linewidths=1.5,          
                             edgecolors='white')      

        if show_labels and selected_nodes:
            for node in selected_nodes:
                if node in pos:
                    x, y = pos[node]
                    txt = ax.text(x, y, str(node),
                               fontsize=15,
                               color='black',
                               weight='bold',
                               ha='center',
                               va='center')
                    txt.set_path_effects([
                        PathEffects.withStroke(linewidth=3, foreground='white')
                    ])

    legend_elements = []

    def extract_number(cluster_name):
        match = re.search(r'(\D+)_(\d+)', cluster_name)
        if match:
            prefix, number = match.groups()
            return prefix, int(number)
        return cluster_name, 0

    sorted_cluster_names = sorted(cluster_colors.keys(),
                                 key=lambda x: (extract_number(x)[0], extract_number(x)[1]))

    for cluster_name in sorted_cluster_names:
        if cluster_name in cluster_info:
            num_nodes = len(cluster_info[cluster_name]['nodes'])
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[cluster_name],
                          markersize=10, label=f'{cluster_name} ({num_nodes} nodes)',
                          markeredgecolor='white', markeredgewidth=1.5)  
            )


    ax.legend(handles=legend_elements,
             loc='center left',
             bbox_to_anchor=(1, 0.5),
             frameon=True,
             fancybox=True,
             shadow=True,
             fontsize=10)

    plt.title(f'Node Clusters with Compact Elliptical Regions', fontsize=16, pad=10)

    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    save_filename = "cluster_visualization_strict.png"
    save_path = constants.OUTPUT_DIR / save_filename
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fig saved to: {save_path}")
    

if __name__ == "__main__":

    embeddings_file = constants.OUTPUT_DIR / "normalized_pca_embeddings.npy"
    
    embeddings = load_embeddings(embeddings_file)

    k = 15

    cluster_info = {
        'KS_3': {'nodes': [0, 24, 47, 78, 114, 132, 142, 145]},
        'KS_2': {'nodes': [1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 141, 143, 144, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166]},
        'KS_0': {'nodes': [2, 10, 98, 99, 122, 138, 148]},
        'KS_4': {'nodes': [11, 34, 35, 44, 110, 161]},
        'CLF_5': {'nodes': [167, 168, 169, 174, 175, 176, 178, 186, 195, 199, 202, 205, 211, 212, 214, 219, 220, 226, 229, 233, 234, 235, 237, 238, 241, 243, 245, 248, 249, 271, 272, 277, 279, 281, 283, 288, 289, 290, 297, 298, 302, 303, 305, 306, 310, 313, 317, 319, 325, 328, 330, 331, 332]},
        'CLF_1': {'nodes': [170, 171, 172, 173, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 203, 204, 206, 207, 208, 213, 215, 216, 217, 218, 221, 222, 223, 224, 225, 228, 230, 231, 232, 236, 239, 240, 242, 244, 246, 247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 273, 274, 278, 280, 282, 284, 285, 286, 287, 291, 292, 294, 299, 300, 301, 307, 308, 309, 311, 312, 314, 315, 318, 320, 321, 322, 323, 324, 326, 329]},
        'CLF_4': {'nodes': [177, 200, 201, 210, 276, 327]},
        'CLF_3': {'nodes': [209, 227, 275, 316]},
        'CLF_0': {'nodes': [293, 295, 304]},
        'CLF_2': {'nodes': [296]}
    }

    knn_graph = build_knn_graph(embeddings, k)
    node_categories = list(range(len(knn_graph)))

    selected_nodes = [33, 199, 89, 255, 130, 296, 43, 209, 129, 295, 127, 293, 138, 304]

    print("Generating visualization with colored nodes and selected labels...")
    visualize_knn_graph_with_ellipses(
        knn_graph, 
        node_categories, 
        cluster_info, 
        show_nodes=True, 
        show_labels=True, 
        selected_nodes=selected_nodes
    )