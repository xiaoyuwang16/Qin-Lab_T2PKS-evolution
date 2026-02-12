import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

def load_embeddings(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"ERROR_File not find: {file_path}")
    embeddings = np.load(file_path)
    return embeddings

def build_knn_graph(embeddings, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    adj_list = {i: set(indices[i][1:]) for i in range(len(embeddings))}
    return adj_list

class CurvedArrow(FancyArrowPatch):
    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA, posB, *args, **kwargs)
        self.posA = posA
        self.posB = posB
        dx = posB[0] - posA[0]
        dy = posB[1] - posA[1]
        dist = np.sqrt(dx*dx + dy*dy)
        curve_rad = min(0.3, max(0.1, dist * 0.15))  
        self.set_connectionstyle(f"arc3,rad={curve_rad}")

def visualize_knn_graph_with_curved_vectors(adj_list, node_categories, directed_edges, selected_nodes, edge_weights, excluded_nodes=None, show_labels=True):
    if excluded_nodes is None:
        excluded_nodes = []

    G = nx.DiGraph(adj_list)
    fig, ax = plt.subplots(figsize=(10, 12)) 
    pos = nx.spring_layout(G, k=0.25, seed=42)

    x = np.array([pos[i][0] for i in sorted(pos.keys())])
    y = np.array([pos[i][1] for i in sorted(pos.keys())])

    xi = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    yi = np.linspace(y.min() - 0.1, y.max() + 0.1, 200)
    xi, yi = np.meshgrid(xi, yi)

    categories = ['yellow', 'red', 'blue']
    category_masks = {
        'yellow': [i == 0 for i in range(len(node_categories))],
        'red': [1 <= i <= 166 for i in range(len(node_categories))],
        'blue': [167 <= i <= 333 for i in range(len(node_categories))]
    }

    colors = {
        'yellow': constants.COLOR_SCHEME.get('FABF', '#FF9292'), 
        'red': constants.COLOR_SCHEME.get('KSa', '#00DFA2'),     
        'blue': constants.COLOR_SCHEME.get('KSb', '#FFB433')     
    }
    colors = {
        'yellow': '#FF9292',
        'red': '#00DFA2',      
        'blue': '#FFB433'      
    }

    for category in categories:
        mask = category_masks[category]
        if sum(mask) > 0:  
            cat_x = x[mask]
            cat_y = y[mask]

            if len(cat_x) > 2:  
                positions = np.vstack([cat_x, cat_y])
                if category == 'yellow':
                    bw_method = 0.3
                else:
                    bw_method = 0.3

                kernel = gaussian_kde(positions, bw_method=bw_method)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
                zi = zi.reshape(xi.shape)

                if zi.max() > zi.min():
                    zi = (zi - zi.min()) / (zi.max() - zi.min())
                    zi = np.power(zi, 0.9)
                    zi = np.where(zi > 0.01, zi, 0)

                color_rgb = mcolors.hex2color(colors[category])

                if category == 'red':
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        f'{category}_cmap',
                        [(1, 1, 1, 0), (*color_rgb, 0.8), (*color_rgb, 0.9), (*color_rgb, 1.0)],
                        N=256
                    )
                    alpha = 1
                else:
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        f'{category}_cmap',
                        [(1, 1, 1, 0), (*color_rgb, 0.8)],
                        N=256
                    )
                    alpha = 0.8

                ax.contourf(xi, yi, zi, levels=15, cmap=cmap, alpha=alpha, linewidths=0)

    node_colors = []
    node_sizes = []

    for i in range(len(node_categories)):
        if i == 0:
            node_colors.append(colors['yellow'])
            node_sizes.append(250)
        elif 1 <= i <= 166:
            node_colors.append(colors['red'])
            node_sizes.append(120)
        elif 167 <= i <= 333:
            node_colors.append(colors['blue'])
            node_sizes.append(120)
        else:
            node_colors.append('#D3D3D3')
            node_sizes.append(120)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.7, ax=ax, edgecolors='white', linewidths=0.5)

    if 0 in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=300,
                             node_color=colors['yellow'], alpha=0.9, ax=ax,
                             edgecolors='white', linewidths=1.5)

    def rank_scale_weights(weights, min_thickness=0.5, max_thickness=3.0):
        values = np.array(list(weights.values()))
        if len(values) == 0: return {}
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val: return {k: min_thickness for k in weights}
        scaled_values = min_thickness + (max_thickness - min_thickness) * (values - min_val) / (max_val - min_val)
        return dict(zip(weights.keys(), scaled_values))

    edge_thickness_dict = rank_scale_weights(edge_weights)

    filtered_edge_weights = {edge: weight for edge, weight in edge_thickness_dict.items()
                           if edge[0] not in excluded_nodes and edge[1] not in excluded_nodes}

    node0_to_red_edges = [(s, t) for (s, t) in filtered_edge_weights.keys()
                         if s == 0 and 1 <= t <= 166]
    node0_to_blue_edges = [(s, t) for (s, t) in filtered_edge_weights.keys()
                          if s == 0 and 167 <= t <= 333]

    red_to_blue_edges = [(s, t) for (s, t) in filtered_edge_weights.keys()
                        if ((1 <= s <= 166 and 167 <= t <= 333) or
                            (167 <= s <= 333 and 1 <= t <= 166)) and
                           s not in selected_nodes and t not in selected_nodes]

    node0_to_red_edges = sorted(node0_to_red_edges, key=lambda e: filtered_edge_weights[e], reverse=True)[:10]
    node0_to_blue_edges = sorted(node0_to_blue_edges, key=lambda e: filtered_edge_weights[e], reverse=True)[:10]
    red_to_blue_edges = sorted(red_to_blue_edges, key=lambda e: filtered_edge_weights[e], reverse=True)[:5]

    key_edges = node0_to_red_edges + node0_to_blue_edges + red_to_blue_edges
    edge_color = '#C0C0C0'

    for edge in key_edges:
        source, target = edge
        source_pos = pos[source]
        target_pos = pos[target]
        thickness = filtered_edge_weights[edge]

        arrow = CurvedArrow(
            source_pos, target_pos,
            arrowstyle='-|>',
            color=edge_color,
            linewidth=thickness,
            alpha=0.8,
            mutation_scale=15,
            zorder=10
        )
        arrow.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.4)])
        ax.add_patch(arrow)

    if show_labels:
        labels = {i: str(i) for i in selected_nodes if i not in excluded_nodes}
        for node, label in labels.items():
            if node in pos:
                x_coord, y_coord = pos[node]
                font_size = 14 if node == 0 else 10
                txt = ax.text(x_coord, y_coord, label, fontsize=font_size, ha='center', va='center',
                             color='white', fontweight='bold', zorder=15)
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    legend_elements = [
        mpatches.Patch(color=colors['yellow'], label='FAB', alpha=0.9),
        mpatches.Patch(color=colors['red'], label='KS', alpha=0.9),
        mpatches.Patch(color=colors['blue'], label='CLF', alpha=0.9),
        plt.Line2D([0], [0], color=edge_color, marker='>', markersize=10,
                  label='FAB to KS (Top 10)', linestyle='-', linewidth=2),
        plt.Line2D([0], [0], color=edge_color, marker='>', markersize=10,
                  label='FAB to CLF (Top 10)', linestyle='-', linewidth=2),
        plt.Line2D([0], [0], color=edge_color, marker='>', markersize=10,
                  label='Between KS/CLF Groups (Top 5)', linestyle='-', linewidth=1.5)
    ]

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, fancybox=True, shadow=True, fontsize=9)

    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    title_suffix = "(With Labels)" if show_labels else "(Without Labels)"
    plt.title(f'Density Plot with Key Evolutionary Paths {title_suffix}', fontsize=14, pad=10)
    plt.show()

    print(f"\n--- Edge Analysis {title_suffix} ---")
    print("FAB to KS (Top 10):")
    for i, edge in enumerate(node0_to_red_edges):
        print(f"{i+1}. Edge {edge}: weight={edge_weights[edge]:.4f}")

if __name__ == "__main__":
   
    embeddings_file = constants.OUTPUT_DIR / "normalized_pca_embeddings.npy"
    new_edge_weights_file = constants.OUTPUT_DIR / "new_edge_weights_pca.pkl"

    print(f"Loading Embeddings: {embeddings_file}")
    print(f"Loading Edge Weights: {new_edge_weights_file}")

    try:
        embeddings = load_embeddings(embeddings_file)
        
        if not new_edge_weights_file.exists():
            raise FileNotFoundError(f"Can't find weight file: {new_edge_weights_file}")
            
        with open(new_edge_weights_file, "rb") as f:
            new_edge_weights = pickle.load(f)

        k = 15 
        selected_nodes = [0, 2, 10, 24, 47, 78, 98, 99, 110, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 276, 293, 295, 296, 304, 316, 130, 296, 44, 210, 11, 177, 161, 327, 34, 200, 35, 201, 134, 300]
        nodes_to_exclude = [293]

        knn_graph = build_knn_graph(embeddings, k)
        node_categories = list(range(len(knn_graph)))

        directed_edges = [(edge[0], edge[1]) for edge in new_edge_weights.keys()]
        edge_weights = {edge: weight for edge, weight in new_edge_weights.items()}


        print("\nGenerating dense graph with tags...")
        visualize_knn_graph_with_curved_vectors(knn_graph, node_categories, directed_edges, selected_nodes, edge_weights,
                                               excluded_nodes=nodes_to_exclude, show_labels=True)

        print("\n" + "="*50 + "\n")
        print("Generating dense graph without tags...")
        visualize_knn_graph_with_curved_vectors(knn_graph, node_categories, directed_edges, selected_nodes, edge_weights,
                                               excluded_nodes=nodes_to_exclude, show_labels=False)

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure the presence of 'embeddings_normalized_pca.npy' and 'new_edge_weights_pca.pkl' in the output directory.")