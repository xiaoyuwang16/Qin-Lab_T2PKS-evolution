import sys
import pickle
import random
from pathlib import Path
import constants 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import numpy as np

def main():

    input_file = constants.OUTPUT_DIR / 'selected_edges_data.pkl'
    output_image = constants.OUTPUT_DIR / 'network_with_highlighted_edges.png'


    if not input_file.exists():
        print(f"Error_File not found: {input_file}")
        print("Check 'selected_edges_data_pca.pkl' in the output dir.")
        return

    print(f"Loading: {input_file} ...")
    with open(input_file, 'rb') as f:
        selected_edges = pickle.load(f)

    G = nx.DiGraph()

    selected_nodes_to_label = [0, 2, 10, 24, 47, 78, 98, 99, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 293, 295, 304, 316]

    original_total_weights = []
    edge_data = []  #(edge_tuple, total_weight, color_value)
    existing_nodes = set()

    for (query_index, result_index), data in selected_edges.items():
        existing_nodes.add(query_index)
        existing_nodes.add(result_index)

        weight1 = data['total_weight1']
        weight2 = data['total_weight2']
        total_weight = weight1 + weight2
        original_total_weights.append(total_weight)

        if weight1 > weight2:
            source, target = result_index, query_index
        else:
            source, target = query_index, result_index

        G.add_edge(source, target, weight=total_weight)

        if total_weight > 0:
            color_value = np.abs((weight1 - weight2) / total_weight)
        else:
            color_value = 0

        edge_data.append(((source, target), total_weight, color_value))

    existing_selected_nodes = [node for node in selected_nodes_to_label if node in existing_nodes]
    missing_nodes = set(selected_nodes_to_label) - set(existing_selected_nodes)

    print(f"Number of nodes: {len(existing_nodes)}")
    print(f"Nodes with tags: {len(existing_selected_nodes)}")
    if missing_nodes:
        print(f"Cautious:following nodes missing: {missing_nodes}")

    print("Calculating layout...")
    pos = nx.kamada_kawai_layout(G)


    color_scheme = {
        'yellow': '#fcba03',   
        'green': '#00DFA2',    
        'orange': '#FFB433',   
    }
    

    vector_labels = {
        'yellow': 'FAB',
        'green': 'KS',
        'orange': 'CLF'
    }


    node_colors = []
    graph_nodes_list = list(G.nodes())
    
    for node in graph_nodes_list:
        if node == 0:
            node_colors.append(color_scheme['yellow'])
        elif 1 <= node <= 166:
            node_colors.append(color_scheme['green'])
        elif 167 <= node <= 333:
            node_colors.append(color_scheme['orange'])
        else:
            node_colors.append('#808080') 

    min_node_size = 100
    max_node_size = 300
    node_sizes = [max(min_node_size, min(max_node_size, 20 * (1 + 0.8 * G.degree(node)))) for node in graph_nodes_list]


    edge_color_map = plt.cm.GnBu
    edge_color_normalized = mcolors.Normalize(vmin=0, vmax=1)
    arrow_size = 20
    arrow_style = '->'
    light_gray = '#D3D3D3'


    fig, ax = plt.subplots(figsize=(8, 8))


    if original_total_weights:
        weight_threshold = np.percentile(original_total_weights, 99)
    else:
        weight_threshold = 0
        
    print(f"Threshold for the top 1% of original total weights: {weight_threshold:.6f}")

    high_weight_data = []
    low_weight_data = []

    for edge, weight, color_value in edge_data:
        if weight >= weight_threshold:
            high_weight_data.append((edge, weight, color_value))
        else:
            low_weight_data.append((edge, weight, color_value))


    sample_size = max(1, int(len(low_weight_data) * 0.1)) if low_weight_data else 0
    sampled_low_weight_data = random.sample(low_weight_data, sample_size) if low_weight_data else []
    
    print(f"High edges sampled: {len(high_weight_data)}")
    print(f"Low edges sampled: {len(sampled_low_weight_data)} (Total {len(low_weight_data)})")

    sampled_low_weight_edges = [edge for edge, _, _ in sampled_low_weight_data]
    if sampled_low_weight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=sampled_low_weight_edges, 
                              edge_color=light_gray, alpha=0.2,
                              arrowsize=arrow_size * 0.8, arrowstyle=arrow_style, ax=ax,
                              node_size=node_sizes, node_shape='o',
                              connectionstyle=f'arc3, rad=0.1', width=1)

    high_weight_edges = [edge for edge, _, _ in high_weight_data]
    high_weight_colors = [edge_color_map(edge_color_normalized(color)) for _, _, color in high_weight_data]
    
    if high_weight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=high_weight_edges, 
                              edge_color=high_weight_colors, alpha=0.9,
                              arrowsize=arrow_size * 1.5, arrowstyle=arrow_style, ax=ax,
                              node_size=node_sizes, node_shape='o',
                              connectionstyle=f'arc3, rad=0.1', width=2.0)

    nx.draw_networkx_nodes(G, pos, nodelist=graph_nodes_list, 
                          node_size=node_sizes, node_color=node_colors, 
                          alpha=0.8, ax=ax, edgecolors='white', linewidths=1)

    labels = {node: str(node) for node in existing_selected_nodes}
    for node, label in labels.items():
        if node in pos:
            x, y = pos[node]
            fontsize = 12 if node == 0 else 9
            txt = ax.text(x, y, label, fontsize=fontsize,
                         ha='center', va='center',
                         color='white', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])


    sm = plt.cm.ScalarMappable(cmap=edge_color_map, norm=edge_color_normalized)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.95, 0.3, 0.03, 0.4])
    cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax)
    cbar.set_label('Edge Weight Difference (Absolute Ratio)', fontsize=12)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=vector_labels[color],
                                  markerfacecolor=color_scheme[color], markersize=10,
                                  markeredgecolor='white', markeredgewidth=1)
                       for color in color_scheme]
    
    legend_elements.append(plt.Line2D([0], [0], color=edge_color_map(0.7), linestyle='-', lw=3.0, alpha=0.9,
                                     label='Top 1% Edges'))
    legend_elements.append(plt.Line2D([0], [0], color=light_gray, linestyle='-', lw=1.0, alpha=0.2,
                                     label='10% Sampled Lower Weight Edges'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             bbox_to_anchor=(1.0, 0.0), frameon=False)

    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)

    print(f"Fig saved to: {output_image}")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()

    if high_weight_data:
        print("\nTop 10 highest weighted edges:")
        sorted_edges = sorted(edge_data, key=lambda x: x[1], reverse=True)
        for i, (edge, weight, _) in enumerate(sorted_edges[:10]):
            print(f"{i+1}. Edge {edge}: Weight = {weight:.6f}")

    high_weight_percentage = (len(high_weight_data) / len(edge_data)) * 100 if edge_data else 0
    print(f"\nPercentage of high-weighted edges relative to the total number of edges: {high_weight_percentage:.2f}%")

if __name__ == "__main__":
    main()
