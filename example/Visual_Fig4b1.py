import networkx as nx
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patheffects as PathEffects
import random
from pathlib import Path
import sys

def main():

    input_file = constants.OUTPUT_DIR / "selected_edges_data.pkl"
    output_image_file = constants.OUTPUT_DIR / "network_with_bidirectional_highlighted_edges.png"

    if not input_file.exists():
        input_file = Path("output") / "selected_edges_data_pca.pkl"
        if not input_file.exists():
            print(f"Error: Input file '{input_file.name}' not found.")
            print("Please ensure 'selected_edges_data_pca.pkl' is in the script directory.")
            sys.exit(1)

    print(f"Loading data from: {input_file}")

    with open(input_file, 'rb') as f:
        selected_edges = pickle.load(f)

    G = nx.DiGraph()

    selected_nodes_ref = [0, 2, 10, 24, 47, 78, 98, 99, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 293, 295, 304, 316]

    existing_nodes = set()
    single_edges_data = []  # (edge, total_weight, color_value)
    double_edges_data = []  # (edge, total_weight, color_value)

    print("Processing edges...")

    for (query_index, result_index), data in selected_edges.items():
        existing_nodes.add(query_index)
        existing_nodes.add(result_index)
        
        weight1 = data['total_weight1']  
        weight2 = data['total_weight2']  
        total_weight = weight1 + weight2
        weight_diff = abs(weight1 - weight2)

        if weight1 + weight2 > 0:
            color_value = np.abs((weight1 - weight2) / (weight1 + weight2))
        else:
            color_value = 0

        if weight_diff > 0.5:
            if weight1 > weight2:
                G.add_edge(result_index, query_index, weight=total_weight)
                single_edges_data.append(((result_index, query_index), total_weight, color_value))
            else:
                G.add_edge(query_index, result_index, weight=total_weight)
                single_edges_data.append(((query_index, result_index), total_weight, color_value))
        else:
            G.add_edge(query_index, result_index, weight=total_weight)
            G.add_edge(result_index, query_index, weight=total_weight)
            double_edges_data.append(((query_index, result_index), total_weight, color_value))
            double_edges_data.append(((result_index, query_index), total_weight, color_value))

    existing_selected_nodes = [node for node in selected_nodes_ref if node in existing_nodes]
    missing_nodes = set(selected_nodes_ref) - set(existing_selected_nodes)

    print(f"Total unique nodes in graph: {len(existing_nodes)}")
    print(f"Selected nodes present: {existing_selected_nodes}")
    if missing_nodes:
        print(f"Selected nodes missing: {missing_nodes}")

    print("Calculating layout (Kamada-Kawai)...")
    pos = nx.kamada_kawai_layout(G)

    
    color_scheme = {
        'yellow': '#fcba03',  
        'red': '#00DFA2',     
        'blue': '#FFB433',   
    }

    vector_labels = {
        'yellow': 'FAB',
        'red': 'KS',
        'blue': 'CLF'
    }

    node_colors = []
    for node in G.nodes():
        if node == 0:
            node_colors.append(color_scheme['yellow'])
        elif 1 <= node <= 166:
            node_colors.append(color_scheme['red'])
        elif 167 <= node <= 333:
            node_colors.append(color_scheme['blue'])
        else:
            node_colors.append('#CCCCCC') 

    min_node_size = 100
    max_node_size = 300
    node_sizes = [max(min_node_size, min(max_node_size, 20 * (1 + 0.8 * G.degree(node)))) for node in G.nodes()]

    single_edge_color_map = plt.cm.Reds      
    double_edge_color_map = plt.cm.summer    
    edge_color_normalized = mcolors.Normalize(vmin=0, vmax=1)

    arrow_size = 20
    arrow_style = '->'
    light_gray = '#D3D3D3'

    fig, ax = plt.subplots(figsize=(8, 8))


    single_weights = [weight for _, weight, _ in single_edges_data]
    double_weights = [weight for _, weight, _ in double_edges_data]

    single_threshold = np.percentile(single_weights, 99) if single_weights else 0
    double_threshold = np.percentile(double_weights, 99) if double_weights else 0

    single_high_weight_data = []
    single_low_weight_data = []
    for edge, weight, color_value in single_edges_data:
        if weight >= single_threshold:
            single_high_weight_data.append((edge, weight, color_value))
        else:
            single_low_weight_data.append((edge, weight, color_value))

    double_high_weight_data = []
    double_low_weight_data = []
    for edge, weight, color_value in double_edges_data:
        if weight >= double_threshold:
            double_high_weight_data.append((edge, weight, color_value))
        else:
            double_low_weight_data.append((edge, weight, color_value))

    sampled_single_low = random.sample(single_low_weight_data, max(1, int(len(single_low_weight_data) * 0.1))) if single_low_weight_data else []
    sampled_double_low = random.sample(double_low_weight_data, max(1, int(len(double_low_weight_data) * 0.1))) if double_low_weight_data else []


    sampled_single_edges = [edge for edge, _, _ in sampled_single_low]
    sampled_double_edges = [edge for edge, _, _ in sampled_double_low]

    if sampled_single_edges:
        nx.draw_networkx_edges(G, pos, edgelist=sampled_single_edges, edge_color=light_gray, alpha=0.2,
                              arrowsize=arrow_size * 0.8, arrowstyle=arrow_style, ax=ax,
                              connectionstyle='arc3, rad=0.1', width=1)

    if sampled_double_edges:
        nx.draw_networkx_edges(G, pos, edgelist=sampled_double_edges, edge_color=light_gray, alpha=0.2,
                              arrowsize=arrow_size * 0.8, arrowstyle=arrow_style, ax=ax,
                              connectionstyle='arc3, rad=0.1', width=1)

    single_hw_edges = [edge for edge, _, _ in single_high_weight_data]
    single_hw_colors = [single_edge_color_map(edge_color_normalized(color)) for _, _, color in single_high_weight_data]

    if single_hw_edges:
        nx.draw_networkx_edges(G, pos, edgelist=single_hw_edges, edge_color=single_hw_colors, alpha=0.9,
                              arrowsize=arrow_size * 1.5, arrowstyle=arrow_style, ax=ax,
                              connectionstyle='arc3, rad=0.1', width=2.0)

    double_hw_edges = [edge for edge, _, _ in double_high_weight_data]
    double_hw_colors = [double_edge_color_map(edge_color_normalized(color)) for _, _, color in double_high_weight_data]

    if double_hw_edges:
        nx.draw_networkx_edges(G, pos, edgelist=double_hw_edges, edge_color=double_hw_colors, alpha=0.9,
                              arrowsize=arrow_size * 1.5, arrowstyle=arrow_style, ax=ax,
                              connectionstyle='arc3, rad=0.1', width=2.0)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax,
                          edgecolors='white', linewidths=1)

    labels = {node: str(node) for node in existing_selected_nodes}
    for node, label in labels.items():
        x, y = pos[node]
        fontsize = 12 if node == 0 else 9
        txt = ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',
                     color='white', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])


    sm_single = plt.cm.ScalarMappable(cmap=single_edge_color_map, norm=edge_color_normalized)
    sm_single.set_array([])
    cbar_ax_single = fig.add_axes([0.95, 0.55, 0.03, 0.3])
    cbar_single = plt.colorbar(sm_single, ax=ax, cax=cbar_ax_single)
    cbar_single.set_label('Single-Direction Edge\nWeight Difference', fontsize=10)

    sm_double = plt.cm.ScalarMappable(cmap=double_edge_color_map, norm=edge_color_normalized)
    sm_double.set_array([])
    cbar_ax_double = fig.add_axes([0.95, 0.15, 0.03, 0.3])
    cbar_double = plt.colorbar(sm_double, ax=ax, cax=cbar_ax_double)
    cbar_double.set_label('Double-Direction Edge\nWeight Difference', fontsize=10)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=vector_labels[key],
                                  markerfacecolor=color_scheme[key], markersize=10,
                                  markeredgecolor='white', markeredgewidth=1)
                       for key in color_scheme]

    legend_elements.append(plt.Line2D([0], [0], color=single_edge_color_map(0.7), linestyle='-', lw=3.0, alpha=0.9,
                                     label='Top 1% Single-Direction Edges'))
    legend_elements.append(plt.Line2D([0], [0], color=double_edge_color_map(0.7), linestyle='-', lw=3.0, alpha=0.9,
                                     label='Top 1% Double-Direction Edges'))
    legend_elements.append(plt.Line2D([0], [0], color=light_gray, linestyle='-', lw=1.0, alpha=0.2,
                                     label='10% Sampled Lower Weight Edges'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=False,
             bbox_to_anchor=(0.0, 1.0))

    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)

    print(f"Saving image to: {output_image_file}")
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
    
    plt.show()


    print("\n=== Top 10 High Weight Unidirectional Edges ===")
    print("Note: Arrow direction indicates Source -> Target")
    if single_high_weight_data:
        sorted_single = sorted(single_edges_data, key=lambda x: x[1], reverse=True)
        for i, (edge, weight, _) in enumerate(sorted_single[:10]):
            source, target = edge
            print(f"{i+1}. Edge {source} -> {target}: Weight = {weight:.6f}")

    print("\n=== Top 10 High Weight Bidirectional Edges ===")
    if double_high_weight_data:
        sorted_double = sorted(double_edges_data, key=lambda x: x[1], reverse=True)
        for i, (edge, weight, _) in enumerate(sorted_double[:10]):
            source, target = edge
            print(f"{i+1}. Edge {source} -> {target}: Weight = {weight:.6f}")

    print("\n=== Network Statistics ===")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {len(single_edges_data) + len(double_edges_data)}")
    print(f"Unidirectional Edges: {len(single_edges_data)}")
    print(f"Bidirectional Edges: {len(double_edges_data)}")


if __name__ == "__main__":
    main()