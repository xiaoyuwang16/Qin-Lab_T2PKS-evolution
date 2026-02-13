import networkx as nx
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
import random
from pathlib import Path
import sys
import constants 

def main():

    input_file = constants.OUTPUT_DIR / "selected_edges_data.pkl"
    output_image_file = constants.OUTPUT_DIR / "network_with_bidirectional_highlighted_edges.png"


    print(f"Loading data from: {input_file}")

    with open(input_file, 'rb') as f:
        selected_edges = pickle.load(f)

    G = nx.DiGraph()

    selected_nodes = [0, 2, 10, 24, 47, 78, 98, 99, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 293, 295, 304, 316]

    original_total_weights = []
    single_edges_data = []  
    double_edges_data = []  

    existing_nodes = set()

    for (query_index, result_index), data in selected_edges.items():
        existing_nodes.add(query_index)
        existing_nodes.add(result_index)

        weight1 = data['total_weight1']
        weight2 = data['total_weight2']
        total_weight = weight1 + weight2
        original_total_weights.append(total_weight)
        weight_diff = abs(weight1 - weight2)

        if weight1 + weight2 > 0:
            color_value = np.abs((weight1 - weight2) / (weight1 + weight2))
        else:
            color_value = 0

        if weight_diff > 0.5:
            if weight1 > weight2:
                source, target = result_index, query_index 
            else:
                source, target = query_index, result_index

            G.add_edge(source, target, weight=total_weight)

            single_edges_data.append(((source, target), total_weight, color_value, weight1, weight2))
        else:
            G.add_edge(query_index, result_index, weight=total_weight)
            G.add_edge(result_index, query_index, weight=total_weight)

            double_edges_data.append(((query_index, result_index), total_weight, color_value, weight1, weight2))
            double_edges_data.append(((result_index, query_index), total_weight, color_value, weight1, weight2))

    existing_selected_nodes = [node for node in selected_nodes if node in existing_nodes]

    print(f"Nodes in the graph: {len(existing_nodes)}")
    print(f"Selected nodes that exist in the graph: {existing_selected_nodes}")
    print(f"Total selected nodes: {len(selected_nodes)}")
    print(f"Count of selected nodes in graph: {len(existing_selected_nodes)}")

    single_weights = [weight for _, weight, _, _, _ in single_edges_data]
    double_weights = [weight for _, weight, _, _, _ in double_edges_data]

    single_threshold = np.percentile(single_weights, 99) if single_weights else 0
    double_threshold = np.percentile(double_weights, 99) if double_weights else 0

    single_high_weight_data = []
    double_high_weight_data = []

    for edge, weight, color_value, weight1, weight2 in single_edges_data:
        if weight >= single_threshold:
            single_high_weight_data.append((edge, weight, color_value, weight1, weight2))

    for edge, weight, color_value, weight1, weight2 in double_edges_data:
        if weight >= double_threshold:
            double_high_weight_data.append((edge, weight, color_value, weight1, weight2))

    def get_node_color(node):
        if node == 0:
            return '#fcba03'  
        elif 1 <= node <= 166:
            return '#00DFA2'  
        elif 167 <= node <= 333:
            return '#FFB433'  
        else:
            return '#CCCCCC'  

    def get_node_type(node):
        if node == 0:
            return 'FAB'
        elif 1 <= node <= 166:
            return 'KS'
        elif 167 <= node <= 333:
            return 'CLF'
        else:
            return 'Other'

    high_weight_nodes = set()
    for edge, _, _, _, _ in single_high_weight_data + double_high_weight_data:
        high_weight_nodes.update(edge)

    all_high_weights = [weight for _, weight, _, _, _ in single_high_weight_data + double_high_weight_data]
    min_weight = min(all_high_weights) if all_high_weights else 0
    max_weight = max(all_high_weights) if all_high_weights else 1
    print(f"Weight Range: {min_weight:.6f} - {max_weight:.6f}")

    def create_grouped_chord_diagram():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')

        node_groups = {'FAB': [], 'KS': [], 'CLF': []}
        for node in high_weight_nodes:
            node_type = get_node_type(node)
            if node_type in node_groups:
                node_groups[node_type].append(node)

        for group in node_groups.values():
            group.sort()

        total_nodes = sum(len(group) for group in node_groups.values())
        group_colors = {'FAB': '#fcba03', 'KS': '#00DFA2', 'CLF': '#FFB433'}  

        angle_start = 0
        node_positions = {}

        for group_name, nodes in node_groups.items():
            if not nodes:
                continue

            group_size = len(nodes)
            group_angle_span = (group_size / total_nodes) * 2 * np.pi * 0.9  

            if group_size > 1:
                node_angles = np.linspace(angle_start, angle_start + group_angle_span, group_size)
            else:
                node_angles = [angle_start + group_angle_span / 2]

            radius = 1.2

            arc_angles = np.linspace(angle_start, angle_start + group_angle_span, 100)
            arc_x = (radius + 0.1) * np.cos(arc_angles)
            arc_y = (radius + 0.1) * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, color=group_colors[group_name], linewidth=8, alpha=0.3, zorder=0)

            mid_angle = angle_start + group_angle_span / 2
            label_radius = radius + 0.35
            label_x = label_radius * np.cos(mid_angle)
            label_y = label_radius * np.sin(mid_angle)
            ax.text(label_x, label_y, group_name, ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=group_colors[group_name],
                            alpha=0.9, edgecolor='none'),  
                   zorder=5)

            for node, angle in zip(nodes, node_angles):
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                node_positions[node] = (x, y, angle)

                circle_size = 0.08 if node == 0 else 0.05
                circle = plt.Circle((x, y), circle_size, facecolor=group_colors[group_name],
                                  alpha=1, zorder=3, edgecolor='white', linewidth=1)
                ax.add_patch(circle)

                if node in existing_selected_nodes:
                    label_radius_node = radius - 0.15
                    label_x = label_radius_node * np.cos(angle)
                    label_y = label_radius_node * np.sin(angle)

                    fontsize = 14 if node == 0 else 12
                    ax.text(label_x, label_y, str(node), ha='center', va='center',
                           fontsize=fontsize, fontweight='bold', color='black',
                           zorder=4)

            angle_start += group_angle_span + 0.2  

        single_edge_color_map = plt.cm.Reds       
        double_edge_color_map = plt.cm.summer     
        edge_color_normalized = mcolors.Normalize(vmin=0, vmax=1)

        for edge, weight, color_value, weight1, weight2 in single_high_weight_data:
            source, target = edge
            if source in node_positions and target in node_positions:
                x1, y1, angle1 = node_positions[source]
                x2, y2, angle2 = node_positions[target]

                curvature = min(0.2 + weight / max_weight * 0.5, 0.8)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                control_x = mid_x * (1 - curvature)
                control_y = mid_y * (1 - curvature)

                t = np.linspace(0, 1, 100)
                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2

                color = single_edge_color_map(edge_color_normalized(color_value))

                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                linewidth = 2 + np.log10(1 + normalized_weight * 9) * 4

                alpha = 0.6 + color_value * 0.3

                ax.plot(curve_x, curve_y, color=color, linewidth=linewidth, alpha=alpha, zorder=1)

                arrow_scale = 0.015 + (linewidth / 2) * 0.025

                arrow_idx = int(len(curve_x) * 0.7)
                if arrow_idx < len(curve_x) - 1:
                    dx = curve_x[arrow_idx + 1] - curve_x[arrow_idx]
                    dy = curve_y[arrow_idx + 1] - curve_y[arrow_idx]
                    ax.arrow(curve_x[arrow_idx], curve_y[arrow_idx], dx*8, dy*8,
                            head_width=arrow_scale, head_length=arrow_scale, fc=color, ec=color,
                            alpha=alpha, zorder=2)

        for edge, weight, color_value, weight1, weight2 in double_high_weight_data:
            source, target = edge
            if source in node_positions and target in node_positions:
                x1, y1, angle1 = node_positions[source]
                x2, y2, angle2 = node_positions[target]

                curvature = min(0.2 + weight / max_weight * 0.5, 0.8)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                control_x = mid_x * (1 - curvature)
                control_y = mid_y * (1 - curvature)

                t = np.linspace(0, 1, 100)
                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2

                color = double_edge_color_map(edge_color_normalized(color_value))

                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                linewidth = 2 + np.log10(1 + normalized_weight * 9) * 4

                alpha = 0.6 + color_value * 0.3

                ax.plot(curve_x, curve_y, color=color, linewidth=linewidth, alpha=alpha, zorder=1)

        legend_elements = []

        for node_type, color in group_colors.items():
            if any(get_node_type(node) == node_type for node in high_weight_nodes):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                 label=node_type, markerfacecolor=color,
                                                 markersize=14, markeredgecolor='white'))

        legend_elements.append(plt.Line2D([0], [0], color=single_edge_color_map(0.7),
                                         linestyle='-', lw=4.0, alpha=0.9,
                                         label='Top 1% Single-Direction Edges'))
        legend_elements.append(plt.Line2D([0], [0], color=double_edge_color_map(0.7),
                                         linestyle='-', lw=4.0, alpha=0.9,
                                         label='Top 1% Double-Direction Edges'))

        ax.legend(handles=legend_elements, loc='center left',
                          bbox_to_anchor=(1.05, 0.5), fontsize=16,
                          frameon=False)  

        plt.title('Grouped Chord Diagram - Network Visualization\n(Top 1% Single vs Double Direction Edges)',
                 fontsize=16, fontweight='bold', pad=30)

        return fig, ax

    print("Generating Chord Diagram...")
    fig, ax = create_grouped_chord_diagram()

    sm_single = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_single.set_array([])
    cbar_ax_single = fig.add_axes([0.02, 0.55, 0.02, 0.3])
    cbar_single = plt.colorbar(sm_single, ax=ax, cax=cbar_ax_single)
    cbar_single.set_label('Single-Direction\nWeight Difference', fontsize=10)

    sm_double = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_double.set_array([])
    cbar_ax_double = fig.add_axes([0.02, 0.15, 0.02, 0.3])
    cbar_double = plt.colorbar(sm_double, ax=ax, cax=cbar_ax_double)
    cbar_double.set_label('Double-Direction\nWeight Difference', fontsize=10)

    print(f"Saving image to: {output_image_file}")
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\n=== Chord Diagram Statistics ===")
    print(f"Displayed Unidirectional High-Weight Edges: {len(single_high_weight_data)}")
    print(f"Displayed Bidirectional High-Weight Edges: {len(double_high_weight_data)}")
    print(f"Total Nodes Involved: {len(high_weight_nodes)}")

    all_weights_sorted = sorted(all_high_weights, reverse=True)
    print(f"\nWeight Distribution:")
    print(f"Max Weight: {all_weights_sorted[0]:.6f}")
    print(f"Min Weight: {all_weights_sorted[-1]:.6f}")
    print(f"Median Weight: {np.median(all_weights_sorted):.6f}")

    type_counts = {}
    for node in high_weight_nodes:
        node_type = get_node_type(node)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1


if __name__ == "__main__":
    main()
