import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from pathlib import Path


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

def main():
    print("=== Generating Grouped Chord Diagram) ===")

    input_file = constants.OUTPUT_DIR / "selected_edges_data.pkl"
    output_image_file = constants.OUTPUT_DIR / "network_grouped_chord_diagram_enhanced_width.png"

    if not input_file.exists():
        print(f"Error_File not found: {input_file}")
        return

    print(f"Loading: {input_file}")
    with open(input_file, 'rb') as f:
        selected_edges = pickle.load(f)
    print(f"Loading finished, including {len(selected_edges)} edges")

    selected_nodes_ref = [0, 2, 10, 24, 47, 78, 98, 99, 114, 122, 132, 138, 142, 145, 148, 209, 227, 275, 293, 295, 304, 316]
    
    original_total_weights = []
    edge_data = []
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

        if weight1 + weight2 > 0:
            color_value = np.abs((weight1 - weight2) / (weight1 + weight2))
        else:
            color_value = 0

        edge_data.append(((source, target), total_weight, color_value))

    existing_selected_nodes = set([node for node in selected_nodes_ref if node in existing_nodes])

    if not original_total_weights:
        print("No edges found")
        return

    weight_threshold = np.percentile(original_total_weights, 99)
    print(f"Top 1% edges: {weight_threshold:.6f}")
    
    high_weight_data = []
    for edge, weight, color_value in edge_data:
        if weight >= weight_threshold:
            high_weight_data.append((edge, weight, color_value))

    high_weight_nodes = set()
    for edge, _, _ in high_weight_data:
        high_weight_nodes.update(edge)

    print(f"Reserving {len(high_weight_data)} high-weighted edges，related {len(high_weight_nodes)} nodes")

    weights = [w for _, w, _ in high_weight_data]
    if not weights:
        print("Warning: not enough edges for visualization")
        return
        
    min_weight = min(weights)
    max_weight = max(weights)

    print("Generating...")
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
    if total_nodes == 0:
        return

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
                              alpha=0.9, zorder=3, edgecolor='white', linewidth=1)
            ax.add_patch(circle)

            if node in existing_selected_nodes:
                label_radius_node = radius - 0.15
                lx = label_radius_node * np.cos(angle)
                ly = label_radius_node * np.sin(angle)
                fontsize = 14 if node == 0 else 12
                ax.text(lx, ly, str(node), ha='center', va='center',
                       fontsize=fontsize, fontweight='bold', color='black',
                       zorder=4)

        angle_start += group_angle_span + 0.2

    edge_color_map = plt.cm.GnBu
    edge_color_normalized = mcolors.Normalize(vmin=0, vmax=1)

    for edge, weight, color_value in high_weight_data:
        source, target = edge
        if source in node_positions and target in node_positions:
            x1, y1, _ = node_positions[source]
            x2, y2, _ = node_positions[target]

            curvature = min(0.2 + weight / max_weight * 0.5, 0.8)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            control_x = mid_x * (1 - curvature)
            control_y = mid_y * (1 - curvature)

            t = np.linspace(0, 1, 100)
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2

            color = edge_color_map(edge_color_normalized(color_value))
            normalized_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0
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

    legend_elements = []
    for node_type, color in group_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         label=node_type, markerfacecolor=color,
                                         markersize=14, markeredgecolor='white'))
    
    legend_elements.append(plt.Line2D([0], [0], color=edge_color_map(0.7),
                                     linestyle='-', lw=4.0, alpha=0.9,
                                     label='Top 1% Edges'))

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.99, 1.03), fontsize=16, frameon=False)

    plt.title('Grouped Chord Diagram - Network Visualization by Node Types\n(Top 1% High Weight Edges with Enhanced Line Width)',
             fontsize=16, fontweight='bold', pad=30)

    print(f"saving fig to: {output_image_file}")
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show() 

if __name__ == "__main__":
    main()