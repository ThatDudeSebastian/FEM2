import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import argparse

def plot_mesh(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    try:
        mesh = meshio.read(filepath)
        print(f"Loaded {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return

    points = mesh.points
    print(f"Nodes: {points.shape[0]}")
    
    # Check if we can just use X and Y (if Z is negligible)
    if points.shape[1] >= 2:
        x, y = points[:, 0], points[:, 1]
    else:
        print("Error: Mesh points do not have at least 2 dimensions.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    legend_handles = []
    
    print("\nElement Blocks:")
    
    has_plotted_elements = False
    
    for i, cell_block in enumerate(mesh.cells):
        cell_type = cell_block.type
        data = cell_block.data
        count = len(data)
        print(f"  - Type: {cell_type:<15} Count: {count}")
        
        color = colors[i % len(colors)]
        
        # 2D Elements
        if cell_type in ['triangle', 'triangle3', 'triangle6', 'quad', 'quad4', 'quad8', 'quad9']:
            # For higher order elements, just take the corner nodes for plotting the patch
            if cell_type in ['triangle6']:
                plot_conn = data[:, :3]
            elif cell_type in ['quad8', 'quad9']:
                plot_conn = data[:, :4]
            else:
                plot_conn = data
                
            polys = [points[elem_idx, :2] for elem_idx in plot_conn]
            pc = PolyCollection(polys, facecolors=color, edgecolors='k', linewidths=0.5, alpha=0.4, label=cell_type)
            ax.add_collection(pc)
            
            import matplotlib.patches as mpatches
            legend_handles.append(mpatches.Patch(color=color, alpha=0.4, label=f"{cell_type} ({count})"))
            has_plotted_elements = True
            
        # 1D Elements (Lines)
        elif cell_type in ['line', 'line3']:
             # Could plot lines if needed, but often clutter. Let's plot them if they are the ONLY thing or requested.
             # For now, maybe just scatter points for boundaries?
             # Let's plot lines just in case they are important (trusses etc)
             pass 

    if not has_plotted_elements:
        print("No standard 2D elements found to plot (triangle/quad). Plotting node scatter only.")
        ax.scatter(x, y, s=2, c='k', marker='.')
    
    # --- EXTRACT ALL BC NODE SETS ---
    # Merge from point_sets and cell_sets (where 1D elements are tagged)
    all_bc_sets = {}
    
    # 1. Start with explicit point sets
    for name, indices in mesh.point_sets.items():
        all_bc_sets[name] = set(indices)
        
    # 2. Add from cell sets (e.g. physical groups on lines)
    for name, block_masks in mesh.cell_sets.items():
        node_indices = set()
        # block_masks is a list of masks/indices, one for each cell block
        for block_idx, mask in enumerate(block_masks):
            if mask is None or len(mask) == 0:
                continue
                
            # Check if mask is an array-like object
            import numpy as np
            mask_arr = np.array(mask)
            
            if mask_arr.size == 0:
                continue
                
            block = mesh.cells[block_idx]
            
            try:
                if mask_arr.dtype == bool:
                    # Boolean mask: must match block length
                    if len(mask_arr) != len(block.data):
                        # Some versions might have global masks, skip or handle
                        continue
                    if any(mask_arr):
                        tagged_nodes = block.data[mask_arr].flatten()
                        node_indices.update(tagged_nodes)
                else:
                    # Integer index array: must be within bounds [0, len(block.data)-1]
                    valid_mask = mask_arr[mask_arr < len(block.data)]
                    if len(valid_mask) > 0:
                        tagged_nodes = block.data[valid_mask].flatten()
                        node_indices.update(tagged_nodes)
            except Exception as e:
                print(f"    Warning: Could not process mask for group '{name}' in block {block_idx}: {e}")
                continue
        
        if node_indices:
            if name in all_bc_sets:
                all_bc_sets[name].update(node_indices)
            else:
                all_bc_sets[name] = node_indices

    # filter to only Fixed and Loaded as requested
    filtered_bc_sets = {}
    for name, nodes in all_bc_sets.items():
        if name.lower() in ['fixed', 'loaded']:
            filtered_bc_sets[name] = nodes
    all_bc_sets = filtered_bc_sets

    # --- PLOT BOUNDARY CONDITIONS ---
    bc_markers = {'Fixed': 'ro', 'Loaded': 'y^', 'Support': 'rs', 'Einspannung': 'rd'}
    
    print("\nBoundary Conditions (Filtered: Fixed, Loaded):")
    for set_name, nodes in all_bc_sets.items():
        node_indices = list(nodes)
        if len(node_indices) == 0:
            continue
            
        print(f"  - Set: {set_name:<15} Nodes: {len(node_indices)}")
        
        # Determine format
        fmt = 'ko' # Default black dots
        marker_label = set_name
        
        for key, style in bc_markers.items():
            if key.lower() in set_name.lower():
                fmt = style
                break
                
        # Sub-sample if too many nodes for visualization clarity
        plot_indices = node_indices
        if len(node_indices) > 500:
            import numpy as np
            plot_indices = np.random.choice(node_indices, 500, replace=False)
            print(f"    (Visualizing 500/{len(node_indices)} nodes for '{set_name}')")

        ax.plot(x[plot_indices], y[plot_indices], fmt, markersize=4, label=f"BC: {set_name}", zorder=10)
        
        # Add to handles for legend (ax.plot already handles this if label exists)
    
    ax.autoscale()
    ax.set_aspect('auto')
    ax.set_xlabel('X in mm')
    ax.set_ylabel('Y in mm')
    
    # Update legend to include BCs
    # Handles from PC patches + plot labels
    handles, labels = ax.get_legend_handles_labels()
    # Unique labels only
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    
    # Place legend outside right
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.title(f"Mesh: {os.path.basename(filepath)}\nNodes: {len(points)}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    print("\nDisplaying plot...")
    plt.savefig(os.path.join(script_dir, "pre_mesh.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Fix case sensitivity and default path
    default_file = os.path.join(script_dir, "mesh", "Radausschnitt_Quad8.msh")
    
    if len(sys.argv) > 1:
        file_to_plot = sys.argv[1]
    else:
        file_to_plot = default_file
        if not os.path.exists(file_to_plot):
             # Try other common names
             alt = os.path.join(script_dir, "mesh", "Beam_Quad8.msh")
             if os.path.exists(alt): file_to_plot = alt

        print(f"No file specified. Using: {file_to_plot}")
    
    plot_mesh(file_to_plot)
