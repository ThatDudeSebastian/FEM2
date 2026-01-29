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

    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if legend_handles:
        ax.legend(handles=legend_handles)
    
    plt.title(f"Mesh: {os.path.basename(filepath)}\nNodes: {len(points)}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.join(script_dir, "mesh", "Beam_Quad8.msh")
    
    if len(sys.argv) > 1:
        file_to_plot = sys.argv[1]
    else:
        file_to_plot = default_file
        print(f"No file specified. Using default: {file_to_plot}")
    
    plot_mesh(file_to_plot)
