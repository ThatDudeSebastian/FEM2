import meshio
import matplotlib.pyplot as plt
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
        print(f"Loaded {filepath}")
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return

    points = mesh.points
    print(f"Points shape: {points.shape}")

    fig = plt.figure(figsize=(10, 8))
    
    # Check if 2D or 3D
    is_3d = points.shape[1] == 3 and not (points[:, 2] == 0).all()
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], s=1, c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

    # Plot elements (edges) if possible
    # This can be slow for large meshes, so maybe just points for now or a subset
    print("Plotting points...")
    
    plt.title(f"Mesh: {os.path.basename(filepath)}")
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.join(script_dir, "mesh", "Radausschnitt.msh")
    
    if len(sys.argv) > 1:
        file_to_plot = sys.argv[1]
    else:
        file_to_plot = default_file
        print(f"No file specified. Using default: {file_to_plot}")
    
    plot_mesh(file_to_plot)
