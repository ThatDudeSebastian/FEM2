import meshio
import os
import torch
import sys

# Add src to path so we can import mesh_utils if needed, 
# but for now let's just test meshio directly on the file
# and then test load_mesh from mesh_utils

sys.path.append("src")
try:
    from mesh_utils import load_mesh
except ImportError:
    print("Could not import mesh_utils. Make sure you run this from the project root.")

filename = "src/mesh/Rad.msh"

if not os.path.exists(filename):
    print(f"File {filename} not found.")
else:
    print(f"Attempting to read {filename} with meshio...")
    try:
        mesh = meshio.read(filename)
        print("Success reading with meshio!")
        print(f"Points: {mesh.points.shape}")
        for cell in mesh.cells:
            print(f"Cell type: {cell.type}, Count: {len(cell.data)}")
    except Exception as e:
        print(f"Failed to read with meshio: {e}")

    print("-" * 20)
    print(f"Attempting to read {filename} with mesh_utils.load_mesh...")
    try:
        # device='cpu' is default
        nodes, conns, point_sets, cell_sets = load_mesh(filename)
        print("Success reading with mesh_utils!")
        print(f"Nodes shape: {nodes.shape}")
        if isinstance(conns, dict):
            for k, v in conns.items():
                print(f"Connectivity '{k}': {v.shape}")
        else:
            print(f"Connectivity: {conns.shape}")
        
    except Exception as e:
        print(f"Failed to read with mesh_utils: {e}")
