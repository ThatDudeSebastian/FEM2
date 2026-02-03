import os
import sys
# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from mesh_utils import load_mesh
import torch

def test_load_mesh():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file = os.path.join(script_dir, "src", "mesh", "Beam_Quad8.msh")
    
    if not os.path.exists(mesh_file):
        print(f"Skipping test, mesh file not found at {mesh_file}")
        return

    try:
        results = load_mesh(mesh_file, primary_element_type='quad8')
        print(f"load_mesh returned {len(results)} items.")
        if len(results) == 5:
            print("SUCCESS: load_mesh returned 5 values as expected.")
            x, conn, pt_sets, cell_sets, cells = results
            print(f"Nodes shape: {x.shape}")
            print(f"Connectivity shape: {conn.shape}")
            print(f"Point sets: {list(pt_sets.keys())}")
            print(f"Cell sets: {list(cell_sets.keys())}")
            print(f"Cells blocks: {len(cells)}")
        else:
            print(f"FAILURE: Expected 5 values, got {len(results)}")
    except Exception as e:
        print(f"FAILURE: load_mesh crashed with error: {e}")

if __name__ == "__main__":
    test_load_mesh()
