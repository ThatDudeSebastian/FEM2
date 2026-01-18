import sys
import os

print("Step 1: Importing meshio...")
import meshio
print("✓ meshio imported")

print("\nStep 2: Checking file exists...")
filepath = os.path.join(os.path.dirname(__file__), "mesh", "Rad_shell_mesh_HDF5.cgns")
print(f"Path: {filepath}")
print(f"Exists: {os.path.exists(filepath)}")
print(f"Size: {os.path.getsize(filepath)} bytes")

print("\nStep 3: Reading CGNS file...")
try:
    mesh = meshio.read(filepath)
    print("✓ File read successfully")
    
    print("\nStep 4: Extracting info...")
    print(f"Nodes: {mesh.points.shape}")
    print(f"Cell types: {[c.type for c in mesh.cells]}")
    for cell in mesh.cells:
        print(f"  {cell.type}: {cell.data.shape}")
    
    print(f"\nPoint sets: {list(mesh.point_sets.keys())}")
    print(f"Cell sets: {list(mesh.cell_sets_dict.keys())}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
