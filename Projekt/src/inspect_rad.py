import meshio
import os

try:
    mesh = meshio.read("mesh/Balken_9.msh")
    print("Mesh loaded successfully.")
    print(f"Point Sets: {list(mesh.point_sets.keys())}")
    print(f"Cell Sets: {list(mesh.cell_sets.keys())}")
    print("Cells:")
    for cell in mesh.cells:
        print(f"  Type: {cell.type}, Count: {len(cell.data)}, Shape: {cell.data.shape}")
except Exception as e:
    print(f"Error: {e}")
