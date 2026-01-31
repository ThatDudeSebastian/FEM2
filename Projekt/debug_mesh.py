import meshio
import os

filepath = "src/mesh/Wheel_Refined.msh"
if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
else:
    mesh = meshio.read(filepath)
    print("CELL SETS:")
    for key in mesh.cell_sets:
        print(f"  {key}")
    print("\nPOINT SETS:")
    for key in mesh.point_sets:
        print(f"  {key}")
    
    print("\nCELL BLOCKS:")
    for block in mesh.cells:
        print(f"  Type: {block.type}, Cells: {len(block.data)}")
