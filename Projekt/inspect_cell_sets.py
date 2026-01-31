import meshio
import os
import numpy as np

filepath = "src/mesh/Wheel_Refined.msh"
if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
else:
    mesh = meshio.read(filepath)
    print("CELL SETS:")
    for name, block_masks in mesh.cell_sets.items():
        print(f"  Set: {name}")
        for i, mask in enumerate(block_masks):
            if isinstance(mask, np.ndarray):
                if mask.dtype == bool:
                    active = np.sum(mask)
                else:
                    active = len(mask)
                print(f"    Block {i} ({mesh.cells[i].type}): Type={mask.dtype}, Count={len(mask)}, Active={active}, CellBlockDataSize={len(mesh.cells[i].data)}")
            else:
                print(f"    Block {i} ({mesh.cells[i].type}): Mask is {type(mask)}")
