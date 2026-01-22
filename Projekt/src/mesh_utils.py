import meshio
import torch
import numpy as np
import os


def load_mesh(filepath, device="cpu", primary_element_type=None):
    """
    Loads a mesh from a file using meshio.
    Supports Gmsh (.msh), Abaqus (.inp), and other meshio-supported formats.

    Args:
        filepath (str): Path to the mesh file.
        device (str): Torch device ('cpu' or 'cuda').
        primary_element_type (str, optional): The element type to extract (e.g., 'line', 'quad', 'triangle').
                                              If None, returns a dictionary of all element types found.

    Returns:
        tuple: (nodes, connectivity)
            - nodes: torch.Tensor of shape (N, 2) or (N, 3)
            - connectivity: torch.Tensor of shape (Nel, Nen) or dict of such tensors
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    # Detect format before calling meshio to handle custom formats cleanly
    try:
        with open(filepath, "r") as f:
            first_line = f.readline()

        if first_line.startswith("MESH "):
            x, c = load_custom_mesh_format(filepath, device)
            # Return empty sets for custom format as it doesn't support them yet
            if primary_element_type:
                return x, c, {}, {}
            return x, {"custom": c}, {}, {}

        mesh = meshio.read(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to read mesh: {e}")

    # Extract nodes (coordinates)
    # We use double precision as in the existing code.
    # Return as (N, 2) if 2D, else (N, 3)
    dim = mesh.points.shape[1]
    if np.all(mesh.points[:, 2] == 0):
        nodes = torch.tensor(mesh.points[:, :2], device=device, dtype=torch.double)
    else:
        nodes = torch.tensor(mesh.points, device=device, dtype=torch.double)

    conns = {}
    for cell_block in mesh.cells:
        cell_type = cell_block.type
        data = torch.tensor(cell_block.data, device=device, dtype=torch.long)

        if cell_type in conns:
            conns[cell_type] = torch.cat((conns[cell_type], data), dim=0)
        else:
            conns[cell_type] = data

    # Extract sets (for boundary conditions)
    # cell_sets_dict: {set_name: {cell_type: [indices]}}
    cell_sets = mesh.cell_sets_dict
    # Point sets: {set_name: [indices]}
    point_sets = mesh.point_sets

    if primary_element_type:
        if primary_element_type in conns:
            return nodes, conns[primary_element_type], point_sets, cell_sets
        else:
            # Check if primary_element_type matches the custom format's elements
            # The custom format doesn't have multiple types, so we just return it
            # if the caller is asking for something like 'quad' or 'line'
            try:
                x, c = load_custom_mesh_format(filepath, device)
                return x, c, {}, {}
            except:
                available = list(conns.keys())
                raise ValueError(
                    f"Requested element type '{primary_element_type}' not found. Available: {available}"
                )

    return nodes, conns, point_sets, cell_sets


def get_geometry_from_file(filepath, element_type=None, device="cpu"):
    """
    Compatibility helper that returns nodes and connectivity,
    matching the signature of manual geometry functions.
    """
    x, conn, _, _ = load_mesh(
        filepath, device=device, primary_element_type=element_type
    )
    return x, conn


def get_bcs_from_sets(point_sets, set_name, dofs=[0, 1], value=0.0):
    """
    Converts a point set (from meshio) into the BC format used in the solver:
    [[node, dof, value], ...]
    """
    bcs = []
    if set_name in point_sets:
        node_indices = point_sets[set_name]
        for node in node_indices:
            for dof in dofs:
                bcs.append([int(node), int(dof), float(value)])
    return bcs


def load_custom_mesh_format(filepath, device="cpu"):
    """
    Parser for the custom format used in the source files:
    'MESH dimension 3 ElemType Quadrilateral Nnode 4'
    """
    nodes = []
    elements = []
    mode = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Coordinates"):
                mode = "nodes"
                continue
            elif line.startswith("Elements"):
                mode = "elements"
                continue
            elif line.startswith("End"):
                mode = None
                continue

            if mode == "nodes":
                parts = line.split()
                # Skip index (parts[0])
                nodes.append([float(parts[1]), float(parts[2])])
            elif mode == "elements":
                parts = line.split()
                # Skip index (parts[0])
                # Convert to 0-indexed (original is 1-indexed)
                elements.append([int(p) - 1 for p in parts[1:]])

    x = torch.tensor(nodes, device=device, dtype=torch.double)
    conn = torch.tensor(elements, device=device, dtype=torch.long)
    return x, conn


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "mesh", "Rad.msh")
    try:
        x, conn, pt_sets, cell_sets = load_mesh(path)
        print(f"Successfully loaded {path}")
        print(f"Nodes: {x.shape}")
        if isinstance(conn, dict):
            for k, v in conn.items():
                print(f"Elements ({k}): {v.shape}")
        else:
            print(f"Elements: {conn.shape}")
        print(f"Point sets: {list(pt_sets.keys())}")
        print(f"Cell sets: {list(cell_sets.keys())}")
    except Exception as e:
        print(f"Test failed: {e}")