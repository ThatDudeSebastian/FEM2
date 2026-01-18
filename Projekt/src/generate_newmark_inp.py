
import os

def create_inp(filename="newmark_task.inp", use_node_sets=True):
    # Parameters from Afg2_Newmark.py
    length = 9.0
    height = 0.6943
    nx = 20
    ny = 2
    
    # Grid sizes
    n_pts_x = 2 * nx + 1
    n_pts_y = 2 * ny + 1
    
    # 1. Generate Nodes
    # Flatten loop order: X outer, Y inner (column-major effectively for grid[i][j])
    # i goes 0..40, j goes 0..4
    
    # Steps
    dx = length / (n_pts_x - 1)
    dy = height / (n_pts_y - 1)
    
    nodes = []
    # Loop i (x) then j (y) to match meshgrid(..., indexing='ij').flatten()
    # verify: index = i * n_pts_y + j
    for i in range(n_pts_x):
        x = i * dx
        for j in range(n_pts_y):
            y = j * dy
            nodes.append((x, y))
            
    num_nodes = len(nodes)
    
    # Output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "HA4_src_task", filename)
    
    with open(output_path, "w") as f:
        f.write("*Heading\n")
        f.write(f"Generated from Newmark Task parameters (Sets={use_node_sets})\n")
        
        # Write Nodes
        f.write("*Node\n")
        for k, (x, y) in enumerate(nodes):
            # Abaqus uses 1-based indexing
            f.write(f"{k+1}, {x:.6f}, {y:.6f}, 0.0\n")
            
        # Write Elements (Q8 -> S8R as fallback for meshio)
        f.write("*Element, type=S8R\n")
        
        elem_id = 1
        num_nodes_y = n_pts_y
        
        for j in range(ny):
            for i in range(nx):
                # Node indices (0-based) from logic
                # logic: k = i_idx * num_nodes_y + j_idx
                
                # Corner nodes
                idx_bl = (2 * i) * num_nodes_y + (2 * j)          # Bottom-Left
                idx_br = (2 * (i + 1)) * num_nodes_y + (2 * j)    # Bottom-Right
                idx_tr = (2 * (i + 1)) * num_nodes_y + (2 * (j + 1)) # Top-Right
                idx_tl = (2 * i) * num_nodes_y + (2 * (j + 1))      # Top-Left
                
                # Midside nodes
                idx_b = (2 * i + 1) * num_nodes_y + (2 * j)       # Bottom mid
                idx_r = (2 * (i + 1)) * num_nodes_y + (2 * j + 1)   # Right mid
                idx_t = (2 * i + 1) * num_nodes_y + (2 * (j + 1))   # Top mid
                idx_l = (2 * i) * num_nodes_y + (2 * j + 1)       # Left mid
                
                node_ids = [idx_bl+1, idx_br+1, idx_tr+1, idx_tl+1, 
                            idx_b+1, idx_r+1, idx_t+1, idx_l+1]
                
                f.write(f"{elem_id}, " + ", ".join(map(str, node_ids)) + "\n")
                elem_id += 1
                
        # 4. Node Sets (Groups) - Optional but recommended
        if use_node_sets:
            # Set "Fixed": Nodes at x=0
            fixed_indices = [j + 1 for j in range(n_pts_y)] 

            # Set "Loaded": Nodes at right edge (x=L)
            start_right = (n_pts_x - 1) * n_pts_y
            loaded_indices = [start_right + j + 1 for j in range(n_pts_y)]

            f.write("*Nset, nset=Fixed\n")
            f.write(", ".join(map(str, fixed_indices)) + "\n")

            f.write("*Nset, nset=Loaded\n")
            f.write(", ".join(map(str, loaded_indices)) + "\n")

    print(f"Created {output_path} with {num_nodes} nodes and {elem_id-1} elements {'(mit Node Sets)' if use_node_sets else '(ohne Node Sets)'}.")

if __name__ == "__main__":
    # 1. Standard file (with sets)
    create_inp("newmark_task.inp", use_node_sets=True)
    
    # 2. Explicit variant WITH sets
    create_inp("newmark_task_with_sets.inp", use_node_sets=True)
    
    # 3. Explicit variant WITHOUT sets
    create_inp("newmark_task_no_sets.inp", use_node_sets=False)
