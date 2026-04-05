import gmsh
import sys
import os
import math


def create_wheel_mesh():
    gmsh.initialize()
    gmsh.model.add("Wheel2D")

    # Dimensions
    R_inner = 100.0  # Inner radius (hole) in mm
    R_outer = 460.0  # Outer radius in mm

    # Radial partitions (3 Layers)
    R1 = 150.0
    R2 = 250.0
    radii = [R_inner, R1, R2, R_outer]

    # Total span of the model: right half only (symmetry)
    # 2.5 degrees from -87.5 to -90 (center/bottom)
    angle_start = math.radians(-87.5)
    angle_bottom = math.radians(-90)

    # Load area angles (exactly 5 nodes = 4 intervals)
    # Load area angles
    load_half_angle = 0.015
    angle_load_right = angle_bottom + load_half_angle

    # Must be strictly decreasing to define geometric ranges properly
    angles = [angle_start, angle_load_right, angle_bottom]

    # Mesh size settings
    # We want it fine at the contact, and coarse at the hole.
    # "for every 2 elements at contact, there is 1 starting on second partition"
    # We will implement this by using the 'coef' and 'setTransfiniteCurve' thoughtfully.

    # Radial divisions
    # Outer layer (R2-R_outer): Very fine
    n_rad_outer = 8
    # Middle layer: Transitioning middle part
    n_rad_middle = 8
    # Inner layer: Biased towards Hub
    n_rad_inner = 8

    # Circumferential divisions
    # n_load reduced roughly by half since symmetric
    n_load = 4  # Nodes on half the contact strip
    n_side = 3

    # Create center point
    center = gmsh.model.geo.addPoint(0, 0, 0)

    # Grid of points [Radius_Index][Angle_Index]
    pts = []
    for r_idx, r in enumerate(radii):
        row = []
        for a_idx, a in enumerate(angles):
            # We don't need MeshSize if using Transfinite, but we can't do 2:1
            # circumferential matching in a pure structured grid without hanging nodes.
            # So we will use the structured grid but with aggressive radial coarsening.
            p = gmsh.model.geo.addPoint(r * math.cos(a), r * math.sin(a), 0)
            row.append(p)
        pts.append(row)

    # Create Arcs
    circ_arcs = []
    for r_idx in range(4):
        row = []
        for a_idx in range(2):
            row.append(
                gmsh.model.geo.addCircleArc(
                    pts[r_idx][a_idx], center, pts[r_idx][a_idx + 1]
                )
            )
        circ_arcs.append(row)

    # Create Radial Lines
    rad_lines = []
    for a_idx in range(3):
        row = []
        for r_idx in range(3):
            row.append(gmsh.model.geo.addLine(pts[r_idx][a_idx], pts[r_idx + 1][a_idx]))
        rad_lines.append(row)

    # Create Surfaces (3 Layers x 2 Sectors = 6)
    surfaces = []
    for r_idx in range(3):  # Layers
        for a_idx in range(2):  # Sectors
            l = gmsh.model.geo.addCurveLoop(
                [
                    circ_arcs[r_idx][a_idx],
                    rad_lines[a_idx + 1][r_idx],
                    -circ_arcs[r_idx + 1][a_idx],
                    -rad_lines[a_idx][r_idx],
                ]
            )
            s = gmsh.model.geo.addPlaneSurface([l])
            surfaces.append(s)

    # Meshing Constraints
    for a_idx in range(3):
        # Layer 0: Inner - Bias towards R_inner (coef > 1 means growth away from start)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            rad_lines[a_idx][0], n_rad_inner, coef=1.25
        )
        # Layer 1: Middle - Bias towards contact part (R2)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            rad_lines[a_idx][1], n_rad_middle, coef=1
        )
        # Layer 2: Outer - Inflation layer (fine towards R_outer)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            rad_lines[a_idx][2], n_rad_outer, coef=0.8
        )

    # Circumferential
    for r_idx in range(4):
        gmsh.model.geo.mesh.setTransfiniteCurve(circ_arcs[r_idx][0], n_side)
        gmsh.model.geo.mesh.setTransfiniteCurve(circ_arcs[r_idx][1], n_load)

    # Recombine for Quads
    for s in surfaces:
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)

    gmsh.model.geo.synchronize()

    # Physical Groups
    gmsh.model.addPhysicalGroup(1, circ_arcs[0], name="Fixed")
    gmsh.model.addPhysicalGroup(1, [circ_arcs[3][1]], name="Loaded")
    gmsh.model.addPhysicalGroup(2, surfaces, name="Wheel")
    
    # Symmetry Boundary Group
    # The left-most boundary of our right-half domain is a_idx=2 (the bottom line at -90 deg)
    sym_lines = [rad_lines[2][i] for i in range(3)]
    gmsh.model.addPhysicalGroup(1, sym_lines, name="Symmetry")

    # Mesh Options
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # Quad8

    gmsh.model.mesh.generate(2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "mesh")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Radausschnitt_Quad8_coarse_v3.msh")
    gmsh.write(output_file)
    print(f"Refined graded mesh saved to {output_file}")

    gmsh.finalize()
    
    print("\nVisualizing setup plot for geometry...")
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from mesh_utils import load_mesh
    
    try:
        x, conn, pt_sets, cell_sets, mesh_cells = load_mesh(
            output_file, device=torch.device("cpu"), primary_element_type="quad8"
        )
        element_type = "quad8"
    except:
        x, conn, pt_sets, cell_sets, mesh_cells = load_mesh(
            output_file, device=torch.device("cpu"), primary_element_type="quad"
        )
        element_type = "quad4"

    x_np = x.detach().numpy()
    nnp, nel = x_np.shape[0], conn.shape[0]

    limit = float(np.max(np.abs(x_np)) * 1.1)
    if limit > 10.0:
        x_np = x_np * 1e-3
        limit = limit * 1e-3

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title(f"Radgeometrie (Half-Symmetry)\nNodes: {nnp} | Elements: {nel}", fontweight="bold")

    if element_type == "quad8":
        idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    else:
        idx = [0, 1, 2, 3, 0]
        
    for e in range(nel):
        n_idx = conn[e].numpy()
        ax1.plot(x_np[n_idx[idx], 0]*1000, x_np[n_idx[idx], 1]*1000, color="black", lw=0.4, alpha=0.15)
        
    ax1.set_aspect("auto")
    ax1.set_xlabel("x [mm]", fontweight="bold")
    ax1.set_ylabel("y [mm]", fontweight="bold")
    
    # Highlight fixed nodes
    fixed_nodes = set()
    for name, indices in pt_sets.items():
        if "Fixed" in name:
            fixed_nodes.update(np.array(indices).astype(int).tolist())
    for n in fixed_nodes:
        ax1.scatter(x_np[n,0]*1000, x_np[n,1]*1000, color="green", marker="o", s=15, alpha=0.5, zorder=5)

    # Highlight loaded surface
    loaded_nodes = set()
    for name, indices in pt_sets.items():
        if "Loaded" in name:
            loaded_nodes.update(np.array(indices).astype(int).tolist())
    for n in loaded_nodes:
        xn, yn = x_np[n,0]*1000, x_np[n,1]*1000
        norm = math.sqrt(xn*xn + yn*yn)
        vec = np.array([-xn, -yn]) / norm if norm > 0 else np.array([0, -1])
        arrow_length = 0.08 * limit * 1000
        gap = 0.02 * limit * 1000
        total_dist = gap + arrow_length
        ax1.arrow(
            xn - total_dist * vec[0],
            yn - total_dist * vec[1],
            arrow_length * vec[0],
            arrow_length * vec[1],
            head_width=0.015 * limit * 1000,
            head_length=0.02 * limit * 1000,
            fc="red",
            ec="red",
            zorder=6,
        )

    # Highlight symmetry line
    sym_nodes = list()
    for name, indices in pt_sets.items():
        if "Symmetry" in name:
            sym_nodes.extend(np.array(indices).astype(int).tolist())
    if len(sym_nodes) > 0:
        min_y = np.min(x_np[sym_nodes, 1]) * 1000
        max_y = np.max(x_np[sym_nodes, 1]) * 1000
        ax1.plot([0, 0], [min_y, max_y], color="cyan", linestyle="--", lw=2, zorder=4)
        
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], color='black', lw=1, alpha=0.5, label='FE Mesh'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, alpha=0.7, label='Fixed Hub Nodes'),
        mlines.Line2D([0], [0], color='red', marker='>', markersize=8, lw=0, label='Loaded Surface / Force direction')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_wheel_mesh()
