import gmsh
import sys
import os
import math

def create_wheel_mesh():
    gmsh.initialize()
    gmsh.model.add("Wheel2D")

    # Dimensions
    R_inner = 100.0  # Inner radius (hole)
    R_outer = 300.0  # Outer radius
    
    # Radial partitions (3 Layers)
    R1 = 150.0
    R2 = 250.0
    radii = [R_inner, R1, R2, R_outer]
    
    # Total span of the model: 60 degrees centered at bottom (-90 deg)
    angle_start = math.radians(-87.5)
    angle_end = math.radians(-92.5)
    angle_bottom = math.radians(-90)
    
    # Load area angles (exactly 5 nodes = 4 intervals)
    load_half_angle = 0.015 
    angle_load_right = angle_bottom + load_half_angle
    angle_load_left = angle_bottom - load_half_angle
    
    angles = [angle_start, angle_load_right, angle_load_left, angle_end]
    
    # Mesh size settings
    # We want it fine at the contact, and coarse at the hole.
    # "for every 2 elements at contact, there is 1 starting on second partition"
    # We will implement this by using the 'coef' and 'setTransfiniteCurve' thoughtfully.
    
    # Radial divisions
    # Outer layer (R2-R_outer): Very fine
    n_rad_outer = 16
    # Middle layer: Transitioning towards contact
    n_rad_middle = 8
    # Inner layer: Biased towards hole
    n_rad_inner = 5
    
    # Circumferential divisions
    # We'll keep the partitions but let's use the progression to coarse them inwards
    n_load = 10 # Nodes at the bottom contact strip
    n_side = 4
    
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
        for a_idx in range(3):
            row.append(gmsh.model.geo.addCircleArc(pts[r_idx][a_idx], center, pts[r_idx][a_idx+1]))
        circ_arcs.append(row)
        
    # Create Radial Lines
    rad_lines = []
    for a_idx in range(4):
        row = []
        for r_idx in range(3):
            row.append(gmsh.model.geo.addLine(pts[r_idx][a_idx], pts[r_idx+1][a_idx]))
        rad_lines.append(row)
        
    # Create Surfaces (3 Layers x 3 Sectors = 9)
    surfaces = []
    for r_idx in range(3): # Layers
        for a_idx in range(3): # Sectors
            l = gmsh.model.geo.addCurveLoop([
                circ_arcs[r_idx][a_idx], 
                rad_lines[a_idx+1][r_idx], 
                -circ_arcs[r_idx+1][a_idx], 
                -rad_lines[a_idx][r_idx]
            ])
            s = gmsh.model.geo.addPlaneSurface([l])
            surfaces.append(s)
            
    # Meshing Constraints
    for a_idx in range(4):
        # Layer 0: Inner - Bias towards R_inner (coef > 1 means growth away from start)
        gmsh.model.geo.mesh.setTransfiniteCurve(rad_lines[a_idx][0], n_rad_inner, coef=-1.2)
        # Layer 1: Middle - Bias towards contact part (R2)
        gmsh.model.geo.mesh.setTransfiniteCurve(rad_lines[a_idx][1], n_rad_middle, coef=1)
        # Layer 2: Outer - Inflation layer (fine towards R_outer)
        gmsh.model.geo.mesh.setTransfiniteCurve(rad_lines[a_idx][2], n_rad_outer, coef=0.85)
            
    # Circumferential
    for r_idx in range(4):
        gmsh.model.geo.mesh.setTransfiniteCurve(circ_arcs[r_idx][0], n_side)
        gmsh.model.geo.mesh.setTransfiniteCurve(circ_arcs[r_idx][1], n_load)
        gmsh.model.geo.mesh.setTransfiniteCurve(circ_arcs[r_idx][2], n_side)
        
    # Recombine for Quads
    for s in surfaces:
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)

    gmsh.model.geo.synchronize()
    
    # Physical Groups
    gmsh.model.addPhysicalGroup(1, circ_arcs[0], name="Fixed")
    gmsh.model.addPhysicalGroup(1, [circ_arcs[3][1]], name="Loaded")
    gmsh.model.addPhysicalGroup(2, surfaces, name="Wheel")
    
    # Mesh Options
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1) # Quad8
    
    gmsh.model.mesh.generate(2)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "mesh")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Radausschnitt_Quad8.msh")
    gmsh.write(output_file)
    print(f"Refined graded mesh saved to {output_file}")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_wheel_mesh()
