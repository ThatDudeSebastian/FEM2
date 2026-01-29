import gmsh
import sys
import os

def create_beam_mesh():
    gmsh.initialize()
    gmsh.model.add("Beam2D")

    # Dimensions
    L = 10.0
    H = 1.0
    
    # Mesh density (Number of nodes/elements along edges)
    # n_L elements along Length, n_H elements along Height
    n_L = 10
    n_H = 2

    # Points
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(L, 0, 0)
    p3 = gmsh.model.geo.addPoint(L, H, 0)
    p4 = gmsh.model.geo.addPoint(0, H, 0)

    # Lines
    l1 = gmsh.model.geo.addLine(p1, p2) # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3) # Right
    l3 = gmsh.model.geo.addLine(p3, p4) # Top
    l4 = gmsh.model.geo.addLine(p4, p1) # Left

    # Curve Loop and Surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Synchronization needed before meshing settings?
    # Usually transfinite settings apply to Geo entities.
    
    # Transfinite Meshing (Structured Grid)
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_L + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, n_H + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, n_L + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, n_H + 1)
    
    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    
    # Recombine to get Quads
    gmsh.model.geo.mesh.setRecombine(2, s)

    gmsh.model.geo.synchronize()

    # Physical Groups (Boundary Conditions)
    # Left edge (Fixed)
    gmsh.model.addPhysicalGroup(1, [l4], name="Fixed")
    # Right edge (Loaded)
    gmsh.model.addPhysicalGroup(1, [l2], name="Loaded")
    # Surface (Volume/Material)
    gmsh.model.addPhysicalGroup(2, [s], name="Beam")

    # Mesh Options
    gmsh.option.setNumber("Mesh.ElementOrder", 2) # Quadratic elements
    # Important: Enable Serendipity elements (8-node quads instead of 9-node)
    # 0 = complete (9-node for Q2), 1 = serendipity (8-node for Q2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1) 

    # Generate Mesh
    gmsh.model.mesh.generate(2)

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "src", "mesh")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "Beam_Quad8.msh")
    gmsh.write(output_file)
    print(f"Mesh saved to {output_file}")

    gmsh.finalize()

if __name__ == "__main__":
    create_beam_mesh()
