import gmsh
import sys
import os
import math

class MeshGenerator:
    def __init__(self, name="Mesh"):
        self.name = name
        self.model = None

    def initialize(self):
        if not gmsh.is_initialized():
            gmsh.initialize()
        self.model = gmsh.model
        self.model.add(self.name)

    def finalize(self):
        if gmsh.is_initialized():
            gmsh.finalize()

    def generate_mesh(self, order=2, optimize=True):
        # Global mesh options
        # 2nd order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        # 1 = Serendipity (8-node quad), 0 = 9-node
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        
        # Generate 2D mesh
        self.model.mesh.generate(2)
        
        if optimize:
            self.model.mesh.optimize("Netgen")

    def save(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "src", "mesh")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_file = os.path.join(output_dir, filename)
        gmsh.write(output_file)
        print(f"Mesh saved to {output_file}")


class BeamGenerator(MeshGenerator):
    def __init__(self, L=10.0, H=1.0, n_L=10, n_H=2):
        super().__init__("Beam2D")
        self.L = L
        self.H = H
        self.n_L = n_L
        self.n_H = n_H

    def create_geometry(self):
        # Points
        p1 = self.model.geo.addPoint(0, 0, 0)
        p2 = self.model.geo.addPoint(self.L, 0, 0)
        p3 = self.model.geo.addPoint(self.L, self.H, 0)
        p4 = self.model.geo.addPoint(0, self.H, 0)

        # Lines
        l1 = self.model.geo.addLine(p1, p2) # Bottom
        l2 = self.model.geo.addLine(p2, p3) # Right
        l3 = self.model.geo.addLine(p3, p4) # Top
        l4 = self.model.geo.addLine(p4, p1) # Left

        # Surface
        cl = self.model.geo.addCurveLoop([l1, l2, l3, l4])
        s = self.model.geo.addPlaneSurface([cl])

        # Transfinite
        self.model.geo.mesh.setTransfiniteCurve(l1, self.n_L + 1)
        self.model.geo.mesh.setTransfiniteCurve(l2, self.n_H + 1)
        self.model.geo.mesh.setTransfiniteCurve(l3, self.n_L + 1)
        self.model.geo.mesh.setTransfiniteCurve(l4, self.n_H + 1)
        
        self.model.geo.mesh.setTransfiniteSurface(s)
        self.model.geo.mesh.setRecombine(2, s)

        self.model.geo.synchronize()

        # Physical Groups
        self.model.addPhysicalGroup(1, [l4], name="Fixed")
        self.model.addPhysicalGroup(1, [l2], name="Loaded")
        self.model.addPhysicalGroup(2, [s], name="Beam")


class HalfWheelGenerator(MeshGenerator):
    def __init__(self, R_in=0.5, R_out=2.0, n_rad=4, n_tan=12, n_refine=20):
        super().__init__("HalfWheel")
        self.R_in = R_in
        self.R_out = R_out
        self.n_rad = n_rad
        self.n_tan = n_tan
        self.n_refine = n_refine # Number of elements in the contact sector (260-280)

    def create_geometry(self):
        # Generate Lower Half-Wheel (180 to 360 degrees, or -180 to 0)
        # Partitioned into 3 Sectors for Contact Refinement:
        # 1. Left Leg: 180 (-180) to 260 (-100) degrees
        # 2. Contact Area: 260 (-100) to 280 (-80) degrees (Centered at 270/-90)
        # 3. Right Leg: 280 (-80) to 360 (0) degrees
        
        # Center
        pc = self.model.geo.addPoint(0, 0, 0)
        
        # --- Define Points ---
        
        # Angles in radians (for manual coordinate calculation if needed, 
        # but addCircleArc usually takes start/center/end points)
        # We need points at 180, 260, 280, 0 (360)
        
        def get_point_coords(angle_deg, radius):
            rad = math.radians(angle_deg)
            return radius * math.cos(rad), radius * math.sin(rad), 0
            
        # 180 deg
        p_in_180 = self.model.geo.addPoint(*get_point_coords(180, self.R_in))
        p_out_180 = self.model.geo.addPoint(*get_point_coords(180, self.R_out))
        
        # 260 deg
        p_in_260 = self.model.geo.addPoint(*get_point_coords(260, self.R_in))
        p_out_260 = self.model.geo.addPoint(*get_point_coords(260, self.R_out))
        
        # 280 deg
        p_in_280 = self.model.geo.addPoint(*get_point_coords(280, self.R_in))
        p_out_280 = self.model.geo.addPoint(*get_point_coords(280, self.R_out))
        
        # 360/0 deg
        p_in_0 = self.model.geo.addPoint(*get_point_coords(0, self.R_in))
        p_out_0 = self.model.geo.addPoint(*get_point_coords(0, self.R_out))
        
        
        # --- Sector 1: Left Leg (180 to 260) ---
        l_top_left = self.model.geo.addLine(p_out_180, p_in_180) # Top Horizontalish (-x axis)
        l_arc_in_left = self.model.geo.addCircleArc(p_in_180, pc, p_in_260)
        l_div_left = self.model.geo.addLine(p_in_260, p_out_260) # Divider at 260
        l_arc_out_left = self.model.geo.addCircleArc(p_out_180, pc, p_out_260)
        
        # Loop: p_out_180 -> p_in_180 -> p_in_260 -> p_out_260 -> p_out_180
        # l_top_left (+)
        # l_arc_in_left (+)
        # l_div_left (+)
        # l_arc_out_left (Reverse: Out_180 -> Out_260 is forward? No, Arc is CCW usually. 
        # Wait, Gmsh CircleArc is usually Start -> Center -> End (CCW).
        # 180 -> 260 is CCW. So l_arc_out_left is p_out_180 -> p_out_260.
        # Loop expects p_out_260 -> p_out_180. So Reverse (-l_arc_out_left).
        cl_1 = self.model.geo.addCurveLoop([l_top_left, l_arc_in_left, l_div_left, -l_arc_out_left])
        s_1 = self.model.geo.addPlaneSurface([cl_1])
        
        
        # --- Sector 2: Contact Area (260 to 280) ---
        l_arc_in_contact = self.model.geo.addCircleArc(p_in_260, pc, p_in_280)
        l_div_right = self.model.geo.addLine(p_in_280, p_out_280) # Divider at 280
        l_arc_out_contact = self.model.geo.addCircleArc(p_out_260, pc, p_out_280)
        
        # Loop: p_in_260 -> p_in_280 -> p_out_280 -> p_out_260 -> p_in_260
        # l_arc_in_contact (+)
        # l_div_right (+)
        # l_arc_out_contact (Reverse: Out_260->Out_280 is CCW. Loop needs Out_280->Out_260. So -)
        # l_div_left (Reverse: In_260->Out_260. Loop needs Out_260->In_260. So -)
        cl_2 = self.model.geo.addCurveLoop([l_arc_in_contact, l_div_right, -l_arc_out_contact, -l_div_left])
        s_2 = self.model.geo.addPlaneSurface([cl_2])
        
        
        # --- Sector 3: Right Leg (280 to 360) ---
        l_arc_in_right = self.model.geo.addCircleArc(p_in_280, pc, p_in_0)
        l_top_right = self.model.geo.addLine(p_in_0, p_out_0) # Top Horizontalish (+x axis)
        l_arc_out_right = self.model.geo.addCircleArc(p_out_280, pc, p_out_0)
        
        # Loop: p_in_280 -> p_in_0 -> p_out_0 -> p_out_280 -> p_in_280
        # l_arc_in_right (+)
        # l_top_right (+)
        # l_arc_out_right (Reverse: Out_280->Out_0. Loop needs Out_0->Out_280. So -)
        # l_div_right (Reverse: In_280->Out_280. Loop needs Out_280->In_280. So -)
        cl_3 = self.model.geo.addCurveLoop([l_arc_in_right, l_top_right, -l_arc_out_right, -l_div_right])
        s_3 = self.model.geo.addPlaneSurface([cl_3])
        
        
        # --- Transfinite Meshing & Refinement ---
        
        # Radial lines: l_top_left, l_div_left, l_div_right, l_top_right
        # Constant radial density
        for l in [l_top_left, l_div_left, l_div_right, l_top_right]:
             self.model.geo.mesh.setTransfiniteCurve(l, self.n_rad + 1)
             
        # Tangential lines
        
        # Sector 1 (Left Leg): 80 degrees span
        n_tan_leg = max(5, int(self.n_tan * (80/180))) 
        for l in [l_arc_in_left, l_arc_out_left]:
            # Bias towards end (260 deg)
            self.model.geo.mesh.setTransfiniteCurve(l, n_tan_leg + 1, "Progression", 0.9)
            
        # Sector 2 (Contact): 20 degrees span
        # Use explicit n_refine parameter
        print(f"Refining contact area with {self.n_refine} elements.")
        for l in [l_arc_in_contact, l_arc_out_contact]:
            self.model.geo.mesh.setTransfiniteCurve(l, self.n_refine + 1)
            
        # Sector 3 (Right Leg): 80 degrees span
        # Bias towards start (280 deg)
        for l in [l_arc_in_right, l_arc_out_right]:
            self.model.geo.mesh.setTransfiniteCurve(l, n_tan_leg + 1, "Progression", 1/0.9)
            
            
        # Surface Transfinite
        for s in [s_1, s_2, s_3]:
            self.model.geo.mesh.setTransfiniteSurface(s)
            self.model.geo.mesh.setRecombine(2, s)
            
        self.model.geo.synchronize()
        
        # --- Physical Groups ---
        self.model.addPhysicalGroup(1, [l_arc_in_left, l_arc_in_contact, l_arc_in_right], name="Inner")
        self.model.addPhysicalGroup(1, [l_arc_out_left, l_arc_out_contact, l_arc_out_right], name="Outer")
        # Left Top Edge
        self.model.addPhysicalGroup(1, [l_top_left], name="TopLeft")
        # Right Top Edge
        self.model.addPhysicalGroup(1, [l_top_right], name="TopRight")
        
        self.model.addPhysicalGroup(2, [s_1, s_2, s_3], name="HalfWheel")


if __name__ == "__main__":
    # Example usage
    
    # 1. Create Beam
    bg = BeamGenerator()
    bg.initialize()
    bg.create_geometry()
    bg.generate_mesh()
    bg.save("Beam_Quad8.msh")
    bg.finalize()
    
    # 2. Create Half Wheel
    # n_tan should be even ideally
    hw = HalfWheelGenerator(R_in=0.1, R_out=2.0, n_rad=30, n_tan=30, n_refine=30)
    hw.initialize()
    hw.create_geometry()
    hw.generate_mesh()
    hw.save("HalfWheel_Quad8.msh")
    hw.finalize()
    
    # Update default plot to confirm
    print("Done. You can verify with:")
    print("  python src/plot_mesh.py src/mesh/HalfWheel_Quad8.msh")
