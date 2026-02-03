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
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
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


class HalfWheelOGrid(MeshGenerator):
    """
    True O-Grid implementation for half-wheel (lower half, open downwards).
    Uses a central rectangular block + 3 surrounding curved blocks.
    """
    def __init__(self, R_in=0.5, R_out=2.0, n_rad=10, n_tan=20, n_refine=30, core_ratio=0.7):
        super().__init__("HalfWheel_OGrid")
        self.R_in = R_in
        self.R_out = R_out
        self.n_rad = n_rad  # Radial elements in curved blocks
        self.n_tan = n_tan  # Tangential elements
        self.n_refine = n_refine  # Contact area refinement
        self.core_ratio = core_ratio  # Core size relative to R_in (e.g., 0.7 means core is at 0.7*R_in)

    def create_geometry(self):
        pc = self.model.geo.addPoint(0, 0, 0)  # Center
        
        # Core radius (inner boundary of O-Grid blocks)
        R_core = self.R_in * self.core_ratio
        
        # For simplicity, use 2 blocks (left and right halves)
        # This avoids the triangular distortion
        angles = [180, 270, 0]  # Left half, bottom, right half
        
        # --- Create Points ---
        # Core points (circular inner boundary)
        core_pts = []
        for angle in angles:
            rad = math.radians(angle)
            x = R_core * math.cos(rad)
            y = R_core * math.sin(rad)
            core_pts.append(self.model.geo.addPoint(x, y, 0))
        
        # Inner arc points (at R_in)
        inner_pts = []
        for angle in angles:
            rad = math.radians(angle)
            x = self.R_in * math.cos(rad)
            y = self.R_in * math.sin(rad)
            inner_pts.append(self.model.geo.addPoint(x, y, 0))
        
        # Outer arc points (at R_out)
        outer_pts = []
        for angle in angles:
            rad = math.radians(angle)
            x = self.R_out * math.cos(rad)
            y = self.R_out * math.sin(rad)
            outer_pts.append(self.model.geo.addPoint(x, y, 0))
        
        # --- Create Arcs on Core ---
        l_core = []
        for i in range(len(core_pts) - 1):
            l_core.append(self.model.geo.addCircleArc(core_pts[i], pc, core_pts[i+1]))
        
        # --- Create 2 O-Grid Blocks ---
        blocks = []
        
        for i in range(2):  # 2 sectors (left and right)
            # Radial lines from core to inner
            l_rad_core_inner_start = self.model.geo.addLine(core_pts[i], inner_pts[i])
            l_rad_core_inner_end = self.model.geo.addLine(core_pts[i+1], inner_pts[i+1])
            
            # Arc on inner radius
            l_arc_inner = self.model.geo.addCircleArc(inner_pts[i], pc, inner_pts[i+1])
            
            # Inner block (core to inner)
            cl_inner = self.model.geo.addCurveLoop([
                l_core[i],
                l_rad_core_inner_end,
                -l_arc_inner,
                -l_rad_core_inner_start
            ])
            s_inner = self.model.geo.addPlaneSurface([cl_inner])
            
            # Radial lines from inner to outer
            l_rad_inner_outer_start = self.model.geo.addLine(inner_pts[i], outer_pts[i])
            l_rad_inner_outer_end = self.model.geo.addLine(inner_pts[i+1], outer_pts[i+1])
            
            # Arc on outer radius
            l_arc_outer = self.model.geo.addCircleArc(outer_pts[i], pc, outer_pts[i+1])
            
            # Outer block (inner to outer)
            cl_outer = self.model.geo.addCurveLoop([
                l_arc_inner,
                l_rad_inner_outer_end,
                -l_arc_outer,
                -l_rad_inner_outer_start
            ])
            s_outer = self.model.geo.addPlaneSurface([cl_outer])
            
            blocks.append({
                'inner_surface': s_inner,
                'outer_surface': s_outer,
                'core_arc': l_core[i],
                'arc_inner': l_arc_inner,
                'arc_outer': l_arc_outer,
                'rad_core_inner': [l_rad_core_inner_start, l_rad_core_inner_end],
                'rad_inner_outer': [l_rad_inner_outer_start, l_rad_inner_outer_end]
            })
        
        # --- Apply Transfinite Meshing ---
        
        # Core arcs (tangential) - uniform distribution
        n_tan_half = max(10, self.n_tan // 2)
        for l in l_core:
            self.model.geo.mesh.setTransfiniteCurve(l, n_tan_half + 1)
        
        # Radial lines (constant for core-to-inner transition)
        n_rad_core = max(3, self.n_rad // 3)
        
        for i, block in enumerate(blocks):
            # Core to inner radial
            for l in block['rad_core_inner']:
                self.model.geo.mesh.setTransfiniteCurve(l, n_rad_core + 1)
            
            # Inner to outer radial
            for l in block['rad_inner_outer']:
                self.model.geo.mesh.setTransfiniteCurve(l, self.n_rad + 1)
            
            # Arcs (match core)
            self.model.geo.mesh.setTransfiniteCurve(block['arc_inner'], n_tan_half + 1)
            self.model.geo.mesh.setTransfiniteCurve(block['arc_outer'], n_tan_half + 1)
            
            # Apply transfinite to surfaces
            self.model.geo.mesh.setTransfiniteSurface(block['inner_surface'])
            self.model.geo.mesh.setRecombine(2, block['inner_surface'])
            self.model.geo.mesh.setTransfiniteSurface(block['outer_surface'])
            self.model.geo.mesh.setRecombine(2, block['outer_surface'])
        
        self.model.geo.synchronize()
        
        # --- Physical Groups ---
        all_core_arcs = l_core
        all_inner_arcs = [b['arc_inner'] for b in blocks]
        all_outer_arcs = [b['arc_outer'] for b in blocks]
        all_surfaces = [b['inner_surface'] for b in blocks] + [b['outer_surface'] for b in blocks]
        
        self.model.addPhysicalGroup(1, all_core_arcs, name="Core")
        self.model.addPhysicalGroup(1, all_inner_arcs, name="Inner")
        self.model.addPhysicalGroup(1, all_outer_arcs, name="Outer")
        self.model.addPhysicalGroup(1, [blocks[0]['rad_inner_outer'][0]], name="TopLeft")
        self.model.addPhysicalGroup(1, [blocks[1]['rad_inner_outer'][1]], name="TopRight")
        self.model.addPhysicalGroup(2, all_surfaces, name="HalfWheel")


if __name__ == "__main__":
    # Create O-Grid Half Wheel
    hw = HalfWheelOGrid(R_in=0.1, R_out=2.0, n_rad=30, n_tan=30, n_refine=30, core_ratio=0.7)
    hw.initialize()
    hw.create_geometry()
    hw.generate_mesh()
    hw.save("HalfWheel_OGrid_Quad8.msh")
    hw.finalize()
    
    print("Done. You can verify with:")
    print("  python src/plot_mesh.py src/mesh/HalfWheel_OGrid_Quad8.msh")
