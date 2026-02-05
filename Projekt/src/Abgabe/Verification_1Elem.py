import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import io
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# ============ IMPORT MATERIAL MODEL =======
# ==========================================
print("Importing Material Model from Diskretisierer_Balken.py...")

# Redirect stdout to suppress Diskretisierer_Balken output (if any remains)
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from Diskretisierer_Balken import von_mises_return, get_shape_data, E, nu, sigma_y, H, r
except ImportError as e:
    sys.stdout = old_stdout
    print(f"CRITICAL ERROR: Could not import from Diskretisierer_Balken.py: {e}")
    sys.exit(1)
finally:
    sys.stdout = old_stdout

print(f"SUCCESS. Material: E={E/1e9:.1f} GPa, sig_y={sigma_y/1e6:.1f} MPa, H={H/1e6:.1f} MPa, r={r}")

# ==========================================
# ============ CONFIGURATION ===============
# ==========================================
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")
width = 1.0  # Unit thickness

# Plane Strain Constants (Re-calculate to be safe or import if avail)
mu = E / (2.0 * (1.0 + nu))
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
K_mod = E / (3.0 * (1.0 - 2.0 * nu)) 

def clone_state(st):
    return {"ep": st["ep"].clone(), "k": float(st["k"]), "a": st["a"].clone()}

# ==========================================
# ============ 1-ELEMENT MESH ==============
# ==========================================
# 1x1 Square, Q8
# Nodes: Corners(0,1,2,3), Mids(4,5,6,7)
x = torch.tensor([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], 
    [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]
], dtype=torch.float64)

conn = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
nel = 1
nnp = 8
ndf = 2

# Gauss Points 3x3
nqp = 9
qpt = torch.zeros(nqp, 2); w8 = torch.zeros(nqp)
val_a = math.sqrt(3.0/5.0); w1 = 5.0/9.0; w2 = 8.0/9.0
vals = [-val_a, 0, val_a]; ws = [w1, w2, w1]
k=0
for i in range(3):
    for j in range(3):
        qpt[k,0] = vals[j]; qpt[k,1] = vals[i]
        w8[k] = ws[i] * ws[j]
        k+=1

# Initialize State
state_gp = [[{"ep": torch.zeros(3,3), "k": 0.0, "a": torch.zeros(3,3)} for _ in range(nqp)] for _ in range(nel)]

# ==========================================
# ============ LOADING PROTOCOL ============
# ==========================================
# Driven: Right Edge (x=1) -> Nodes 1, 2, 5
# Fixed: Left Edge (x=0) -> Nodes 0, 3, 7 (Fix X)
# Fix Rigid Body Y: Node 0 (Fix Y)

fixed_nodes_x = [0, 3, 7]
driven_nodes_x = [1, 2, 5]
fixed_nodes_y = [0] 

# Cyclic Loading: 0 -> +0.5% -> -0.5% -> +0.5%
max_strain = 0.005 
L0 = 1.0
u_max = max_strain * L0

# 1 Cycle: 0 -> max -> -max -> 0
t1 = np.linspace(0, u_max, 21)
t2 = np.linspace(u_max, -u_max, 81)[1:]
t3 = np.linspace(-u_max, 0, 41)[1:]
u_schedule = np.concatenate([t1, t2, t3])

results_eps_xx = []
results_sig_xx = []
results_sig_vm = []
results_eps_p_eq = []
results_u = []

u = torch.zeros(nnp*ndf, 1)

print("Starting 1-Element Verification...")
print(f"Total Steps: {len(u_schedule)}")

for step, u_target in enumerate(tqdm(u_schedule)):
    
    # Update BCs
    drlt_mask = torch.zeros(nnp*ndf, 1)
    drlt_val  = torch.zeros(nnp*ndf, 1)
    
    for n in fixed_nodes_x:
        drlt_mask[n*ndf+0] = 1.0; drlt_val[n*ndf+0] = 0.0
    for n in fixed_nodes_y:
        drlt_mask[n*ndf+1] = 1.0; drlt_val[n*ndf+1] = 0.0
    for n in driven_nodes_x:
        drlt_mask[n*ndf+0] = 1.0; drlt_val[n*ndf+0] = u_target
        
    free_dofs = torch.nonzero(drlt_mask < 0.5)[:, 0]
    
    # Enforce BCs
    u = u * (1.0 - drlt_mask) + drlt_val
    
    # Newton Loop
    state_gp_step_start = [[clone_state(s) for s in el_s] for el_s in state_gp]
    
    for it in range(20):
        K = torch.zeros(nnp*ndf, nnp*ndf)
        fint = torch.zeros(nnp*ndf, 1)
        state_gp_iter = [[None]*nqp for _ in range(nel)]
        
        for e in range(nel):
            n_idx = conn[e]
            edofs = []
            for n in n_idx: edofs.extend([int(n)*ndf, int(n)*ndf+1])
            edofs_t = torch.tensor(edofs, dtype=torch.long)
            
            xe = x[n_idx].t()
            ue = u.flatten()[edofs_t].reshape(-1, ndf).t()
            
            Ke_e = torch.zeros(2*nnp, 2*nnp)
            fe_e = torch.zeros(2*nnp, 1)
            
            for q in range(nqp):
                N, gamma = get_shape_data(qpt[q], nen=8)
                Je = xe.mm(gamma)
                detJ = torch.det(Je)
                dv = detJ * w8[q] * width
                G = torch.linalg.solve(Je.T, gamma.T).T
                
                eps2 = 0.5 * (ue.mm(G) + ue.mm(G).t())
                
                # Call Material Routine
                sig2, Ct2, st_new, s33 = von_mises_return(eps2, state_gp_step_start[e][q])
                state_gp_iter[e][q] = st_new
                
                # Assemble
                for A in range(nnp):
                    # Internal Force
                    # fe_A = sum(sig_ij * dN_Ax_j)
                    # Vectorized: sig @ G[A] is (2,1)
                    force_node = sig2 @ G[A].unsqueeze(1) 
                    fe_e[2*A : 2*A+2, 0] += dv * force_node.squeeze()
                    
                    for B in range(nnp):
                        # KAB = G[A]_i * C_ijkl * G[B]_l
                        KAB = torch.tensordot(G[A], torch.tensordot(Ct2, G[B], [[3],[0]]), [[0],[0]])
                        Ke_e[2*A : 2*A+2, 2*B : 2*B+2] += dv * KAB
            
            idx = edofs_t
            K[idx.unsqueeze(1), idx] += Ke_e
            fint[idx] += fe_e

        # Residual (No External Force except Reaction, but Solver solves Balance)
        # Since this is displacement driven, we only solve for Free DOFs.
        # R = f_ext - fint = 0 - fint (on free nodes)
        R = -fint
        Rf = R[free_dofs]
        
        if torch.norm(Rf) < 1e-6:
            state_gp = state_gp_iter
            break
            
        du = torch.linalg.solve(K[free_dofs][:, free_dofs], Rf)
        u[free_dofs] += du
        
    # Store Results (Center GP = 4)
    st_res = state_gp[0][4]
    
    # Recalc stress at center for plotting
    N0, gamma0 = get_shape_data(torch.tensor([0.0, 0.0]), nen=8)
    xe = x.t()
    Je = xe.mm(gamma0)
    G0 = torch.linalg.solve(Je.T, gamma0.T).T
    
    edofs = []
    for n in conn[0]: edofs.extend([int(n)*ndf, int(n)*ndf+1])
    ue = u.flatten()[torch.tensor(edofs, dtype=torch.long)].reshape(-1, 2).t()
    
    eps_center = 0.5 * (ue.mm(G0) + ue.mm(G0).t())
    sig_center, _, _, s33 = von_mises_return(eps_center, st_res)
    
    s11, s22, s12 = sig_center[0,0], sig_center[1,1], sig_center[0,1]
    svm = math.sqrt(
        s11**2 + s22**2 + s33.item()**2 
        - (s11*s22 + s22*s33.item() + s33.item()*s11) 
        + 3.0*s12**2
    )
    
    ep3 = st_res["ep"]
    ep_dev = ep3 - (torch.trace(ep3)/3.0)*torch.eye(3)
    ep_eq = math.sqrt((2.0/3.0) * torch.sum(ep_dev*ep_dev))
    
    if step == 20: # Peak Load (approx)
        u_peak = u.clone()
        svm_peak = svm
    
    results_u.append(u_target)
    results_sig_vm.append(svm)
    results_eps_p_eq.append(ep_eq)
    results_sig_xx.append(s11)
    results_eps_xx.append(eps_center[0,0].item())

# ==========================================
# ============ PLOTTING ====================
# ==========================================
print("Plotting results...")
results_sig_vm = np.array(results_sig_vm)
results_eps_p_eq = np.array(results_eps_p_eq)
results_sig_xx = np.array(results_sig_xx)
results_eps_xx = np.array(results_eps_xx)

# 1. Verification of Hardening Law
# Plot only the monotonic loading loading (first 21 steps) for cleaner comparison
n_mono = 21

plt.figure(figsize=(10, 6))
plt.plot(results_eps_p_eq[:n_mono], results_sig_vm[:n_mono]/1e6, 'b-', lw=3, label='Simulation (Monotonic)')
plt.plot(results_eps_p_eq, results_sig_vm/1e6, 'b--', lw=1, alpha=0.3, label='Simulation (Full Cycle)')

# Analytical Line
ep_range = np.linspace(0, results_eps_p_eq.max(), 100)
sig_anal = (sigma_y + H * ep_range) / 1e6
plt.plot(ep_range, sig_anal, 'r--', lw=2, label=f'Analytical (Sigma_y + {H/1e6:.0f}*ep)')

plt.xlabel("Equivalent Plastic Strain [-]")
plt.ylabel("Von Mises Stress [MPa]")
plt.title(f"Verification: Hardening Law (r={r})")
plt.legend()
plt.grid(True)
plt.savefig("Verify_Hardening.png", dpi=150)

# 2. Hysteresis Loop
plt.figure(figsize=(10, 6))
plt.plot(results_eps_xx, results_sig_xx/1e6, 'k-o', markersize=3, label="Sim S11-E11")
plt.xlabel("Strain E11 [-]")
plt.ylabel("Stress S11 [MPa]")
plt.title("Verification: Hysteresis (Bauschinger Effect)")
plt.grid(True)
plt.savefig("Verify_Hysteresis.png", dpi=150)

# 3. Element Visualization (Peak Load)
if 'u_peak' in locals():
    print("Plotting Element Visualization at Peak Load...")
    from matplotlib.collections import PolyCollection
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scale factor for displacement (exaggerated)
    scale = 10.0
    
    # Original coordinates (Corners for plotting Q8 as Quad)
    quad_nodes = [0, 1, 2, 3] 
    coords_orig = x[quad_nodes].numpy()
    
    # Deformed coordinates
    u_peak_resh = u_peak.reshape(-1, 2)
    coords_def = coords_orig + scale * u_peak_resh[quad_nodes].numpy()
    
    verts = [coords_def]
    
    # Plot 1: Displacement (Deformed Mesh)
    ax1.set_title(f"Deformed Shape (Scale {scale}x)", fontweight='bold')
    pc1 = PolyCollection(verts, edgecolors='k', facecolors='lightgray', alpha=0.8)
    ax1.add_collection(pc1)
    
    # Overlay original
    poly_orig = plt.Polygon(coords_orig, fill=False, edgecolor='k', linestyle='--', label='Original')
    ax1.add_patch(poly_orig)
    
    ax1.scatter(coords_orig[[0,3],0], coords_orig[[0,3],1], marker='>', color='g', s=100, label='Fixed X')
    ax1.quiver(coords_orig[[1,2],0], coords_orig[[1,2],1], 1, 0, color='r', scale=5, label='Driven X')
    
    ax1.set_xlim(-0.5, 1.5 + (u_peak_resh[:,0].max()*scale).item())
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Von Mises Stress Field
    ax2.set_title(f"Von Mises Stress [MPa] (Peak)", fontweight='bold')
    pc2 = PolyCollection(verts, cmap='jet', edgecolors='k')
    pc2.set_array(np.array([svm_peak/1e6])) 
    ax2.add_collection(pc2)
    plt.colorbar(pc2, ax=ax2, label="Stress [MPa]")
    
    ax2.set_xlim(-0.5, 1.5 + (u_peak_resh[:,0].max()*scale).item())
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.savefig("Verify_Element_Plot.png", dpi=150)
    print("Saved element plot to Verify_Element_Plot.png")

print("Saved plots to Verify_Hardening.png, Verify_Hysteresis.png, Verify_Element_Plot.png")
plt.xlabel("Strain E11 [-]")
plt.ylabel("Stress S11 [MPa]")
plt.title("Verification: Hysteresis (Bauschinger Effect)")
plt.grid(True)
plt.savefig("Verify_Hysteresis.png", dpi=150)

print("Saved plots to Verify_Hardening.png and Verify_Hysteresis.png")
