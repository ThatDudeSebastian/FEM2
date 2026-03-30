
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PolyCollection
import numpy as np
import math
import os
from tqdm import tqdm
from mesh_utils import load_mesh, get_bcs_from_sets
import time

try:
    import fem_postprocessing as fp
    MODULAR_POST = True
except ImportError:
    MODULAR_POST = False

# ============ SETTINGS & CONFIG ===========
# SI-Einheiten
torch.set_default_dtype(torch.float64)
device = torch.device("cpu") 

interactive_hover = True 

# --- Mesh Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file = os.path.abspath(os.path.join(script_dir, "mesh", "Radausschnitt_Quad8.msh"))

print(f"Loading mesh from: {mesh_file}")

# --- Material Parameters ---
E = 210e9 # [Pa]
nu = 0.25


sigma_y = 350e6    # Fließspannung [Pa]
H = 209e7          # Gesamt-Verfestigungsmodul (H_iso + H_kin) [Pa]
r = 0.35            # Faktor der Mischung (0=rein kinematisch, 1=rein isotrop)

# --- Force ---
F_total = -5.3e4     # Normalkraft [N]
a_hz = 0.0035        # erste Halbachse nach Knothe [m]
b_hz = 0.0035 # zweite Halbachse nach Knothe [m]

p0_hz = (2.0 * F_total) / (math.pi * a_hz * b_hz) # [Pa]

def p_hertz(s):
    if abs(s) >= a_hz: return 0.0
    return p0_hz * math.sqrt(max(0.0, 1.0 - (s / a_hz) ** 2))

# --- Cyclic force loading ---
n_cycles = 1.0
n_steps_per_cycle = 20
n_steps = n_cycles * n_steps_per_cycle
F_amp = F_total
use_ramp_in = True
n_ramp_steps = 5

newton_max = 200 # Increased for Modified Newton
newton_tol = 1e-4

# ==========================================
# ============ MESH LOADING ============
# ==========================================
try:
    x, conn, pt_sets, cell_sets, mesh_cells = load_mesh(mesh_file, device=device, primary_element_type="quad8")
    element_type = "quad8"
except:
    x, conn, pt_sets, cell_sets, mesh_cells = load_mesh(mesh_file, device=device, primary_element_type="quad")
    element_type = "quad4"

Lx_val = float((x[:, 0].max() - x[:, 0].min()).item())
if Lx_val > 10.0:
    print("UNIT INFO: Converting from mm to m.")
    x = x * 1e-3

nnp, nel, nen = x.shape[0], conn.shape[0], conn.shape[1]
ndf = 2 

# ==========================================
# ============ BOUNDARY CONDITIONS ========
# ==========================================
def extract_nodes_from_sets(mesh_pt_sets, mesh_cell_sets, mesh_cells, target_names):
    node_indices = set()
    for name, indices in mesh_pt_sets.items():
        if any(tn.lower() in name.lower() for tn in target_names):
            node_indices.update(np.array(indices).astype(int).tolist())
    if mesh_cell_sets:
        for set_name, block_masks in mesh_cell_sets.items():
            if any(tn.lower() in set_name.lower() for tn in target_names):
                for b_idx, mask in enumerate(block_masks):
                    if mask is not None and len(mask) > 0 and b_idx < len(mesh_cells):
                        mask_arr = np.array(mask)
                        block_data = mesh_cells[b_idx].data
                        if mask_arr.dtype == bool:
                            if len(mask_arr) == len(block_data): node_indices.update(block_data[mask_arr].flatten())
                        else:
                            try:
                                indices = mask_arr.astype(int)
                                node_indices.update(block_data[indices].flatten())
                            except: pass
    return list(node_indices)

drlt_bcs = []
neum_bcs = []

fixed_nodes = extract_nodes_from_sets(pt_sets, cell_sets, mesh_cells, ["Fixed", "Support", "Hub", "Innen"])
if not fixed_nodes:
    r_dist = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    min_r = torch.min(r_dist)
    fixed_nodes = torch.where(torch.abs(r_dist - min_r) < 1e-4)[0]
else:
    fixed_nodes = torch.tensor(fixed_nodes, dtype=torch.long)

for n in fixed_nodes:
    drlt_bcs.append([int(n), 0, 0.0])
    drlt_bcs.append([int(n), 1, 0.0])

loaded_nodes = extract_nodes_from_sets(pt_sets, cell_sets, mesh_cells, ["Loaded", "Last", "Tread", "Aussen"])
if not loaded_nodes:
    r_dist = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    max_r = torch.max(r_dist)
    loaded_nodes = torch.where(torch.abs(r_dist - max_r) < 1e-4)[0].tolist()
loaded_group_set = set(loaded_nodes)

# ==========================================
# ============ LOAD CALCULATION ============
# ==========================================
x_np = x.detach().cpu().numpy()
wheel_center = np.array([0.0, 0.0])

q8_edges = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
edge_count = {}
edge_data = {}
for e in range(nel):
    for (i1, i2, im) in q8_edges:
        n_idx = conn[e]; n1, n2, nm = int(n_idx[i1]), int(n_idx[i2]), int(n_idx[im])
        key = tuple(sorted((n1, n2)))
        edge_count[key] = edge_count.get(key, 0) + 1
        edge_data[key] = (e, (n1, n2, nm))
boundary_edges = [edge_data[k][1] for k, c in edge_count.items() if c == 1]

b_nodes = set()
for (n1, n2, nm) in boundary_edges: [b_nodes.add(n) for n in [n1, n2, nm]]
b_nodes = np.array(list(b_nodes), dtype=int)
ymin = x_np[b_nodes, 1].min()
i_contact = b_nodes[np.argmin(x_np[b_nodes, 1])]
x_contact = x_np[i_contact]; n_in = -x_contact/np.linalg.norm(x_contact); t_hat = np.array([-n_in[1], n_in[0]])

contact_edges = []
for (n1, n2, nm) in boundary_edges:
    if loaded_group_set and not (n1 in loaded_group_set or n2 in loaded_group_set or nm in loaded_group_set): continue
    s_loc = float(np.dot((x_np[nm] - x_contact), t_hat))
    if abs(s_loc) <= 1.2 * a_hz: contact_edges.append((n1, n2, nm))

if len(contact_edges) > 0:
    print(f"HERTZ: Found {len(contact_edges)} edges. Integrating pressure...")
    gp, gw = [-math.sqrt(3/5), 0, math.sqrt(3/5)], [5/9, 8/9, 5/9]
    def N_line(xi): return np.array([0.5*xi*(xi-1), 0.5*xi*(xi+1), 1-xi*xi])
    def dN_line(xi): return np.array([xi-0.5, xi+0.5, -2*xi])
    node_F = {}
    for (n1, n2, nm) in contact_edges:
        X1, X2, Xm = x_np[n1], x_np[n2], x_np[nm]
        for xi, w in zip(gp, gw):
            Nsh, dNsh = N_line(xi), dN_line(xi)
            xg = Nsh[0]*X1 + Nsh[1]*X2 + Nsh[2]*Xm
            ds = np.linalg.norm(dNsh[0]*X1 + dNsh[1]*X2 + dNsh[2]*Xm)
            p = p_hertz(np.dot((xg - x_contact), t_hat))
            fvec = (p * b_hz * ds * w) * (-xg/np.linalg.norm(xg))
            for node, Ni in zip([n1, n2, nm], Nsh):
                if node not in node_F: node_F[node] = np.zeros(2)
                node_F[node] += Ni * fvec
    for node, Fxy in node_F.items():
        neum_bcs.append([int(node), 0, float(Fxy[0])])
        neum_bcs.append([int(node), 1, float(Fxy[1])])
    load_nodes = torch.tensor(sorted(list(node_F.keys())), dtype=torch.long)
else:
    print("BC INFO: Falling back to direct force on Tread.")
    f_total_vec = abs(F_total / len(loaded_nodes))
    for n_idx in loaded_nodes:
        vec = -x_np[n_idx] / np.linalg.norm(x_np[n_idx])
        neum_bcs.append([int(n_idx), 0, float(f_total_vec * vec[0])])
        neum_bcs.append([int(n_idx), 1, float(f_total_vec * vec[1])])
    load_nodes = torch.tensor(loaded_nodes, dtype=torch.long)

drlt = torch.tensor(drlt_bcs, dtype=torch.float64).reshape(-1, 3)
neum = torch.tensor(neum_bcs, dtype=torch.float64).reshape(-1, 3)

# ==========================================
# ============ SETUP VISUALIZATION =========
# ==========================================
limit = float(torch.max(torch.abs(x)) * 1.1)
fig1, ax1 = plt.subplots(figsize=(12, 10))
ax1.set_title(f"Simulation Setup\nNodes: {nnp} | Elements: {nel}", fontweight='bold')
if nen == 8:
    idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    for e in range(nel): ax1.plot(x[conn[e][idx], 0], x[conn[e][idx], 1], color='black', lw=0.4, alpha=0.15)
else:
    for e in range(nel): ax1.plot(x[torch.cat([conn[e], conn[e][:1]]), 0], x[torch.cat([conn[e], conn[e][:1]]), 1], color='black', lw=0.4, alpha=0.15)

for n in fixed_nodes: ax1.scatter(x[n, 0], x[n, 1], color='green', marker='o', s=15, alpha=0.5, zorder=5)
for bc in neum_bcs:
    if abs(bc[2]) < 1e-6 or bc[1] == 0: continue
    n = int(bc[0]); xn, yn = x_np[n]; vec = -x_np[n]/np.linalg.norm(x_np[n])
    ax1.arrow(xn - 0.1*limit*vec[0], yn - 0.1*limit*vec[1], 0.08*limit*vec[0], 0.08*limit*vec[1], head_width=0.015*limit, head_length=0.02*limit, fc='red', ec='red', zorder=6)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3); plt.show()

# ==========================================
# ============ PRE-CALCULATION & QUADRATURE ============
# ==========================================
if element_type == 'quad8':
    nqp=9; qpt=torch.zeros(9,2); w8=torch.zeros(9); a=math.sqrt(0.6); v=[-a,0,a]; w=[5/9,8/9,5/9]
    for i in range(3):
        for j in range(3): qpt[i*3+j,0]=v[j]; qpt[i*3+j,1]=v[i]; w8[i*3+j]=w[i]*w[j]
else:
    nqp=4; a=1/math.sqrt(3); qpt=torch.tensor([[-a,-a],[a,-a],[a,a],[-a,a]]); w8=torch.ones(4)

# Pre-calculate Shape Functions & Gradients for ALL QPs
# We need N (nqp, nen) and G_iso (nqp, nen, 2)
N_vec = torch.zeros(nqp, nen, dtype=torch.float64)
G_iso_vec = torch.zeros(nqp, nen, 2, dtype=torch.float64)

for q in range(nqp):
    xi, eta = qpt[q, 0], qpt[q, 1]
    if element_type == 'quad8':
        # Q8 Shape Functions
        # 0:BL, 1:BR, 2:TR, 3:TL, 4:B, 5:R, 6:T, 7:L
        N = torch.zeros(nen); G = torch.zeros(nen, 2)
        N[0]=0.25*(1-xi)*(1-eta)*(-xi-eta-1); N[1]=0.25*(1+xi)*(1-eta)*(xi-eta-1)
        N[2]=0.25*(1+xi)*(1+eta)*(xi+eta-1);  N[3]=0.25*(1-xi)*(1+eta)*(-xi+eta-1)
        N[4]=0.5*(1-xi*xi)*(1-eta); N[5]=0.5*(1+xi)*(1-eta*eta); N[6]=0.5*(1-xi*xi)*(1+eta); N[7]=0.5*(1-xi)*(1-eta*eta)
        
        G[0,0]=0.25*(1-eta)*(-1)*(-xi-eta-1)+0.25*(1-xi)*(1-eta)*(-1)
        G[1,0]=0.25*(1-eta)*(1)*(xi-eta-1)+0.25*(1+xi)*(1-eta)*(1)
        G[2,0]=0.25*(1+eta)*(1)*(xi+eta-1)+0.25*(1+xi)*(1+eta)*(1)
        G[3,0]=0.25*(1+eta)*(-1)*(-xi+eta-1)+0.25*(1-xi)*(1+eta)*(-1)
        G[4,0]=0.5*(-2*xi)*(1-eta); G[5,0]=0.5*(1-eta*eta); G[6,0]=0.5*(-2*xi)*(1+eta); G[7,0]=0.5*(-1)*(1-eta*eta)
        
        G[0,1]=0.25*(1-xi)*(-1)*(-xi-eta-1)+0.25*(1-xi)*(1-eta)*(-1)
        G[1,1]=0.25*(1+xi)*(-1)*(xi-eta-1)+0.25*(1+xi)*(1-eta)*(-1)
        G[2,1]=0.25*(1+xi)*(1)*(xi+eta-1)+0.25*(1+xi)*(1+eta)*(1)
        G[3,1]=0.25*(1-xi)*(1)*(-xi+eta-1)+0.25*(1-xi)*(1+eta)*(1)
        G[4,1]=0.5*(1-xi*xi)*(-1); G[5,1]=0.5*(1+xi)*(-2*eta); G[6,1]=0.5*(1-xi*xi)*(1); G[7,1]=0.5*(1-xi)*(-2*eta)
    else:
        # Q4
        N[0]=0.25*(1-xi)*(1-eta); N[1]=0.25*(1+xi)*(1-eta); N[2]=0.25*(1+xi)*(1+eta); N[3]=0.25*(1-xi)*(1+eta)
        G[0,0]=-0.25*(1-eta); G[0,1]=-0.25*(1-xi); G[1,0]=0.25*(1-eta); G[1,1]=-0.25*(1+xi)
        G[2,0]=0.25*(1+eta); G[2,1]=0.25*(1+xi); G[3,0]=-0.25*(1+eta); G[3,1]=0.25*(1-xi)
    
    N_vec[q] = N
    G_iso_vec[q] = G

# Identity Tensors (Global)
I3 = torch.eye(3, dtype=torch.float64)
I4s = torch.zeros(3,3,3,3, dtype=torch.float64)
for i in range(3):
    for j in range(3):
        for k2 in range(3):
            for l2 in range(3):
                I4s[i,j,k2,l2] = 0.5*((1.0 if (i==k2 and j==l2) else 0.0) + (1.0 if (i==l2 and j==k2) else 0.0))
IoxI = torch.einsum("ij,kl->ijkl", I3, I3)
Idev_sym = I4s - (1.0/3.0)*IoxI
Idev_sym_2d = Idev_sym[:2,:2,:2,:2].contiguous() # For 2D parts
IoxI_2d = IoxI[:2,:2,:2,:2].contiguous() # For 2D parts

# Elastic Tangent (Plane Strain)
C4 = torch.zeros(2,2,2,2, dtype=torch.float64)
mu = E / (2.0*(1.0+nu)); K_mod = E / (3.0*(1.0-2.0*nu)) # K is taken by Loop index
lam = (E*nu)/((1+nu)*(1-2*nu))
C4[0,0,0,0]=C4[1,1,1,1]=lam+2*mu; C4[0,0,1,1]=C4[1,1,0,0]=lam; C4[0,1,0,1]=C4[1,0,0,1]=C4[0,1,1,0]=C4[1,0,1,0]=mu
C4_flat = C4.reshape(4,4) # For matrix mult if needed

# Gather Indices for Batched Assembly
print("Preparing Gather Indices...")
gather_indices = torch.zeros(nel, nen*ndf, dtype=torch.long)
for e in range(nel):
    indices = []
    for n in conn[e]:
        indices.extend([int(n)*ndf, int(n)*ndf+1])
    gather_indices[e] = torch.tensor(indices, dtype=torch.long)

# ==========================================
# ============ VECTORIZED J2 ============
# ==========================================
def von_mises_return_batch(eps2_batch, ep_batch, k_batch, a_batch):
    # Input shapes: 
    # eps2_batch: (nel, nqp, 2, 2)
    # ep_batch (alpha): (nel, nqp, 3, 3)
    # k_batch: (nel, nqp)
    # a_batch (backstress): (nel, nqp, 3, 3)
    
    # Constants and Dimensions
    batch_size, n_qp, _, _ = eps2_batch.shape
    
    # Construct 3D Strain (Plane Strain: eps33 = 0)
    eps = torch.zeros(batch_size, n_qp, 3, 3, dtype=torch.float64)
    eps[:, :, :2, :2] = eps2_batch
    
    # 1. Trial State
    # s_trial = 2*mu*(eps - ep) + K*tr(eps)*I
    trace_eps = (eps[:, :, 0, 0] + eps[:, :, 1, 1]).unsqueeze(-1).unsqueeze(-1) # (nel, nqp, 1, 1)
    
    # Deviatoric trial stress
    s_trial = 2.0 * mu * (eps - ep_batch) + lam * trace_eps * I3 

    # Deviatoric part of s_trial
    trace_s = (s_trial[:, :, 0, 0] + s_trial[:, :, 1, 1] + s_trial[:, :, 2, 2]) / 3.0
    s_dev = s_trial - trace_s.unsqueeze(-1).unsqueeze(-1) * I3
    
    # Backstress: alpha_h = (2/3) * (1-r) * H * a
    alpha_h = (2.0/3.0) * (1.0 - r) * H * a_batch
    
    s_red = s_dev - alpha_h # Relative stress
    
    # Norm of relative stress
    norm_red = torch.norm(s_red, dim=(2,3)) # (nel, nqp)
    
    # Yield function
    # phi = ||s_rel|| - sqrt(2/3) * (sy + r*H*k)
    f_yield = norm_red - math.sqrt(2.0/3.0) * (sigma_y + r * H * k_batch)
    
    # --- MASKING ---
    # Where f_yield > 0: PLASTIC
    mask = f_yield > 0 # (nel, nqp) boolean
    
    # Initialize returns with ELASTIC values
    sig_out = s_trial.clone()
    Ct_out = torch.zeros(batch_size, n_qp, 2, 2, 2, 2, dtype=torch.float64)
    # Expand C4 to batch size
    Ct_out[:] = C4 # Default to elastic tangent
    
    ep_new = ep_batch.clone()
    k_new = k_batch.clone()
    a_new = a_batch.clone()
    
    # --- PLASTIC CORRECTION (Only for masked) ---
    if torch.any(mask):
        # Extract active values
        norm_red_act = norm_red[mask]
        phi_act = f_yield[mask]
        
        # Directions
        # v = s_red / norm_red
        v = s_red[mask] / norm_red_act.unsqueeze(-1).unsqueeze(-1) # (N_act, 3, 3)
        
        # Plastic Multiplier
        B = 2.0*mu + (2.0/3.0)*H
        dlam = phi_act / B
        
        # Update Internal Vars
        # ep_n = ep + dlam * v
        d_ep = dlam.unsqueeze(-1).unsqueeze(-1) * v
        ep_new[mask] += d_ep
        k_new[mask] += dlam * math.sqrt(2.0/3.0)
        a_new[mask] += d_ep # Kinematic var evolution
        
        # Update Stress
        # sig = s_trial - 2*mu*dlam*v
        sig_out[mask] -= 2.0 * mu * d_ep
        
        # Consistent Tangent
        # Modified Newton: Keep Elastic Ct_out!
        # Consistent Tangent logic is commented out below to prevent numerical instability
        """
        A_fac = 2.0 * mu
        ratio = dlam / (norm_red_act + 1e-20)
        c1 = 2.0 * mu * (1.0 - A_fac * ratio)
        c2 = 2.0 * mu * A_fac * (ratio - 1.0/B)
        
        vv = torch.einsum("nij,nkl->nijkl", v, v)
        c1_exp = c1.view(-1, 1, 1, 1, 1)
        c2_exp = c2.view(-1, 1, 1, 1, 1)
        
        Ct_plast = (K_mod * IoxI.unsqueeze(0) + 
                    c1_exp * Idev_sym.unsqueeze(0) +
                    c2_exp * vv)
        Ct_out[mask] = Ct_plast[:, :2, :2, :2, :2]
        """
        
    return sig_out[:, :, :2, :2], Ct_out, ep_new, k_new, a_new

# ==========================================
# ============ GEO PRE-CALCULATION (Small Strain) ============
# ==========================================
print("Pre-calculating geometric tensors (G, dv)...")

# Gather Element Coords
xe = x[conn].transpose(1, 2) # (nel, 2, nen)

# Jacobian: Je = xe @ G_iso
Je_batch = torch.einsum("eak,qkb->eaqb", xe, G_iso_vec) # (nel, 2, nqp, 2)
Je_batch = Je_batch.permute(0, 2, 1, 3) # (nel, nqp, 2, 2)

# Determinant
det_Je = torch.det(Je_batch) # (nel, nqp)

# Inverse Jacobian
Je_inv = torch.linalg.inv(Je_batch) # (nel, nqp, 2, 2)

# Global Gradient G = inv(Je) @ G_iso
G_global = torch.einsum("qnj,eqji->eqni", G_iso_vec, Je_inv)

# dv = det * w * thickness
dv_global = det_Je * w8.unsqueeze(0) * b_hz # (nel, nqp)

# Clean up memory
del xe, Je_batch, det_Je, Je_inv
import gc; gc.collect()

print(f"DEBUG info:")
print(f"  Fixed Nodes: {len(fixed_nodes)}")
print(f"  Loaded Nodes: {len(loaded_nodes)}")
print(f"  Min det(Je): {dv_global.min().item()/b_hz:.4e} (Should be > 0)")
print(f"  Max det(Je): {dv_global.max().item()/b_hz:.4e}")

# ==========================================
# ============ CORE SOLVER (VECTORIZED) ============
# ==========================================
# State Initialization (Tensors)
ep_state = torch.zeros(nel, nqp, 3, 3, dtype=torch.float64)
k_state = torch.zeros(nel, nqp, dtype=torch.float64)
a_state = torch.zeros(nel, nqp, 3, 3, dtype=torch.float64)
u = torch.zeros(nnp*ndf, 1)
free_dofs = torch.nonzero(1 - torch.zeros(nnp*ndf, 1).index_fill_(0, (drlt[:, 0]*2 + drlt[:, 1]).long(), 1))[:, 0]

disp_pl, load_hist, load_target_hist, fac_used_hist = [], [], [], []
eps_p_xx_hist, sig_yy_hist, eps_p_eq_hist, sig_eq_hist = [], [], [], []
k_hist, a_hist, sig_1_hist, sig_2_hist = [], [], [], []
fac_conv, fac_scale, step = 0.0, 1.0, 1
sim_start_time = time.time()

# Pre-calculate gather indices flat for scattering
rows_flat = gather_indices.unsqueeze(2).expand(nel, nen*ndf, nen*ndf).flatten()
cols_flat = gather_indices.unsqueeze(1).expand(nel, nen*ndf, nen*ndf).flatten()

print("Starting Vectorized Solver...")
pbar = tqdm(total=int(n_steps), desc="Solving")

while step <= n_steps:
    pbar.n = step-1; pbar.refresh()
    
    # Calculate target load factor
    fac_target = (min(1, step/n_ramp_steps) if use_ramp_in else 1) * math.sin(2*math.pi*(step-1)/n_steps_per_cycle)
    
    # Save State (Backup for Cutback)
    ep_old = ep_state.clone()
    k_old = k_state.clone()
    a_old = a_state.clone() 
    u_old = u.clone()
    cutbacks = 0
    
    while True:
        fac = fac_conv + fac_scale * (fac_target - fac_conv)
        print(f"  Substep: fac_conv={fac_conv:.4f} -> fac={fac:.4f} (scale={fac_scale:.3e})")
        
        # External Force Vector
        f_ext = torch.zeros(nnp*ndf, 1, dtype=torch.float64)
        if len(neum) > 0:
            # Vectorized BC application
            indices = (neum[:, 0]*2 + neum[:, 1]).long()
            values = neum[:, 2] * fac
            f_ext.index_add_(0, indices, values.unsqueeze(1))
        
        u = u_old.clone()
        converged = False
        
        # Substep State (Updated ITERATIVELY inside Newton)
        
        for it in range(newton_max):
            # 1. Gather Displacements
            ue_flat = u.squeeze()[gather_indices] # (nel, nen*ndf)
            ue = ue_flat.reshape(nel, nen, 2) # (nel, nen, 2)
            
            # 2. Compute Strain
            grad_u = torch.einsum("end,eqni->eqdi", ue, G_global)
            eps = 0.5 * (grad_u + grad_u.transpose(-1, -2)) # (nel, nqp, 2, 2)
            
            # 3. Material Law (Batched)
            # Use state from OLD converged step as base for trial
            sig, Ct, ep_new, k_new, a_new = von_mises_return_batch(eps, ep_old, k_old, a_old)
            
            # 4. Integrate Internal Force
            fe = torch.einsum("eqij,eqnj,eq->eni", sig, G_global, dv_global)
            fe_flat = fe.reshape(nel, nen*ndf)
            
            # 5. Integrate Stiffness Matrix
            Ke_tensor = torch.einsum("eqak,eqikjl,eqbl,eq->eaibj", G_global, Ct, G_global, dv_global)
            Ke_flat = Ke_tensor.reshape(nel, nen*ndf, nen*ndf)
            
            # 6. Global Assembly
            Kt = torch.zeros(nnp*ndf, nnp*ndf, dtype=torch.float64)
            f_int = torch.zeros(nnp*ndf, 1, dtype=torch.float64)
            
            # Index Put (Scatter Add)
            Kt.index_put_((rows_flat, cols_flat), Ke_flat.flatten(), accumulate=True)
            f_int.view(-1).index_put_((gather_indices.flatten(),), fe_flat.flatten(), accumulate=True)
            
            # 7. Solve
            R = f_ext - f_int
            Rf = R[free_dofs]
            fext_norm = torch.norm(f_ext[free_dofs])
            rel = float(torch.norm(Rf)) / max(float(fext_norm), 1.0)
            
            print(f"    it {it:2d}: rel={rel:.3e}, ||Rf||={torch.norm(Rf):.3e}")
            
            if rel < newton_tol:
                # Update State Tensors
                ep_state = ep_new
                k_state = k_new
                a_state = a_new
                
                # --- HISTORIE TRACKING ---
                el_containing = torch.where(conn == i_contact)
                if len(el_containing[0]) > 0:
                    el_idx = el_containing[0][0].item()
                else:
                    el_idx = 0
                
                # History Append
                # Stress at el_idx, qp=0
                sig_val = sig[el_idx, 0]
                ep_val_xx = ep_new[el_idx, 0, 0, 0]
                
                eps_p_xx_hist.append(float(ep_val_xx))
                sig_yy_hist.append(float(sig_val[1,1]))
                load_hist.append(float(fac*F_total))
                disp_pl.append(float(u[i_contact*2+1]))
                
                load_target_hist.append(float(fac_target * F_total))
                fac_used_hist.append(float(fac))
                k_hist.append(float(k_new[el_idx, 0]))
                a_hist.append(a_new[el_idx, 0])
                
                # Eq Stress
                s11, s22, s12 = float(sig_val[0,0]), float(sig_val[1,1]), float(sig_val[0,1])
                s33 = nu*(s11+s22)
                tr = (s11+s22+s33)/3.0
                svm = math.sqrt(1.5*((s11-tr)**2 + (s22-tr)**2 + (s33-tr)**2 + 2*s12**2))
                sig_eq_hist.append(svm)
                
                # Eq Plastic Strain
                ep_tr = ep_new[el_idx, 0]
                ep_dev = ep_tr - (torch.trace(ep_tr)/3)*torch.eye(3)
                ep_eq = math.sqrt((2/3)*torch.sum(ep_dev**2).item())
                eps_p_eq_hist.append(ep_eq)
                
                # Principal
                R_sig = math.sqrt(0.25*(s11-s22)**2 + s12**2)
                sig_1_hist.append((0.5*(s11+s22)+R_sig)/1e6)
                sig_2_hist.append((0.5*(s11+s22)-R_sig)/1e6)
                
                converged = True
                print(f"    -> Converged at iteration {it}")
                break
            
            # Solve Linear System
            # Invert Kt or Solve
            try:
                du = torch.zeros_like(u)
                du[free_dofs] = torch.linalg.solve(Kt[free_dofs][:, free_dofs], Rf)
                u += du
            except Exception as e:
                print(f"    !! Solver Error: {e}")
                converged = False
                break
                
        if converged:
            fac_conv = fac
            if abs(fac_target - fac_conv) < 1e-3:
                step += 1
                break
            fac_scale = min(1.0, fac_scale * 1.1)
        else:
            print(f"  !! Cutback: substep failed to converge. Reducing scale.")
            fac_scale *= 0.5
            cutbacks += 1
            if fac_scale < 1e-6 or cutbacks > 15:
                raise RuntimeError(f"Convergence failed at step {step} (Scale: {fac_scale:.2e})")

pbar.close()
sim_duration = time.time() - sim_start_time

# ==========================================
# ============ POST & PLOT =============
# ==========================================
print("\nPost-processing spatial results...")
element_results = {"u_norm": np.zeros(nel), "svm": np.zeros(nel)}
u_np = u.detach().numpy().flatten()
x_def = x_np + u_np.reshape(-1, 2)

# Vectorized Post-Processing
# 1. Displacements
ue_flat = u.squeeze()[gather_indices] # (nel, nen*ndf)
ue = ue_flat.reshape(nel, nen, 2)
u_qp = torch.einsum("end,qn->eqd", ue, N_vec)
u_norm_qp = torch.norm(u_qp, dim=2) # (nel, nqp)
element_results["u_norm"] = torch.mean(u_norm_qp, dim=1).detach().numpy() * 1000.0 # mm

# 2. Stresses
# Compute final strains
grad_u = torch.einsum("end,eqni->eqdi", ue, G_global)
eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))

# Compute final stresses using final state
# ep_state, k_state, a_state are already updated to the END of the simulation
sig_final, _, _, _, _ = von_mises_return_batch(eps, ep_state, k_state, a_state)

# Von Mises
# sig_final: (nel, nqp, 2, 2)
s11 = sig_final[:, :, 0, 0]
s22 = sig_final[:, :, 1, 1]
s12 = sig_final[:, :, 0, 1]
s33 = nu * (s11 + s22)
tr = (s11 + s22 + s33) / 3.0

sd11, sd22, sd33 = s11-tr, s22-tr, s33-tr
sd12 = s12
svm_sq = 1.5 * (sd11**2 + sd22**2 + sd33**2 + 2*sd12**2)
svm = torch.sqrt(svm_sq) # (nel, nqp)

element_results["svm"] = torch.mean(svm, dim=1).detach().numpy() / 1e6 # MPa

# Local Plot (Quick check)
if not MODULAR_POST:
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f"Final State: Displacement & Von Mises Stress (Step {step-1})", fontweight='bold')
    
    verts = [x_def[conn[e][(list(range(4)) + [0]) if element_type!="quad8" else [0,4,1,5,2,6,3,7,0]]] for e in range(nel)]
    pc_u = PolyCollection(verts, cmap='viridis', edgecolors='none', alpha=0.9)
    pc_u.set_array(element_results["u_norm"])
    ax2.add_collection(pc_u); plt.colorbar(pc_u, ax=ax2, label="Verschiebung [mm]")
    ax2.set_title("Verschiebungsbetrag"); ax2.set_aspect('equal'); ax2.autoscale_view()
    
    pc_s = PolyCollection(verts, cmap='jet', edgecolors='none', alpha=0.9)
    pc_s.set_array(element_results["svm"])
    ax3.add_collection(pc_s); plt.colorbar(pc_s, ax=ax3, label="Spannung [MPa]")
    ax3.set_title("Von Mises Spannung"); ax3.set_aspect('equal'); ax3.autoscale_view()
    plt.show()

# Modular Post-Processing (Redundant)
if MODULAR_POST:
    fig_spatial = fp.plot_spatial_results(
        f"Final State (Vectorized): Displacement & Von Mises (Step {step-1})",
        x_def, conn, element_type, element_results, interactive=interactive_hover,
        save_path=os.path.join(script_dir, "results", "spatial_results_vec.png")
    )
    plt.show()
    
    fig_hist = fp.plot_history_overview(
        disp_pl, load_hist, load_target_hist, eps_p_xx_hist, sig_yy_hist, eps_p_eq_hist, sig_eq_hist,
        sig_1_hist, sig_2_hist, k_hist, a_hist, sigma_y, H, r,
        save_path=os.path.join(script_dir, "results", "history_results_vec.png")
    )
    plt.show()

print(f"Simulation completed in {sim_duration:.2f} seconds.")