
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

# ============ SETTINGS & CONFIG ===========

torch.set_default_dtype(torch.float64)
device = torch.device("cpu") 

interactive_hover = True 

# --- Mesh Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file = os.path.abspath(os.path.join(script_dir, "mesh", "Radausschnitt_Quad8.msh"))

print(f"Loading mesh from: {mesh_file}")

# --- Material Parameters ---
E = 205e9
nu = 0.3
width = 0.00583

sigma_y = 350e6    # Fließspannung
H = 209e7          # Gesamt-Verfestigungsmodul (H_iso + H_kin)
r = 0.35            # Faktor der Mischung (0=rein kinematisch, 1=rein isotrop)

# --- Force ---
F_total = -5.3e4     # Normalkraft
a_hz = 0.0035        # Hertz halbbreite

p0_hz = (2.0 * F_total) / (math.pi * a_hz * width)

def p_hertz(s):
    if abs(s) >= a_hz: return 0.0
    return p0_hz * math.sqrt(max(0.0, 1.0 - (s / a_hz) ** 2))

# --- Cyclic force loading ---
n_cycles = 1.0
n_steps_per_cycle = 60
n_steps = n_cycles * n_steps_per_cycle
F_amp = F_total
use_ramp_in = True
n_ramp_steps = 5

newton_max = 50
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
            fvec = (p * width * ds * w) * (-xg/np.linalg.norm(xg))
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
fig1, ax1 = plt.subplots(figsize=(8, 8))
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
# ============ MATERIAL & SHAPE ============
# ==========================================
def get_shape_data(xi, nen):
    e, n = xi[0], xi[1]
    N = torch.zeros(nen); G = torch.zeros(nen, 2)
    if nen == 4:
        N[0]=0.25*(1-e)*(1-n); N[1]=0.25*(1+e)*(1-n); N[2]=0.25*(1+e)*(1+n); N[3]=0.25*(1-e)*(1+n)
        G[0,0]=-0.25*(1-n); G[0,1]=-0.25*(1-e); G[1,0]=0.25*(1-n); G[1,1]=-0.25*(1+e)
        G[2,0]=0.25*(1+n); G[2,1]=0.25*(1+e); G[3,0]=-0.25*(1+n); G[3,1]=0.25*(1-e)
    elif nen == 8:
        # 0:BL, 1:BR, 2:TR, 3:TL, 4:B, 5:R, 6:T, 7:L
        N[0]=0.25*(1-e)*(1-n)*(-e-n-1); N[1]=0.25*(1+e)*(1-n)*(e-n-1)
        N[2]=0.25*(1+e)*(1+n)*(e+n-1);  N[3]=0.25*(1-e)*(1+n)*(-e+n-1)
        N[4]=0.5*(1-e*e)*(1-n); N[5]=0.5*(1+e)*(1-n*n); N[6]=0.5*(1-e*e)*(1+n); N[7]=0.5*(1-e)*(1-n*n)
        G[0,0]=0.25*(1-n)*(-1)*(-e-n-1)+0.25*(1-e)*(1-n)*(-1)
        G[1,0]=0.25*(1-n)*(1)*(e-n-1)+0.25*(1+e)*(1-n)*(1)
        G[2,0]=0.25*(1+n)*(1)*(e+n-1)+0.25*(1+e)*(1+n)*(1)
        G[3,0]=0.25*(1+n)*(-1)*(-e+n-1)+0.25*(1-e)*(1+n)*(-1)
        G[4,0]=0.5*(-2*e)*(1-n); G[5,0]=0.5*(1-n*n); G[6,0]=0.5*(-2*e)*(1+n); G[7,0]=0.5*(-1)*(1-n*n)
        G[0,1]=0.25*(1-e)*(-1)*(-e-n-1)+0.25*(1-e)*(1-n)*(-1)
        G[1,1]=0.25*(1+e)*(-1)*(e-n-1)+0.25*(1+e)*(1-n)*(-1)
        G[2,1]=0.25*(1+e)*(1)*(e+n-1)+0.25*(1+e)*(1+n)*(1)
        G[3,1]=0.25*(1-e)*(1)*(-e+n-1)+0.25*(1-e)*(1+n)*(1)
        G[4,1]=0.5*(1-e*e)*(-1); G[5,1]=0.5*(1+e)*(-2*n); G[6,1]=0.5*(1-e*e)*(1); G[7,1]=0.5*(1-e)*(-2*n)
    return N, G

if element_type == 'quad8':
    nqp=9; qpt=torch.zeros(9,2); w8=torch.zeros(9); a=math.sqrt(0.6); v=[-a,0,a]; w=[5/9,8/9,5/9]
    for i in range(3):
        for j in range(3): qpt[i*3+j,0]=v[j]; qpt[i*3+j,1]=v[i]; w8[i*3+j]=w[i]*w[j]
else:
    nqp=4; a=1/math.sqrt(3); qpt=torch.tensor([[-a,-a],[a,-a],[a,a],[-a,a]]); w8=torch.ones(4)

C4 = torch.zeros(2,2,2,2); mu=E/(2*(1+nu)); lam=(E*nu)/((1+nu)*(1-2*nu))
C4[0,0,0,0]=C4[1,1,1,1]=lam+2*mu; C4[0,0,1,1]=C4[1,1,0,0]=lam; C4[0,1,0,1]=C4[1,0,0,1]=C4[0,1,1,0]=C4[1,0,1,0]=mu

# Pre-calculate 4th-order identity tensors for speed
I3 = torch.eye(3, dtype=torch.float64)
I4s = torch.zeros(3,3,3,3, dtype=torch.float64)
for i in range(3):
    for j in range(3):
        for k2 in range(3):
            for l2 in range(3):
                I4s[i,j,k2,l2] = 0.5*((1.0 if (i==k2 and j==l2) else 0.0) + (1.0 if (i==l2 and j==k2) else 0.0))
IoxI = torch.einsum("ij,kl->ijkl", I3, I3)
Idev_sym = I4s - (1.0/3.0)*IoxI

def von_mises_return(eps2, state):
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    I3 = torch.eye(3, dtype=torch.float64)
    
    eps = torch.zeros(3, 3, dtype=torch.float64)
    eps[:2, :2] = eps2
    
    ep = state["ep"] 
    k = float(state["k"])
    a = state["a"] 
    
    # 1. Trial state
    # s = 2*mu*(eps_dev - ep) + K*trace(eps)*I
    s_trial = 2.0*mu*(eps - ep - (torch.trace(eps - ep)/3.0)*I3) + K*torch.trace(eps)*I3
    
    # 2. Yield surface check
    s_dev_tr = s_trial - (torch.trace(s_trial)/3.0)*I3
    alpha_h_tr = -(2.0/3.0)*(1.0-r)*H*a
    s_red_tr = s_dev_tr + alpha_h_tr
    norm_red = torch.norm(s_red_tr)
    
    phi = norm_red - math.sqrt(2.0/3.0)*(sigma_y + r*H*k)
    
    if phi <= 0:
        return s_trial[:2, :2], C4, {"ep": ep, "k": k, "a": a}
    
    # 3. Plastic step (Return Mapping)
    v = s_red_tr / (norm_red + 1e-15)
    B = 2.0*mu + (2.0/3.0)*H
    dlam = phi / B
    
    # Update internal vars
    ep_n = ep + dlam * v
    k_n = k + dlam * math.sqrt(2.0/3.0)
    a_n = a + dlam * v
    
    # Ensure deviatoric
    ep_n = ep_n - (torch.trace(ep_n)/3.0)*I3
    a_n = a_n - (torch.trace(a_n)/3.0)*I3
    
    # Stress update
    sig_n = s_trial - 2.0*mu*dlam*v
    
    # 4. Consistent Tangent
    A_fac = 2.0*mu
    c1 = 2.0*mu * (1.0 - (A_fac * dlam / (norm_red + 1e-15)))
    c2 = 2.0*mu * A_fac * (dlam / (norm_red + 1e-15) - 1.0/B)
    
    vv = torch.einsum("ij,kl->ijkl", v, v)
    Ct_4th = K*IoxI + c1*Idev_sym + c2*vv
    Ct2 = Ct_4th[:2, :2, :2, :2]
    
    return sig_n[:2, :2], Ct2, {"ep": ep_n, "k": k_n, "a": a_n}

# ==========================================
# ============ CORE SOLVER ============
# ==========================================
def clone_state_gp(state_gp):
    return [[{k: v.clone() if torch.is_tensor(v) else v for k, v in qp.items()} for qp in el] for el in state_gp]

state_gp = [[{"ep":torch.zeros(3,3), "k": 0.0, "a": torch.zeros(3,3)} for q in range(nqp)] for e in range(nel)]
u = torch.zeros(nnp*ndf, 1)
free_dofs = torch.nonzero(1 - torch.zeros(nnp*ndf, 1).index_fill_(0, (drlt[:, 0]*2 + drlt[:, 1]).long(), 1))[:, 0]

disp_pl, load_hist, load_target_hist = [], [], []
eps_p_xx_hist, sig_yy_hist, eps_p_eq_hist, sig_eq_hist = [], [], [], []
fac_conv, fac_scale, step = 0.0, 1.0, 1

# Pre-calculate element DOF indices for speed
element_dofs = []
for el in range(nel):
    indices = []
    for n in conn[el]:
        indices.extend([int(n)*ndf, int(n)*ndf+1])
    element_dofs.append(torch.tensor(indices, dtype=torch.long))

pbar = tqdm(total=int(n_steps), desc="Solving")
while step <= n_steps:
    pbar.n = step-1; pbar.refresh()
    
    # Calculate target load factor
    fac_target = (min(1, step/n_ramp_steps) if use_ramp_in else 1) * math.sin(2*math.pi*(step-1)/n_steps_per_cycle)
    print(f"\nSTEP {step}/{n_steps} | Target Load Factor: {fac_target:.4f}")
    
    state_gp_old, u_old, cutbacks = clone_state_gp(state_gp), u.clone(), 0
    
    while True:
        fac = fac_conv + fac_scale * (fac_target - fac_conv)
        print(f"  Substep: fac_conv={fac_conv:.4f} -> fac={fac:.4f} (scale={fac_scale:.3e})")
        
        f_ext = torch.zeros(nnp*ndf, 1)
        for bc in neum: f_ext[int(bc[0])*2 + int(bc[1])] = fac * bc[2]
        
        u, converged = u_old.clone(), False
        
        # Iteration-level storage (only cloned once per substep, NOT per iteration)
        state_gp_current_substep = clone_state_gp(state_gp_old)
        
        for it in range(newton_max):
            Kt, fint = torch.zeros(nnp*ndf, nnp*ndf), torch.zeros(nnp*ndf, 1)
            
            # Temporary state for THIS iteration's trial calculation
            # We don't actually need to deep-clone EVERYTHING every time if we're careful,
            # but for J2 we need the trial state based on the converged state of the PREVIOUS substep.
            
            for el in range(nel):
                n_idx = conn[el]
                xe = x[n_idx].t()
                edofs = element_dofs[el]
                ue = u[edofs].reshape(-1, 2).t()
                
                Ke, fe = torch.zeros(nen*ndf, nen*ndf), torch.zeros(nen*ndf, 1)
                
                for q in range(nqp):
                    N, Gsh = get_shape_data(qpt[q], nen)
                    Je = xe @ Gsh
                    dv = torch.det(Je) * w8[q] * width
                    G = torch.linalg.solve(Je.T, Gsh.T).T
                    
                    eps = 0.5 * (ue @ G + (ue @ G).t())
                    
                    # Compute stresses and tangent based on the state at the end of the LAST converged substep
                    sig, Ct, state_gp_current_substep[el][q] = von_mises_return(eps, state_gp_old[el][q])
                    
                    for A in range(nen):
                        for B in range(nen):
                            Ke[A*2:A*2+2, B*2:B*2+2] += dv * torch.tensordot(G[A], torch.tensordot(Ct, G[B], [[3],[0]]), [[0], [0]])
                        fe[A*2:A*2+2, 0] += dv * (sig @ G[A].unsqueeze(1)).squeeze()
                
                Kt[edofs.unsqueeze(1), edofs] += Ke
                fint[edofs] += fe
            
            R = f_ext - fint
            Rf = R[free_dofs]
            fext_norm = torch.norm(f_ext[free_dofs])
            rel = float(torch.norm(Rf)) / max(float(fext_norm), 1.0)
            
            print(f"    it {it:2d}: rel={rel:.3e}, ||Rf||={torch.norm(Rf):.3e}")
            
            if rel < newton_tol:
                state_gp, converged = state_gp_current_substep, True
                # Tracking point stats
                st_tr = state_gp[i_contact // 4 if element_type == "quad8" else 0][0]
                eps_p_xx_hist.append(float(st_tr["ep"][0,0]))
                sig_yy_hist.append(float(sig[1,1]))
                load_hist.append(float(fac*F_total))
                disp_pl.append(float(u[i_contact*2+1]))
                print(f"    -> Converged at iteration {it}")
                break
                
            du_f = torch.linalg.solve(Kt[free_dofs][:, free_dofs], Rf)
            u[free_dofs] += du_f
            
        if converged:
            fac_conv = fac
            state_gp_old, u_old = clone_state_gp(state_gp), u.clone()
            if abs(fac_target - fac_conv) < 1e-3:
                step += 1
                break
            fac_scale = min(1.0, fac_scale * 1.5)
        else:
            print(f"  !! Cutback: substep failed to converge. Reducing factor scale.")
            fac_scale *= 0.5
            cutbacks += 1
            if fac_scale < 1e-4 or cutbacks > 8:
                raise RuntimeError(f"Step {step} failed to converge after {cutbacks} cutbacks.")

pbar.close()

# ==========================================
# ============ POST & PLOT =============
# ==========================================
def yield_curve_sigma12_closed(alpha, beta):
    sig_eff = sigma_y + r*H*alpha; theta = np.linspace(0, 2*np.pi, 200)
    x = sig_eff*np.cos(theta); y = sig_eff*np.sin(theta); s12 = np.zeros_like(x)
    s1 = (x - y/math.sqrt(3)); s2 = (x + y/math.sqrt(3)); s3 = -(s1+s2)
    return s1 + (beta[0,0]+beta[1,1]).item(), s2 + (beta[1,1]-beta[0,0]).item()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(disp_pl, load_hist); plt.title("Force-Displacement"); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(eps_p_xx_hist, sig_yy_hist); plt.title("Hysteresis"); plt.grid(True)
plt.tight_layout(); plt.show()
print("Simulation complete.")