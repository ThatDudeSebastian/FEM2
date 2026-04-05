
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
# SI-Einheiten
torch.set_default_dtype(torch.float64)
device = torch.device("cpu") 

interactive_hover = True 

# --- Mesh Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file = os.path.abspath(os.path.join(script_dir, "mesh", "Radausschnitt_Quad8.msh"))

print(f"Loading mesh from: {mesh_file}")

# --- Material Parameters ---
E = 205e9 # [Pa]
nu = 0.29


sigma_y = 695e6    # Fließspannung [Pa]
H = 2091e6          # Gesamt-Verfestigungsmodul (H_iso + H_kin) [Pa]
r = 0.35            # Faktor der Mischung (0=rein kinematisch, 1=rein isotrop)

# --- Force ---
F_total = -5.3e4     # Normalkraft [N]
a_hz = 0.005143        # erste Halbachse nach Knothe [m]
b_hz = a_hz # zweite Halbachse nach Knothe [m]

p0_hz = (2.0 * F_total) / (math.pi * a_hz * b_hz) # [Pa]

def p_hertz(s):
    if abs(s) >= a_hz: return 0.0
    return p0_hz * math.sqrt(max(0.0, 1.0 - (s / a_hz) ** 2))

# --- Cyclic force loading ---
n_cycles = 1
n_steps_per_cycle = 20
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
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
try:
    fig1.savefig(os.path.join(script_dir, "setup_mesh.png"), dpi=300)
except Exception as e:
    print(f"Error saving setup_mesh.png: {e}")
plt.show()

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
        return s_trial, C4, {"ep": ep, "k": k, "a": a}
    
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
    
    # Return FULL 3D stress tensor to capture sigma_33 correctly for plotting
    return sig_n, Ct2, {"ep": ep_n, "k": k_n, "a": a_n}



def yield_curve_sigma12_closed(alpha, beta, n=361):
    # Effective yield stress (isotropic hardening)
    sig_eff = float(sigma_y + (r * H) * alpha)

    # Shift (kinematic hardening) in sigma11-sigma22 plane
    b11 = float(beta[0, 0])
    b22 = float(beta[1, 1])

    # Quadratic form matrix for tau12=0 von Mises in (s11,s22):
    # s11^2 - s11*s22 + s22^2 = sig_eff^2
    # => v^T A v = sig_eff^2, A = [[1, -1/2],[-1/2, 1]]
    lam1 = 1.5
    lam2 = 0.5
    e1 = np.array([1.0, -1.0]) / np.sqrt(2.0)
    e2 = np.array([1.0,  1.0]) / np.sqrt(2.0)

    thetas = np.linspace(0.0, 2.0*np.pi, n)
    xs = np.empty(n)
    ys = np.empty(n)

    for i, th in enumerate(thetas):
        v = (sig_eff * (np.cos(th) / np.sqrt(lam1) * e1 +
                        np.sin(th) / np.sqrt(lam2) * e2))
        xs[i] = v[0] + b11
        ys[i] = v[1] + b22

    return xs, ys

# ==========================================
# ============ CORE SOLVER ============
# ==========================================
def clone_state_gp(state_gp):
    return [[{k: v.clone() if torch.is_tensor(v) else v for k, v in qp.items()} for qp in el] for el in state_gp]

state_gp = [[{"ep":torch.zeros(3,3), "k": 0.0, "a": torch.zeros(3,3)} for q in range(nqp)] for e in range(nel)]
u = torch.zeros(nnp*ndf, 1)
free_dofs = torch.nonzero(1 - torch.zeros(nnp*ndf, 1).index_fill_(0, (drlt[:, 0]*2 + drlt[:, 1]).long(), 1))[:, 0]

disp_pl, load_hist, load_target_hist, fac_used_hist = [], [], [], []
eps_p_xx_hist, sig_yy_hist, eps_p_eq_hist, sig_eq_hist = [], [], [], []
k_hist, a_hist, sig_1_hist, sig_2_hist = [], [], [], []
k_hist, a_hist, sig_1_hist, sig_2_hist = [], [], [], []
fac_conv, fac_scale, step = 0.0, 1.0, 1
sim_start_time = time.time()

# --- Tracking Variables ---
track_el = None
track_q = None
best_val = -1.0

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
    fac_target = (min(1, step/n_ramp_steps) if use_ramp_in else 1) * abs(math.sin(2*math.pi*(step-1)/n_steps_per_cycle))
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
        
        # Temporary storage for tracking potential hotspot in this iter
        eps_eq_tmp = {}
        eps_p_xx_tmp, sig_yy_tmp, eps_p_eq_tmp, sig_eq_tmp = {}, {}, {}, {}

        for it in range(newton_max):
            # Reset best_val for this iteration if we are searching (not needed, global is better)
            # actually we search only in first iter of first loaded step
            
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
                    dv = torch.det(Je) * w8[q] * b_hz
                    G = torch.linalg.solve(Je.T, Gsh.T).T
                    
                    eps = 0.5 * (ue @ G + (ue @ G).t())
                    sim_sig = None

                    # Compute stresses and tangent based on the state at the end of the LAST converged substep
                    # Returns FULL 3x3 stress tensor now
                    sig, Ct, state_gp_current_substep[el][q] = von_mises_return(eps, state_gp_old[el][q])
                    
                    # --- TRACKING LOGIC ---
                    # Calculate values using TRUE 3D stress
                    s11, s22, s33 = float(sig[0,0]), float(sig[1,1]), float(sig[2,2])
                    s12 = float(sig[0,1])
                    
                    tr_s = s11+s22+s33
                    s_dev = torch.tensor([[s11-tr_s/3, s12, 0],[s12, s22-tr_s/3, 0],[0,0, s33-tr_s/3]])
                    svm = math.sqrt(1.5 * torch.sum(s_dev*s_dev).item())
                    
                    st_cur = state_gp_current_substep[el][q]
                    ep_tr = st_cur["ep"]; ep_dev = ep_tr - (torch.trace(ep_tr)/3.0)*torch.eye(3)
                    ep_eq = math.sqrt((2.0/3.0) * torch.sum(ep_dev*ep_dev).item())

                    key = (el, q)
                    eps_p_xx_tmp[key] = float(ep_tr[0,0])
                    sig_yy_tmp[key] = s22
                    eps_p_eq_tmp[key] = ep_eq
                    sig_eq_tmp[key] = svm
                    
                    # Auto-detect hotspot (first time we have load)
                    if track_el is None and it == 0 and torch.norm(f_ext[free_dofs]).item() > 1.0:
                         # [CRITICAL FIX] Avoid Hub Singularity!
                         # Only consider elements in the outer 20% of the radius (Rim/Tread)
                         # Calculate approximate radius of this element
                         # xe is (2, nen) -> mean over nodes
                         r_el = float(torch.mean(torch.sqrt(xe[0,:]**2 + xe[1,:]**2)))
                         
                         # Determine global max radius only once (or estimate)
                         # We know wheel is roughly centered. xmax is ~0.46m?
                         # Let's use a safe threshold. If radius is small, skip.
                         # Better: calculate r_max outside or assuming standard wheel size.
                         # Let's assume r > 0.2m is safe for a train wheel (usually r=0.46m).
                         # Or simpler: use `Lx_val` from earlier?
                         # Let's use a hard threshold for now or logic relative to max coordinate.
                         r_threshold = 0.2 # meters. Hub is usually < 0.15m.
                         
                         if r_el > r_threshold:
                             if svm > best_val:
                                best_val = svm
                                track_el = el 
                                track_q = q
                    # ----------------------
                    
                    for A in range(nen):
                        for B in range(nen):
                            Ke[A*2:A*2+2, B*2:B*2+2] += dv * torch.tensordot(G[A], torch.tensordot(Ct, G[B], [[3],[0]]), [[0], [0]])
                        fe[A*2:A*2+2, 0] += dv * (sig[:2, :2] @ G[A].unsqueeze(1)).squeeze()
                
                Kt[edofs.unsqueeze(1), edofs] += Ke
                fint[edofs] += fe
            
            R = f_ext - fint
            Rf = R[free_dofs]
            fext_norm = torch.norm(f_ext[free_dofs])
            rel = float(torch.norm(Rf)) / max(float(fext_norm), 1.0)
            
            print(f"    it {it:2d}: rel={rel:.3e}, ||Rf||={torch.norm(Rf):.3e}")
            
            if rel < newton_tol:
                state_gp, converged = state_gp_current_substep, True
                
                # Consolidate Tracking Selection
                if track_el is None and torch.norm(f_ext[free_dofs]).item() > 1.0:
                    # Fallback if somehow missed
                    track_el, track_q = 0, 0
                    print("WARNING: Tracking element not found, defaulting to 0,0")
                if track_el is not None:
                     if it==0: print(f"Tracking: El {track_el} QP {track_q} (Max Stress approx {best_val/1e6:.1f} MPa)")

                # Use tracked values
                if track_el is not None:
                     k_key = (track_el, track_q)
                     # Fallback to current values if available (it should be)
                     if k_key in eps_p_xx_tmp:
                        eps_p_xx_hist.append(eps_p_xx_tmp[k_key])
                        sig_yy_hist.append(sig_yy_tmp[k_key] / 1e6) # [MPa]
                        eps_p_eq_hist.append(eps_p_eq_tmp[k_key])
                        sig_eq_hist.append(sig_eq_tmp[k_key] / 1e6) # [MPa]
                        
                        st_tr = state_gp[track_el][track_q]
                        k_hist.append(float(st_tr["k"]))
                        a_hist.append(st_tr["a"].clone())
                        
                     else:
                        # Should not happen if logic is correct
                        pass

                load_hist.append(float(fac*F_total))
                # Tracking Element Displacement (approximate via node 0 of element?)
                if track_el is not None:
                    n_tr = conn[track_el][0] # first node
                    # displacement in mm for plotting? User asked for MPa, usually disp in mm is good too.
                    # But kept as m in hist for now? Or mm?
                    # The plot 1 uses meters. Let's keep meters for disp in history to be safe or mm?
                    # Hannes_main uses disp_pl in [m] for history plot (label says [m]).
                    disp_pl.append(float(u[int(n_tr)*2+1])) # Y-disp [m]
                else:
                    disp_pl.append(0.0)
                
                # Enhanced Tracking
                load_target_hist.append(float(fac_target * F_total))
                fac_used_hist.append(float(fac))
                
                # Re-calculate exact stress state of tracked element for Path
                if track_el is not None:
                     # Re-compute for tracked QP to get full tensor
                     st_tr = state_gp[track_el][track_q]
                     n_idx = conn[track_el]
                     xe = x[n_idx].t()
                     edofs = element_dofs[track_el]
                     ue = u[edofs].reshape(-1, 2).t()
                     N, Gsh = get_shape_data(qpt[track_q], nen)
                     Je = xe @ Gsh; G = torch.linalg.solve(Je.T, Gsh.T).T
                     eps = 0.5 * (ue @ G + (ue @ G).t())
                     sig_fin, _, _ = von_mises_return(eps, state_gp_old[track_el][track_q]) # Use converged state input
                     
                     # Result from `von_mises_return` (sig) is the stress corresponding to `eps` and updated state.
                     
                     s11, s22, s33 = float(sig_fin[0,0]), float(sig_fin[1,1]), float(sig_fin[2,2])
                     s12 = float(sig_fin[0,1])
                     
                     # Calculate Principal Stresses for History Path
                     # (User explicitly requested Principal Space)
                     # Eigenvalues of 2D block [s11, s12; s12, s22]
                     curr_center = 0.5 * (s11 + s22)
                     curr_radius = math.sqrt(0.25*(s11-s22)**2 + s12**2)
                     
                     s1 = curr_center + curr_radius
                     s2 = curr_center - curr_radius
                     
                     sig_1_hist.append(s1 / 1e6) # [MPa]
                     sig_2_hist.append(s2 / 1e6) # [MPa]
                     
                     # Store final State rotation for Yield Surface Plot alignment
                     # Angle of First Principal axis w.r.t Global X
                     # tan(2theta) = 2*s12 / (s11 - s22)
                     if abs(s11-s22) > 1e-9:
                         theta_p = 0.5 * math.atan2(2*s12, s11-s22)
                     else:
                         theta_p = 0.0 if abs(s12) < 1e-9 else (math.pi/4 if s12>0 else -math.pi/4)
                     
                     final_theta_p = theta_p

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
sim_duration = time.time() - sim_start_time

# ==========================================
# ============ POST & PLOT =============
# ==========================================
print("\nPost-processing spatial results...")
element_results = {"u_norm": np.zeros(nel), "svm": np.zeros(nel)}
u_np = u.detach().numpy().flatten()
x_def = x_np + u_np.reshape(-1, 2)

for el in range(nel):
    indices = element_dofs[el]
    ue = u[indices].reshape(-1, 2).t()
    xe = x[conn[el]].t()
    svm_el, u_norm_el = 0.0, 0.0
    for q in range(nqp):
        N, Gsh = get_shape_data(qpt[q], nen)
        Je = xe @ Gsh; G = torch.linalg.solve(Je.T, Gsh.T).T
        eps = 0.5*(ue@G + (ue@G).t())
        sig, _, _ = von_mises_return(eps, state_gp[el][q])
        s11, s22, s12 = float(sig[0,0]), float(sig[1,1]), float(sig[0,1])
        s33 = float(sig[2,2])  # Plane Strain: σ₃₃ kommt aus 3D-Materialmodell, NICHT ν(σ₁₁+σ₂₂)
        tr = (s11 + s22 + s33)/3.0
        sd = torch.tensor([[s11-tr, s12, 0], [s12, s22-tr, 0], [0, 0, s33-tr]])
        svm_el += math.sqrt(1.5 * torch.sum(sd*sd).item())
        u_vals = (ue @ N).detach().numpy()
        u_norm_el += np.linalg.norm(u_vals)
    element_results["svm"][el] = svm_el / nqp / 1e6 # MPa
    element_results["u_norm"][el] = (u_norm_el / nqp) * 1000 # mm

# Figure 1: Spatial Results
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"Final State: Displacement & Von Mises Stress (Step {step-1})", fontweight='bold')

verts = [x_def[conn[e][idx if element_type=="quad8" else [0,1,2,3,0]]] for e in range(nel)]
pc_u = PolyCollection(verts, cmap='viridis', edgecolors='none', alpha=0.9)
pc_u.set_array(element_results["u_norm"])
ax2.add_collection(pc_u); plt.colorbar(pc_u, ax=ax2, label="Verschiebung [mm]")
ax2.set_title("Verschiebungsbetrag"); ax2.set_aspect('equal'); ax2.autoscale_view()

pc_s = PolyCollection(verts, cmap='jet', edgecolors='none', alpha=0.9)
pc_s.set_array(element_results["svm"])
ax3.add_collection(pc_s); plt.colorbar(pc_s, ax=ax3, label="Spannung [MPa]")
ax3.set_title("Von Mises Spannung"); ax3.set_aspect('equal'); ax3.autoscale_view()

if interactive_hover:
    ann2 = ax2.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
    ann3 = ax3.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
    ann2.set_visible(False); ann3.set_visible(False)
    def hover(event):
        for ax, ann, pc, key, unit in [(ax2, ann2, pc_u, "u_norm", "mm"), (ax3, ann3, pc_s, "svm", "MPa")]:
            if event.inaxes == ax:
                cont, ind = pc.contains(event)
                if cont:
                    i = ind["ind"][0]; b = pc.get_paths()[i].get_extents()
                    ann.xy = [(b.x0+b.x1)/2, (b.y0+b.y1)/2]
                    ann.set_text(f"Elem: {i}\n{element_results[key][i]:.2f} {unit}")
                    ann.set_visible(True); fig.canvas.draw_idle(); return
            ann.set_visible(False)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)

# Figure 2: History Analysis
fig_hist = plt.figure(figsize=(14, 10))
plt.subplot(2,2,1); plt.plot(disp_pl, np.array(load_hist)/1000.0, 'r', label="Actual")
plt.title("Force-Displacement"); plt.xlabel("U_y [m]"); plt.ylabel("Force [kN]"); plt.grid(True);
# Subplot 2: Hysteresis Stress-Strain
plt.subplot(2, 2, 2)
plt.plot(eps_p_xx_hist, sig_yy_hist, lw=2, color='blue')
plt.xlabel(r"$\varepsilon^{p}_{xx}$ [-]")
plt.ylabel(r"$\sigma_{xx}$ [MPa]")
plt.title("Hysterese: Axialspannung vs. Plast. Dehnung")
plt.grid(True)

# Subplot 3: Equivalent Hysteresis
plt.subplot(2, 2, 3)
plt.plot(eps_p_eq_hist, sig_eq_hist, lw=2, color='green')
plt.xlabel(r"$\varepsilon^p_\mathrm{eq}$ [-]")
plt.ylabel(r"$\sigma_\mathrm{eq}$ [MPa]")
plt.title("Äquivalente Spannung vs. Akkumulierte Plast. Dehnung")
plt.grid(True)


# Subplot 4: Yield Surface (Principal Stress Space)
plt.subplot(2, 2, 4)
plt.title("Fließfläche im HS-Raum (r=0.35)")

# Check if final_theta_p exists (it should if tracking ran)
if 'final_theta_p' not in locals():
    final_theta_p = 0.0

if len(a_hist) > 0:
    a_final = a_hist[-1]
    # beta tensor (kinematic shift) in Global Frame
    beta_final = (2.0/3.0)*(1.0-r)*H*a_final
    
    # Rotate beta into Principal Frame
    # The Principal Axis 1 is at angle `theta_p`.
    co = math.cos(final_theta_p)
    si = math.sin(final_theta_p)
    
    # Beta components global
    b11 = float(beta_final[0,0])
    b22 = float(beta_final[1,1])
    b12 = float(beta_final[0,1])
    
    # Projection to 1-2 Principal Frame (of Stress)
    # b_p11 = c^2 b11 + 2cs b12 + s^2 b22
    # b_p22 = s^2 b11 - 2cs b12 + c^2 b22
    b1_proj = co*co*b11 + 2.0*co*si*b12 + si*si*b22
    b2_proj = si*si*b11 - 2.0*co*si*b12 + co*co*b22
    
    # Isotropic part
    k_final = k_hist[-1]
    alpha_iso = k_final
    
    # Generate centered ellipse geometry (beta=0)
    xs0, ys0 = yield_curve_sigma12_closed(alpha_iso, torch.zeros(3,3).double())
    # Shift manualy by Projected Backstress Center
    xs_cyc = xs0 + b1_proj
    ys_cyc = ys0 + b2_proj
    
    # Initial Yield (Centered at 0)
    xs00, ys00 = yield_curve_sigma12_closed(0.0, torch.zeros(3,3).double())
    plt.plot(xs00/1e6, ys00/1e6, 'k--', label="Initial (Yield)", lw=2.5)

    plt.plot(xs_cyc/1e6, ys_cyc/1e6, 'r-', label="Aktuell (Hardened)", lw=3)
    
    # Plot Center (Projected)
    plt.plot(b1_proj/1e6, b2_proj/1e6, 'rx', markersize=10, markeredgewidth=2, label="Center (Backstress)")
    
    # Limit Lines (Relative to Backstress)
    # s2 = s1 + (b2-b1) +/- 2/sqrt(3)*Y
    sig_eff_cur = sigma_y + r*H*k_final
    limit_dist = (2.0/math.sqrt(3.0)) * sig_eff_cur
    shift = b2_proj - b1_proj
    
    sig_range = np.linspace(np.min(xs_cyc/1e6)-1000, np.max(xs_cyc/1e6)+1000, 100)
    
    # Line 1 & 2
    plt.plot(sig_range, sig_range + (shift/1e6) + (limit_dist/1e6), 'r:', alpha=0.6, label="PE Limit")
    plt.plot(sig_range, sig_range + (shift/1e6) - (limit_dist/1e6), 'r:', alpha=0.6)
    
    # Bounds logic
    x_min_surf = np.min(xs_cyc/1e6); x_max_surf = np.max(xs_cyc/1e6)
    y_min_surf = np.min(ys_cyc/1e6); y_max_surf = np.max(ys_cyc/1e6)
    
    if len(sig_1_hist) > 0:
        x_min = min(x_min_surf, min(sig_1_hist))
        x_max = max(x_max_surf, max(sig_1_hist))
        y_min = min(y_min_surf, min(sig_2_hist))
        y_max = max(y_max_surf, max(sig_2_hist))
        margin_x = 0.2*max(x_max-x_min, 100)
        margin_y = 0.2*max(y_max-y_min, 100)
        plt.xlim(x_min-margin_x, x_max+margin_x)
        plt.ylim(y_min-margin_y, y_max+margin_y)
    else:
        plt.axis('equal')

    plt.plot(sig_1_hist, sig_2_hist, 'b-', lw=1.5, alpha=0.7, label="Stress Path")
    if len(sig_1_hist) > 0:
        plt.plot(sig_1_hist[-1], sig_2_hist[-1], 'co', markersize=6, label="Current State")
    
    plt.xlabel(r"$\sigma_{xx}$ [MPa]")
    plt.ylabel(r"$\sigma_{yy}$ [MPa]")
    plt.title(f"Fließfläche im HS-Raum (r={r:.2f})")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Add Simulation Time Text
    plt.figtext(0.5, 0.01, f"Simulation Duration: {sim_duration:.2f} s", ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":3})
    
    # Add origin crosshair
    plt.axhline(0, color='black', lw=1.0, alpha=0.3)
    plt.axvline(0, color='black', lw=1.0, alpha=0.3)

else:
    plt.text(0.5, 0.5, "Keine plastischen Daten", ha='center')

plt.tight_layout()
try:
    fig.savefig(os.path.join(script_dir, "spatial_results.png"), dpi=300)
    fig_hist.savefig(os.path.join(script_dir, "history_results.png"), dpi=300)
except Exception as e:
    print(f"Error saving results: {e}")
plt.show()

print(f"Simulation complete. Duration: {sim_duration:.2f}s")

print(f"Maximaler Hertzscher Druck p0: {p0_hz / 1e6:.2f} MPa")
print(f"Maximale Von-Mises-Spannung:   {max(element_results['svm']):.2f} MPa")
print(f"Maximale Verformung (Total):   {max(element_results['u_norm']):.4f} mm")