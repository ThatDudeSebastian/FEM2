
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

# 1. Standard-Datei (Newmark Task)
mesh_file = os.path.abspath(os.path.join(script_dir, "HA4_src_task", "newmark_task.inp"))

# 2. BENUTZER-DATEI (Hier einkommentieren, um die Standard-Datei zu überschreiben)
# mesh_file = "/Users/hanne/Desktop/mein_neues_mesh.msh"
# mesh_file = "/Users/hanne/Desktop/mein_neues_mesh.inp"

print(f"Loading mesh from: {mesh_file}")

# For Newmark task, we used Q8 elements (8 nodes)
element_type = 'quad8' 

# --- Material Parameters ---
E = 205e9
nu = 0.3
width = 0.00583

sigma_y = 350e6    # Fließspannung
H = 209e7          # Gesamt-Verfestigungsmodul (H_iso + H_kin)
r = 0.0            # Faktor der Mischung (0=rein kinematisch, 1=rein isotrop)

# --- Force ---
F_total = -2500000.0 

# --- Cyclic force loading ---
n_cycles = 2.0
n_steps_per_cycle = 60
n_steps = n_cycles * n_steps_per_cycle

F_amp = F_total          # Amplitude der zyklischen Kraft (N), z.B. -40000
use_ramp_in = True
n_ramp_steps = 5         # sanftes Einschwingen für Newton


# ==========================================
# ============ MESH LOADING ============
# ==========================================
x, conn, pt_sets, cell_sets, _ = load_mesh(mesh_file, device=device, primary_element_type=element_type)

print("Lx =", (x[:,0].max()-x[:,0].min()).item(), "Ly =", (x[:,1].max()-x[:,1].min()).item())
e0 = conn[0]
pts = x[e0[:4]]  # grob
print("approx element size =", torch.norm(pts[1]-pts[0]).item())


# Fix for "Floating Nodes" (Center nodes of Q9 grid not used by Q8 elements)
# User requested NOT to remove nodes. So we keep x and conn as is.
# We will stabilize these unused nodes later by adding them to the boundary conditions.

nnp, nel, nen = x.shape[0], conn.shape[0], conn.shape[1]
ndf = 2 

# Need to import get_bcs_from_sets? Actually we can access pt_sets directly.
# pt_sets is a dictionary: {"Fixed": [indices...], "Loaded": [indices...]}

# --- Boundary Conditions ---
drlt_bcs = []
neum_bcs = []
x_min, x_max = torch.min(x[:, 0]), torch.max(x[:, 0])

# Logic: Check if "Fixed" set exists. If so, use it. Else use coords.
print("-" * 20)
if "Fixed" in pt_sets and len(pt_sets["Fixed"]) > 0:
    print("BC INFO: Using 'Fixed' Node Set from .inp file.")
    fixed_indices = pt_sets["Fixed"]
    # Check if indices are 0-based? meshio usually returns 0-based.
    # Note: pt_sets values are numpy arrays.
    fixed_nodes = torch.tensor(fixed_indices, dtype=torch.long)
else:
    print("BC INFO: Node Set 'Fixed' NOT found. Fallback to coordinate search.")
    # Tolerance for fine mesh
    # x_min/max are scalar tensors, so we use item() for cleaner python float arithmetic if needed, 
    # or keep torch ops. Newmark task grid is regular.
    tol = 1e-3 # Stricter tolerance likely needed for fine grid
    fixed_nodes = torch.where(torch.abs(x[:, 0] - x_min) < tol)[0]
# RB für Biegung
# for n in fixed_nodes:
#     drlt_bcs.append([int(n), 0, 0.0]) # Fix X
#     drlt_bcs.append([int(n), 1, 0.0]) # Fix Y
# 1) u_x = 0 auf der linken Kante (alle fixed_nodes)

# ----------------------------------
# Determine load nodes (right edge)
# ----------------------------------
tol = 1e-8
load_nodes = torch.where(torch.abs(x[:, 0] - x_max) < tol)[0]


print(f"BC INFO: Number of load nodes (right edge) = {len(load_nodes)}")


for n in fixed_nodes:
    drlt_bcs.append([int(n), 0, 0.0])  # Fix X

ys = x[fixed_nodes, 1]
n_ref = int(fixed_nodes[torch.argmin(ys)])
drlt_bcs.append([n_ref, 1, 0.0])      # nur 1x uy=0 gegen RB



# Load at right edge in Y
if "Loaded" in pt_sets and len(pt_sets["Loaded"]) > 0:
    print("BC INFO: Using 'Loaded' Node Set from .inp file.")
    load_indices = pt_sets["Loaded"]
    load_nodes = torch.tensor(load_indices, dtype=torch.long)
else:
    print("BC INFO: Node Set 'Loaded' NOT found. Fallback to coordinate search.")
    tol = 1e-3
    load_nodes = torch.where(torch.abs(x[:, 0] - x_max) < tol)[0]

# Biegung y-Last
# f_per_node = F_total / max(1, len(load_nodes))
# for n in load_nodes:
#     neum_bcs.append([int(n), 1, f_per_node])
# [V7 FIX] Q8 Consistent Nodal Loads (Simpson's Rule 1/6 : 4/6 : 1/6)
# Uniform force on Q8 quadratic edge nodes causes distortion ("collapsed elements").
# We must distribute F_total according to shape functions.
if 'quad8' in element_type or nen == 8:
    print("BC INFO: Calculating consistent nodal loads for Q8 elements...")
    node_weights = torch.zeros(nnp)
    
    # Identify right edge elements
    for e in range(nel):
        n_idx = conn[e] # 0-7
        # Right edge in local Q8 indexing (Counter-Clockwise from BL):
        # 0(BL), 1(BR), 2(TR), 3(TL), 4(B), 5(R), 6(T), 7(L)
        # Right edge nodes: 1 (Corner), 2 (Corner), 5 (Midside)
        edge_nodes = [n_idx[1], n_idx[2], n_idx[5]]
        
        # Check if this edge is on the load boundary aka all nodes in load_nodes
        # (We converted load_nodes to tensor earlier, let's use set for speed or just tensor checks)
        # load_nodes is a tensor of indices.
        
        on_boundary = True
        for en in edge_nodes:
            if not torch.isin(en, load_nodes):
                on_boundary = False
                break
        
        if on_boundary:
            # Add weights: Corners get 1, Midside gets 4
            # Logic: Integral of N_i over edge length L_e. 
            # int N_corner = L_e/6, int N_mid = 4L_e/6.
            # We treat L_e as uniform.
            node_weights[edge_nodes[0]] += 1.0
            node_weights[edge_nodes[1]] += 1.0
            node_weights[edge_nodes[2]] += 4.0

    # Extract sum of weights for loaded nodes only
    total_weight = torch.sum(node_weights[load_nodes])
    if total_weight == 0:
        print("WARNING: No Q8 boundary edges found matching load_nodes! Fallback to uniform.")
        f_per_node = F_total / max(1, len(load_nodes))
        for n in load_nodes:
            neum_bcs.append([int(n), 0, f_per_node])
    else:
        print(f"BC INFO: Distributed Force using Q8 consistent weights (Total W={total_weight})")
        for n in load_nodes:
            w = node_weights[n]
            if w > 0:
                f_n = (w / total_weight) * F_total
                neum_bcs.append([int(n), 0, float(f_n)])

else:
    # Linear elements / Fallback
    f_per_node = F_total / max(1, len(load_nodes))
    for n in load_nodes:
        neum_bcs.append([int(n), 0, f_per_node])  # dof=0 => x

    
print("-" * 20)

drlt = torch.tensor(drlt_bcs, dtype=torch.float64).reshape(-1, 3)
neum = torch.tensor(neum_bcs, dtype=torch.float64).reshape(-1, 3)


# ==========================================
# ============ CORE SOLVER ============
# ==========================================
def get_shape_data(xi, nen):
    e, n = xi[0], xi[1]
    N = torch.zeros(nen); gamma = torch.zeros(nen, 2)
    if nen == 4:
        # Q4 Shape Functions
        N[0]=0.25*(1-e)*(1-n); N[1]=0.25*(1+e)*(1-n); N[2]=0.25*(1+e)*(1+n); N[3]=0.25*(1-e)*(1+n)
        gamma[0,0]=-0.25*(1-n); gamma[0,1]=-0.25*(1-e); gamma[1,0]=0.25*(1-n); gamma[1,1]=-0.25*(1+e)
        gamma[2,0]=0.25*(1+n); gamma[2,1]=0.25*(1+e); gamma[3,0]=-0.25*(1+n); gamma[3,1]=0.25*(1-e)
    elif nen == 8:
        # Q8 Shape Functions (from Afg2_Newmark.py)
        # 0:BL, 1:BR, 2:TR, 3:TL, 4:B, 5:R, 6:T, 7:L
        # Corner nodes
        N[0] = 0.25 * (1-e)*(1-n)*(-e-n-1); N[1] = 0.25 * (1+e)*(1-n)*(e-n-1)
        N[2] = 0.25 * (1+e)*(1+n)*(e+n-1);  N[3] = 0.25 * (1-e)*(1+n)*(-e+n-1)
        # Midside nodes
        N[4] = 0.5 * (1-e*e)*(1-n); N[5] = 0.5 * (1+e)*(1-n*n)
        N[6] = 0.5 * (1-e*e)*(1+n); N[7] = 0.5 * (1-e)*(1-n*n)

        # Derivatives dN/de (gamma[:,0]) and dN/dn (gamma[:,1])
        # d/de
        gamma[0, 0] = 0.25 * (1-n)*(-1)*(-e-n-1) + 0.25*(1-e)*(1-n)*(-1)
        gamma[1, 0] = 0.25 * (1-n)*(1)*(e-n-1) + 0.25*(1+e)*(1-n)*(1)
        gamma[2, 0] = 0.25 * (1+n)*(1)*(e+n-1) + 0.25*(1+e)*(1+n)*(1)
        gamma[3, 0] = 0.25 * (1+n)*(-1)*(-e+n-1) + 0.25*(1-e)*(1+n)*(-1)
        gamma[4, 0] = 0.5 * (-2*e)*(1-n);      gamma[5, 0] = 0.5 * (1)*(1-n*n)
        gamma[6, 0] = 0.5 * (-2*e)*(1+n);      gamma[7, 0] = 0.5 * (-1)*(1-n*n)
        # d/dn
        gamma[0, 1] = 0.25 * (1-e)*(-1)*(-e-n-1) + 0.25*(1-e)*(1-n)*(-1)
        gamma[1, 1] = 0.25 * (1+e)*(-1)*(e-n-1) + 0.25*(1+e)*(1-n)*(-1)
        gamma[2, 1] = 0.25 * (1+e)*(1)*(e+n-1) + 0.25*(1+e)*(1+n)*(1)
        gamma[3, 1] = 0.25 * (1-e)*(1)*(-e+n-1) + 0.25*(1-e)*(1+n)*(1)
        gamma[4, 1] = 0.5 * (1-e*e)*(-1);      gamma[5, 1] = 0.5 * (1+e)*(-2*n)
        gamma[6, 1] = 0.5 * (1-e*e)*(1);       gamma[7, 1] = 0.5 * (1-e)*(-2*n)
    
    return N, gamma

# Gauss Quadrature 
# For Q4 we used 2x2. For Q8, 3x3 is standard (nqp=9).
if 'quad8' in element_type or '8' in element_type:
    nqp=9
    qpt = torch.zeros(nqp, 2); w8 = torch.zeros(nqp)
    a = math.sqrt(3.0 / 5.0); w1 = 5.0/9.0; w2 = 8.0/9.0
    # Tensor product rule 3x3
    vals = [-a, 0, a]; ws = [w1, w2, w1]
    k=0
    for i in range(3):
        for j in range(3):
            qpt[k,0] = vals[j]; qpt[k,1] = vals[i] # Inner loop x, outer loop y (or vice versa, symmetric)
            w8[k] = ws[i] * ws[j]
            k+=1
else:
    # Q4 defaults
    nqp=4; a=1.0/math.sqrt(3.0); qpt=torch.tensor([[-a,-a],[a,-a],[a,a],[-a,a]]); w8=torch.ones(4)
C4 = torch.zeros(2, 2, 2, 2)
# Plane Strain parameters
mu = E / (2.0 * (1.0 + nu))
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

# C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
# 1111 -> lam + 2mu
C4[0,0,0,0] = lam + 2.0*mu
C4[1,1,1,1] = lam + 2.0*mu
# 1122 -> lam
C4[0,0,1,1] = lam
C4[1,1,0,0] = lam
# 1212 -> mu (and symmetric indices for shear)
C4[0,1,0,1] = mu
C4[1,0,0,1] = mu
C4[0,1,1,0] = mu
C4[1,0,1,0] = mu



def von_mises_return(eps2, state):
    """
    PDF-konformer J2-Return-Mapping (Box 6.5) mit gemischter Verfestigung.
    Interne Variablen:
      ep : plastische Dehnung (3x3, deviatorisch)
      k  : isotrope Variable
      a  : kinematische Variable (3x3, deviatorisch)
    Hardening-Spannungen nach PDF:
      kappa = - r H k
      alpha_h = -(2/3)(1-r)H a
    Yield:
      Phi = || s_dev + alpha_h || - sqrt(2/3)*(sigma_y - kappa) <= 0
    """

    mu = E / (2.0*(1.0+nu))
    K  = E / (3.0*(1.0-2.0*nu))

    I3 = torch.eye(3, dtype=torch.float64)

    # --- build 3D strain for plane strain: eps33=0, eps13=eps23=0 ---
    eps = torch.zeros(3,3, dtype=torch.float64)
    eps[0:2, 0:2] = eps2

    ep = state["ep"]
    k  = float(state["k"])
    a  = state["a"]

    # enforce deviatoric a, ep (numerical hygiene)
    a  = a - (torch.trace(a)/3.0) * I3
    ep = ep - (torch.trace(ep)/3.0) * I3

    # elastic trial stress
    e_el = eps - ep
    tr_e = torch.trace(e_el)
    dev_e = e_el - (tr_e/3.0)*I3
    sig_trial = 2.0*mu*dev_e + K*tr_e*I3

    tr_s = torch.trace(sig_trial)
    p = tr_s/3.0
    s_tr = sig_trial - p*I3

    # hardening stresses (PDF)
    kappa = - r * H * k
    alpha_h = -(2.0/3.0) * (1.0 - r) * H * a

    # reduced deviatoric trial stress
    s_red_tr = s_tr + alpha_h
    norm_red = torch.sqrt(torch.sum(s_red_tr*s_red_tr) + 1e-30)

    # Phi_tr
    Phi_tr = norm_red - math.sqrt(2.0/3.0) * (sigma_y - kappa)

    # elastic
    if float(Phi_tr) <= 0.0:
        # elastic tangent (3D)
        I4s = torch.zeros(3,3,3,3, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                for k2 in range(3):
                    for l2 in range(3):
                        I4s[i,j,k2,l2] = 0.5*((1.0 if (i==k2 and j==l2) else 0.0) +
                                              (1.0 if (i==l2 and j==k2) else 0.0))
        IoxI = torch.einsum("ij,kl->ijkl", I3, I3)
        Pdev = I4s - (1.0/3.0)*IoxI
        C3 = K*IoxI + 2.0*mu*Pdev

        Ct2 = C3[0:2, 0:2, 0:2, 0:2]
        sig2 = sig_trial[0:2, 0:2]
        state_new = {"ep": ep, "k": k, "a": a}
        return sig2, Ct2, state_new

    # plastic (Box 6.5)
    v = s_red_tr / norm_red  # direction

    B = 2.0*mu + (2.0/3.0)*H
    A = 2.0*mu

    dlam = float(Phi_tr) / float(B)
    dlam_t = torch.tensor(dlam, dtype=torch.float64)

    # update internal variables (PDF)
    ep_new = ep + dlam_t * v
    k_new  = k  + dlam * math.sqrt(2.0/3.0)
    a_new  = a  + dlam_t * v

    # keep deviatoric
    ep_new = ep_new - (torch.trace(ep_new)/3.0) * I3
    a_new  = a_new  - (torch.trace(a_new)/3.0) * I3

    # stress update (PDF Eq. 6.53)
    s_new = s_tr - A * dlam_t * v
    sig = s_new + p*I3

    # consistent algorithmic tangent (PDF Eq. 6.61-6.63)
    I4s = torch.zeros(3,3,3,3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            for k2 in range(3):
                for l2 in range(3):
                    I4s[i,j,k2,l2] = 0.5*((1.0 if (i==k2 and j==l2) else 0.0) +
                                          (1.0 if (i==l2 and j==k2) else 0.0))
    IoxI = torch.einsum("ij,kl->ijkl", I3, I3)
    Idev_sym = I4s - (1.0/3.0)*IoxI

    c1 = 2.0*mu * (1.0 - (A/float(norm_red))*dlam)
    c2 = 2.0*mu * A * (dlam/float(norm_red) - 1.0/float(B))

    vv = torch.einsum("ij,kl->ijkl", v, v)
    C3 = K*IoxI + c1*Idev_sym + c2*vv

    Ct2 = C3[0:2, 0:2, 0:2, 0:2]
    sig2 = sig[0:2, 0:2]

    state_new = {"ep": ep_new, "k": k_new, "a": a_new}
    return sig2, Ct2, state_new






# Funktion für Fließdiagramm

def yield_curve_s11_t12_pdf(k, a, n=361):
    # Effective yield stress per PDF: sigma_y - kappa with kappa = - r H k  -> sigma_y + r H k
    sig_eff = float(sigma_y + r * H * float(k))

    I3 = torch.eye(3, dtype=torch.float64)
    a_dev = a - (torch.trace(a)/3.0) * I3
    alpha_h = -(2.0/3.0) * (1.0 - r) * H * a_dev  # hardening stress

    # yield: || s + alpha_h || = sqrt(2/3) * sig_eff
    # in (sigma11, tau12) cut with others ~0: sqrt(s11^2 + 3*t12^2) = sig_eff
    thetas = np.linspace(0.0, 2*np.pi, n)

    # center shift in s-space is -alpha_h (because ||s + alpha|| = R)
    shift_s11 = -float(alpha_h[0, 0])
    shift_t12 = -float(alpha_h[0, 1])

    xs = sig_eff*np.cos(thetas) + shift_s11
    ys = (sig_eff/np.sqrt(3.0))*np.sin(thetas) + shift_t12
    return xs, ys


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

def yield_curve(alpha, beta, n=361, rmax=3.0e9):
    # rmax groß genug wählen, damit Schnittpunkt sicher existiert (Pa)
    thetas = np.linspace(0, 2*np.pi, n)
    xs, ys = [], []

    for th in thetas:
        c, s = math.cos(th), math.sin(th)

        # f(r) along ray: (sig1, sig2) = r*(c, s)
        def f_of_r(rr):
            return yield_function_sigma12(rr*c, rr*s, alpha, beta)

        # bracket
        r_lo = 0.0
        f_lo = f_of_r(r_lo)  # sollte <=0 sein
        r_hi = rmax
        f_hi = f_of_r(r_hi)

        # falls rmax zu klein war: expand bis f_hi > 0
        tries = 0
        while f_hi <= 0.0 and tries < 20:
            r_hi *= 2.0
            f_hi = f_of_r(r_hi)
            tries += 1

        if f_hi <= 0.0:
            # kein Schnitt gefunden -> skip oder append nan
            xs.append(np.nan); ys.append(np.nan)
            continue

        # bisection
        for _ in range(60):
            r_mid = 0.5*(r_lo+r_hi)
            f_mid = f_of_r(r_mid)
            if f_mid > 0:
                r_hi = r_mid
            else:
                r_lo = r_mid

        r_star = r_hi
        xs.append(r_star*c)
        ys.append(r_star*s)

    return np.array(xs), np.array(ys)




def clone_state_gp(state_gp):
    # schneller als deepcopy: klont nur Tensors, k als float
    out = []
    for e_list in state_gp:
        row = []
        for st in e_list:
            row.append({
                "ep": st["ep"].clone(),
                "k": float(st["k"]),
                "a": st["a"].clone()
            })
        out.append(row)
    return out



state_gp = [[{"ep": torch.zeros(3,3, dtype=torch.float64),
              "k": 0.0,
              "a": torch.zeros(3,3, dtype=torch.float64)}
             for q in range(nqp)] for e in range(nel)]


state_gp_old = clone_state_gp(state_gp)



K = torch.zeros(nnp*ndf, nnp*ndf); f_ext = torch.zeros(nnp*ndf, 1)

print("Assembling linear stiffness matrix...")
for el in tqdm(range(nel), desc="Assembly"):
    n_idx = conn[el]; xe = x[n_idx].t(); Ke = torch.zeros(nen*ndf, nen*ndf)
    for q in range(nqp):
        N, gamma = get_shape_data(qpt[q], nen)
        Je = xe.mm(gamma)
        detJ = torch.det(Je)
        if detJ <= 0:
            print(f"WARNING: Element {el}, QP {q}: det(J) = {detJ.item()} <= 0")
        
        dv = detJ * w8[q] * width; G = torch.linalg.solve(Je.T, gamma.T).T
        for A in range(nen):
            for B in range(nen):
                KAB = torch.tensordot(G[A], torch.tensordot(C4, G[B], [[3],[0]]), [[0], [0]])
                Ke[A*ndf:A*ndf+ndf, B*ndf:B*ndf+ndf] += dv * KAB
    idx = []; [idx.extend([int(n)*ndf, int(n)*ndf+1]) for n in n_idx]
    idx_t = torch.tensor(idx)
    K[idx_t.unsqueeze(1), idx_t] += Ke

# --- Stabilization for Floating Nodes ---
# Identify nodes with no stiffness (zero diagonal in K)
k_diag = torch.diag(K)
# Check 2-norm of the 2x2 blocks or just check both dofs.
# If a node is unconnected, both X and Y diagonal entries will be exactly zero.
zero_dofs = torch.where(torch.abs(k_diag) < 1e-20)[0]

if len(zero_dofs) > 0:
    print(f"DEBUG: Found {len(zero_dofs)} floating DOFs (unused nodes). Stabilizing by fixing them to 0.")
    # Add to drlt_mask to treat them as fixed (Value 0 by default in drlt_vals)
    # We update drlt_mask before calculating free_dofs
    # Note: drlt_mask is created below, so we'll just inject these indices into a list to append to drlt_bcs logic
    # OR we just modify the mask creation loop.
    # Let's collect them now and set mask later.
    floating_dofs_list = zero_dofs.tolist()
else:
    floating_dofs_list = []

drlt_mask = torch.zeros(nnp*ndf, 1)
for bc in drlt: drlt_mask[int(bc[0])*ndf + int(bc[1])] = 1.0

# Apply stabilization
# Apply stabilization
fixed_floating_count = 0
for dof in floating_dofs_list:
    # Safety Check: Don't fix loaded DOFs!
    is_loaded = False
    for bc in neum:
        load_dof = int(bc[0])*ndf + int(bc[1])
        if int(dof) == load_dof:
            is_loaded = True
            break
    
    if not is_loaded:
        drlt_mask[int(dof)] = 1.0
        fixed_floating_count += 1
    else:
        print(f"CRITICAL WARNING: Node {int(dof)//ndf} DOF {int(dof)%ndf} has zero stiffness but IS LOADED. Not fixing it! (Check Mesh/Element Type)")

print(f"Stabilization: Fixed {fixed_floating_count} floating DOFs (skipped loaded ones).")

free_dofs = torch.nonzero(1.0 - drlt_mask)[:, 0]
for bc in neum: f_ext[int(bc[0])*ndf + int(bc[1])] = bc[2]

# ==========================================
# ============ NONLINEAR SOLVER ============
# ==========================================
eps_norm_max = 0.0

newton_tol = 1e-3
newton_max = 45   # Schritte

track_node = int(load_nodes[0])
track_dof = track_node*ndf + 0


disp_pl = []
load_hist = []
load_target_hist = []  # [V5 CHANGE] debug: target load history (fac_target*F_total)
fac_used_hist = []  # [V5.1 CHANGE] actually used fac (=fac_scale*fac_target) per converged step
fac_target_hist = []  # [V5.1 CHANGE] target fac (=ramp*cyclic) per converged step
fac_scale_hist = []  # [V5.1 CHANGE] fac_scale per converged step

u = torch.zeros(nnp*ndf,1)

def ramp(step, n_ramp_steps):
    if not use_ramp_in or n_ramp_steps <= 0:
        return 1.0
    return min(1.0, step / n_ramp_steps)

def cyclic_factor(step, n_steps_per_cycle):
    # step starts at 1
    return math.sin(2.0 * math.pi * (step-1) / n_steps_per_cycle)

track_el = 0
track_q = 0
tracking_elem = nel - 1

k_hist = []  # isotrope Variable k
a_hist  = []  # kinematische Variable a

eps_yy_hist = []
sig_yy_hist = []
eps_eq_hist = []
sig_eq_hist = []
eps_p_eq_hist = []
eps_p_eq_hist = []
eps_p_xx_hist = []
sig_1_hist = [] # Principal Stresses History
sig_2_hist = []

total_start_time = time.time()



step = 1

fac_scale_min = 1e-3          # Cutback-Untergrenze
max_cutbacks = 8              # max Versuche pro Step

fac_conv = 0.0   # zuletzt konvergierter Lastfaktor -> Konvergenz bei Beginn Plastizizät

fac_tol = 1e-3

pbar = tqdm(total=int(n_steps), desc="Simulation Steps")
while step <= n_steps:
    pbar.n = step - 1
    pbar.refresh()

    print(f"\nSTEP {step}/{n_steps}")
    t_start_step = time.time()

    fac_scale = 1.0               # aktuelle Skalierung des Step-Inkrements

    # freeze converged state from previous step
    state_gp_old = clone_state_gp(state_gp)
    u_old = u.clone()  # [V5 CHANGE] reset u on cutback retry

    # load factor target for this step (independent of cutbacks)
    fac_target = ramp(step, n_ramp_steps) * cyclic_factor(step, n_steps_per_cycle)

    cutbacks = 0  # [V5 CHANGE] count retries for THIS step

    substeps = 0
    max_substeps = 50


    # [V5 CHANGE] retry-loop for same step (cutbacks)
    while True:


        fac = fac_conv + fac_scale * (fac_target - fac_conv)
        print(f"   substep: fac_conv={fac_conv:.6f}, fac_target={fac_target:.6f}, fac_scale={fac_scale:.3e}, fac={fac:.6f}")

        

        # external force vector for this step (recomputed after cutback)
        f_ext = torch.zeros(nnp*ndf, 1)
        for bc in neum:
            f_ext[int(bc[0])*ndf + int(bc[1])] = fac * bc[2]

        # reset to last converged state before attempting Newton
        u = u_old.clone()  # [V5 CHANGE]
        converged = False

       

        for it in range(newton_max):

             # trial state from frozen state
            state_gp_iter = clone_state_gp(state_gp_old)

            # --- auto-pick init (only once, at a non-trivial load level) ---
            if track_el is None and it == 0 and torch.norm(f_ext[free_dofs]).item() > 1.0:
                best_val = -1.0
                best_el = 0
                best_q = 0

            # --- temporary storage for hysteresis tracking (per Newton iteration) ---
            eps_yy_tmp = {}
            sig_yy_tmp = {}
            eps_eq_tmp = {}
            sig_eq_tmp = {}
            
            eps_xx_tmp = {}  
            sig_xx_tmp = {}

            eps_p_eq_tmp = {}   
            eps_p_xx_tmp = {}     

            Kt = torch.zeros_like(K)
            fint = torch.zeros_like(f_ext)
            

            
            for el in range(nel):
                n_idx = conn[el]
                xe = x[n_idx].t()

                edofs = []
                for n in n_idx:
                    edofs.extend([int(n)*ndf, int(n)*ndf+1])

                ue = u[edofs].reshape(-1, 2).t()

                Ke = torch.zeros(nen*ndf, nen*ndf)
                fe = torch.zeros(nen*ndf, 1)

                for q in range(nqp):
                    N, gamma = get_shape_data(qpt[q], nen)
                    Je = xe.mm(gamma)
                    dv = torch.det(Je) * w8[q] * width

                    G = torch.linalg.solve(Je.T, gamma.T).T  # später optimieren

                    eps = 0.5 * (ue.mm(G) + (ue.mm(G)).t())

                    sig, Ct, state_gp_iter[el][q] = von_mises_return(eps, state_gp_old[el][q])

                    # --- tracking quantities for every qp ---
                    key = (el, q)
                    eps_xx_tmp[key] = eps[0, 0].item()
                    sig_xx_tmp[key] = sig[0, 0].item()


                    svm = math.sqrt(
                        sig[0,0].item()**2 + sig[1,1].item()**2
                        - sig[0,0].item()*sig[1,1].item()
                        + 3.0*sig[0,1].item()**2
                    )
                    evm = math.sqrt(
                        eps[0,0].item()**2 + eps[1,1].item()**2
                        - eps[0,0].item()*eps[1,1].item()
                        + 3.0*eps[0,1].item()**2
                    )
                    # äquivalente plastische Dehnung aus state_gp_iter (aktualisierter Zustand!)
                    ep3 = state_gp_iter[el][q]["ep"]  # 3x3
                    I3 = torch.eye(3, dtype=torch.float64)
                    ep_dev = ep3 - (torch.trace(ep3)/3.0) * I3
                    ep_eq = math.sqrt((2.0/3.0) * torch.sum(ep_dev*ep_dev).item())

                    eps_p_eq_tmp[key] = ep_eq
                    eps_p_xx_tmp[key] = ep3[0,0].item()


                    eps_eq_tmp[key] = evm
                    sig_eq_tmp[key] = svm

                    # auto-pick hotspot in first Newton iter (only if not set yet)
                    if track_el is None and it == 0 and torch.norm(f_ext[free_dofs]).item() > 1.0:
                        if evm > best_val:
                            best_val = evm
                            best_el = el
                            best_q = q

                    # --- update eps max ---
                    eps_norm_max = max(eps_norm_max, float(torch.sqrt(torch.sum(eps*eps))))

                    # assemble Ke, fe
                    for A in range(nen):
                        for B in range(nen):
                            KAB = torch.tensordot(
                                G[A],
                                torch.tensordot(Ct, G[B], [[3], [0]]),
                                [[0], [0]]
                            )
                            Ke[A*ndf:A*ndf+2, B*ndf:B*ndf+2] += dv * KAB

                        fe[A*ndf:A*ndf+2, 0] += dv * (sig @ G[A].unsqueeze(1)).squeeze()

                idx = torch.tensor(edofs)
                Kt[idx.unsqueeze(1), idx] += Ke
                fint[idx] += fe

    
            # finalize tracking point once (after first assembled state at nonzero load)
            if track_el is None and it == 0 and torch.norm(f_ext[free_dofs]).item() > 1.0:
                track_el = best_el
                track_q = best_q
                print(f"Tracking GP set to el={track_el}, q={track_q}, eps_eq≈{best_val:.3e}")

            # --- read tracking values for this iteration ---
            if track_el is None or track_q is None:
                # fallback: pick any available key (e.g. first qp) to avoid KeyError
                fallback_key = next(iter(eps_yy_tmp.keys()))
                track_eps_xx = eps_xx_tmp[fallback_key]
                track_sig_xx = sig_xx_tmp[fallback_key]
                track_eps_eq = eps_eq_tmp[fallback_key]
                track_sig_eq = sig_eq_tmp[fallback_key]
                track_eps_p_eq = eps_p_eq_tmp[fallback_key]
                track_eps_p_xx = eps_p_xx_tmp[fallback_key]

            else:
                track_eps_xx = eps_xx_tmp[(track_el, track_q)]
                track_sig_xx = sig_xx_tmp[(track_el, track_q)]
                track_eps_eq = eps_eq_tmp[(track_el, track_q)]
                track_sig_eq = sig_eq_tmp[(track_el, track_q)]
                track_eps_p_eq = eps_p_eq_tmp[(track_el, track_q)]
                track_eps_p_xx = eps_p_xx_tmp[(track_el, track_q)]


                


            # residual
            R = f_ext - fint
            Rf = R[free_dofs]

            Rf_norm = torch.norm(Rf)
            fext_norm = torch.norm(f_ext[free_dofs])

            # [V5 CHANGE] robust relative criterion (avoid division by tiny load near zero crossing)
            denom = max(float(fext_norm), 1.0)
            rel = float(Rf_norm) / denom

            # [KORREKTUR] Immer printen oder für step >= 6
            if True: 
                print(f"  it {it}: rel={rel:.3e}, ||Rf||={Rf_norm.item():.3e}")

            if rel < newton_tol:
                print(f" converged (rel={rel:.3e}, it={it})")

                # commit state
                state_gp = state_gp_iter
                st_tr = state_gp[track_el][track_q]

                k_hist.append(float(st_tr["k"]))
                a_hist.append(st_tr["a"].clone())

                # histories ONLY here
                eps_yy_hist.append(track_eps_xx)
                sig_yy_hist.append(track_sig_xx)
                eps_eq_hist.append(track_eps_eq)
                sig_eq_hist.append(track_sig_eq)
                eps_p_eq_hist.append(track_eps_p_eq)
                eps_p_xx_hist.append(track_eps_p_xx)   # plst. dehnung xx



                # [V5 CHANGE] store applied load from f_ext (not target)
                F_applied = 0.0
                for bc in neum:
                    dof = int(bc[0])*ndf + int(bc[1])
                    F_applied += float(f_ext[dof])
                
                F_target = float(fac_target * F_total)

                disp_pl.append(u[track_dof].item())
                load_hist.append(F_applied)
                load_target_hist.append(F_target)

                fac_used_hist.append(float(fac))  # [V5.1 CHANGE]
                fac_target_hist.append(float(fac_target))  # [V5.1 CHANGE]
                fac_scale_hist.append(float(fac_scale))  # [V5.1 CHANGE]


                converged = True

                # If Newton was "hard", reduce next increment a bit
                if it > 20:
                    fac_scale = max(0.5, fac_scale * 0.7)
                elif it < 6:
                    fac_scale = min(1.0, fac_scale * 1.1)
                else:
                    fac_scale = min(1.0, fac_scale * 1.0)

                break

                
            # Newton update (MUSS pro Iteration passieren)
            du = torch.zeros_like(u)
            du_f = torch.linalg.solve(Kt[free_dofs][:, free_dofs], Rf)
            du[free_dofs] = du_f
            # einfaches Dämpfungskonzept
            omega = 1.0
            u += omega * du


    


       
        # ---- after Newton loop: accept or cutback ----
        if converged:
            substeps += 1
            if substeps > max_substeps:
                raise RuntimeError(f"Step {step}: too many substeps (>{max_substeps}).")

            # Teilinkrement akzeptieren
            fac_conv = fac

            # sehr wichtig: neuen konvergierten Zustand als Startpunkt für das nächste Substep setzen
            state_gp_old = clone_state_gp(state_gp)
            u_old = u.clone()
            
            # --- TRACK STRESS PATH (Tracking Element) ---
            # Extract state for Element 'tracking_elem' (the last one), QP 0
            # Re-calculate stress from displacement (u) and state_new (state_gp)
            
            # Get displacement for tracking element
            n_idx_tr = conn[tracking_elem]
            # Gather flat, then reshape to (nen, ndf) -> (8,2)
            dof_indices_tr = []
            for n in n_idx_tr:
                dof_indices_tr.extend([int(n)*ndf, int(n)*ndf+1])
            ue_tr = u[dof_indices_tr].clone().reshape(nen, ndf)
            
            # QP 0
            N_tr, gamma_tr = get_shape_data(qpt[0], nen) # using QP 0
            xe_tr = x[n_idx_tr].t()
            Je_tr = xe_tr.mm(gamma_tr)
            G_tr = torch.linalg.solve(Je_tr.T, gamma_tr.T).T
            
            # grad_u = du/dX = ue.T * G  -> (2,8)*(8,2) = (2,2)
            grad_u_tr = ue_tr.t().mm(G_tr)
            eps_tr = 0.5 * (grad_u_tr + grad_u_tr.t())
            
            # Use CURRENT converged state (state_gp is the new one for next step, so it holds the just-converged updated vars)
            st_tr_q0 = state_gp[tracking_elem][0] 
            sig_tr, _, _ = von_mises_return(eps_tr, st_tr_q0)
            
            s11 = sig_tr[0,0].item()
            s22 = sig_tr[1,1].item()
            s12 = sig_tr[0,1].item()
            s_avg = (s11 + s22) / 2.0
            s_R = math.sqrt(((s11 - s22) / 2.0)**2 + s12**2)
            # Append to history
            sig_1_hist.append((s_avg + s_R) / 1e6)
            sig_2_hist.append((s_avg - s_R) / 1e6)
            # ---------------------------------------------

            # Ziel erreicht? -> globalen Step verlassen
            rem = fac_target - fac_conv
            if abs(rem) < fac_tol:
                fac_conv = fac_target   # snap exakt aufs Ziel
                fac_conv = fac_target   # snap exakt aufs Ziel
                step += 1
                t_end_step = time.time()
                print(f" -> Step completed in {t_end_step - t_start_step:.2f} s")
                break
            else:
             # Ziel noch nicht erreicht -> im selben Step weiter Richtung fac_target
             # Wichtig: NICHT zu aggressiv vergrößern, sonst fällst du sofort wieder in Cutbacks zurück
             fac_scale = min(1.0, fac_scale * 1.25)   # oder 1.1 wenn sehr nonlinear
             cutbacks = 0
             continue





        # not converged -> cutback and retry SAME step
        fac_scale *= 0.5
        cutbacks += 1
        if fac_scale < fac_scale_min or cutbacks > max_cutbacks:
            raise RuntimeError(f"Step {step} did not converge even after cutbacks.")
    print(f"  -> cutback: retry step {step} with fac_scale={fac_scale:.3e}")
    # loop continues with smaller fac_scale

pbar.n = int(n_steps)
pbar.refresh()
pbar.close()


# kontroll print
if len(disp_pl) > 0:
    print("u_track_end =", disp_pl[-1])
else:
    print("disp_pl is empty (no converged steps?)")

# linear reference
 #u_lin=torch.zeros(nnp*ndf,1)
 #for bc in neum: f_ext[int(bc[0])*ndf+int(bc[1])]=bc[2]
 #u_lin[free_dofs]=torch.linalg.solve(K[free_dofs][:,free_dofs],f_ext[free_dofs])

# ==========================================
# ============ POSTPROCESS NONLINEAR =========
# ==========================================

def clone_state_dict(st):
    out = {}
    for k, v in st.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out

element_results = {"svm": [], "u_norm": []}

print("Post-processing results...")
for e in tqdm(range(nel), desc="Post-processing"):
    n_idx = conn[e]
    xe = x[n_idx].t()  # (2, nen)

    # element dofs
    edofs = []
    for n in n_idx:
        edofs.extend([int(n)*ndf, int(n)*ndf+1])

    ue = u[edofs].reshape(-1, 2).t()  # (2, nen)

    # --- displacement magnitude per element (mean nodal magnitude) ---
    u_elem = u[edofs].reshape(-1, 2)  # (nen,2)
    u_norm = float(torch.sqrt(torch.sum(u_elem*u_elem, dim=1)).mean())
    element_results["u_norm"].append(u_norm * 1e3)  # [mm]

    # --- dv-weighted mean von Mises per element ---
    svm_sum = 0.0
    dv_sum  = 0.0

    for q in range(nqp):
        N, gamma = get_shape_data(qpt[q], nen)          # gamma: (nen,2)
        Je = xe.mm(gamma)                               # (2,2)
        detJ = torch.det(Je)
        if detJ <= 0:
            print(f"WARNING (post): Element {e}, QP {q}: det(J) = {detJ.item()} <= 0")

        dv = float(detJ * w8[q] * width)

        # spatial grads of shape functions
        G = torch.linalg.solve(Je.T, gamma.T).T         # (nen,2)

        # small strain tensor (2x2)
        eps = 0.5 * (ue.mm(G) + (ue.mm(G)).t())

        # IMPORTANT: use stored converged state at this Gauss point (do not modify it)
        st_q = clone_state_dict(state_gp[e][q])

        sig2, Ct2, _ = von_mises_return(eps, st_q)

        # von Mises in plane (tau12 included) using your formula:
        svm = math.sqrt(
            sig2[0,0].item()**2 + sig2[1,1].item()**2
            - sig2[0,0].item()*sig2[1,1].item()
            + 3.0*sig2[0,1].item()**2
        )

        svm_sum += svm * dv
        dv_sum  += dv

    svm_mean = svm_sum / max(dv_sum, 1e-30)
    element_results["svm"].append(svm_mean / 1e6)  # [MPa]


# ==========================================
# ============ VISUALIZATION ============
# ==========================================
# [V6 FIX] Scale factor was too high, causing element inversion in plots.
# Heuristic: max displacement in plot should be around 30% of element size, not 15% of body size.
# Element size approx 0.45.
elem_size_approx = 0.45
scale_auto = 0.5 * elem_size_approx / (torch.max(torch.abs(u)) + 1e-25)
# Cap scale to avoid "explosion" if u is very small, but also limit it if u is large.
scale = min(float(scale_auto), 50.0)
print(f"Plotting: auto-calculated scale factor = {scale:.2f}")

total_end_time = time.time()
sim_duration = total_end_time - total_start_time
print(f"TOTAL SIMULATION TIME: {sim_duration:.2f} s")

x_def = x + scale * u.reshape(-1, 2)
# Re-order connectivity for plotting if Q8 (matplotlib only likes linear quads nicely, or we just plot corners)
# Q8 order: BL, BR, TR, TL, ...
# We'll use just the first 4 nodes for the patch plot to keep it simple and working
plot_conn = conn[:, :4] if nen >= 4 else conn
limit = torch.max(torch.abs(x)) * 1.1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7.5))

# 1. Setup
# 1. Setup
ax1.set_title("1. Setup & Randbedingungen", fontweight='bold')
# Plot edges including midside nodes is tricky with simple plot, let's trace 0-4-1-5-2-6-3-7-0 if Q8
if nen == 8:
    trace_idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    for e in range(nel):
        els = conn[e][trace_idx]
        ax1.plot(x[els,0], x[els,1], color='black', lw=0.4, alpha=0.15)
else:
    for e in range(nel):
        pts = np.append(conn[e].numpy(), conn[e][0].numpy())
        ax1.plot(x[pts,0], x[pts,1], color='black', lw=0.4, alpha=0.15)
for bc in drlt:
    n, d = int(bc[0]), int(bc[1])
    xn, yn = x[n, 0].item(), x[n, 1].item()
    if d == 0: ax1.scatter(xn - 1.5, yn, color='green', marker='>', s=100, edgecolors='black', zorder=5)
    else: ax1.scatter(xn, yn - 1.5, color='green', marker='^', s=100, edgecolors='black', zorder=5)
for n_f in neum[:, 0].unique().long():
    ax1.arrow(x[n_f, 0].item(), x[n_f, 1].item()+6, 0, -4.5, head_width=1.8, head_length=1.8, fc='red', ec='red', zorder=6)
ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit); ax1.set_aspect('equal')

# 2. Deformed Displacement [mm]
ax2.set_title(f"2. Verschiebung [mm] (Skal. {scale:.1f}x)", fontweight='bold')
if nen == 8:
    trace_idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    for e in range(nel):
        els = conn[e][trace_idx]
        ax2.plot(x[els,0], x[els,1], color='gray', lw=0.3, ls='--', alpha=0.3)
else:
    for e in range(nel):
        pts = np.append(conn[e].numpy(), conn[e][0].numpy())
        ax2.plot(x[pts,0], x[pts,1], color='gray', lw=0.3, ls='--', alpha=0.3)

# Use plot_conn (first 4 nodes) for PolyCollection to avoid errors with 8-node polygons
verts_def = [x_def[plot_conn[e]].numpy() for e in range(nel)]
pc_u = PolyCollection(verts_def, cmap='viridis', edgecolors='k', lw=0.5)
pc_u.set_array(np.array(element_results["u_norm"]))
ax2.add_collection(pc_u); cbar2 = plt.colorbar(pc_u, ax=ax2, label="Verschiebung [mm]")
ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit); ax2.set_aspect('equal')

# 3. Discrete Stress
ax3.set_title("3. Von Mises Spannung [MPa]", fontweight='bold')
pc_s = PolyCollection(verts_def, cmap='jet', edgecolors='k', lw=0.5)
s_vals = np.array(element_results["svm"])
pc_s.set_array(s_vals)

# Fix colorbar noise if field is uniform
s_min, s_max = s_vals.min(), s_vals.max()
print(f"DEBUG PLOT: Stress Range: Min={s_min:.4e}, Max={s_max:.4e}, Delta={s_max-s_min:.4e}")

if s_max - s_min < 0.1:
    print(" -> Field is uniform. Enforcing fixed colorbar limits to identify noise.")
    s_mid = 0.5 * (s_min + s_max)
    pc_s.set_clim(s_mid - 0.1, s_mid + 0.1)

ax3.add_collection(pc_s); cbar3 = plt.colorbar(pc_s, ax=ax3, label="Spannung [MPa]")
ax3.set_xlim(-limit, limit); ax3.set_ylim(-limit, limit); ax3.set_aspect('equal')

# --- Hover Fix: Global figure annotation for "on top of everything" ---
# We use one annotation per subplot but ensure it doesn't clip and has high zorder
# Actually, the best way to be in front of colorbar (another axes) is to use figure coordinates or 
# adjust axes zorder.
ax2.set_zorder(1); ax3.set_zorder(1)
cbar2.ax.set_zorder(0); cbar3.ax.set_zorder(0) # Put colorbars behind

# Beautiful speech bubble style
ann_style = dict(boxstyle="round4,pad=0.5", fc="white", ec="gray", lw=1.5, alpha=0.95)

ann2 = ax2.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", 
                    bbox=ann_style, arrowprops=dict(arrowstyle="->"), zorder=1000, clip_on=False)
ann3 = ax3.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", 
                    bbox=ann_style, arrowprops=dict(arrowstyle="->"), zorder=1000, clip_on=False)
ann2.set_visible(False); ann3.set_visible(False)

def hover(event):
    ann2.set_visible(False); ann3.set_visible(False)
    if event.inaxes == ax2:
        cont, ind = pc_u.contains(event)
        if cont: 
            i = ind["ind"][0]; b = pc_u.get_paths()[i].get_extents()
            ann2.xy = [(b.x0+b.x1)/2, (b.y0+b.y1)/2]
            if ann2.xy[0] > 0.5 * (ax2.get_xlim()[0] + ax2.get_xlim()[1]): ann2.xyann = (-100, 20)
            else: ann2.xyann = (20, 20)
            ann2.set_text(f"Element: {i}\nWert: {element_results['u_norm'][i]:.4f} mm")
            ann2.set_visible(True)
    elif event.inaxes == ax3:
        cont, ind = pc_s.contains(event)
        if cont:
            i = ind["ind"][0]; b = pc_s.get_paths()[i].get_extents()
            ann3.xy = [(b.x0+b.x1)/2, (b.y0+b.y1)/2]
            if ann3.xy[0] > 0.5 * (ax3.get_xlim()[0] + ax3.get_xlim()[1]): ann3.xyann = (-100, 20)
            else: ann3.xyann = (20, 20)
            ann3.set_text(f"Element: {i}\nWert: {element_results['svm'][i]:.2f} MPa")
            ann3.set_visible(True)
    fig.canvas.draw_idle()

if interactive_hover:
    fig.canvas.mpl_connect("motion_notify_event", hover)

# checken ob pllasitizität eintritt    
print("max k (isotrop):", max([state_gp[e][q]["k"] for e in range(nel) for q in range(nqp)]))

beta_max = 0.0
for e in range(nel):
    for q in range(nqp):
        b = state_gp[e][q]["a"]
        beta_max = max(beta_max, float(torch.sqrt(torch.sum(b*b))))
print("max ||beta||:", beta_max)  # Verifizierung ob steigt bei r=0 oder 1

print("max ||eps||:", eps_norm_max) # dehnungen prüfen ob wirklich kleine dehnung (-0,05) ist

n = min(len(disp_pl), len(load_hist), len(load_target_hist))
disp_pl = disp_pl[:n]
load_hist = load_hist[:n]
load_target_hist = load_target_hist[:n]


print("len(disp_pl)      =", len(disp_pl))
print("len(load_hist)    =", len(load_hist))
print("len(fac_used_hist)=", len(fac_used_hist))
print("len(k_hist)       =", len(k_hist))
print("len(a_hist)       =", len(a_hist))

assert len(disp_pl) == len(load_hist) == len(fac_used_hist), "History arrays are inconsistent!"
assert len(k_hist) == len(a_hist) == len(fac_used_hist), "k/a history inconsistent!"

plt.figure(figsize=(15, 10))

# Subplot 1: Force-Displacement
plt.subplot(2, 2, 1)
plt.plot(disp_pl, load_hist, 'r', lw=2, label="Plastisch")
plt.plot(disp_pl, load_target_hist, 'k--', lw=1, label="Target")
plt.xlabel("Verschiebung [m]")
plt.ylabel("Last [N]")
plt.title("Last–Verschiebungs–Kurve")
plt.grid(True)
plt.legend()

# Subplot 2: Hysteresis Stress-Strain
plt.subplot(2, 2, 2)
plt.plot(eps_p_xx_hist, sig_yy_hist, lw=2, color='blue')
plt.xlabel(r"$\varepsilon^{p}_{xx}$ [-]")
plt.ylabel(r"$\sigma_{xx}$ [Pa]")
plt.title("Hysterese: Axialspannung vs. Plast. Dehnung")
plt.grid(True)

# Subplot 3: Equivalent Hysteresis
plt.subplot(2, 2, 3)
plt.plot(eps_p_eq_hist, sig_eq_hist, lw=2, color='green')
plt.xlabel(r"$\varepsilon^p_\mathrm{eq}$ [-]")
plt.ylabel(r"$\sigma_\mathrm{eq}$ [Pa]")
plt.title("Äquivalente Sannung vs. Akkumulierte Plast. Dehnung")
plt.grid(True)


# Subplot 4: Yield Surface (Principal Stress Space)
plt.subplot(2, 2, 4)
if len(k_hist) > 0:
    # plot: Initial state should be virgin material (k=0, a=0)
    # This guarantees we see the difference even if the first saved step is already hardened.
    k0 = 0.0
    a0 = torch.zeros(3,3)
    
    # Current state
    k1 = k_hist[-1]
    a1 = a_hist[-1]

    # Function signature: yield_curve_sigma12_closed(alpha, beta)
    # alpha = isotropic variable (k)
    # beta  = backstress tensor (a)
    # Calculate Backstress Tensor (Stress Units) from Internal Variable a (Strain Units)
    # alpha_h (PDF) = -(2/3)(1-r)H * a
    # Center of yield surface X = -alpha_h = (2/3)(1-r)H * a
    factor = (2.0/3.0) * (1.0 - r) * H
    beta0 = factor * a0
    beta1 = factor * a1

    x0, y0 = yield_curve_sigma12_closed(k0, beta0)
    x1, y1 = yield_curve_sigma12_closed(k1, beta1)

    plt.plot(x0/1e6, y0/1e6, 'k--', lw=2.5, label="Initial (Yield)")
    plt.plot(x1/1e6, y1/1e6, 'r-',  lw=2.5, label="Aktuell (Hardened)")
    
    plt.plot(x1/1e6, y1/1e6, 'r-',  lw=2.5, label="Aktuell (Hardened)")
    
    # Plane Strain Limit Lines (s1 - s2 = +/- 2/sqrt(3) * sig_eff)
    sig_eff_current = sigma_y + r*H*k1
    limit_val = (2.0 / math.sqrt(3.0)) * sig_eff_current / 1e6
    
    # Determine plot bounds based on Yield Surface and History
    x_min_plot = min(np.min(x1/1e6), np.min(sig_1_hist))
    x_max_plot = max(np.max(x1/1e6), np.max(sig_1_hist))
    margin = (x_max_plot - x_min_plot) * 0.3
    x_range = np.linspace(x_min_plot - margin, x_max_plot + margin, 100)

    # Line 1: s1 - s2 = limit_val => s2 = s1 - limit_val
    # Line 2: s1 - s2 = -limit_val => s2 = s1 + limit_val
    plt.plot(x_range, x_range - limit_val, 'r:', lw=1.5, alpha=0.6, label="PE Limit")
    plt.plot(x_range, x_range + limit_val, 'r:', lw=1.5, alpha=0.6)
    
    # Plot Stress Path Trajectory
    if len(sig_1_hist) > 0:
        plt.plot(sig_1_hist, sig_2_hist, 'b-', lw=1.5, alpha=0.7, label="Stress Path")
        plt.plot(sig_1_hist[-1], sig_2_hist[-1], 'bo', markersize=6, label="Current State")
    
    plt.xlim(x_min_plot - margin, x_max_plot + margin)
    plt.ylim(x_min_plot - margin, x_max_plot + margin)  # Keep roughly square aspect
    plt.xlabel(r"HS $\sigma_1$ [MPa]")
    plt.ylabel(r"HS $\sigma_2$ [MPa]")
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
plt.show()