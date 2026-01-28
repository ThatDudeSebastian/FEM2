import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PolyCollection
import numpy as np
import math
import os
from mesh_utils import load_mesh, get_bcs_from_sets

# ==========================================
# ============ SETTINGS & CONFIG ===========
# ==========================================
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
# --- Material Parameters ---
# --- Material Parameters ---
E = 220e9
nu = 0.3
width = 0.00583

sigma_y = 350e6     # Fließspannung
H = 1.0e9          # Isotrope Verfestigung

# --- Force ---
F_total = -40000.0 # From Afg2_Newmark.py (was -5 MN). 1000 N in Y (pointing down)

# ==========================================
# ============ MESH LOADING ============
# ==========================================
x, conn, pt_sets, cell_sets = load_mesh(mesh_file, device=device, primary_element_type=element_type)

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
    tol = 1e-4 # Stricter tolerance likely needed for fine grid
    fixed_nodes = torch.where(torch.abs(x[:, 0] - x_min) < tol)[0]

for n in fixed_nodes:
    drlt_bcs.append([int(n), 0, 0.0]) # Fix X
    drlt_bcs.append([int(n), 1, 0.0]) # Fix Y

# Load at right edge in Y
if "Loaded" in pt_sets and len(pt_sets["Loaded"]) > 0:
    print("BC INFO: Using 'Loaded' Node Set from .inp file.")
    load_indices = pt_sets["Loaded"]
    load_nodes = torch.tensor(load_indices, dtype=torch.long)
else:
    print("BC INFO: Node Set 'Loaded' NOT found. Fallback to coordinate search.")
    tol = 1e-4
    load_nodes = torch.where(torch.abs(x[:, 0] - x_max) < tol)[0]

f_per_node = F_total / max(1, len(load_nodes))
for n in load_nodes:
    neum_bcs.append([int(n), 1, f_per_node])
    
print("-" * 20)

drlt = torch.tensor(drlt_bcs); neum = torch.tensor(neum_bcs)

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
C4 = torch.zeros(2, 2, 2, 2); fac = E / (1 - nu**2)
C4[0,0,0,0]=C4[1,1,1,1]=fac; C4[0,0,1,1]=C4[1,1,0,0]=fac*nu; C4[0,1,0,1]=C4[1,0,0,1]=C4[0,1,1,0]=C4[1,0,1,0]=E/(2*(1+nu))


def von_mises_return(eps, state):
    ep = state["ep"]
    alpha = state["alpha"]

    sig_trial = torch.tensordot(C4, eps-ep, dims=2)

    I = torch.eye(2, device=eps.device)
    s = sig_trial - torch.trace(sig_trial)/2 * I

    seq = torch.sqrt(1.5*torch.sum(s*s))

    f = seq - (sigma_y + H*alpha)

    if f <= 0:
        return sig_trial, C4, state

    mu = E/(2*(1+nu))
    dgamma = f/(3*mu+H)

    n = s/seq
    sig = sig_trial - 2*mu*dgamma*n

    ep_new = ep + 1.5*dgamma*n
    alpha_new = alpha + dgamma

    Ct = C4 - (2*mu)**2/(3*mu+H)*(
        torch.einsum("ij,kl->ijkl", n, n)
    )

    return sig, Ct, {"ep":ep_new, "alpha":alpha_new}

state_gp = [[{"ep":torch.zeros(2,2),"alpha":0.0} for q in range(nqp)] for e in range(nel)]


K = torch.zeros(nnp*ndf, nnp*ndf); f_ext = torch.zeros(nnp*ndf, 1)

for el in range(nel):
    n_idx = conn[el]; xe = x[n_idx].t(); Ke = torch.zeros(nen*ndf, nen*ndf)
    for q in range(nqp):
        N, gamma = get_shape_data(qpt[q], nen)
        Je = xe.mm(gamma)
        detJ = torch.det(Je)
        if detJ <= 0:
            print(f"WARNING: Element {el}, QP {q}: det(J) = {detJ.item()} <= 0")
        
        dv = detJ * w8[q] * width; G = gamma.mm(torch.inverse(Je))
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
for dof in floating_dofs_list:
    drlt_mask[int(dof)] = 1.0

free_dofs = torch.nonzero(1.0 - drlt_mask)[:, 0]
for bc in neum: f_ext[int(bc[0])*ndf + int(bc[1])] = bc[2]

# ==========================================
# ============ NONLINEAR SOLVER ============
# ==========================================

n_steps = 20
newton_tol = 1e-8
newton_max = 30

track_node = int(load_nodes[0])
track_dof = track_node*ndf + 1

disp_lin = []
disp_pl = []
load_hist = []

u = torch.zeros(nnp*ndf,1)

for step in range(1,n_steps+1):

    fac = step/n_steps
    print(f"\nSTEP {step}/{n_steps}")

    f_ext = torch.zeros(nnp*ndf,1)
    for bc in neum:
        f_ext[int(bc[0])*ndf+int(bc[1])] = fac*bc[2]

    for it in range(newton_max):

        Kt = torch.zeros_like(K)
        fint = torch.zeros_like(f_ext)

        for el in range(nel):
            n_idx = conn[el]; xe = x[n_idx].t()
            edofs=[]
            for n in n_idx: edofs.extend([int(n)*ndf,int(n)*ndf+1])
            ue = u[edofs].reshape(-1,2).t()

            Ke = torch.zeros(nen*ndf,nen*ndf)
            fe = torch.zeros(nen*ndf,1)

            for q in range(nqp):
                N,gamma = get_shape_data(qpt[q],nen)
                Je = xe.mm(gamma)
                dv = torch.det(Je)*w8[q]*width
                G = gamma.mm(torch.inverse(Je))

                eps = 0.5*(ue.mm(G)+(ue.mm(G)).t())
                sig,Ct,state_gp[el][q] = von_mises_return(eps,state_gp[el][q])

                for A in range(nen):
                    for B in range(nen):
                        KAB = torch.tensordot(G[A],
                                torch.tensordot(Ct,G[B],[[3],[0]]),[[0],[0]])
                        Ke[A*ndf:A*ndf+2,B*ndf:B*ndf+2]+=dv*KAB

                    fe[A*ndf:A*ndf+2,0] += dv * (sig @ G[A].unsqueeze(1)).squeeze()

            idx=torch.tensor(edofs)
            Kt[idx.unsqueeze(1),idx]+=Ke
            fint[idx]+=fe

        R = f_ext-fint
        Rf = R[free_dofs]

        if torch.norm(Rf)<newton_tol:
            print(" converged")
            break

        du=torch.zeros_like(u)
        du_f=torch.linalg.solve(Kt[free_dofs][:,free_dofs],Rf)
        du[free_dofs]=du_f
        u+=du

    load_hist.append(fac*F_total)
    disp_pl.append(u[track_dof].item())

# linear reference
u_lin=torch.zeros(nnp*ndf,1)
for bc in neum: f_ext[int(bc[0])*ndf+int(bc[1])]=bc[2]
u_lin[free_dofs]=torch.linalg.solve(K[free_dofs][:,free_dofs],f_ext[free_dofs])

# ==========================================
# ============ POSTPROCESS NONLINEAR =========
# ==========================================

element_results = {"svm": [], "u": []}

for e in range(nel):
    n_idx = conn[e]
    xe = x[n_idx].t()
    edofs = []
    for n in n_idx:
        edofs.extend([int(n)*ndf, int(n)*ndf+1])

    ue = u[edofs].reshape(-1, 2).t()

    N, gamma = get_shape_data(torch.tensor([0.0, 0.0]), nen)
    Je = xe.mm(gamma)
    G = gamma.mm(torch.inverse(Je))

    eps = 0.5 * (ue.mm(G) + (ue.mm(G)).t())
    sig = torch.tensordot(C4, eps, dims=2)

    svm = torch.sqrt(sig[0,0]**2 + sig[1,1]**2 
                     - sig[0,0]*sig[1,1] 
                     + 3*sig[0,1]**2)

    element_results["svm"].append(float(svm / 1e6))
    element_results["u"].append(float(torch.norm(torch.mean(ue.t(), dim=0))) * 1000)



for i in range(1,n_steps+1):
    disp_lin.append(i/n_steps*u_lin[track_dof].item())


# ==========================================
# ============ VISUALIZATION ============
# ==========================================
scale = 0.15 * torch.max(torch.abs(x)) / (torch.max(torch.abs(u)) + 1e-25)
x_def = x + scale * u.reshape(-1, 2)
# Re-order connectivity for plotting if Q8 (matplotlib only likes linear quads nicely, or we just plot corners)
# Q8 order: BL, BR, TR, TL, ...
# We'll use just the first 4 nodes for the patch plot to keep it simple and working
plot_conn = conn[:, :4] if nen >= 4 else conn
limit = torch.max(torch.abs(x)) * 1.3

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
pc_u.set_array(np.array(element_results["u"]))
ax2.add_collection(pc_u); cbar2 = plt.colorbar(pc_u, ax=ax2, label="Verschiebung [mm]")
ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit); ax2.set_aspect('equal')

# 3. Discrete Stress
ax3.set_title("3. Von Mises Spannung [MPa]", fontweight='bold')
pc_s = PolyCollection(verts_def, cmap='jet', edgecolors='k', lw=0.5)
pc_s.set_array(np.array(element_results["svm"]))
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
            ann2.set_text(f"Element: {i}\nWert: {element_results['u'][i]:.4f} mm")
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
print("max plastic alpha:", max([state_gp[e][q]["alpha"] for e in range(nel) for q in range(nqp)]))

plt.figure(figsize=(7,5))
plt.plot(disp_lin,load_hist,'k--',label="linear")
plt.plot(disp_pl,load_hist,'r',lw=2,label="plastisch")
plt.xlabel("Verschiebung [m]")
plt.ylabel("Last [N]")
plt.title("Last–Verschiebungs–Kurve")
plt.grid(True)
plt.legend()
plt.show()

plt.tight_layout(); plt.show()
