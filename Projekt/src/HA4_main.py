import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Optimization Settings ---
import torch
# Set number of threads for PyTorch operations
# Adjust based on CPU. "8" requested by user.
NUM_CORES = 16
torch.set_num_threads(NUM_CORES)
os.environ['OMP_NUM_THREADS'] = str(NUM_CORES)
print(f"DEBUG: Configured PyTorch with {NUM_CORES} threads.")

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PolyCollection
import numpy as np
import math
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from mesh_utils import load_mesh, get_bcs_from_sets

# ==========================================
# ============ SETTINGS & CONFIG ===========
# ==========================================
torch.set_default_dtype(torch.float64)
device = torch.device("cpu") 

interactive_hover = True

# --- Mesh Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Standard-Datei
mesh_file = os.path.abspath(os.path.join(script_dir, "HA4_src_task", "newmark_task.inp"))

print(f"Loading mesh from: {mesh_file}")

# For Newmark task, we used Q8 elements (8 nodes)
element_type = 'quad8'

# --- Material Parameters ---
E = 220e9
nu = 0.3
width = 0.00583

sigma_y = 350e6     # Fließspannung
H = 1.0e9          # Isotrope Verfestigung

# --- Force ---
F_total = -40000.0

# ==========================================
# ============ MESH LOADING ============
# ==========================================
try:
    x, conn, pt_sets, cell_sets = load_mesh(mesh_file, device=device, primary_element_type='quad8')
    element_type = 'quad8'
except Exception as e:
    print(f"Standard load failed: {e}")
    print("Attempting fallbacks...")
    try:
        x, conn, pt_sets, cell_sets = load_mesh(mesh_file, device=device, primary_element_type='quad')
        element_type = 'quad4'
    except:
        # Load whatever is there
        x, c_dict, pt_sets, cell_sets = load_mesh(mesh_file, device=device, primary_element_type=None)
        
        print(f"DEBUG: Available element types: {list(c_dict.keys())}")
        
        # Priority list for 2D analysis
        vp = ['quad8', 'quad9', 'quad', 'triangle', 'triangle6']
        
        found_type = None
        for t in vp:
            if t in c_dict:
                conn = c_dict[t]
                element_type = t
                found_type = t
                break
        
        if found_type is None:
            # Fallback to whatever is not 'vertex' or 'line' if possible
            available = [k for k in c_dict.keys() if k not in ['vertex', 'line', 'point']]
            if available:
                k = available[0]
                conn = c_dict[k]
                element_type = k
                print(f"WARNING: Selected non-standard type '{k}'.")
            else:
                # Last resort
                k = list(c_dict.keys())[0]
                conn = c_dict[k]
                element_type = k
                print(f"WARNING: Only found '{k}'. Solver may fail if this is 0D/1D.")

print(f"Loaded mesh with element type: {element_type}")
if isinstance(conn, torch.Tensor) and len(conn.shape) == 1:
     # Force 2D if 1D array (e.g. line with 1 node? or flattened)
     # This is risky, but avoids crash. Ideally skip.
     print("DEBUG: Reshaping 1D connectivity to (N, 1)")
     conn = conn.unsqueeze(1)
if 'pt_sets' in locals():
    print(f"Found Point Sets: {list(pt_sets.keys())}")
else:
    print("Warning: pt_sets not defined.")

if isinstance(conn, torch.Tensor):
    print(f"DEBUG: conn.shape = {conn.shape}")
    if len(conn.shape) < 2:
        print("ERROR: Connectivity matrix is 1D. Expected 2D (Nel x Nen).")
        # Attempt to reshape if it looks like a flattened array of triangles/quads
        # BUT we don't know NEN without guessing.
        exit(1)
elif isinstance(conn, np.ndarray): #(fallback)
    print(f"DEBUG: conn.shape (numpy) = {conn.shape}")

# Proceed


nnp, nel, nen = x.shape[0], conn.shape[0], conn.shape[1]
ndf = 2 

# --- Boundary Conditions ---
drlt_bcs = []
neum_bcs = []
x_min, x_max = torch.min(x[:, 0]), torch.max(x[:, 0])

print("-" * 20)
# BC Strategy:
bc_found = False
fixed_indices = []

# Check for sets
for name in ["Fixed", "Support", "Lager", "Einspannung"]:
    if name in pt_sets and len(pt_sets[name]) > 0:
        print(f"BC INFO: Using '{name}' Node Set from file.")
        fixed_indices = pt_sets[name]
        bc_found = True
        break

if not bc_found:
    print("BC INFO: No standard 'Fixed' set found. Trying geometric fallback (Inner Radius).")
    r = torch.sqrt(x[:,0]**2 + x[:,1]**2)
    min_r = torch.min(r)
    tol = 1e-3 + min_r.item() * 0.05 
    fixed_nodes = torch.where(r < min_r + tol)[0]
    print(f"BC INFO: Fixed {len(fixed_nodes)} nodes at inner radius (R < {min_r.item():.4f} + tol)")
else:
    fixed_nodes = torch.tensor(fixed_indices, dtype=torch.long)

for n in fixed_nodes:
    drlt_bcs.append([int(n), 0, 0.0]) # Fix X
    drlt_bcs.append([int(n), 1, 0.0]) # Fix Y

# Load at right edge in Y (or largest X if "Loaded" set missing)
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
        # Q8 Shape Functions 
        # Corner nodes
        N[0] = 0.25 * (1-e)*(1-n)*(-e-n-1); N[1] = 0.25 * (1+e)*(1-n)*(e-n-1)
        N[2] = 0.25 * (1+e)*(1+n)*(e+n-1);  N[3] = 0.25 * (1-e)*(1+n)*(-e+n-1)
        # Midside nodes
        N[4] = 0.5 * (1-e*e)*(1-n); N[5] = 0.5 * (1+e)*(1-n*n)
        N[6] = 0.5 * (1-e*e)*(1+n); N[7] = 0.5 * (1-e)*(1-n*n)

        # Derivatives
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
if 'quad8' in element_type or '8' in element_type:
    nqp=9
    qpt = torch.zeros(nqp, 2); w8 = torch.zeros(nqp)
    a = math.sqrt(3.0 / 5.0); w1 = 5.0/9.0; w2 = 8.0/9.0
    vals = [-a, 0, a]; ws = [w1, w2, w1]
    k=0
    for i in range(3):
        for j in range(3):
            qpt[k,0] = vals[j]; qpt[k,1] = vals[i] 
            w8[k] = ws[i] * ws[j]
            k+=1
else:
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

    Ct = C4 - (2*mu)**2/(3*mu+H)*(torch.einsum("ij,kl->ijkl", n, n))

    return sig, Ct, {"ep":ep_new, "alpha":alpha_new}

state_gp = [[{"ep":torch.zeros(2,2),"alpha":0.0} for q in range(nqp)] for e in range(nel)]

# --- Pre-Calculate Sparse Indices for Assembly ---
# Each element adds an (nen*ndf) x (nen*ndf) block to the global matrix
# To avoid looping over indices, we construct global row and col arrays once.
print("Pre-calculating sparse matrix indices...")
edof_size = nen * ndf
num_entries_per_element = edof_size * edof_size
total_entries = nel * num_entries_per_element

# Create a master list of DOF indices for all elements
# conn shape: [nel, nen]. We need [nel, nen*ndf]
# Example: node i -> 2*i, 2*i+1
# Expand connectivity to DOFs
conn_dofs = torch.zeros((nel, edof_size), dtype=torch.long)
for i in range(nen):
    conn_dofs[:, 2*i]   = conn[:, i] * ndf
    conn_dofs[:, 2*i+1] = conn[:, i] * ndf + 1

# Broadcast to create row and col indices for COO
# rows: repeat conn_dofs for each column
# cols: repeat conn_dofs for each row (transposed block wise)
row_indices = conn_dofs.unsqueeze(2).expand(nel, edof_size, edof_size).reshape(-1).numpy()
col_indices = conn_dofs.unsqueeze(1).expand(nel, edof_size, edof_size).reshape(-1).numpy()

print(f"Sparse Indices Ready. Total non-zeros (potential): {total_entries}")

# --- Stabilization for Floating Nodes ---
# We can't check diagonal of K easily before assembly.
# We'll just assume all nodes in the mesh are connected somewhere.
# If there are truly floating nodes, we need to find "unused" nodes in connectivity.
used_nodes = torch.unique(conn)
all_nodes = torch.arange(nnp, dtype=torch.long)
unused_nodes = torch.tensor(np.setdiff1d(all_nodes.numpy(), used_nodes.numpy()), dtype=torch.long)

floating_dofs_list = []
if len(unused_nodes) > 0:
    print(f"DEBUG: Found {len(unused_nodes)} unused nodes. Stabilizing.")
    for n in unused_nodes:
        floating_dofs_list.extend([n.item()*ndf, n.item()*ndf+1])

drlt_mask = torch.zeros(nnp*ndf, 1)
for bc in drlt: drlt_mask[int(bc[0])*ndf + int(bc[1])] = 1.0
for dof in floating_dofs_list: drlt_mask[int(dof)] = 1.0

free_dofs = torch.nonzero(1.0 - drlt_mask)[:, 0].numpy() # Numpy for Scipy Slicing
free_dofs_torch = torch.from_numpy(free_dofs)

# ==========================================
# ============ NONLINEAR SOLVER ============
# ==========================================

n_steps = 15
newton_tol = 1e-8
newton_max = 30

track_node = int(load_nodes[0])
track_dof = track_node*ndf + 1

disp_lin = []
disp_pl = []
load_hist = []

u = torch.zeros(nnp*ndf,1)

# Outer Loop: Load Steps
step_pbar = tqdm(range(1, n_steps+1), desc="Load Steps")
for step in step_pbar:
    fac = step/n_steps
    # print(f"\nSTEP {step}/{n_steps}") # Handled by pbar

    f_ext = torch.zeros(nnp*ndf,1)
    for bc in neum:
        f_ext[int(bc[0])*ndf+int(bc[1])] = fac*bc[2]

    # Inner Loop: Newton Iterations
    newton_pbar = tqdm(range(newton_max), desc="  Newton", leave=False)
    for it in newton_pbar:
        # Flattened list of all K values
        all_Ke_values = []
        fint = torch.zeros_like(f_ext)

        # Loop elements (physics logic remains)
        # Note: Ideally this would be batched, but straightforward port is ok for now.
        # Loop is slow in Python, but Assembly was OOM without sparse. 
        # For 50k elements, this loop takes time.
        
        # Optimization: We can access x and u globally.
        # Let's try to be efficient inside using numpy accumulation if possible, 
        # but torch is needed for physics.
        
        # Collecting Ke values in a numpy array is faster than appending to list?
        # A list append is actually quite fast.
        
        # To speed up access: converting entire x and u to easy indexing if needed not huge help.
        
        Ke_list = []
        fe_indices = []
        fe_values = []
        
        # Assembly Loop
        # Show pbar only if enough elements to matter (>500)
        show_assembly_bar = nel > 500
        iter_range = tqdm(range(nel), desc="    Assembly", leave=False) if show_assembly_bar else range(nel)
        
        for el in iter_range:
            n_idx = conn[el]; xe = x[n_idx].t()
            edofs_idx = conn_dofs[el] # Use precalculated
            ue = u[edofs_idx].reshape(-1,2).t()
            
            Ke = torch.zeros(nen*ndf,nen*ndf)
            fe = torch.zeros(nen*ndf,1)
            
            # Element loop
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

            # Store Ke flattened
            Ke_list.append(Ke.flatten().detach().numpy())
            
            # Store fe (sparse add later or dense add now)
            # Since f_int is a vector, dense add is fine? 
            # Actually f_int is small (vector). Torch scatter add is good.
            fint[edofs_idx] += fe

        # Build Sparse Matrix
        # Concatenate all Ke values
        data = np.concatenate(Ke_list)
        
        # Create Sparse Matrix
        Kt_sparse = sp.coo_matrix((data, (row_indices, col_indices)), shape=(nnp*ndf, nnp*ndf)).tocsr()
        
        R = f_ext - fint
        Rf = R[free_dofs_torch]
        Rf_numpy = Rf.detach().numpy() # Convert to numpy for Scipy

        norm_Rf = np.linalg.norm(Rf_numpy)
        if it % 5 == 0: print(f"  Iter {it}: |R| = {norm_Rf:.4e}")
        
        if norm_Rf < newton_tol:
            # print(" converged")
            newton_pbar.set_postfix({"R": f"{norm_Rf:.2e}", "Status": "Converged"})
            break
        else:
            newton_pbar.set_postfix({"R": f"{norm_Rf:.2e}"})

        # Solve sparse system
        Kt_free = Kt_sparse[free_dofs, :][:, free_dofs]
        
        # Solver
        du_f_numpy = spsolve(Kt_free, Rf_numpy)
        du_f = torch.from_numpy(du_f_numpy).reshape(-1, 1)
        
        # Update u
        u[free_dofs_torch] += du_f

    load_hist.append(fac*F_total)
    disp_pl.append(u[track_dof].item())

# --- Linear Reference (optional, skip if too slow or just do one step) ---
# We can use the last Kt (tangent stiffness) or initial K. Initial K is elastic.
# Let's assemble Elastic K one time for reference? Or just skip linear plot.
# Lets skip linear reference assembly to save time unless requested.
# Or just approximate it with initial step?

# ==========================================
# ============ POSTPROCESS NONLINEAR =========
# ==========================================

element_results = {"svm": [], "u": []}

for e in range(nel):
    n_idx = conn[e]
    xe = x[n_idx].t()
    edofs_idx = conn_dofs[e]
    ue = u[edofs_idx].reshape(-1, 2).t()

    # Center Gauss Point or Average for SVM
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


# ==========================================
# ============ VISUALIZATION ============
# ==========================================
# (Same visualization code as before, just kept intact)
scale = 0.15 * torch.max(torch.abs(x)) / (torch.max(torch.abs(u)) + 1e-25)
x_def = x + scale * u.reshape(-1, 2)
plot_conn = conn[:, :4] if nen >= 4 else conn
limit = torch.max(torch.abs(x)) * 1.3

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7.5))

# 2. Deformed Displacement [mm]
ax2.set_title(f"Verschiebung [mm] (Skal. {scale:.1f}x)", fontweight='bold')
verts_def = [x_def[plot_conn[e]].numpy() for e in range(nel)]
pc_u = PolyCollection(verts_def, cmap='viridis', edgecolors='k', lw=0.1) # Thinner lines for dense mesh
pc_u.set_array(np.array(element_results["u"]))
ax2.add_collection(pc_u); cbar2 = plt.colorbar(pc_u, ax=ax2, label="Verschiebung [mm]")
ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit); ax2.set_aspect('equal')

# 3. Discrete Stress
ax3.set_title("Von Mises Spannung [MPa]", fontweight='bold')
pc_s = PolyCollection(verts_def, cmap='jet', edgecolors='k', lw=0.1)
pc_s.set_array(np.array(element_results["svm"]))
ax3.add_collection(pc_s); cbar3 = plt.colorbar(pc_s, ax=ax3, label="Spannung [MPa]")
ax3.set_xlim(-limit, limit); ax3.set_ylim(-limit, limit); ax3.set_aspect('equal')

plt.tight_layout()
title = "Deformations"
plt.savefig(title + ".png")
plt.show()
title = "Last–Verschiebungs–Kurve"
plt.figure(figsize=(7,5))
plt.plot(disp_pl,load_hist,'r',lw=2,label="plastisch")
plt.xlabel("Verschiebung [m]")
plt.ylabel("Last [N]")
plt.title(title)
plt.grid(True)
plt.legend()
plt.savefig(title + ".png")
plt.show()
