import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===============================
#=========== INPUT =============
#===============================
ndm = 2 # Dimension
ndf = 2 # DOF per element
nen = 2 # Nodes per element

#=========== GEOMETRY ==========
x = torch.tensor([[0, 0],
     [1, 0],
     [2, 0],
     [4, 0],
     [6, 0],
     [8, 0],
     [9, 0],
     [1, 0.4],
     [3, 0.4],
     [5, 0.4],
     [7, 0.4],
     [8, 0.4],
     [9, -1],
     [6, -2],
     [12, -2],
     [6, -4.5],
     [12, -4.5],
    ], device=device, dtype=torch.double)

#Number of node points
nnp = x.size(dim=0)

conn = torch.tensor([
[0, 7],
[1, 7],
[1, 2],
[2, 7],
[2, 8],
[2, 3],
[3, 8],
[3, 9],
[3, 4],
[4, 9],
[4, 10],
[4, 5],
[5, 10],
[5, 11],
[5, 6],
[6, 11],
[7, 8],
[8, 9],
[9, 10],
[10, 11],
[6, 12],
[12, 13],
[12, 14],
[13, 14],
[13, 15],
[14 ,16],
], device=device)

# Number of elements
nel = conn.size(dim=0)

# Number of quadrature points
nqp = 1

# tolerance of Newton iteration
tol = 1e-6
maxiter = 20 #Max. iterations in Newton loop
#=========== MATERIAL ==========
E = 210e9
Area = 2 * 0.001
eta = 1e11

#=========== Boundary conditions ==========

# Dirichlet boundary condition
#      node  ldof  scale
drlt = torch.tensor([
[0, 0, 0.],
[0, 1, 0.],
[1, 0, 0.],
[1, 1, 0.]
], device=device, dtype=torch.double)

# Neumann boundary condition
#      node  ldof  scale
neum = torch.tensor([
  [15, 1, 0, -3090*9.81],
  [16, 1, 0, -3090*9.81]
], device=device, dtype=torch.double)

# loadcurves
timesteps = torch.tensor([0, 1, 2, 3, 4, 5], device=device, dtype=torch.double)
loadsteps = torch.tensor([[0, 1, 1, 1, 1, 1]], device=device, dtype=torch.double)
dt = 0.1 # time step

times = torch.arange(timesteps[0].item(), timesteps[-1].item(), step=dt)

print(loadsteps.shape)
# Use 'linear' interpolation for 1D time signals to avoid artifacts
loadsteps_interpolated = torch.nn.functional.interpolate(loadsteps.unsqueeze(0), size=len(times), mode='linear', align_corners=True).squeeze(0)
# Ensure correct shape/broadcast


# BC variables
allDofs = torch.linspace(0, nnp*ndf, 1, device=device)
numDrltDofs = drlt.size(dim=0)
drltDofs = torch.zeros((numDrltDofs,1), device=device)
for i in range(numDrltDofs):
    drltDofs[i] = drlt[i, 0]*ndf + drlt[i, 1]
drltDofs = drltDofs.int()

#freeDofs = torch.from_numpy(np.setdiff1d(allDofs.numpy(),drltDofs.numpy()))
#freeDofs = freeDofs.int()

plt.ion() # Interactive mode on
fig, ax = plt.subplots(figsize=(10, 8))
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

# --- Pre-initialize Plot Artists for Performance & Hover ---
plot_lines = []
# Create a line object for each element initially
# We start with initial coords
x_init = x.cpu().numpy()
for i in range(conn.size(dim=0)):
    p1 = x_init[conn[i, 0], :]
    p2 = x_init[conn[i, 1], :]
    # Use generic color first
    ln, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3, picker=5) 
    plot_lines.append(ln)

# Nodes
sc_nodes = ax.scatter(x_init[:, 0], x_init[:, 1], color='black', s=10)

# BCs (Static, only plot once)
for i in range(drlt.size(dim=0)):
    node_idx = drlt[i, 0].long()
    ax.scatter(x_init[node_idx, 0], x_init[node_idx, 1], color="red", marker='s', s=40, zorder=5)

for i in range(neum.size(dim=0)):
    node_idx = neum[i, 0].long()
    ax.scatter(x_init[node_idx, 0], x_init[node_idx, 1], color="green", marker='^', s=40, zorder=5)

# Setup Annotation for Hover
annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind, line_idx, x_mid, y_mid, stress):
    annot.xy = (x_mid, y_mid)
    annot.set_text(f"Element {line_idx}\nSigma: {stress:.2f} MPa")
    annot.get_bbox_patch().set_alpha(0.9)

def on_hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        # Brute force distance check for robustness in dynamic plot
        # (Pickers can be finicky when data changes rapidly)
        found = False
        min_dist = float("inf")
        closest_idx = -1
        
        # Check distance to all line centers
        # We need the CURRENT plot data which is in 'plot_lines'
        m_x, m_y = event.xdata, event.ydata
        if m_x is None or m_y is None: return
        
        for i, ln in enumerate(plot_lines):
            x_data, y_data = ln.get_data()
            # Midpoint
            mid_x = (x_data[0] + x_data[1])/2
            mid_y = (y_data[0] + y_data[1])/2
            
            # Simple euclidean distance to midpoint
            dist = np.sqrt((mid_x - m_x)**2 + (mid_y - m_y)**2)
            if dist < 0.5: # Threshold in data units
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    found = True
                    c_mid_x, c_mid_y = mid_x, mid_y
        
        if found:
            # Get stress from the global sigma storage
            # Assuming sigma is updated in the loop. We need access to 'element_stresses' variable
            # We make 'element_stresses' global or accessible.
            # For now, we rely on the loop variable 'element_stresses' if it exists in scope, 
            # but inside function it might be tricky.
            # Better: store stress in the line object itself as an attribute
            
            st = getattr(plot_lines[closest_idx], 'stress_val', 0.0)
            update_annot(closest_idx, closest_idx, c_mid_x, c_mid_y, st / 1e6) # Display in MPa
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

# Initial Limits
margin = 2.0
x_min, x_max = x_init[:, 0].min(), x_init[:, 0].max()
y_min, y_max = x_init[:, 1].min(), x_init[:, 1].max()
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_aspect('equal', adjustable='box')





#===============================
#=========== SOLVER ============
#===============================

def gauss(nqp, ndm):
    if nqp == 1:
        xi = torch.tensor([0], device=device, dtype=torch.double)
        w8 = torch.tensor([2], device=device, dtype=torch.double)
    else:
        raise(Exception("Not implemented"))

    return xi, w8


def jacobian (xe, gamma, nen, ndm):
    Jq = xe * gamma
    #print ("Jq: ", Jq)
    detJq = torch.det(Jq)
    invJq = torch.inverse(Jq)

    return detJq, invJq

def shape(xi, nen, ndm):
    N = torch.zeros(nen, 1, device=device, dtype=torch.double)
    gamma = torch.zeros(nen, 1, device=device, dtype=torch.double)

    if nen == 2:
        N[0,0] = 0.5 * (1 - xi)
        N[1,0] = 0.5 * (1 + xi)

        gamma[0,0] = -0.5
        gamma[1,0] = 0.5
    else:
        raise(Exception("Not implemented"))

    return N, gamma

I = torch.eye(ndm, ndm, device=device)
xi, w8 = gauss(nqp, ndm)

u = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)
K = torch.zeros(nnp*ndf, nnp*ndf, device=device, dtype=torch.double)
K_tilde = K
fext = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)
fint = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)
fvol = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)
frea = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)

gdof = torch.zeros(nen*ndf,1, device=device, dtype=torch.double)

Ke = torch.zeros(nen*ndf, nen*ndf, device=device, dtype=torch.double)
fvole = torch.zeros(nen*ndf, 1, device=device, dtype=torch.double)
finte = torch.zeros(nen*ndf, 1, device=device, dtype=torch.double)

xe = torch.zeros(ndm, nen, dtype=torch.double)

sigma = torch.zeros(nel, nqp, ndf, device=device, dtype=torch.double)

for tt in range(times.size(0)):
    t = times[tt]
    print("time = ", t)

    u_d = torch.zeros_like(u)
    #      node  ldof  scale

    u_d[(drlt[:, 0] * ndf).long() + drlt[:, 1].long(), 0] = drlt[:, 2]
    #      node  ldof  scale
    fpre = torch.zeros_like(u, dtype=torch.double)
    fpre[(neum[:, 0] * ndf).int() + neum[:, 1].int(), 0] = neum[:, 3] * loadsteps_interpolated[neum[:, 2].int() , tt]

    #print ("fpre: ", fpre)



    rsn = 1
    for iter in range(maxiter):
        K.zero_()
        fext.zero_()
        fint.zero_()
        fvol.zero_()
        frea.zero_()

        # Enforce Dirichlet BCs at start of iteration (important for first step)
        u[drltDofs.view(-1).long()] = u_d[drltDofs.view(-1).long()]


        for e in range(nel):
            # Extract displacements for the element first to update geometry
            ue_curr = torch.zeros(ndm, nen, device=device, dtype=torch.double)
            for i in range(nen):
                gi = conn[e, i]
                ue_curr[0, i] = u[gi*ndf + 0] # x-disp
                ue_curr[1, i] = u[gi*ndf + 1] # y-disp

            # Update element coordinates (Nonlinear Geometry)
            xe_initial = x[conn[e, :], :].transpose(dim0=0, dim1=1)
            xe = xe_initial + ue_curr



            gdof_list = []
            for node in range(nen):
                global_node = conn[e, node].item()
                start_dof = global_node * ndf
                gdof_list.extend([start_dof, start_dof+1])
            
            gdof = torch.tensor(gdof_list, device=device, dtype=torch.long)

            Ke.zero_()
            fvole.zero_()
            finte.zero_()

            # --- Co-rotational Truss Formulation ---
            
            # 1. Geometry
            # x0: Initial coordinates (2x2)
            x0 = xe_initial # as defined above
            # x: Current coordinates (2x2)
            xc = xe # as updated above
            
            # Lengths
            dx0 = x0[:, 1] - x0[:, 0]
            dxc = xc[:, 1] - xc[:, 0]
            
            L0 = torch.norm(dx0)
            L = torch.norm(dxc)
            
            # Current direction vector (n)
            n = dxc / L # (cos, sin)
            
            # 2. Strain (Engineering Strain for large displacement)
            # eps = (L - L0) / L0
            # (Or Green-Lagrange: 0.5 * (L**2 - L0**2) / L0**2)
            eps = (L - L0) / L0
            
            # 3. Stress & Force
            sig = E * eps
            force = sig * Area
            
            # 4. Internal Force Vector (fint)
            # f_local = [-N, N] (along axis)
            # f_global = [-N*c, -N*s, N*c, N*s]
            f_vec = torch.cat((-n * force, n * force)) # Size 4
            finte += f_vec.unsqueeze(1) # Add to element internal force
            
            # Store stress
            sigma[e, 0, :] = sig

            # 5. Tangent Stiffness Matrix (Ke)
            # K = Km (Material) + Kg (Geometric)
            
            # Material Stiffness: Km = (EA / L0) * (n * n.T)
            # Note: stiffness relates to change in displacement.
            # Use L0 for engineering stress measure usually, or L depending on formulation. 
            # Consistent linearization of F = EA/L0 * (L-L0) * n:
            # Km = EA/L0 * n*n.T
            # Kg = F/L * (I - n*n.T)
            
            n_tensor = n.unsqueeze(1) # 2x1
            P = torch.matmul(n_tensor, n_tensor.t()) # 2x2 projection n*n.T
            I2 = torch.eye(2, device=device, dtype=torch.double)
            I_min_P = I2 - P # Transverse projection
            
            Km_block = (E * Area / L0) * P
            Kg_block = (force / L) * I_min_P
            
            Kt_block = Km_block + Kg_block
            
            # Assemble 4x4 Ke
            # [ Kt  -Kt ]
            # [-Kt   Kt ]
            Ke.zero_()
            Ke[0:2, 0:2] = Kt_block
            Ke[0:2, 2:4] = -Kt_block
            Ke[2:4, 0:2] = -Kt_block
            Ke[2:4, 2:4] = Kt_block

            # Skip the old loop logic


            #print("K[gdof, gdof]: ", K[gdof, gdof])
            #print("Ke: ", Ke)
            #print("gdof: ", gdof)

            for i in range(gdof.shape[0]):
                for j in range(gdof.shape[0]):
                    K[gdof[i], gdof[j]] += Ke[i, j]


            fvol[gdof] += fvole
            fint[gdof] += finte

        fext = fpre + fvol
        rsd_F = fext - fint
        
        # Zero out residual at Dirichlet boundaries
        if drltDofs.numel() > 0:
            rsd_F[drltDofs.view(-1).long()] = 0.0

        rsn = torch.norm(rsd_F)
        print(f"  Iter {iter}: Residual Norm = {rsn:.4e}")

        
        # Check convergence
        if rsn > tol:
            rhs = rsd_F.clone()

            # Apply BCs to K for solving
            K_solve = K.clone()
            idx = drltDofs.view(-1).long()
            
            if idx.numel() > 0:
                K_solve[idx, :] = 0.0
                K_solve[:, idx] = 0.0
                K_solve[idx, idx] = 1.0
                rhs[idx] = 0.0
            
            # Reduce artificial stabilization now that we have Kg, but keep a tiny bit for the first step (zero stress)
            # if stress is zero, Kg is zero, so we still need some help for the initial mechanism.
            K_solve.diagonal().add_(1e-2) 

            # Solve for du
            try:
                du = torch.linalg.solve(K_solve, rhs)

            except RuntimeError as e:
                print("Error solving Linear System:", e)
                break

            u += du
            # print ("u updated norm: ", torch.norm(u))

            # Enforce BCs again exactly
            u[idx] = u_d[idx]
        else:
            break

        iter += 1
        if iter > maxiter:
            raise(Exception("maxiter exceeded"))

    fext = torch.matmul(K_tilde, u)
    frea = torch.zeros_like(fext)
    frea[drltDofs] = fext[drltDofs] - fvol[drltDofs]

    xplt = x.cpu() + 1.0 * u.view(x.shape).cpu() # Deformational scale 1.0
    xplt_np = xplt.numpy()
    
    # Calculate stresses for coloring in MPa
    # sigma is (nel, nqp, ndf). For 1D truss, take first component of first QP
    element_stresses_pa = sigma[:, 0, 0].cpu().detach().numpy()
    element_stresses_mpa = element_stresses_pa / 1e6
    
    # Update Plot Data directly
    ax.set_title(f"Deformation at Time t={t:.2f}")

    # Create colormap based on MPa
    norm = plt.Normalize(vmin=element_stresses_mpa.min(), vmax=element_stresses_mpa.max())
    cmap = plt.cm.jet
    
    for i in range(conn.size(dim=0)):
        # Get coordinates
        p1 = xplt_np[conn[i, 0], :]
        p2 = xplt_np[conn[i, 1], :]
        
        ln = plot_lines[i]
        ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        c = cmap(norm(element_stresses_mpa[i]))
        ln.set_color(c)
        
        # Store stress for hover (keep as raw Pa or MPa? Let's keep consistency)
        ln.stress_val = element_stresses_pa[i] # keeping Pa for hover calculation logic if needed, but display is handled by update_annot

    # Update Nodes
    sc_nodes.set_offsets(xplt_np)

    # Colorbar handling
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Plot colorbar into the dedicated axis 'cax' (does not resize ax)
    # limit cax clearing to avoid flickering if possible, or clear it:
    cax.clear() 
    fig.colorbar(sm, cax=cax, label="Spannung (MPa)") 
    
    # fig.canvas.draw()    # Often slow
    # fig.canvas.flush_events() 
    plt.pause(0.01)

plt.ioff()
plt.show()