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
    ], device=device)

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
], device=device)

# Neumann boundary condition
#      node  ldof  scale
neum = torch.tensor([
  [15, 1, 0, -3090*9.81],
  [16, 1, 0, -3090*9.81]
], device=device)

# loadcurves
timesteps = torch.tensor([0, 1, 2, 3, 4, 5], device=device, dtype=torch.float)
loadsteps = torch.tensor([[0, 1, 1, 1, 1, 1]], device=device, dtype=torch.float)
dt = 0.1 # time step

times = torch.arange(timesteps[0].item(), timesteps[-1].item(), step=dt)

print(loadsteps.shape)
loadsteps_interpolated = torch.nn.functional.interpolate(loadsteps.unsqueeze(0).unsqueeze(0), scale_factor = (1, int(1/dt+0.5)), mode='bilinear').squeeze(0).squeeze(0)
#print("loadsteps_interpolated: ", loadsteps_interpolated)

# BC variables
allDofs = torch.linspace(0, nnp*ndf, 1, device=device)
numDrltDofs = drlt.size(dim=0)
drltDofs = torch.zeros((numDrltDofs,1), device=device)
for i in range(numDrltDofs):
    drltDofs[i] = drlt[i, 0]*ndf + drlt[i, 1]
drltDofs = drltDofs.int()

#freeDofs = torch.from_numpy(np.setdiff1d(allDofs.numpy(),drltDofs.numpy()))
#freeDofs = freeDofs.int()

xplt = x.cpu()
plt.figure()
for i in range(conn.size(dim=0)):
  plt.plot (xplt[[conn[i,0], conn[i,1]],0], xplt[[conn[i,0], conn[i,1]],1], 'x-', linewidth=3, color="k")
  plt.gca().set_aspect('equal', adjustable='box')

for i in range(drlt.size(dim=0)):
    plt.scatter(xplt[drlt[i, 0].int(), 0], xplt[drlt[i, 0].int(), 1], color="red")
    #TODO signal direction/ dof

for i in range(neum.size(dim=0)):
    plt.scatter(xplt[neum[i, 0].int(), 0], xplt[neum[i, 0].int(), 1], color="green")
    # TODO signal direction/ dof

plt.show()


#===============================
#=========== SOLVER ============
#===============================

def gauss(nqp, ndm):
    if nqp == 1:
        xi = [0]
        w8 = [2]
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
    N = torch.zeros(nen, 1)
    gamma = torch.zeros(nen, 1)

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

u = torch.zeros(nnp*ndf, 1, device=device)
K = torch.zeros(nnp*ndf, nnp*ndf, device=device)
K_tilde = K
fext = torch.zeros(nnp*ndf, 1, device=device)
fint = torch.zeros(nnp*ndf, 1, device=device)
fvol = torch.zeros(nnp*ndf, 1, device=device)
frea = torch.zeros(nnp*ndf, 1, device=device)

gdof = torch.zeros(nen*ndf,1, device=device)

Ke = torch.zeros(nen*ndf, nen*ndf, device=device)
fvole = torch.zeros(nen*ndf, 1, device=device)
finte = torch.zeros(nen*ndf, 1, device=device)

xe = torch.zeros(ndm, nen)

sigma = torch.zeros(nel, nqp, ndf, device=device)

for tt in range(times.size(0)):
    t = times[tt]
    print("time = ", t)

    u_d = torch.zeros_like(u)
    #      node  ldof  scale

    u_d[(drlt[:, 0] * ndf).int() + drlt[:, 1].int(), 0] = drlt[:, 2]
    #      node  ldof  scale
    fpre = torch.zeros_like(u)
    fpre[(neum[:, 0] * ndf).int() + neum[:, 1].int(), 0] = neum[:, 3] * loadsteps_interpolated[neum[:, 2].int() , tt]
    #print ("fpre: ", fpre)



    rsn = 1
    for iter in range(maxiter):
        K.zero_()
        fext.zero_()
        fint.zero_()
        fvol.zero_()
        frea.zero_()

        for e in range(nel):
            xe = x[conn[e, :], :].transpose(dim0=0, dim1=1)
            #for i in range(ndm):
            #    xe[i, :] = x[conn[e, :], i]


            for node in range(nen):
                gdof[node*ndf-1:node*ndf] = torch.arange((conn[e,node]*ndf-1).item(), (conn[e,node]*ndf).item(), step=1, device=device)

            gdof = gdof.squeeze().int()

            Ke.zero_()
            fvole.zero_()
            finte.zero_()

            for q in range(nqp):
                N, gamma = shape(xi[q], nen, ndm)

                l = torch.sqrt((xe[0, 1] - xe[0,0]).pow(2) + (xe[1,1]-xe[1,0]).pow(2))
                cosphi = (xe[0, 1] - xe[0,0])/l
                sinphi = (xe[1,1] - xe[1, 0])/l
                G_matrix = torch.tensor([[cosphi, sinphi, 0, 0],[0, 0, cosphi, sinphi]], device=device)
                #print("G_matrix: ", G_matrix)
                g_A = gamma[0, 0]
                g_B = gamma[1, 0]

                xe_loc = torch.tensor([0, l], device=device)
                #print (xe_loc, gamma.squeeze(), nen, ndm)
                #detJq, invJq = jacobian(xe_loc, gamma.squeeze(), nen, ndm)
                #print("gamma: ", gamma)
                detJq = (xe_loc * gamma.squeeze()).sum()
                #print("detJq: ", detJq)
                invJq = 1/detJq

                ue = torch.zeros(ndf, nen, device=device)

                for i in range(nen):
                    gi = conn[e, i]
                    #print("ue[:, i]: ", ue[:, i].shape)
                    #print("u[gi*ndf -1 : gi*ndf, 0]: ", u[gi*ndf : (gi+1)*ndf, 0].shape)
                    ue[:, i] = u[gi*ndf : (gi+1)*ndf, 0]

                ue_vec = ue.view(4)
                ue_loc = torch.matmul(G_matrix, ue_vec)

                G = gamma * invJq
                #print("G: ", G)
                #print("ue_loc: ", ue_loc)
                eps = torch.matmul(ue_loc.unsqueeze(0), G)

                #print("eps: ", eps)
                sig = E * eps
                #print("sig: ", sig)
                C = E

                finte += (g_A * G_matrix[0, :] + g_B * G_matrix[1, :]).unsqueeze(0).transpose(0, 1) * Area * sig * w8[q]

                Ke += (g_A * G_matrix[0, :] + g_B * G_matrix[1, :]).unsqueeze(0).transpose(0,1) * C * Area * (g_A * G_matrix[0, :] + g_B * G_matrix[1, :]) * invJq * w8[q]
                sigma[e, q, :] = sig

            #print("K[gdof, gdof]: ", K[gdof, gdof])
            #print("Ke: ", Ke)
            #print("gdof: ", gdof)
            for i in range(gdof.shape[0]):
                for j in range(gdof.shape[0]):
                    K[gdof[i], gdof[j]] += Ke[i, j]


            fvol[gdof] += fvole
            fint[gdof] += finte

        fext = fpre + fvol
        rsd_F = # TODO
        rsn = torch.norm(rsd_F) + torch.norm(u_d - u[drltDofs])

        if rsn > tol:
            rhs = #TODO

            #du = torch.zeros_like(u)
            # TODO Calculate du according to equation (8.6) and (8.7)
            u += du
            print ("u: ", u)

            # TODO Backup hidden/ state variables from material model
            u[drltDofs] = u_d[drltDofs]
        else:
            break

        iter += 1
        if iter > maxiter:
            raise(Exception("maxiter exceeded"))

    fext = K_tilde * u
    frea = torch.zeros_like(fext)
    frea[drltDofs] = fext[drltDofs] - fvol[drltDofs]

    xplt = x.cpu() + 1000000 * u.view(x.shape)
    plt.figure()
    for i in range(conn.size(dim=0)):
        plt.plot(xplt[[conn[i, 0], conn[i, 1]], 0], xplt[[conn[i, 0], conn[i, 1]], 1], 'x-', linewidth=3, color="cyan")
        #TODO: choose color based on strain or stress in the truss

        plt.gca().set_aspect('equal', adjustable='box')

    plt.show()