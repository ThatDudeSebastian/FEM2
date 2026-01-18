"""
fem_quad8_plastic_newton.py (standalone)

- QUAD8 plane strain, small strain
- J2 (von Mises) plasticity with linear isotropic + kinematic hardening
- Radial return mapping + consistent tangent
- Global Newton-Raphson with load stepping
- Optional penalty contact against rigid rail y >= yRail

Run (Windows PowerShell):
  python fem_quad8_plastic_newton.py --mesh-dir .

Disable contact:
  python fem_quad8_plastic_newton.py --mesh-dir . --no-contact

Mesh files required in --mesh-dir:
  circle8_n.msh : columns [node_id, x, y]
  circle8_e.msh : columns [elem_id, n1, n2, n3, n4, n5, n6, n7, n8]
(IDs are 1-based in file, converted to 0-based internally.)
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# Tensor utilities (embedded 3D)
# =============================================================================

def dev(A: np.ndarray) -> np.ndarray:
    """Deviatoric part of a 3x3 tensor."""
    A = np.asarray(A, dtype=float)
    return A - (np.trace(A) / 3.0) * np.eye(3)


def double_contract(A: np.ndarray, B: np.ndarray) -> float:
    """A:B = sum_ij A_ij B_ij."""
    return float(np.sum(np.asarray(A, float) * np.asarray(B, float)))


def outer2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(A ⊗ B)_{ijkl} = A_{ij} B_{kl}."""
    return np.einsum("ij,kl->ijkl", np.asarray(A, float), np.asarray(B, float))


def I4sym_tensor() -> np.ndarray:
    """I4sym_{ijkl} = 0.5*(δik δjl + δil δjk)."""
    I = np.eye(3)
    return 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))


def pack_matlab_9(A: np.ndarray) -> np.ndarray:
    """Pack 3x3 tensor into length-9 vector using MATLAB column-major order."""
    A = np.asarray(A, dtype=float).reshape((3, 3))
    return A.reshape((9,), order="F")


def unpack_matlab_9(v: np.ndarray) -> np.ndarray:
    """Unpack length-9 vector (MATLAB column-major) into 3x3 tensor."""
    v = np.asarray(v, dtype=float).reshape((9,))
    return v.reshape((3, 3), order="F")


# =============================================================================
# Contraction helper: T1 dot T4 dot T1
# =============================================================================

def t1_dot_t4_dot_t1(G_A: np.ndarray, C4: np.ndarray, G_B: np.ndarray) -> np.ndarray:
    """
    T2(j,k) = sum_{i,l} G_A(i) * C4(i,j,k,l) * G_B(l)

    G_A, G_B : (ndm,)
    C4       : (ndm,ndm,ndm,ndm)
    returns  : (ndm,ndm)
    """
    G_A = np.asarray(G_A, dtype=float).reshape(-1)
    G_B = np.asarray(G_B, dtype=float).reshape(-1)
    C4 = np.asarray(C4, dtype=float)
    ndm = C4.shape[0]

    T2 = np.zeros((ndm, ndm), dtype=float)
    for i in range(ndm):
        Gi = G_A[i]
        for l in range(ndm):
            T2 += (Gi * G_B[l]) * C4[i, :, :, l]
    return T2


# =============================================================================
# Material model: J2 von Mises with iso+kin hardening (radial return)
# =============================================================================

def vmises_iso_kin_rr(eps_in: np.ndarray, matparam: np.ndarray, state_old: np.ndarray):
    """
    Small strain J2 plasticity with linear isotropic + kinematic hardening.
    Plane strain is handled by embedding into 3D with eps33 = 0.

    State layout (length 19), MATLAB compatible:
      state = [epsp(:); k; a(:)]
      epsp: 3x3 plastic strain tensor (deviatoric)
      k   : accumulated plastic strain (scalar)
      a   : backstress-like tensor (deviatoric)

    matparam = [E, nu, sigy0, H, rho, r]
      r in [0,1] mixes isotropic vs kinematic.

    Returns:
      sigma      (3,3)
      C4         (3,3,3,3) consistent algorithmic tangent
      state_new  (19,)
      is_plastic bool
    """
    eps_in = np.asarray(eps_in, dtype=float)

    # embed to 3D
    if eps_in.shape == (2, 2):
        eps = np.zeros((3, 3), dtype=float)
        eps[:2, :2] = eps_in
    elif eps_in.shape == (3, 3):
        eps = eps_in.copy()
    else:
        raise ValueError("eps_in must be 2x2 or 3x3.")
    eps = 0.5 * (eps + eps.T)

    mp = np.asarray(matparam, dtype=float).reshape(-1)
    E = float(mp[0])
    nu = float(mp[1])
    sigy0 = float(mp[2])
    H = float(mp[3])
    r = float(mp[5]) if mp.size >= 6 else 0.5

    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    if state_old is None or (isinstance(state_old, np.ndarray) and state_old.size == 0):
        state_old = np.zeros(19, dtype=float)
    state_old = np.asarray(state_old, dtype=float).reshape(-1)
    if state_old.size != 19:
        raise ValueError("state_old must have length 19.")

    epsp_n = unpack_matlab_9(state_old[:9])
    k_n = float(state_old[9])
    a_n = unpack_matlab_9(state_old[10:])

    # enforce symmetry + deviatoric
    epsp_n = dev(0.5 * (epsp_n + epsp_n.T))
    a_n = dev(0.5 * (a_n + a_n.T))

    # trial
    ee_tr = 0.5 * ((eps - epsp_n) + (eps - epsp_n).T)
    s_tr = 2.0 * G * dev(ee_tr)

    kappa_tr = -r * H * k_n
    alpha_tr = -(2.0 / 3.0) * (1.0 - r) * H * a_n

    s_red_tr = s_tr + alpha_tr
    norm_sred = np.sqrt(max(0.0, double_contract(s_red_tr, s_red_tr)))
    Phi_tr = norm_sred - np.sqrt(2.0 / 3.0) * (sigy0 - kappa_tr)

    I = np.eye(3)
    I_outer = outer2(I, I)
    I4sym = I4sym_tensor()
    I4dev = I4sym - (1.0 / 3.0) * I_outer

    tol = 1e-14 * max(1.0, sigy0)

    if (Phi_tr <= tol) or (norm_sred < 1e-30):
        # elastic
        sigma = K * np.trace(ee_tr) * I + s_tr
        C4 = K * I_outer + 2.0 * G * I4dev
        return sigma, C4, state_old.copy(), False

    # plastic step
    nu_dir = s_red_tr / norm_sred
    nu_dir = dev(0.5 * (nu_dir + nu_dir.T))

    denom = 2.0 * G + (2.0 / 3.0) * H
    dlam = Phi_tr / denom

    epsp_np1 = dev(0.5 * (epsp_n + dlam * nu_dir + (epsp_n + dlam * nu_dir).T))
    k_np1 = k_n + dlam * np.sqrt(2.0 / 3.0)
    a_np1 = dev(0.5 * (a_n + dlam * nu_dir + (a_n + dlam * nu_dir).T))

    A = 2.0 * G + (2.0 / 3.0) * (1.0 - r) * H
    s_np1 = s_tr - dlam * A * nu_dir

    ee_np1 = eps - epsp_np1
    sigma = K * np.trace(ee_np1) * I + s_np1

    # consistent tangent
    c1 = 2.0 * G * (1.0 - (A * dlam) / norm_sred)
    c2 = 2.0 * G * A * ((dlam / norm_sred) - (1.0 / denom))
    C4 = K * I_outer + c1 * I4dev + c2 * outer2(nu_dir, nu_dir)

    state_new = np.zeros(19, dtype=float)
    state_new[:9] = pack_matlab_9(epsp_np1)
    state_new[9] = k_np1
    state_new[10:] = pack_matlab_9(a_np1)
    return sigma, C4, state_new, True


# =============================================================================
# Input loader (circle quad8)
# =============================================================================

def input_circle_quad8(mesh_dir: str = ".", do_plot: bool = True):
    """
    Loads mesh and defines BCs/loads similar to MATLAB input_circle_quad8.m.

    Returns:
      ndm, ndf, nnp, nel, nen, x, elem, matparam, drlt, neum, b
    """
    # nodes: [id, x, y]
    N = np.loadtxt(f"{mesh_dir}/circle8_n.msh")
    x = np.column_stack((N[:, 1], N[:, 2]))
    nnp, ndm = x.shape
    ndf = ndm

    # elems: [id, n1..n8]
    E = np.loadtxt(f"{mesh_dir}/circle8_e.msh")
    connectivity = E[:, 1:].astype(int)
    nel, nen = connectivity.shape
    conn0 = connectivity - 1  # 0-based

    elem = [{"cn": conn0[e, :].copy()} for e in range(nel)]

    if do_plot:
        plt.figure()
        for e in range(nel):
            cn = elem[e]["cn"]
            loop = np.r_[cn[:4], cn[0]]
            plt.plot(x[loop, 0], x[loop, 1], marker="x", linewidth=1)
            plt.plot(x[cn[4:], 0], x[cn[4:], 1], marker="x", linestyle="None")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("QUAD8 mesh")
        plt.tight_layout()

    # material parameters
    matparam = np.zeros(6, dtype=float)
    matparam[0] = 210e9     # E
    matparam[1] = 0.3       # nu
    matparam[2] = 100.0     # sigma_y0
    matparam[3] = 700.0     # H
    matparam[4] = 2.7e-03   # rho
    matparam[5] = 0.5       # r

    # Neumann loads (MATLAB list, converted to 0-based)
    neum_nodes_1based = np.array([
        611, 634, 656, 675, 692, 707, 722, 730, 746, 750, 749,
        743, 729, 720, 704, 687, 672, 652, 629, 606, 582
    ], dtype=int)
    neum_nodes = neum_nodes_1based - 1

    Fmax = -5e6
    nn = neum_nodes.size
    neum = np.zeros((nn, 3), dtype=float)
    neum[:, 0] = neum_nodes
    neum[:, 1] = 2  # y dof
    for i in range(nn):
        neum[i, 2] = Fmax * np.sin(np.pi * i / (nn - 1))

    if do_plot:
        plt.plot(x[neum_nodes, 0], x[neum_nodes, 1], "ro", markersize=4)

    # minimal Dirichlet anchor: fix ux at one interior node (to remove rigid motion)
    load_nodes = np.unique(neum_nodes)
    xmid = 0.5 * (x[:, 0].max() + x[:, 0].min())
    ymid = 0.5 * (x[:, 1].max() + x[:, 1].min())

    all_nodes = np.arange(nnp, dtype=int)
    cand = np.setdiff1d(all_nodes, load_nodes, assume_unique=False)

    ymin, ymax = x[:, 1].min(), x[:, 1].max()
    Ly = ymax - ymin
    tol_edge = 1e-3 * max(Ly, 1.0)
    edge_nodes = np.where((x[:, 1] <= ymin + tol_edge) | (x[:, 1] >= ymax - tol_edge))[0]
    cand = np.setdiff1d(cand, edge_nodes, assume_unique=False)

    if cand.size == 0:
        cand = np.setdiff1d(all_nodes, load_nodes, assume_unique=False)

    dist2 = (x[cand, 0] - xmid) ** 2 + (x[cand, 1] - ymid) ** 2
    node_ux_fix = int(cand[np.argmin(dist2)])

    # drlt: [node, ldof(1..2), value]
    drlt = np.array([[node_ux_fix, 1, 0.0]], dtype=float)

    if do_plot:
        plt.plot(x[node_ux_fix, 0], x[node_ux_fix, 1], "bs", markersize=6)
        plt.show(block=False)

    b = np.array([0.0, 0.0], dtype=float)
    return ndm, ndf, nnp, nel, nen, x, elem, matparam, drlt, neum, b


# =============================================================================
# Master element: QUAD8 with 3x3 Gauss
# =============================================================================

def build_masterelement_quad8():
    ndm = 2
    nen = 8
    nqp = 9

    a = np.sqrt(0.6)
    qpt = np.array([
        [-a, -a],
        [-a,  0],
        [-a,  a],
        [ 0, -a],
        [ 0,  0],
        [ 0,  a],
        [ a, -a],
        [ a,  0],
        [ a,  a],
    ], dtype=float)

    w8 = (1.0 / 81.0) * np.array([25, 40, 25, 40, 64, 40, 25, 40, 25], dtype=float)

    N = np.zeros((nqp, nen), dtype=float)
    gamma = np.zeros((nqp, nen, ndm), dtype=float)

    for q in range(nqp):
        xi = qpt[q, :]

        N[q, 0] = -0.25 * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[0] + xi[1])
        N[q, 1] = -0.25 * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[0] + xi[1])
        N[q, 2] = -0.25 * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[0] - xi[1])
        N[q, 3] = -0.25 * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[0] - xi[1])
        N[q, 4] =  0.5  * (1 - xi[0] ** 2) * (1 - xi[1])
        N[q, 5] =  0.5  * (1 + xi[0]) * (1 - xi[1] ** 2)
        N[q, 6] =  0.5  * (1 - xi[0] ** 2) * (1 + xi[1])
        N[q, 7] =  0.5  * (1 - xi[0]) * (1 - xi[1] ** 2)

        gamma[q, 0, :] = [-0.25 * (-1 + xi[1]) * (2 * xi[0] + xi[1]),
                          -0.25 * (-1 + xi[0]) * (xi[0] + 2 * xi[1])]
        gamma[q, 4, :] = [xi[0] * (-1 + xi[1]),
                          0.5 * (1 + xi[0]) * (-1 + xi[0])]
        gamma[q, 1, :] = [0.25 * (-1 + xi[1]) * (xi[1] - 2 * xi[0]),
                          0.25 * (1 + xi[0]) * (2 * xi[1] - xi[0])]
        gamma[q, 5, :] = [-0.5 * (1 + xi[1]) * (-1 + xi[1]),
                          -xi[1] * (1 + xi[0])]
        gamma[q, 2, :] = [0.25 * (1 + xi[1]) * (2 * xi[0] + xi[1]),
                          0.25 * (1 + xi[0]) * (xi[0] + 2 * xi[1])]
        gamma[q, 6, :] = [-xi[0] * (1 + xi[1]),
                          -0.5 * (1 + xi[0]) * (-1 + xi[0])]
        gamma[q, 3, :] = [-0.25 * (1 + xi[1]) * (xi[1] - 2 * xi[0]),
                          -0.25 * (-1 + xi[0]) * (2 * xi[1] - xi[0])]
        gamma[q, 7, :] = [0.5 * (1 + xi[1]) * (-1 + xi[1]),
                          xi[1] * (-1 + xi[0])]

    return qpt, w8, N, gamma


# =============================================================================
# Main solver
# =============================================================================

def main(mesh_dir=".", use_penalty_contact=True):
    ndm, ndf, nnp, nel, nen, x, elem, matparam, drlt, neum, b = input_circle_quad8(
        mesh_dir=mesh_dir, do_plot=True
    )
    if nen != 8:
        raise RuntimeError("This solver expects QUAD8 (nen=8).")

    ndof = nnp * ndf
    all_dofs = np.arange(ndof, dtype=int)

    # Dirichlet dofs
    drlt = np.asarray(drlt, dtype=float)
    drlt_nodes = drlt[:, 0].astype(int)
    drlt_ldof = drlt[:, 1].astype(int)
    drlt_vals = drlt[:, 2].astype(float)
    drlt_dofs = drlt_nodes * ndf + (drlt_ldof - 1)
    free_dofs = np.setdiff1d(all_dofs, drlt_dofs, assume_unique=False)

    # Neumann dofs
    neum = np.asarray(neum, dtype=float)
    neum_nodes = neum[:, 0].astype(int)
    neum_ldof = neum[:, 1].astype(int)
    neum_vals = neum[:, 2].astype(float)
    neum_dofs = neum_nodes * ndf + (neum_ldof - 1)

    fsur = np.zeros(ndof, dtype=float)
    fsur[neum_dofs] = neum_vals

    qpt, w8, Nq, gammaq = build_masterelement_quad8()
    nqp = 9
    tdm = 3

    # Precompute element dof maps + COO pattern
    nen_dof = nen * ndf
    numKe = nen_dof * nen_dof

    Iall = np.zeros(nel * numKe, dtype=int)
    Jall = np.zeros(nel * numKe, dtype=int)

    for e in range(nel):
        cn = elem[e]["cn"]
        gdof = np.zeros(nen_dof, dtype=int)
        for a in range(nen):
            for d in range(ndf):
                gdof[a * ndf + d] = cn[a] * ndf + d
        elem[e]["gdof"] = gdof

        II = np.repeat(gdof, nen_dof)
        JJ = np.tile(gdof, nen_dof)
        base = e * numKe
        Iall[base:base + numKe] = II
        Jall[base:base + numKe] = JJ

    # State variables per element/qpoint
    nsv = 19
    for e in range(nel):
        elem[e]["state"] = np.zeros((nqp, nsv), dtype=float)
        elem[e]["sigma"] = np.zeros((nqp, 9), dtype=float)
        elem[e]["dv"] = np.zeros(nqp, dtype=float)

    # Precompute geometry-dependent quantities: G and dv
    Gpre = np.zeros((nel, nqp, nen, ndm), dtype=float)
    dvpre = np.zeros((nel, nqp), dtype=float)

    for e in range(nel):
        cn = elem[e]["cn"]
        xe = x[cn, :].T  # (2,8)
        for q in range(nqp):
            # IMPORTANT FIX:
            # xe is (2,8), gamma is (8,2) => Je = (2,2). NO transpose here.
            Je = xe @ gammaq[q, :, :]  # (2,2)
            detJe = np.linalg.det(Je)
            if detJe <= 0:
                raise RuntimeError(f"detJ <= 0 in element {e}")
            dv = detJe * w8[q]
            dvpre[e, q] = dv
            invJe = np.linalg.inv(Je)
            G = gammaq[q, :, :] @ invJe  # (8,2)
            Gpre[e, q, :, :] = G
            elem[e]["dv"][q] = dv

    # Precompute body force vector
    fvol = np.zeros(ndof, dtype=float)
    rho = float(matparam[4])

    for e in range(nel):
        gdof = elem[e]["gdof"]
        fvole = np.zeros(nen_dof, dtype=float)
        for q in range(nqp):
            dv = dvpre[e, q]
            for A in range(nen):
                fvolA = Nq[q, A] * rho * b * dv
                fvole[A * ndf:(A + 1) * ndf] += fvolA
        fvol[gdof] += fvole

    # Penalty contact setup
    if use_penalty_contact:
        yRail = float(x[:, 1].min())
        Ly = float(x[:, 1].max() - x[:, 1].min())
        Lx = float(x[:, 0].max() - x[:, 0].min())
        tolY = 1e-3 * max(Ly, 1.0)
        xmid = 0.5 * (float(x[:, 0].max()) + float(x[:, 0].min()))
        patch_half_width = 0.20 * max(Lx, 1.0)

        bottomCand = np.where(x[:, 1] <= yRail + tolY)[0]
        contactNodes = bottomCand[np.abs(x[bottomCand, 0] - xmid) <= patch_half_width]
        if contactNodes.size == 0:
            contactNodes = bottomCand
        contactNodes = np.unique(contactNodes)

        contactTol = 0.0
        kpAuto = True
        kpScale = 10.0
        print(f"Penalty contact ON: yRail={yRail:.6g}, #contactNodes={contactNodes.size}, kpAuto={kpAuto}, kpScale={kpScale:.3g}")
    else:
        contactNodes = np.array([], dtype=int)

    # Load stepping + Newton params
    numSteps = 20
    loadScale = 1.0
    loadFactor = np.linspace(0.0, 1.0, numSteps)

    maxIter = 30
    tolR = 1e-5
    tolDU = 1e-8

    u = np.zeros(ndof, dtype=float)
    u[drlt_dofs] = drlt_vals

    hist_iter = np.zeros(numSteps, dtype=int)
    hist_maxk = np.zeros(numSteps, dtype=float)
    hist_res = np.zeros(numSteps, dtype=float)

    # Load steps
    for n in range(numSteps):
        print(f"\n=== Load step {n+1} / {numSteps} (factor = {loadFactor[n]:.4g}) ===")
        fext = fvol + loadScale * loadFactor[n] * fsur

        u[drlt_dofs] = drlt_vals
        res0 = max(1e-12, np.linalg.norm(fext[free_dofs]))
        res_prev = np.inf

        stateIter = [None] * nel
        sigmaIter = [None] * nel

        # Newton iterations
        for it in range(1, maxIter + 1):
            Vall = np.zeros(nel * numKe, dtype=float)
            fint = np.zeros(ndof, dtype=float)

            for e in range(nel):
                gdof = elem[e]["gdof"]
                ue = u[gdof].reshape((nen, ndf)).T  # (2,8)

                Ke = np.zeros((nen_dof, nen_dof), dtype=float)
                fe = np.zeros(nen_dof, dtype=float)

                sv_e = np.zeros((nqp, nsv), dtype=float)
                sig_e = np.zeros((nqp, 9), dtype=float)

                for q in range(nqp):
                    dv = dvpre[e, q]
                    G = Gpre[e, q, :, :]  # (8,2)

                    h = np.zeros((tdm, tdm), dtype=float)
                    h[:ndm, :ndm] = ue @ G  # (2,2)
                    eps = 0.5 * (h + h.T)

                    sv_old = elem[e]["state"][q, :]
                    sigma, C4, sv_new, _isPl = vmises_iso_kin_rr(eps, matparam, sv_old)

                    sv_e[q, :] = sv_new
                    sig_e[q, :] = sigma.reshape((9,), order="F")

                    sig2 = sigma[:ndm, :ndm]
                    C2 = C4[:ndm, :ndm, :ndm, :ndm]

                    for A in range(nen):
                        G_A = G[A, :]
                        feA = (G_A @ sig2) * dv
                        fe[A * ndf:(A + 1) * ndf] += feA

                        for B in range(nen):
                            G_B = G[B, :]
                            KAB = t1_dot_t4_dot_t1(G_A, C2, G_B) * dv
                            Ke[A * ndf:(A + 1) * ndf, B * ndf:(B + 1) * ndf] += KAB

                fint[gdof] += fe
                Vall[e * numKe:(e + 1) * numKe] = Ke.reshape((numKe,), order="C")

                stateIter[e] = sv_e
                sigmaIter[e] = sig_e

            Kt = coo_matrix((Vall, (Iall, Jall)), shape=(ndof, ndof)).tocsr()

            # penalty contact contributions
            fcont = np.zeros(ndof, dtype=float)
            Kcont = csr_matrix((ndof, ndof), dtype=float)

            if use_penalty_contact and contactNodes.size > 0:
                uy = u[1::ndf]
                ycur = x[:, 1] + uy

                gap = ycur[contactNodes] - yRail
                activeMask = gap <= contactTol
                activeNodes = contactNodes[activeMask]
                gapA = gap[activeMask]

                if activeNodes.size > 0:
                    dofY = activeNodes * ndf + 1

                    if kpAuto:
                        dofYcand = contactNodes * ndf + 1
                        d = Kt[dofYcand, dofYcand].diagonal()
                        ky_med = np.median(np.abs(d))
                        if (not np.isfinite(ky_med)) or (ky_med <= 0):
                            ky_med = 1.0
                        kp = kpScale * ky_med
                    else:
                        kp = 1.0

                    fcont[dofY] = -kp * gapA
                    Kcont = coo_matrix((np.full(dofY.size, kp), (dofY, dofY)), shape=(ndof, ndof)).tocsr()

            Kt_eff = Kt + Kcont

            # Residual
            R = fext + fcont - fint
            Rf = R[free_dofs]
            res = np.linalg.norm(Rf) / res0

            # Solve increment
            du = np.zeros(ndof, dtype=float)
            du[free_dofs] = spsolve(Kt_eff[free_dofs, :][:, free_dofs], Rf)
            relDu = np.linalg.norm(du[free_dofs]) / max(1.0, np.linalg.norm(u[free_dofs]))

            print(f"  it={it:2d}   ||R||/||f|| = {res:.3e}   ||du||/||u|| = {relDu:.3e}")

            if ((res < tolR) and (relDu < tolDU)) or ((relDu < tolDU) and (res < 5 * tolR)):
                print(f"  -> Converged in {it} iterations.")

                for e in range(nel):
                    elem[e]["state"][:, :] = stateIter[e]
                    elem[e]["sigma"][:, :] = sigmaIter[e]

                hist_iter[n] = it
                hist_res[n] = res

                maxk = 0.0
                for e in range(nel):
                    maxk = max(maxk, float(np.max(elem[e]["state"][:, 9])))
                hist_maxk[n] = maxk

                reac = fint[drlt_dofs] - fext[drlt_dofs]
                Rx = float(np.sum(reac[drlt_ldof == 1]))
                Ry = float(np.sum(reac[drlt_ldof == 2]))
                print(f"  Anchor reaction sum: Rx = {Rx:.6e}, Ry = {Ry:.6e}")
                print(f"  External load sum:   Fx = {np.sum(fext[0::2]):.6e}, Fy = {np.sum(fext[1::2]):.6e}")
                print(f"  Contact force sum:   Fx = {np.sum(fcont[0::2]):.6e}, Fy = {np.sum(fcont[1::2]):.6e}")
                break

            # damping
            alpha = 1.0
            if it > 1 and np.isfinite(res_prev) and (res > 1.2 * res_prev):
                alpha = 0.3
            elif it > 1 and np.isfinite(res_prev) and (res > res_prev):
                alpha = 0.6
            res_prev = res

            u = u + alpha * du
            u[drlt_dofs] = drlt_vals

            if it == maxIter:
                raise RuntimeError("Newton did not converge. Try smaller load steps or reduce kpScale.")

    # Post-processing (final fields at quadrature points)
    plot_vm = np.zeros((nel * nqp, 3), dtype=float)
    plot_k = np.zeros((nel * nqp, 3), dtype=float)
    idx = 0

    for e in range(nel):
        cn = elem[e]["cn"]
        xe = x[cn, :].T
        for q in range(nqp):
            xqp = xe @ Nq[q, :]
            sigma = elem[e]["sigma"][q, :].reshape((3, 3), order="F")

            s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
            s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
            sigma_v = np.sqrt(
                s11**2 + s22**2 + s33**2
                - s11 * s22 - s11 * s33 - s22 * s33
                + 3.0 * (s12**2 + s13**2 + s23**2)
            )

            k = elem[e]["state"][q, 9]

            plot_vm[idx, :2] = xqp
            plot_vm[idx, 2] = sigma_v
            plot_k[idx, :2] = xqp
            plot_k[idx, 2] = k
            idx += 1

    plt.figure()
    plt.scatter(plot_vm[:, 0], plot_vm[:, 1], c=plot_vm[:, 2], s=12)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar()
    plt.title("von Mises stress (final)")

    plt.figure()
    plt.scatter(plot_k[:, 0], plot_k[:, 1], c=plot_k[:, 2], s=12)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar()
    plt.title("Accumulated plastic strain k (final)")

    plt.figure()
    plt.plot(np.arange(1, numSteps + 1), hist_iter, marker="o")
    plt.grid(True)
    plt.xlabel("Load step")
    plt.ylabel("Newton iterations")
    plt.title("Newton iterations per load step")

    plt.figure()
    plt.plot(np.arange(1, numSteps + 1), hist_maxk, marker="o")
    plt.grid(True)
    plt.xlabel("Load step")
    plt.ylabel("max k")
    plt.title("Max accumulated plastic strain vs load step")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=str, default=".", help="Folder containing circle8_n.msh and circle8_e.msh")
    parser.add_argument("--no-contact", action="store_true", help="Disable penalty contact")
    args = parser.parse_args()
    main(mesh_dir=args.mesh_dir, use_penalty_contact=(not args.no_contact))
