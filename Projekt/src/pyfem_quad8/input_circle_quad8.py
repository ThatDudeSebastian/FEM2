"""
input_circle_quad8.py

Python translation of input_circle_quad8.m.

It loads:
  - circle8_n.msh (node list: [id, x, y])
  - circle8_e.msh (element list: [id, n1..n8])

Both are read via numpy.loadtxt (whitespace-delimited).

Returns
-------
ndm, ndf, nnp, nel, nen, x, elem, matparam, drlt, neum, b
- x: (nnp, 2)
- elem: list of dicts with key "cn" (0-based connectivity, length nen)

Important
---------
MATLAB node IDs are 1-based. This function converts to 0-based indices.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def input_circle_quad8(mesh_dir: str = ".", do_plot: bool = True):
    # ---- Load mesh ----
    N = np.loadtxt(f"{mesh_dir}/circle8_n.msh")
    x = np.column_stack((N[:, 1], N[:, 2]))
    nnp, ndm = x.shape
    ndf = ndm

    E = np.loadtxt(f"{mesh_dir}/circle8_e.msh")
    connectivity = E[:, 1:].astype(int)  # node ids (1-based)
    nel, nen = connectivity.shape
    conn0 = connectivity - 1            # to 0-based

    elem = [{"cn": conn0[e, :].copy()} for e in range(nel)]

    # ---- Plot mesh ----
    if do_plot:
        plt.figure()
        for e in range(nel):
            cn = elem[e]["cn"]
            # edges 1..4 like the MATLAB plot
            loop = np.r_[cn[:4], cn[0]]
            plt.plot(x[loop, 0], x[loop, 1], marker="x", linewidth=1)
            plt.plot(x[cn[4:], 0], x[cn[4:], 1], marker="x", linestyle="None")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("QUAD8 mesh")
        plt.tight_layout()

    # ---- Material parameters ----
    matparam = np.zeros(6, dtype=float)
    matparam[0] = 210e9     # E
    matparam[1] = 0.3       # nu
    matparam[2] = 100.0     # sigma_y0
    matparam[3] = 700.0     # H
    matparam[4] = 2.7e-03   # rho
    matparam[5] = 0.5       # r (mixing)

    # ---- Neumann loads (node id list from MATLAB, convert to 0-based) ----
    neum_nodes_1based = np.array([
        611, 634, 656, 675, 692, 707, 722, 730, 746, 750, 749,
        743, 729, 720, 704, 687, 672, 652, 629, 606, 582
    ], dtype=int)
    neum_nodes = neum_nodes_1based - 1

    Fmax = -5e6
    nn = neum_nodes.size
    neum = np.zeros((nn, 3), dtype=float)
    neum[:, 0] = neum_nodes
    neum[:, 1] = 2  # ldof=2 => y
    for i in range(nn):
        neum[i, 2] = Fmax * np.sin(np.pi * i / (nn - 1))

    if do_plot:
        plt.plot(x[neum_nodes, 0], x[neum_nodes, 1], "ro", markersize=4)

    # ---- Minimal Dirichlet anchor: ux=0 at one interior node ----
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

    # drlt: [node, ldof, value]
    drlt = np.array([[node_ux_fix, 1, 0.0]], dtype=float)

    if do_plot:
        plt.plot(x[node_ux_fix, 0], x[node_ux_fix, 1], "bs", markersize=6)

    b = np.array([0.0, 0.0], dtype=float)

    if do_plot:
        plt.show(block=False)

    return ndm, ndf, nnp, nel, nen, x, elem, matparam, drlt, neum, b
