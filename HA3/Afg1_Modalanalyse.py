import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

# ---------------------------------------------------------
# [Einstellung] Verschiebungs-Skalierungsfaktor (200000-fache Vergrößerung → für statische Analyseergebnisse)
# ---------------------------------------------------------
disp_scaling = 500
toplot = True

tdm = 2  # Tensor-Dimension für 2D-Probleme


def analysis():
    E = 210e9  # E-Modul in Pa
    nu = 0.3
    rho = 11503.0  # Dichte in kg/m^3

    F = 1000.0  # Gesamtkraft in N
    ndf = 2  # Anzahl der Freiheitsgrade pro Knoten
    ndm = 2  # 2D-Problem

    # Q8 Element Einstellungen
    nen = 8
    nqp = 9

    global tdm
    # --- Geometrie ---
    length = 9  # m
    height = 0.6943  # m
    width = 0.00583  # m

    # [Änderung 1] Analytische Frequenzen bis zur 7. Eigenform berechenbar
    def get_analytical_freqs(num_modes=7):
        """Compute analytical natural frequencies f_n [Hz] for a cantilever beam."""
        A = width * height
        I_z = width * height**3 / 12.0

        # beta_n * L Werte für Moden 1 bis 7
        betasL = torch.tensor(
            [1.875, 4.694, 7.855, 10.996, 14.137, 17.279, 20.420], dtype=torch.float64
        )[:num_modes]

        const = math.sqrt(E * I_z / (rho * A * length**4))
        omega_ana = betasL**2 * const
        f_ana = omega_ana / (2.0 * math.pi)
        return f_ana

    nx = 20  # Anzahl der Elemente in Längsrichtung
    ny = 2  # Anzahl der Elemente in Höhenrichtung

    # Knoten erzeugen
    x_coords = torch.linspace(0, length, 2 * nx + 1)
    y_coords = torch.linspace(0, height, 2 * ny + 1)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
    x = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)

    # Elemente erzeugen (Q8 Konnektivität)
    elements_list = []
    num_nodes_y = 2 * ny + 1
    for j in range(ny):
        for i in range(nx):
            n1 = (2 * i) * num_nodes_y + (2 * j)
            n2 = (2 * (i + 1)) * num_nodes_y + (2 * j)
            n3 = (2 * (i + 1)) * num_nodes_y + (2 * (j + 1))
            n4 = (2 * i) * num_nodes_y + (2 * (j + 1))
            n5 = (2 * i + 1) * num_nodes_y + (2 * j)
            n6 = (2 * (i + 1)) * num_nodes_y + (2 * j + 1)
            n7 = (2 * i + 1) * num_nodes_y + (2 * (j + 1))
            n8 = (2 * i) * num_nodes_y + (2 * j + 1)
            elements_list.append([n1, n2, n3, n4, n5, n6, n7, n8])

    elems = torch.tensor(elements_list, dtype=torch.long)

    # Nicht verwendete Knoten herausfiltern
    unique_nodes = torch.unique(elems.flatten())
    x = x[unique_nodes]
    node_map = torch.zeros(x_grid.numel(), dtype=torch.long)
    node_map[unique_nodes] = torch.arange(len(unique_nodes))
    elems = node_map[elems]

    # Randbedingungen (Dirichlet)
    left_nodes = torch.where(x[:, 0] == 0)[0]
    drlt_list = [[[node_idx, 0, 0], [node_idx, 1, 0]] for node_idx in left_nodes]
    drlt = torch.tensor([item for sublist in drlt_list for item in sublist])

    # Kraft (Neumann Randbedingung)
    total_force = -F / width
    right_nodes = torch.where(x[:, 0] == length)[0]
    force_per_node = total_force / len(right_nodes)
    neum_list = [[node_idx, 1, force_per_node] for node_idx in right_nodes]
    neum = torch.tensor(neum_list)

    ei = torch.eye(ndm, ndm)
    I = torch.eye(3, 3)

    # Materialtensor
    C4 = torch.zeros(2, 2, 2, 2)
    factor = E / (1 - nu**2)
    C4[0, 0, 0, 0] = factor
    C4[1, 1, 1, 1] = factor
    C4[0, 0, 1, 1] = factor * nu
    C4[1, 1, 0, 0] = factor * nu
    shear_factor = E / (2 * (1 + nu))
    C4[0, 1, 0, 1] = C4[1, 0, 0, 1] = C4[0, 1, 1, 0] = C4[1, 0, 1, 0] = shear_factor

    ############ Preprocessing ###############
    nnp = x.size()[0]
    print("nnp: ", nnp)
    nel = elems.size()[0]

    drlt_mask = torch.zeros(nnp * ndf, 1)
    drlt_vals = torch.zeros(nnp * ndf, 1)
    for i in range(drlt.size()[0]):
        drlt_mask[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = 1
        drlt_vals[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = drlt[i, 2]

    free_mask = torch.ones(nnp * ndf, 1) - drlt_mask
    drltDofs = torch.nonzero(drlt_mask)
    # print("drltDofs", drltDofs)

    drlt_matrix = 1e22 * torch.diag(drlt_mask[:, 0], 0)

    neum_vals = torch.zeros(nnp * ndf, 1)
    for i in range(neum.size()[0]):
        neum_vals[int(neum[i, 0]) * ndf + int(neum[i, 1]), 0] = neum[i, 2]

    # Gauss-Quadratur
    qpt = torch.zeros(nqp, ndm)
    w8 = torch.zeros(nqp, 1)
    a = math.sqrt(3.0 / 5.0)
    w1 = 5.0 / 9.0
    w2 = 8.0 / 9.0

    qpt[0, :] = torch.tensor([-a, -a])
    qpt[1, :] = torch.tensor([0, -a])
    qpt[2, :] = torch.tensor([a, -a])
    qpt[3, :] = torch.tensor([-a, 0])
    qpt[4, :] = torch.tensor([0, 0])
    qpt[5, :] = torch.tensor([a, 0])
    qpt[6, :] = torch.tensor([-a, a])
    qpt[7, :] = torch.tensor([0, a])
    qpt[8, :] = torch.tensor([a, a])

    w8[0] = w1 * w1
    w8[1] = w1 * w2
    w8[2] = w1 * w1
    w8[3] = w2 * w1
    w8[4] = w2 * w2
    w8[5] = w2 * w1
    w8[6] = w1 * w1
    w8[7] = w1 * w2
    w8[8] = w1 * w1

    # Formfunktionen
    masterelem_N = torch.zeros(nqp, nen)
    masterelem_gamma = torch.zeros(nqp, nen, ndm)

    for q in range(nqp):
        xi = qpt[q, :]
        e, n = xi[0], xi[1]
        masterelem_N[q, 0] = 0.25 * (1 - e) * (1 - n) * (-e - n - 1)
        masterelem_N[q, 1] = 0.25 * (1 + e) * (1 - n) * (e - n - 1)
        masterelem_N[q, 2] = 0.25 * (1 + e) * (1 + n) * (e + n - 1)
        masterelem_N[q, 3] = 0.25 * (1 - e) * (1 + n) * (-e + n - 1)
        masterelem_N[q, 4] = 0.5 * (1 - e * e) * (1 - n)
        masterelem_N[q, 5] = 0.5 * (1 + e) * (1 - n * n)
        masterelem_N[q, 6] = 0.5 * (1 - e * e) * (1 + n)
        masterelem_N[q, 7] = 0.5 * (1 - e) * (1 - n * n)

        # Ableitungen d/de
        masterelem_gamma[q, 0, 0] = 0.25 * (1 - n) * (-1) * (-e - n - 1) + 0.25 * (
            1 - e
        ) * (1 - n) * (-1)
        masterelem_gamma[q, 1, 0] = 0.25 * (1 - n) * (1) * (e - n - 1) + 0.25 * (
            1 + e
        ) * (1 - n) * (1)
        masterelem_gamma[q, 2, 0] = 0.25 * (1 + n) * (1) * (e + n - 1) + 0.25 * (
            1 + e
        ) * (1 + n) * (1)
        masterelem_gamma[q, 3, 0] = 0.25 * (1 + n) * (-1) * (-e + n - 1) + 0.25 * (
            1 - e
        ) * (1 + n) * (-1)
        masterelem_gamma[q, 4, 0] = 0.5 * (-2 * e) * (1 - n)
        masterelem_gamma[q, 5, 0] = 0.5 * (1) * (1 - n * n)
        masterelem_gamma[q, 6, 0] = 0.5 * (-2 * e) * (1 + n)
        masterelem_gamma[q, 7, 0] = 0.5 * (-1) * (1 - n * n)

        # Ableitungen d/dn
        masterelem_gamma[q, 0, 1] = 0.25 * (1 - e) * (-1) * (-e - n - 1) + 0.25 * (
            1 - e
        ) * (1 - n) * (-1)
        masterelem_gamma[q, 1, 1] = 0.25 * (1 + e) * (-1) * (e - n - 1) + 0.25 * (
            1 + e
        ) * (1 - n) * (-1)
        masterelem_gamma[q, 2, 1] = 0.25 * (1 + e) * (+1) * (e + n - 1) + 0.25 * (
            1 + e
        ) * (1 + n) * (1)
        masterelem_gamma[q, 3, 1] = 0.25 * (1 - e) * (+1) * (-e + n - 1) + 0.25 * (
            1 - e
        ) * (1 + n) * (1)
        masterelem_gamma[q, 4, 1] = 0.5 * (1 - e * e) * (-1)
        masterelem_gamma[q, 5, 1] = 0.5 * (1 + e) * (-2 * n)
        masterelem_gamma[q, 6, 1] = 0.5 * (1 - e * e) * (1)
        masterelem_gamma[q, 7, 1] = 0.5 * (1 - e) * (-2 * n)

    indices = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7, 0])

    ############## Matrix Assembly ###############
    u = torch.zeros(ndf * nnp, 1)
    edof = torch.zeros(nel, ndf, nen, dtype=int)
    gdof = torch.zeros(nel, ndf * nen, dtype=int)
    for el in range(nel):
        for ien in range(nen):
            for idf in range(ndf):
                edof[el, idf, ien] = ndf * elems[el, ien] + idf
        gdof[el, :] = edof[el, :, :].t().reshape(ndf * nen)

    finte = torch.zeros(nen * ndf, 1)
    fvole = torch.zeros(nen * ndf, 1)

    Ke = torch.zeros(nen * ndf, nen * ndf)
    K = torch.zeros(nnp * ndf, nnp * ndf)
    Me = torch.zeros(nen * ndf, nen * ndf)
    M = torch.zeros(nnp * ndf, nnp * ndf)

    h = torch.zeros(tdm, tdm)
    G_A = torch.zeros(1, ndm)
    fsur = neum_vals
    fint = torch.zeros(ndf * nnp, 1)
    fvol = torch.zeros(ndf * nnp, 1)

    for el in range(nel):
        xe = torch.zeros(ndm, nen)
        for idm in range(ndm):
            xe[idm, :] = x[elems[el, :], idm]
        ue = torch.squeeze(u[edof[el, 0:ndm, :]])

        Ke.zero_()
        Me.zero_()
        for q in range(nqp):
            N = masterelem_N[q]
            gamma = masterelem_gamma[q]
            Je = xe.mm(gamma)
            detJe = torch.det(Je)

            if detJe <= 0:
                print("Error: detJe <= 0")

            dv = detJe * w8[q] * width  # Breitenkorrektur enthalten!
            invJe = torch.inverse(Je)
            G = gamma.mm(invJe)

            h[0:ndm, 0:ndm] = ue.mm(G)
            eps_2d = 0.5 * (h + h.t())[0:ndm, 0:ndm]
            stre_2d = torch.tensordot(C4, eps_2d, dims=2)

            for A in range(nen):
                G_A[0, :] = G[A, :]
                fintA = dv * (G_A.mm(stre_2d))
                finte[A * ndf : A * ndf + ndm] += fintA.t()
                for B in range(nen):
                    KAB = torch.tensordot(
                        G[A, :],
                        (
                            torch.tensordot(
                                C4[0:tdm, 0:tdm, 0:tdm, 0:tdm], G[B, :], [[3], [0]]
                            )
                        ),
                        [[0], [0]],
                    )
                    Ke[A * ndf : A * ndf + ndm, B * ndf : B * ndf + ndm] += dv * KAB
                    MAB = rho * N[A] * N[B] * I[0:ndm, 0:ndm]
                    Me[A * ndf : A * ndf + ndm, B * ndf : B * ndf + ndm] += dv * MAB

        fint[gdof[el, :]] += finte
        fvol[gdof[el, :]] += fvole
        for i in range(gdof.shape[1]):
            K[gdof[el, i], gdof[el, :]] += Ke[i, :]
            M[gdof[el, i], gdof[el, :]] += Me[i, :]

    rsd = free_mask.mul(fsur - fint)
    K_tilde = K + drlt_matrix
    free_dofs = torch.nonzero(free_mask)
    print("free_dofs", free_dofs)
    free_dofs = free_dofs[:, 0]
    print("free_dofs", free_dofs)

    K_free = K[free_dofs, :][:, free_dofs]
    M_free = M[free_dofs, :][:, free_dofs]
    print("K_free:", K_free)
    print("M_free:", M_free)

    # ====== Modalanalyse ======
    # 1) K^-1 * M
    KInv = torch.linalg.inv(K_free)
    AK = KInv.mm(M_free)
    vals_K, vecs_K = torch.linalg.eig(AK)  # Eigenvektoren speichern!

    omega_K = 1.0 / torch.sqrt(vals_K.real)
    mask_K = ~torch.isinf(omega_K) & ~torch.isnan(omega_K) & (omega_K.real > 0)
    omega_K = omega_K[mask_K]
    omega_K_sorted, _ = torch.sort(omega_K)

    # 2) M^-1 * K
    MInv = torch.linalg.inv(M_free)
    AM = MInv.mm(K_free)
    vals_M, _ = torch.linalg.eig(AM)
    omega_M = torch.sqrt(vals_M.real)
    mask_M = ~torch.isinf(omega_M) & ~torch.isnan(omega_M) & (omega_M.real > 0)
    omega_M = omega_M[mask_M]
    omega_M_sorted, _ = torch.sort(omega_M)

    # 3) Geklumpte M^-1 * K
    sumM = torch.sum(M_free, dim=1)
    sumM = torch.abs(sumM) + 1e-12
    diagMInv = torch.diag(1.0 / sumM)
    Op_Lumped = diagMInv.mm(K_free)
    vals_L, _ = torch.linalg.eig(Op_Lumped)
    vals_L = vals_L.real
    vals_L = vals_L[vals_L > 0]
    omega_L = torch.sqrt(vals_L)
    mask_L = ~torch.isinf(omega_L) & ~torch.isnan(omega_L) & (omega_L.real > 0)
    omega_L = omega_L[mask_L]
    omega_L_sorted, _ = torch.sort(omega_L)

    # Ergebnisse ausgeben
    def first_freq(omega_tensor):
        if omega_tensor.numel() == 0:
            return 0.0
        return omega_tensor[0].item() / (2.0 * math.pi)

    f1_K = first_freq(omega_K_sorted)
    f1_M = first_freq(omega_M_sorted)
    f1_L = first_freq(omega_L_sorted)

    print("\n=== 1. Eigenfrequenz (Hz) ===")
    print(f"  K^-1 M   : {f1_K:.4f} Hz")
    print(f"  M^-1 K   : {f1_M:.4f} Hz")
    print(f"  Lumped M : {f1_L:.4f} Hz")

    # [Änderung 2] Anzahl der Modi zum Vergleich auf 7 erhöht
    num_modes_text = 4
    f_ana = get_analytical_freqs(num_modes_text).detach().numpy()
    modes = np.arange(1, num_modes_text + 1)

    def top_f(omega_tensor):
        res = np.zeros(num_modes_text)
        n = min(num_modes_text, omega_tensor.numel())
        if n > 0:
            res[:n] = omega_tensor[:n].detach().numpy() / (2.0 * math.pi)
        return res

    f_k_top = top_f(omega_K_sorted)
    f_m_top = top_f(omega_M_sorted)
    f_l_top = top_f(omega_L_sorted)

    print(f"\n=== Erste {num_modes_text} Frequenzen [Hz] (Analytisch vs FEM) ===")
    for i in range(num_modes_text):
        print(
            f"Mode {i + 1}:  f_ana = {f_ana[i]:.4f} | "
            f"K^-1M = {f_k_top[i]:.4f}, "
            f"M^-1K = {f_m_top[i]:.4f}, "
            f"lumped = {f_l_top[i]:.4f}"
        )

    

    # Statische Lösung
    du = torch.linalg.solve(K_tilde, rsd)
    u += du
    u = free_mask.mul(u) + drlt_mask.mul(drlt_vals)
    fext = K * u
    frea = fext - fvol - fsur



    print("Gesamtmasse:" , torch.sum(M) / ndm, "kg")
    print("Gesamtmasse analytisch:" , width*height*length*rho, "mm^2")

    # ==========================================
    # [Änderung 3] Abbildung 2: Modenformen
    # ==========================================
    if toplot:
        omega_K_unsorted = 1.0 / torch.sqrt(vals_K.real)
        mask = (
            ~torch.isinf(omega_K_unsorted)
            & ~torch.isnan(omega_K_unsorted)
            & (omega_K_unsorted.real > 0)
        )
        valid_indices = torch.nonzero(mask).flatten()
        omega_valid = omega_K_unsorted[mask]
        sorted_indices = torch.argsort(omega_valid)
        final_indices = valid_indices[sorted_indices]

        # Erste drei Modi als einzelne Figures darstellen und speichern
        num_plot_modes = 4
        for i in range(num_plot_modes):
            idx = final_indices[i]
            mode_shape = vecs_K[:, idx].real
            full_mode = torch.zeros(nnp * ndf, 1)
            full_mode[free_dofs, 0] = mode_shape
            u_mode = full_mode.reshape(-1, 2)
            # Automatische Skalierung
            scale_factor = 2 * height / (torch.max(torch.abs(u_mode)) + 1e-20)
            x_mode = x + u_mode * scale_factor

            plt.figure(i + 1, figsize=(10, 6))
            freq = omega_valid[sorted_indices[i]] / (2 * math.pi)
            plt.title(f"Eigenmode {i + 1} ({freq:.2f} Hz)", fontsize=16)
            for e in range(nel):
                els = torch.index_select(elems[e, :], 0, indices)
                plt.plot(x_mode[els, 0], x_mode[els, 1], "b-", linewidth=0.5)
            plt.axis("equal")
            plt.grid(True)
            plt.savefig(f"{i+1} Eigenform Python.png", dpi=600, bbox_inches="tight")
        plt.show()

    # Optional: Vergleichsplot der Frequenzen (kann beibehalten werden)
    plt.figure(4, figsize=(10, 6))
    plt.title("Modale Frequenzen: Analytisch vs FEM")
    plt.plot(modes, f_ana, "ko-", label="Analytisch")
    plt.plot(modes, f_k_top, "rs--", label="K^-1 M")
    plt.plot(modes, f_m_top, "b^:", label="M^-1 K")
    plt.plot(modes, f_l_top, "gx-.", label="Geklumpte M")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Eigenmode")
    plt.ylabel("Frequenz [Hz]")
    plt.savefig("frequenzvergleich.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    start_perfcount = timemodule.perf_counter()
    analysis()
    end_perfcount = timemodule.perf_counter()
    print("Elapsed = {}s".format((end_perfcount - start_perfcount)))

