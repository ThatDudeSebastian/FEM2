import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)
torch.set_num_threads(16)

disp_scaling = 100
toplot = True

tdm = 2 # Tensor dimension for 2
 D problems


def analysis():
    E = 220e9  # E-Modul in Pa (210000MPa)
    nu = 0.3
    rho = 11503.0  # Dichte in kg/m^3

    F = 1000.0  # Gesamtkraft in N
    ndf = 2 # Anzahl der Freiheitsgrade pro Knoten
    ndm = 2 # 2D-Problem (2 Raumdimensionen) Number of dimensions
 
    nen = 8 # 8 Knoten für quadratisches Element (Q8)
    nqp = 9 # 3x3 Gauß-Quadratur für Q8-Elemente


    global tdm
    ###### Geometrie: Balkennetzes ######
    length = 9  # m
    height = 0.6943 # m
    width = 0.00583  # m (für Kraftberechnung, nicht für 2D-Netz)

    nx = 20  # Anzahl der Elemente in Längsrichtung
    ny = 2   # Anzahl der Elemente in Höhenrichtung (wichtig für genauen Verlauf)

    # Erzeuge Knoten
    # Für Q8-Elemente benötigen wir Knoten an den Ecken UND in der Mitte der Kanten.
    # Wir erzeugen daher ein feineres Gitter von potenziellen Knotenpositionen.
    x_coords = torch.linspace(0, length, 2 * nx + 1)
    y_coords = torch.linspace(0, height, 2 * ny + 1)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij') # Gitter von Knotenkoordinaten
    x = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1) # Knotenkoordinaten als Tensor [nnp x ndm]

    # Erzeuge Elemente (Konnektivität)
    elements_list = [] # Liste zur Speicherung der Elemente wird initialisiert
    num_nodes_y = 2 * ny + 1
    for j in range(ny): # Schleife über Elemente in y-Richtung
        for i in range(nx): # Schleife über Elemente in x-Richtung
            # Eckknoten (wie bei Q4, aber mit Schritt 2)
            n1 = (2 * i) * num_nodes_y + (2 * j)          # Unten links
            n2 = (2 * (i + 1)) * num_nodes_y + (2 * j)    # Unten rechts
            n3 = (2 * (i + 1)) * num_nodes_y + (2 * (j + 1)) # Oben rechts
            n4 = (2 * i) * num_nodes_y + (2 * (j + 1))      # Oben links

            # Mittenknoten
            n5 = (2 * i + 1) * num_nodes_y + (2 * j)      # Mitte unten
            n6 = (2 * (i + 1)) * num_nodes_y + (2 * j + 1)  # Mitte rechts
            n7 = (2 * i + 1) * num_nodes_y + (2 * (j + 1))  # Mitte oben
            n8 = (2 * i) * num_nodes_y + (2 * j + 1)      # Mitte links

            # Reihenfolge für Q8: 4 Ecken, dann 4 Mittenknoten
            elements_list.append([n1, n2, n3, n4, n5, n6, n7, n8])
    elems = torch.tensor(elements_list, dtype=torch.long) # Elemente als Tensor [nel x nen]

    # Filtere ungenutzte Knoten heraus (falls das Gitter nicht perfekt passt)
    unique_nodes = torch.unique(elems.flatten())
    x = x[unique_nodes]
    
    # Erstelle eine Mapping-Tabelle von alten zu neuen Knotenindizes
    node_map = torch.zeros(x_grid.numel(), dtype=torch.long)
    node_map[unique_nodes] = torch.arange(len(unique_nodes))
    
    # Wende das Mapping auf die Element-Konnektivität an
    elems = node_map[elems]

    # Einspannung am linken Rand (x=0)
    left_nodes = torch.where(x[:, 0] == 0)[0] # Knoten am linken Rand finden (Spalte 0 ist x-Koordinate)
    drlt_list = [[[node_idx, 0, 0], [node_idx, 1, 0]] for node_idx in left_nodes] # hier werden Dirichlet-Randbedingungen definiert
    # Jeder Eintrag in drlt_list ist eine Liste von [node_idx, dof, value] wobei value = 0 (festgehalten) ist
    # node ist der Knotenindex, dof ist der Freiheitsgrad (0 für x-Richtung, 1 für y-Richtung) das heißt node_idx entspricht nnp Knoten 
    # node_idx ist der Index des Knotens, 0 und 1 sind die Freiheitsgrade (x- und y-Richtung), 0 ist der Wert (festgehalten)
    drlt = torch.tensor([item for sublist in drlt_list for item in sublist]) # Flatten der Liste und in Tensor umwandeln 

    # Querkraft am rechten Rand (x=length)
    total_force = - F
    right_nodes = torch.where(x[:, 0] == length)[0] # x[:,0] ist die x-Koordinate aller Knoten, wo x = length & [0] gibt die Indizes der Knoten zurück
    force_per_node = total_force / len(right_nodes) # Kraft pro Knoten am rechten Rand
    neum_list = [[node_idx, 1, force_per_node] for node_idx in right_nodes]
    neum = torch.tensor(neum_list)

    ############ Identity tensors ###########
    ei = torch.eye(ndm, ndm)

    I = torch.eye(3, 3)

    blk = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    #C4 = blk * I4 + 2 * mu * I4dev
    lame = blk - 2 / 3 * mu

    # Korrekter Materialtensor für ebenen Spannungszustand (plane stress)
    C4 = torch.zeros(2,2,2,2) # 4. Ordnung Tensor für 2D leerer Tensor
    factor = E / (1 - nu**2)
    C4[0,0,0,0] = factor * 1
    C4[1,1,1,1] = factor * 1
    C4[0,0,1,1] = factor * nu
    C4[1,1,0,0] = factor * nu
    
    shear_factor = E / (2 * (1 + nu)) # Schubmodul G
    C4[0,1,0,1] = C4[1,0,0,1] = C4[0,1,1,0] = C4[1,0,1,0] = shear_factor

    ############ Preprocessing ###############
    nnp = x.size()[0] # number of nodal points for example with one element: 4
    print("nnp: ", nnp)
    nel = elems.size()[0] # number of elements bezogen auf elems Tensor z.B. mit einem Element: 1

    drlt_mask = torch.zeros(nnp * ndf, 1) # Maske so lang wie alle Freiheitsgrade (nnp*ndf x 1) in der Sortierung Knoten1_FG1, Knoten1_FG2, Knoten2_FG1, Knoten2_FG2, ... (festgehaltene Freiheitsgrade sind 1, freie 0)

    drlt_vals = torch.zeros(nnp * ndf, 1) 
    for i in range(drlt.size()[0]):
        drlt_mask[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = 1 # Form des Tensors: (nnp*ndf x 1) (ndf Freiheitsgrade pro Knoten) (freie Freiheitsgrade sind 0, festgehaltene 1)
        drlt_vals[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = drlt[i, 2]
    free_mask = torch.ones(nnp * ndf, 1) - drlt_mask # (freie Freiheitsgrade sind 1, festgehaltene 0) (Inverse von drlt_mask)
    drltDofs = torch.nonzero(drlt_mask) # gibt Tensor aus, der zeigt, welche Indizes nicht 0 sind
    # print("drltDofs", drltDofs)

    drlt_matrix = 1e22 * torch.diag(drlt_mask[:, 0], 0) # Große Zahl auf den Diagonalen der festgehaltenen Freiheitsgrade (Penalty-Methode Steifigkeitsmethode)

    neum_vals = torch.zeros(nnp * ndf, 1)
    for i in range(neum.size()[0]):
        neum_vals[int(neum[i, 0]) * ndf + int(neum[i, 1]), 0] = neum[i, 2]

    # Gauß-Punkte und Gewichte für 3x3-Quadratur
    qpt = torch.zeros(nqp, ndm)
    w8 = torch.zeros(nqp, 1)
    a = math.sqrt(3.0 / 5.0)
    w1 = 5.0 / 9.0
    w2 = 8.0 / 9.0
    
    # Definition der 9 Gauß-Punkte und ihrer Gewichte
    qpt[0,:] = torch.tensor([-a, -a])
    qpt[1,:] = torch.tensor([ 0, -a])
    qpt[2,:] = torch.tensor([ a, -a])
    qpt[3,:] = torch.tensor([-a,  0])
    qpt[4,:] = torch.tensor([ 0,  0])
    qpt[5,:] = torch.tensor([ a,  0])
    qpt[6,:] = torch.tensor([-a,  a])
    qpt[7,:] = torch.tensor([ 0,  a])
    qpt[8,:] = torch.tensor([ a,  a])

    w8[0] = w1 * w1
    w8[1] = w1 * w2
    w8[2] = w1 * w1
    w8[3] = w2 * w1
    w8[4] = w2 * w2
    w8[5] = w2 * w1
    w8[6] = w1 * w1
    w8[7] = w1 * w2
    w8[8] = w1 * w1

    # Masterelement - Shape function
    masterelem_N = torch.zeros(nqp, nen) # nqp x nen
    masterelem_gamma = torch.zeros(nqp, nen, ndm) # nqp x nen x ndm

    for q in range(nqp):
        xi = qpt[q, :]
        e, n = xi[0], xi[1]
        # Quadratische Ansatzfunktionen für Q8-Element
        # Eckknoten
        masterelem_N[q, 0] = 0.25 * (1-e)*(1-n)*(-e-n-1)
        masterelem_N[q, 1] = 0.25 * (1+e)*(1-n)*(e-n-1)
        masterelem_N[q, 2] = 0.25 * (1+e)*(1+n)*(e+n-1)
        masterelem_N[q, 3] = 0.25 * (1-e)*(1+n)*(-e+n-1)
        # Mittenknoten
        masterelem_N[q, 4] = 0.5 * (1-e*e)*(1-n)
        masterelem_N[q, 5] = 0.5 * (1+e)*(1-n*n)
        masterelem_N[q, 6] = 0.5 * (1-e*e)*(1+n)
        masterelem_N[q, 7] = 0.5 * (1-e)*(1-n*n)

        # Ableitungen der quadratischen Ansatzfunktionen
        # d/de
        masterelem_gamma[q, 0, 0] = 0.25 * (1-n)*(-1)*(-e-n-1) + 0.25*(1-e)*(1-n)*(-1)
        masterelem_gamma[q, 1, 0] = 0.25 * (1-n)*(1)*(e-n-1) + 0.25*(1+e)*(1-n)*(1)
        masterelem_gamma[q, 2, 0] = 0.25 * (1+n)*(1)*(e+n-1) + 0.25*(1+e)*(1+n)*(1)
        masterelem_gamma[q, 3, 0] = 0.25 * (1+n)*(-1)*(-e+n-1) + 0.25*(1-e)*(1+n)*(-1)
        masterelem_gamma[q, 4, 0] = 0.5 * (-2*e)*(1-n)
        masterelem_gamma[q, 5, 0] = 0.5 * (1)*(1-n*n)
        masterelem_gamma[q, 6, 0] = 0.5 * (-2*e)*(1+n)
        masterelem_gamma[q, 7, 0] = 0.5 * (-1)*(1-n*n)
        # d/dn
        masterelem_gamma[q, 0, 1] = 0.25 * (1-e)*(-1)*(-e-n-1) + 0.25*(1-e)*(1-n)*(-1)
        masterelem_gamma[q, 1, 1] = 0.25 * (1+e)*(-1)*(e-n-1) + 0.25*(1+e)*(1-n)*(-1)
        masterelem_gamma[q, 2, 1] = 0.25 * (1+e)*(1)*(e+n-1) + 0.25*(1+e)*(1+n)*(1)
        masterelem_gamma[q, 3, 1] = 0.25 * (1-e)*(1)*(-e+n-1) + 0.25*(1-e)*(1+n)*(1)
        masterelem_gamma[q, 4, 1] = 0.5 * (1-e*e)*(-1)
        masterelem_gamma[q, 5, 1] = 0.5 * (1+e)*(-2*n)
        masterelem_gamma[q, 6, 1] = 0.5 * (1-e*e)*(1)
        masterelem_gamma[q, 7, 1] = 0.5 * (1-e)*(-2*n)

    # Indizes zum Plotten der gekrümmten Elementkanten
    indices = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7, 0])
    
    # Plot des unverformten Netzes
    plt.subplot(3,3,1)
    plt.title("Unverformtes Netz")
    for e in range(nel): # nel = Gesamtanzahl der Elemente
        els = torch.index_select(elems[e,:], 0, indices)
        plt.plot(x[els, 0], x[els, 1], 'k-', linewidth=0.5)
    plt.plot(x[:, 0], x[:, 1], 'ko', markersize=2)
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    
    # Randbedingungen einzeichnen
    for drlt_bc in drlt:
        node_idx = int(drlt_bc[0])
        if drlt_bc[1] == 0: # x-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], 'g>', markersize=5)
        if drlt_bc[1] == 1: # y-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], 'g^', markersize=5)
    
    for neum_bc in neum:
        node_idx = int(neum_bc[0])
        if neum_bc[1] == 0: # x-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], 'r>', markersize=5)
        if neum_bc[1] == 1: # y-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], 'r^', markersize=5)


    ############## Analysis ###############
    u = torch.zeros(ndf * nnp, 1) # Anfangsverschiebungen Null Tensortype [ndf*nnp x 1] (2*nnp x 1) 
    # print("x", x)
    # node_idx globler Freiheitsgrad index
    edof = torch.zeros(nel, ndf, nen, dtype=int) # element dof
    gdof = torch.zeros(nel, ndf * nen, dtype=int) # globale Freiheitsgrade in der Form [nel x (ndf*nen)]
    for el in range(nel): # Schleife über alle Elemente nel = Gesamtanzahl der Elemente
        for ien in range(nen): # Schleife über Knoten des Elements = nen = 4 für quadratisches Element
            for idf in range(ndf): # Schleife über Freiheitsgrade ndf = 2 für 2D (x und y)
                edof[el, idf, ien] = ndf * elems[el, ien] + idf # global dof of element edof = [nel x ndf x nen] so 3 indizes
        gdof[el, :] = edof[el, :, :].t().reshape(ndf * nen) # reshape to 1D array [ndf*nen] for each element means 1 index

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

    ############################Element#####################################
    for el in range(nel): # Schleife über alle Elemente
        xe = torch.zeros(ndm, nen) # Knotenkoordinaten des Elements Tensor [ndm x nen] [2 x 4] für 2D
        for idm in range(ndm): # idm: 0,1 für 2D (Koordinaten x und y)
            xe[idm, :] = x[elems[el, :], idm] # Knotenkoordinaten des Elements el (idm = 0: x-Koordinaten, idm = 1: y-Koordinaten)

        ue = torch.squeeze(u[edof[el, 0:ndm, :]]) # Verschiebungen des Elements


        Ke.zero_() # Steifigkeitsmatrix des Elements auf Null setzen dynamischer Tensor
        Me.zero_() # Massenmatrix des Elements auf Null setzen für Iteration
        for q in range(nqp): # Schleife über Gauß-Punkte nqp = 4 für 2x2 Gauß-Quadratur
            N = masterelem_N[q] # Shape functions at Gauss point q
            gamma = masterelem_gamma[q] # Shape function derivatives at Gauss point q

            Je = xe.mm(gamma) # Jacobian matrix at Gauss point q
            detJe = torch.det(Je) # Determinante der Jacobian-Matrix at Gauss point q

            if detJe <= 0:
                print("Error: detJe <= 0")

            dv = detJe * w8[q] * width # Volumenelement at Gauss point q (Area in 2D)
            invJe = torch.inverse(Je) # Inverse der Jacobian-Matrix at Gauss point q for gradient transformation

            G = gamma.mm(invJe) # Gradient der Shape functions im globalen Koordinatensystem at Gauss point q wird in unsere globale Koordinaten transformiert (echte Welt)

            h[0:ndm, 0:ndm] = ue.mm(G) # Deformationsgradient im 2D
            eps_2d = 0.5 * (h + h.t())[0:ndm, 0:ndm] # Kleine Dehnungen im 2D (symmetrischer Teil des Deformationsgradienten)
            stre_2d = torch.tensordot(C4, eps_2d, dims=2) # Spannung im 2D (Hooke'sches Gesetz)
            
            for A in range(nen): # Schleife über Knoten des Elements
                G_A[0, :] = G[A, :] # Gradient der Shape function von Knoten A

                fintA = dv * (G_A.mm(stre_2d)) # Innere Kraftbeiträge von Knoten A
                finte[A * ndf: A * ndf + ndm] += fintA.t() # Sammle innere Kräfte für das Element

                for B in range(nen): # Schleife über Knoten des Elements
                    KAB = torch.tensordot(G[A, :], # linker Tensor in der SummeGradient der Shape function von Knoten A [die Summation über die Indizes erfogt schon durch das tensordot]
                                          (torch.tensordot(C4[0:tdm, 0:tdm, 0:tdm, 0:tdm], G[B, :], [[3], [0]])), # Kontrahiere Dimension 3 (0 indizierung) von C4 mit Dim 0 von G_B rechtes Tensorprodukt bestehend aus (A,B,dims) Gradient der Shape function von Knoten B
                                          [[0], [0]]) # Dimensions für das übergeordnete Tensorprodukt Dimension von KAB ist jetzt (ndm x ndm) (2x2 in 2D)
                    Ke[A * ndf:A * ndf + ndm, B * ndf:B * ndf + ndm] += dv * KAB # Steifigkeitsmatrix des Elements addieren über alle Gauß-Punkte (Integration)


                    MAB = rho * N[A] * N[B] * I[0:ndm, 0:ndm] # I(0:ndm, 0:ndm) ist Einheitsmatrix in 2D (2x2) analog Integrationsschema aus der Vorlesung (Innere Produkt der Shape functions mal Dichte mal Einheitsmatrix)
                    Me[A * ndf:A * ndf + ndm, B * ndf:B * ndf + ndm] += dv * MAB # Massenmatrix des Elements addieren über alle Gauß-Punkte (Integration) äußere Summe über alle Knotenpaare (A,B)
                    # die Summen werden durch die range Schleifen realisiert über nen und nqp realisiert

        fint[gdof[el, :]] += finte # Assemblierung der inneren Kräfte
        fvol[gdof[el, :]] += fvole # Assemblierung der Volumenkraft (hier null)
        for i in range(gdof.shape[1]): # Assemblierung der globalen Steifigkeitsmatrix
            K[gdof[el, i], gdof[el, :]] += Ke[i, :] # Assemblierung der globalen Steifigkeitsmatrix
            M[gdof[el, i], gdof[el, :]] += Me[i, :]

            
            

    ############################end of Element#####################################

    

    # --- Demonstration der Singularität von K ---
    # Theoretisch ist K singulär (det(K) = 0), da keine Randbedingungen angewendet wurden.
    # Aufgrund von Fließkomma-Ungenauigkeiten ist die berechnete Determinante eine sehr kleine Zahl, nicht exakt 0.
    # det_K = torch.det(K)
    # print(f"Determinante der rohen K-Matrix: {det_K:.2e}") # Wird eine sehr kleine Zahl sein
    # Der Versuch, K zu invertieren, führt zu einer Matrix mit riesigen Zahlen und ist numerisch instabil.
    # KInvv = torch.linalg.inv(K) # Würde riesige Zahlen produzieren, aber keinen Fehler werfen.

    # fsur entspricht den Neumann-Randbedingungen (externe Kräfte) also Nodes wo Kräfte aufgebracht werden
    # fint sind die internen Kräfte die kommen durch die Verformung des Körpers zustande

    rsd = free_mask.mul(fsur - fint) # Residuum nur für freie Freiheitsgrade

    K_tilde = K + drlt_matrix # Modifizierte Steifigkeitsmatrix mit Penalty für festgehaltene Freiheitsgrade

    # KInvv = torch.linalg.inv(K)
    # print("KInvv:", KInvv)    

    # KInv = torch.linalg.inv(K_tilde)
    # print("KInv:", KInv)
    # MInv = torch.linalg.inv(M)

    
    free_dofs = torch.nonzero(free_mask) # gibt Tensor aus, der zeigt, welche Indizes nicht 0 sind
    # print("free_dofs", free_dofs)
    free_dofs = free_dofs[:, 0] # Umwandlung in 1D Tensor
    # print("free_dofs", free_dofs)

    K_free = K[free_dofs, :][:, free_dofs] # Reduzierte Steifigkeitsmatrix für freie Freiheitsgrade wählt nur die Zeilen und Spalten der freien Freiheitsgrade aus
    # print("K_free shape:", K_free.shape)

    M_free = M[free_dofs, :][:, free_dofs] # Reduzierte Massenmatrix für freie Freiheitsgrade wählt nur die Zeilen und Spalten der freien Freiheitsgrade aus
    # print("M_free shape:", M_free.shape)
    # print("K_free:", K_free)
    # print("M_free:", M_free)

    # Grenzwert für "Null"-Frequenzen (in Hz). Alles darunter wird als 0 ignoriert.
    f_threshold = 0.001 
    anzahl_moden = 10  # Wie viele Moden sollen angezeigt werden?

    print(f"{'='*80}")
    print(f"{'VERGLEICH DER EIGENFREQUENZEN (Hz)':^80}")
    print(f"{'='*80}")

    # ---------------------------------------------------------
    # 1. Ansatz: Invertierte Penalty Steifigkeitsmatrix
    # ---------------------------------------------------------
    KInvvv = torch.linalg.inv(K_tilde) 
    AKK = KInvvv.mm(M)
    LKK, VKK = torch.linalg.eig(AKK) # Eigenwertproblem K^-1 * M

    # Berechnung Frequenzen
    lkk = 1 / torch.sqrt(LKK.real) # Nur Realteil nutzen
    lkk = torch.sort(lkk).values   # Sortieren
    freq_penalty = lkk / (2.0 * math.pi)

    # FILTER: Nur Frequenzen größer als 0 (bzw. Threshold) nehmen
    freq_penalty = freq_penalty[freq_penalty > f_threshold]


    # ---------------------------------------------------------
    # 2. Ansatz: Invertierte Steifigkeitsmatrix (Reduced)
    # ---------------------------------------------------------
    KInv = torch.linalg.inv(K_free) 
    AK = KInv.mm(M_free)
    LK, VK = torch.linalg.eig(AK) # Eigenwertproblem K^-1 * M

    # Berechnung Frequenzen
    lk = 1 / torch.sqrt(LK.real)
    lk = torch.sort(lk).values
    freq_k_inv = lk / (2.0 * math.pi)

    # FILTER: Nur Frequenzen größer als 0 nehmen
    freq_k_inv = freq_k_inv[freq_k_inv > f_threshold]


    # ---------------------------------------------------------
    # 3. Ansatz: Invertierte Massenmatrix
    # ---------------------------------------------------------
    MInv = torch.linalg.inv(M_free) 
    AM = MInv.mm(K_free) # M^-1 * K
    LM, VM = torch.linalg.eig(AM)

    # Berechnung Frequenzen (Hier: sqrt(lambda), da M^-1*K)
    lm = torch.sqrt(LM.real)
    lm = torch.sort(lm).values
    freq_m_inv = lm / (2.0 * math.pi)

    # FILTER: Nur Frequenzen größer als 0 nehmen
    freq_m_inv = freq_m_inv[freq_m_inv > f_threshold]


    # ---------------------------------------------------------
    # 4. Ansatz: Invertierte Lumped Mass Matrix
    # ---------------------------------------------------------
    sumM = torch.sum(M_free, dim=0)
    diagM = torch.diag(sumM)
    diagMInv = torch.linalg.inv(diagM)
    diagMInvK = diagMInv.mm(K_free) # M_lumped^-1 * K

    LdiagMInvK, VdiagMInvK = torch.linalg.eig(diagMInvK)

    # Berechnung Frequenzen
    ldiagMInvK = torch.sqrt(LdiagMInvK.real)
    ldiagMInvK = torch.sort(ldiagMInvK).values
    freq_lumped = ldiagMInvK / (2.0 * math.pi)

    # FILTER: Hier ist es besonders wichtig -> Null-Moden abschneiden
    freq_lumped = freq_lumped[freq_lumped > f_threshold]


    # ---------------------------------------------------------
    # Ausgabe und Vergleich
    # ---------------------------------------------------------

    # Kopfzeile formatieren
    header = f"{'Mode':<6} | {'Penalty (Inv K)':<18} | {'K-Inv (Red)':<18} | {'M-Inv (Red)':<18} | {'Lumped Mass':<18}"
    print(header)
    print("-" * len(header))

    # Schleife über die ersten N Moden
    for i in range(anzahl_moden):
        # Werte abrufen (mit Check, falls weniger Moden existieren als n)
        val1 = freq_penalty[i].item() if i < len(freq_penalty) else float('nan')
        val2 = freq_k_inv[i].item() if i < len(freq_k_inv) else float('nan')
        val3 = freq_m_inv[i].item() if i < len(freq_m_inv) else float('nan')
        val4 = freq_lumped[i].item() if i < len(freq_lumped) else float('nan')

        # Zeilen formatieren
        print(f"{i+1:<6} | {val1:18.4f} | {val2:18.4f} | {val3:18.4f} | {val4:18.4f}")

    print("-" * len(header))

    du = torch.linalg.solve(K_tilde, rsd) # Löse das Gleichungssystem für die Verschiebungsänderung du
    # rsd = K * du - (fint + fvol - fsur)
    u += du # Aktualisiere die Verschiebungen
    u = free_mask.mul(u) + drlt_mask.mul(drlt_vals) # Setze festgehaltene Freiheitsgrade auf vorgegebene Werte hier(0)
    # print("u: ", u)

    fext = K * u # Externe Kräfte aus der aktuellen Verschiebung berechnen (Überprüfung)
    frea = fext - fvol - fsur # Reaktionskräfte berechnen (Überprüfung)

    ###### Post-processing/ plots ########
    u_reshaped = torch.reshape(u, (-1, 2))
    # u.reshape in (-1,2) bedeutet, dass die Anzahl der Zeilen automatisch berechnet wird basierend auf der Gesamtanzahl der Elemente und 2 Spalten (für x und y Verschiebungen)
    # also hier wird u zu einem Tenor der Form [nnp x 2] umgeformt, wobei jede Zeile die Verschiebungen in x und y für jeden Knoten enthält
    # einfach gesagt (x1 y1), (x2 y2), ... durch
    torch.reshape

    x_disped = x + disp_scaling * u_reshaped

    print(x_disped.size())

    voigt = torch.tensor([[0,0], [1,1], [2,2], [0, 1], [0,2], [1,2]])
    # Plot-Titel ändern, um Verschiebungen statt Schubspannungen anzuzeigen
    plot_titles = ["Spannung XX", 
                   "Spannung YY", 
                   "Von-Mises-Spannung", 
                   "Verschiebung u_x", 
                   "Verschiebung u_y"]
    ei = torch.eye(3,3)
    # Die Größe von plotdata ist nicht mehr kritisch, da wir Knotenergebnisse plotten
    plotdata = torch.zeros(len(plot_titles), nel*nqp, 3)

    # Initialisierung für Knoten-Spannungs-Extrapolation
    # Tensor zum Speichern der summierten Spannungswerte an jedem Knoten
    nodal_stresses_sum = torch.zeros(nnp, len(plot_titles))
    # Tensor zum Zählen, wie viele Elemente zu jedem Knoten beitragen (für die Mittelung)
    nodal_contribution_count = torch.zeros(nnp, len(plot_titles))
    
    for i in range(len(plot_titles)):
        ax = plt.subplot(3,3,i+2) # Subplots 2, 3, 4, 5, 6, 7
        for e in range(nel):
            els = torch.index_select(elems[e,:], 0, indices)
            ax.plot( x_disped[els, 0], x_disped[els, 1], 'b-', linewidth=0.5 )

            xe = torch.zeros(ndm, nen)
            for idm in range(ndm):
                xe[idm, :] = x[elems[e, :], idm]
            ue = torch.squeeze(u[edof[e, 0:ndm, :]])

            # Temporärer Speicher für Spannungen an Gauß-Punkten eines Elements
            stresses_at_gauss_points = torch.zeros(nqp)

            for q in range(nqp):
                xgauss = torch.mv ( torch.transpose(x_disped[elems[e, :]], 0, 1),  masterelem_N[q] )

                gamma = masterelem_gamma[q]
                Je = xe.mm(gamma)
                invJe = torch.inverse(Je)
                G = gamma.mm(invJe)
            
                h = torch.zeros(3,3)
                h_2d = ue.mm(G)
                eps_2d = 0.5 * (h_2d + h_2d.t())
                
                stre_2d = torch.tensordot(C4, eps_2d, dims=2)

                stre = torch.zeros(3,3)
                stre[0:2, 0:2] = stre_2d
                eps = torch.zeros(3,3)
                eps[0:2, 0:2] = eps_2d
                eps[2,2] = -nu / (1-nu) * (eps_2d[0,0] + eps_2d[1,1])

                s11, s22, s12 = stre[0,0], stre[1,1], stre[0,1]
                s33 = 0.0 # Annahme für ebenen Spannungszustand
                sigma_v = torch.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*(s12**2)))

                # Logik zur Auswahl der Plot-Werte
                title = plot_titles[i]
                if title == "Spannung XX":
                    stre_val = stre[0,0]
                elif title == "Spannung YY":
                    stre_val = stre[1,1]
                elif title == "Von-Mises-Spannung":
                    stre_val = sigma_v
                elif title == "Dehnung XX":
                    stre_val = eps[0,0]
                else:
                    # Für Verschiebungen ist keine Gauß-Punkt-Berechnung nötig
                    stre_val = 0 
                
                if "Spannung" in title or "Dehnung" in title:
                    stresses_at_gauss_points[q] = stre_val

            # Extrapolation von Gauß-Punkten zu den Knoten des Elements
            N_at_gps = masterelem_N # (9x8)
            try:
                # Least-Squares Extrapolation
                extrapolation_matrix = torch.linalg.inv(N_at_gps.T @ N_at_gps) @ N_at_gps.T
            except torch.linalg.LinAlgError:
                # Fallback auf Pseudo-Inverse, falls Matrix singulär ist
                extrapolation_matrix = torch.linalg.pinv(N_at_gps)

            # Nur für Spannungs- und Dehnungswerte extrapolieren
            if "Spannung" in plot_titles[i] or "Dehnung" in plot_titles[i]:
                nodal_stresses_element = extrapolation_matrix.mv(stresses_at_gauss_points)
                # Addiere die extrapolierten Werte zu den globalen Summen-Tensoren
                element_nodes = elems[e, :]
                nodal_stresses_sum[element_nodes, i] += nodal_stresses_element
                nodal_contribution_count[element_nodes, i] += 1

        # Mittelung der Knotenspannungen nach der Schleife über alle Elemente
        title = plot_titles[i]
        if "Spannung" in title or "Dehnung" in title:
            # Vermeide Division durch Null für Knoten, die nicht verwendet werden

            # hier ist clone() wichtig, um eine Kopie zu erstellen anstatt einen Pointer... damit würde bei späterem Verändern von plot_values_nodal auch u verändert werden
            valid_counts = nodal_contribution_count[:, i].clone()
            valid_counts[valid_counts == 0] = 1
            averaged_nodal_stresses = nodal_stresses_sum[:, i] / valid_counts
            plot_values_nodal = averaged_nodal_stresses.clone()
        elif title == "Verschiebung u_x":
            plot_values_nodal = u_reshaped[:, 0].clone() # Direkter Zugriff auf Knotenergebnis
        elif title == "Verschiebung u_y":
            plot_values_nodal = u_reshaped[:, 1].clone() # Direkter Zugriff auf Knotenergebnis

        # Verwende die gemittelten Knotenspannungen für den Plot 
        if "Spannung" in title:
            plot_values_nodal /= 1e6 # Umrechnung in MPa
            unit_str = " (MPa)"
        elif "Verschiebung" in title:
            plot_values_nodal *= 1000 # Umrechnung in mm
            unit_str = " (mm)"
        else:
            unit_str = "" # Dehnung ist einheitenlos

        min_val_nodal = torch.min(plot_values_nodal)
        max_val_nodal = torch.max(plot_values_nodal)

        ax.set_title(f"{plot_titles[i]}{unit_str}\nMax: {max_val_nodal:.2e}, Min: {min_val_nodal:.2e}")

        # tricontourf für eine glatte Darstellung der Knotenergebnisse
        scatter = ax.tricontourf(x[:, 0], x[:, 1], plot_values_nodal, levels=15, cmap=mpl.cm.jet)
        
        plt.colorbar(scatter, ax=ax)
        
        ax.axis('equal')

    # Plot - Spannungsverlauf bei L/2 
    ax_cross_section = plt.subplot(3,3,7) # Plot an Position 8
    
    # Finde Knoten in der Mitte des Balkens (x = L/2)
    mid_point_x = length / 2.0
    # Finde den x-Wert im Gitter, der der Mitte am nächsten kommt
    closest_x_value = x_coords[torch.argmin(torch.abs(x_coords - mid_point_x))]
    mid_nodes_indices = torch.where(x[:, 0] == closest_x_value)[0]
    
    # Extrahiere y-Koordinaten und Spannung XX für diese Knoten
    y_vals_mid = x[mid_nodes_indices, 1]
    # Spannung XX ist der erste Plot (Index 0)
    stress_xx_counts = nodal_contribution_count[:, 0].clone()
    stress_xx_counts[stress_xx_counts == 0] = 1
    stress_xx_nodal = (nodal_stresses_sum[:, 0] / stress_xx_counts)

    stress_xx_mid = stress_xx_nodal[mid_nodes_indices] / 1e6 # in MPa

    # Sortiere die Werte nach y-Koordinate für einen sauberen Plot
    y_sorted, indices_sorted = torch.sort(y_vals_mid)
    stress_sorted = stress_xx_mid[indices_sorted]

    # Analytische Lösung bei L/2
    I_z = (width * height**3) / 12
    M_z = (1000.0) * (length - mid_point_x) # F * (L-x)
    y_analytical = torch.linspace(0, height, 100)
    sigma_analytical = (M_z / I_z * (y_analytical - height/2)) / 1e6 # in MPa

    ax_cross_section.plot(stress_sorted.numpy(), y_sorted.numpy(), 'bo-', label='FEM Ergebnis')
    ax_cross_section.plot(sigma_analytical.numpy(), y_analytical.numpy(), 'r--', label='Analytische Lösung')
    ax_cross_section.set_title(f"Biegespannung bei x = {mid_point_x:.1f} m")
    ax_cross_section.set_xlabel("Spannung XX (MPa)")
    ax_cross_section.set_ylabel("y-Koordinate (m)")
    ax_cross_section.grid(True)
    ax_cross_section.legend()


# #####NEWMARK-TIME INTEGRATION SCHEME STARTS HERE#####

    ax = plt.subplot(3,3,8) # Subplots im 3x3 Raster an Position 9
    ax_u= plt.subplot(3,3,9)
    
    # Parameter definition for Newmark time integration
    time_intervall = 1 # total time of simulation
    steps = 450
    # print("K_tilde size:", K_tilde.size())
    beta = 0.25
    gamma = 0.5
    dt = time_intervall / steps
    print(f"Aktuelle Schrittweite: {dt}")
    print(f"Maximaler EW: {torch.max(lk)}")
    dt_crit = 2 / torch.max(lk)
    print(f"Kritische Schrittweite: {dt_crit}")
    # Save original external forces (Neumann) and remove them for t = 0
    fsur_static = fsur.clone()  # original load vector
    fsur = torch.zeros_like(fsur) # ensure no external force at t=0 (initially unloaded)
    # dt = time step size. Start from an UNLOADED, UNDEFORMED initial configuration (u_0 = 0)
    # The static solution computed earlier is kept for modal analysis/plot scaling but
    # the dynamic run starts from the undeformed state.
    u_0 = torch.zeros_like(u)  # initial displacements zero (undisturbed at t=0)
    v_0 = torch.zeros_like(u_0) # initial velocity zero
    a_0 = torch.linalg.inv(M).mm((fsur - K.mm(u_0)))
    x_0 = x + torch.reshape(u_0, (-1, 2)) * disp_scaling # initial coordinates (no displacement)
    # fsur are the external forces applied at the nodes (neum_vals) (f_ext in the equation)
    if beta >= 0.1:

        K_eff = K_tilde + 1/(beta*dt*dt) * M
        # x_disped are the displaced coordinates after static analysis in form of tensor [nnp x ndm]
        # so x1, y1, x2, y2, ..., x_nnp, y_nnp so onedimensional array of size (nnp*ndm x 1)

        # Newmark time integration loop
        # Listen initialisieren und leeres Linien-Objekt erstellen
        time_history = [] 
        disp_history = []

        # Add initial sample at t = 0 so u_0 is visible in the plot
        time_history.append(0.0)
        disp_history.append(u_0[-1, 0] * disp_scaling)

        # Wir plotten eine leere blaue Linie ('b-') in ax_u. 
        # Das Komma nach 'line_u' ist wichtig, da plot() eine Liste zurückgibt!
        line_u, = ax_u.plot([], [], 'bo-', linewidth=1, markersize=4, label="u_x Verlauf")
        line_u.set_data(time_history, disp_history)

        for s in range(steps):
            
            ax.cla() 
            ax.set_xlim([ -0.1*length, 1.1*length ])
            ax.set_ylim([ -5*height, 5*height ])
            ax.set_title(f"Time Integration Step {s+1}/{steps} - Zeitschritt {dt} s")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True)

            # Berechnung aktuelle Werte
            time_current = (s+1) * dt
            u_last_node = u_0[-1, 0] * disp_scaling # u_x

            # Werte an die Historie anhängen
            time_history.append(time_current)
            disp_history.append(u_last_node)

            # Plotting für ax_u (Zeitverlauf)
        
            ax_u.set_xlim([0, time_intervall + dt])
           
            ax_u.set_ylim([ -20*disp_scaling*torch.max(u_reshaped), 20*disp_scaling*torch.max(u_reshaped) ])
            
            ax_u.set_title("Verschiebung u_y am letzten Knoten über die Zeit")
            ax_u.set_xlabel("Zeit [s]")
            ax_u.set_ylabel("Verschiebung u_y [m]")
            ax_u.grid(True)

            # existierende Linien-Objekt mit neuen Daten
            line_u.set_data(time_history, disp_history)

            for e in range(nel):
                els = torch.index_select(elems[e,:], 0, indices) # Beispielhaft für ein Element [0,4,1,5,2,6,3,7,0]
                ax.plot( x_0[els, 0], x_0[els, 1], 'b-', linewidth=0.5 ) # plottet die Kanten des verformten Elements iterativ für alle Elemente
            plt.pause(0.05) 
            # x_0 = x_0 + (disp_scaling * u_reshaped) * 0.01 # Dummy update der verformten Koordinaten für die Animation
            # Apply sudden load for t > 0: use the stored Neumann vector fsur_static
            fsur_time = fsur_static  # at first time step time_current=dt>0 so load is applied suddenly
            F_1 = fsur_time + M.mm( (1/(beta*dt*dt)*u_0) + (1/(beta*dt)*v_0) + ((1-2*beta)/(2*beta))*a_0)
            u_1 = torch.linalg.solve(K_eff, F_1)
            a_1 = (1/(beta*dt*dt))*(u_1 - u_0) - (1/(beta*dt))*v_0 - ((1-2*beta)/(2*beta))*a_0
            v_1 = v_0 + dt*((1-gamma)*a_0 + gamma*a_1)
            u_0reshaped = torch.reshape(u_1, (-1, 2)) * disp_scaling
            x_0 = x + u_0reshaped
            u_0 = u_1
            v_0 = v_1
            a_0 = a_1

    else:
        # EXPLIZITES CENTRAL-DIFFERENCE-SCHEME (Beta gegen Null)
        # 0) Reduzierte Matrizen für freie DOFs
        # M_free und K_free sind oben bereits definiert
        M_inv_free = torch.linalg.inv(M_free)

        # 1) Anfangswerte (global)
        u_n = u_0.clone()
        v_n = v_0.clone()
        a_n = a_0.clone()

        # 2) Anfangsschritt: u^{-1} konstruieren
        #    (CDS benötigt u_{n-1}, v_{n}, a_{n}), u^{n-1} = u^n - dt * v_n + 0.5 dt^2 * a_n
        u_prev = u_n - dt * v_n + 0.5 * dt * dt * a_n #Beschleunigung geht halb ein

        
        #nur freie DOFs
        
        u_n_free    = u_n[free_dofs].clone()
        u_prev_free = u_prev[free_dofs].clone()
        # use stored sudden load for t>0
        fsur_free   = fsur_static[free_dofs].clone()

        # Plot-Vorbereitung
        time_history = []
        disp_history = []
        # Add initial sample at t = 0 so u_0 is visible in the plot
        time_history.append(0.0)
        disp_history.append(u_n[-1, 0] * disp_scaling)
        line_u, = ax_u.plot([], [], 'bo-', linewidth=1, markersize=4)
        line_u.set_data(time_history, disp_history)

        # 3) CDS-Zeitschleife
        for s in range(steps):

            # EXPLIZITER SCHRITT:
            #
            # u_{n+1} = dt^2 * M_free^{-1} * ( f_free - K_free * u_n_free )   + 2*u_n_free - u_prev_free
            fint_n_free = K_free.mm(u_n_free)
            rhs_free = fsur_free - fint_n_free

            u_next_free = (dt*dt) * (M_inv_free.mm(rhs_free)) \
                          + 2.0 * u_n_free - u_prev_free

            # CDS typische Approximationen:
            v_free = (u_next_free - u_prev_free) / (2*dt)
            a_free = (u_next_free - 2*u_n_free + u_prev_free) / (dt*dt)

            # Reintegration in globalen Vector u
            u_global = u_n.clone()
            u_global[free_dofs] = u_next_free
            u_global = free_mask.mul(u_global) + drlt_mask.mul(drlt_vals)

            u_0reshaped = torch.reshape(u_global, (-1, 2))
            x_0 = x + disp_scaling * u_0reshaped

            ax.cla()
            ax.set_xlim([-0.1*length, 1.1*length])
            ax.set_ylim([-5*height, 5*height])
            ax.set_title(f"CDS Step {s+1}/{steps}, dt={dt}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True)

            for e in range(nel):
                els = torch.index_select(elems[e,:], 0, indices)
                ax.plot(x_0[els, 0], x_0[els, 1], 'b-', linewidth=0.5)

            # letzte Verschiebung
            tcur = (s+1)*dt
            time_history.append(tcur)
            disp_history.append(u_global[-1,0] * disp_scaling)

            ax_u.set_xlim([0, time_intervall])
            ax_u.set_ylim([-20*disp_scaling*torch.max(u_reshaped),
                           20*disp_scaling*torch.max(u_reshaped)])
            ax_u.set_title("Verschiebung u_y am freien Ende")
            ax_u.set_xlabel("Zeit [s]")
            ax_u.set_ylabel("u_y")
            ax_u.grid(True)

            line_u.set_data(time_history, disp_history)

            plt.pause(0.05)

            # Überschreibung für nächsten Schritt
            u_prev_free = u_n_free.clone()
            u_n_free    = u_next_free.clone()




# #####END OF NEWMARK-TIME INTEGRATION SCHEME#####





    # Dic mit ERgebnisse
    results = {
        "u": u,
        "K": K,
        "M": M,
        "x": x,
        "elems": elems,
        "u_reshaped": u_reshaped,
        "x_disped": x_disped,
        "eigenvalues_K_inv_M": lk.values,
        "eigenvalues_M_inv_K": lm.values,
        "eigenvalues_M_lumped_inv_K": ldiagMInvK.values,
        "nodal_stresses_sum": nodal_stresses_sum,
        "nodal_contribution_count": nodal_contribution_count
    }
    return results


if __name__ == '__main__':

    start_perfcount = timemodule.perf_counter()
    results = analysis()

    u_max = results["u"].abs().max()
    # print(f"Maximale Verschiebung im Tensor (Meter): {u_max:.6f} m")


    end_perfcount = timemodule.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end_perfcount - start_perfcount)))

    plt.show()