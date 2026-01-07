
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys

# Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Laufe auf: {device}")

#===============================
#=========== INPUT =============
#===============================

AUFGABE = 4

# Standard-Parameter
E_modul = 210e9
Area = 0.002
Sy = 960e6     
H = 2e9        
eta = 0.0      

# Flags
dynamic = False
visco = False
plastic = False
hardening = False
plot_live = True

# Solver Parameter
tol = 1e-6
maxiter = 20

# Aufgaben Konfiguration
if AUFGABE == 4:
    # Hooke, Iterativ Area anpassen
    pass # Wird unten in Main Loop gemacht

elif AUFGABE == 5:
    plastic = True
    Area = 0.001308 # Ergebnis aus Task 4 (13.08 cm2)

elif AUFGABE == 6:
    plastic = True
    hardening = True
    Area = 0.001308

elif AUFGABE == 7:
    visco = True
    E_modul = 600e6
    Area = 0.001308 * (210e9 / 600e6) # Äquivalente Steifigkeit (EA) zu Stahl
    eta = 1e8

elif AUFGABE == 8:
    dynamic = True
    plastic = True
    hardening = True
    Area = 0.001308

elif AUFGABE == 9:
    # Vergleich
    plastic = True
    hardening = True
    Area = 0.001308


#===============================
#========== GEOMETRIE ==========
#===============================

ndm = 2
ndf = 2
nen = 2

# Knoten
x_nodes = torch.tensor([
    [0, 0], [1, 0], [2, 0], [4, 0], [6, 0], [8, 0], [9, 0],
    [1, 0.4], [3, 0.4], [5, 0.4], [7, 0.4], [8, 0.4],
    [9, -1], [6, -2], [12, -2], [6, -4.5], [12, -4.5]
], device=device, dtype=torch.double)

# Elemente
conn = torch.tensor([
    [0, 7], [1, 7], [1, 2], [2, 7], [2, 8], [2, 3], [3, 8], [3, 9], [3, 4], [4, 9],
    [4, 10], [4, 5], [5, 10], [5, 11], [5, 6], [6, 11], [7, 8], [8, 9], [9, 10],
    [10, 11], [6, 12], [12, 13], [12, 14], [13, 14], [13, 15], [14, 16]
], device=device, dtype=torch.long)

nnp = x_nodes.shape[0]
nel = conn.shape[0]

# Randbedingungen
# Dirichlet
drlt = torch.tensor([
    [0, 0, 0.], [0, 1, 0.],
    [1, 0, 0.], [1, 1, 0.]
], device=device, dtype=torch.double)

drltDofs = []
for i in range(drlt.shape[0]):
    node = int(drlt[i, 0])
    dof = int(drlt[i, 1])
    drltDofs.append(node*ndf + dof)
drltDofs = torch.tensor(drltDofs, device=device, dtype=torch.long)

# Neumann
neum = torch.tensor([
  [15, 1, -3090*9.81],
  [16, 1, -3090*9.81]
], device=device, dtype=torch.double)

# Lastverläufe
if AUFGABE == 8:
    # Zeitschrittverfahren - Feine Schritte für Konvergenz nötig
    times = torch.linspace(0, 2, 1001, dtype=torch.double) # dt = 0.002
    load_factors = torch.ones_like(times)
    load_factors[0] = 0.0
elif AUFGABE == 9:
    # Zyklus: Belasten -> Halten -> Entlasten
    times = torch.linspace(0, 5, 1001, dtype=torch.double)
    load_factors = torch.zeros_like(times)
    for i, t in enumerate(times):
        if t <= 2.0:
            load_factors[i] = t / 2.0
        elif t <= 3.0:
            load_factors[i] = 1.0
        elif t <= 5.0:
            load_factors[i] = max(0.0, 1.0 - (t - 3.0) / 2.0)
else:
    # Ramp
    times = torch.linspace(0, 5, 21, dtype=torch.double)
    load_factors = torch.linspace(0, 1, 21, dtype=torch.double)

dt = times[1] - times[0]

#===============================
#=========== HELFER ============
#===============================

# Formfunktionen (Dummy für Template-Kompatibilität)
def gauss(nqp, ndm):
    return torch.tensor([0], device=device), torch.tensor([2], device=device)

def shape(xi, nen, ndm):
    return torch.zeros(nen, 1, device=device), torch.zeros(nen, 1, device=device)

#===============================
#=========== SOLVER ============
#===============================

def run_fem(curr_area, curr_E, enable_dyn=False, enable_plastic=False, enable_visco=False, enable_hardening=False):
    
    # Init Variablen
    u = torch.zeros(nnp*ndf, 1, device=device, dtype=torch.double)
    v = torch.zeros_like(u)
    a = torch.zeros_like(u)
    
    # History initialisieren
    history = []
    for i in range(nel):
        history.append({'eps_p': 0.0, 'alpha': 0.0, 'eps_prev': 0.0})

    # Massenmatrix (Lumped)
    M = torch.zeros(nnp*ndf, nnp*ndf, device=device, dtype=torch.double)
    if enable_dyn:
        rho_steel = 7850.0
        for e in range(nel):
            idx = conn[e]
            x_e = x_nodes[idx]
            L_e = torch.norm(x_e[1]-x_e[0])
            m_e = rho_steel * curr_area * L_e
            for n in idx:
                M[n*2, n*2] += m_e/2
                M[n*2+1, n*2+1] += m_e/2
        
        # Rayleigh parameters
        alpha_ray = 0.5
        beta_ray = 0.001
        
        # Newmark consts
        beta_nm = 0.25
        gamma_nm = 0.5

    # Plot Vorbereitung
    if plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_aspect('equal')
        plot_lines = []
        for i in range(nel):
            ln, = ax.plot([], [], 'k-', linewidth=2)
            plot_lines.append(ln)
        sc_nodes = ax.scatter(x_nodes[:,0].cpu(), x_nodes[:,1].cpu(), c='k', s=10)
        
        # Normierung Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1)) # dummy
        cbar = fig.colorbar(sm, ax=ax, label='Stress/Strain Level')

    max_displacement = 0.0
    max_stress_global = 0.0
    results_t = []
    results_u = []
    results_stress_hist = []
    results_strain_hist = []

    # Zeitschleife
    for t_idx, t in enumerate(times):
        lf = load_factors[t_idx]
        
        # Externe Lasten
        f_ext = torch.zeros_like(u)
        for val in neum:
             node, dof, load = int(val[0]), int(val[1]), val[2]
             f_ext[node*ndf + dof] = load * lf
             
        # Dynamik Prädiktor
        if enable_dyn:
            u_pred = u + dt*v + (0.5 - beta_nm)*dt**2 * a
            v_pred = v + (1 - gamma_nm)*dt*a
            a_pred = torch.zeros_like(a)
            
            u = u_pred.clone()
            v = v_pred.clone()
            a = a_pred.clone()
            
            c1 = 1.0 / (beta_nm * dt**2)
            c2 = gamma_nm / (beta_nm * dt)

        # Newton Schleife
        hist_trial = copy.deepcopy(history)
        
        for iter in range(maxiter):
            K_glob = torch.zeros(nnp*ndf, nnp*ndf, device=device, dtype=torch.double)
            f_int = torch.zeros_like(u)
            
            # Speicher für Stress zum Plotten
            elem_stresses = []
            
            for e in range(nel):
                # Nodes
                idx = conn[e]
                # Coords
                x0 = x_nodes[idx].T # 2x2
                
                # Displacements holen
                ue = torch.zeros(ndm, nen, device=device, dtype=torch.double)
                for k in range(nen):
                    glob_node = idx[k]
                    ue[0, k] = u[glob_node*ndf+0]
                    ue[1, k] = u[glob_node*ndf+1]
                
                xc = x0 + ue
                
                # Truss Lengths
                dx0 = x0[:,1] - x0[:,0]
                dxc = xc[:,1] - xc[:,0]
                L0 = torch.norm(dx0)
                L = torch.norm(dxc)
                
                # Strain
                eps = (L - L0) / L0
                
                # Materialgesetz
                # ---------------------------------------------
                sigma = 0.0
                Et = curr_E
                # We need to capture epsilon for history output
                # eps is already calculated above: eps = (L - L0) / L0
                
                if enable_plastic:
                    eps_prev = hist_trial[e]['eps_prev']
                    deps = eps - eps_prev
                    eps_p = hist_trial[e]['eps_p']
                    alpha = hist_trial[e]['alpha']
                    
                    eps_trial = eps # total strain approach simple
                    sig_trial = curr_E * (eps_trial - eps_p)
                    
                    yield_stress = Sy
                    if enable_hardening:
                        yield_stress += H * alpha
                        
                    phi = torch.abs(sig_trial) - yield_stress
                    
                    if phi <= 0:
                        sigma = sig_trial
                        Et = curr_E
                    else:
                        denom = curr_E
                        if enable_hardening: denom += H
                        
                        d_gamma = phi / denom
                        sign_sig = torch.sign(sig_trial)
                        
                        sigma = sig_trial - d_gamma * curr_E * sign_sig
                        
                        # Update trial variables
                        hist_trial[e]['eps_p'] += d_gamma * sign_sig.item()
                        hist_trial[e]['alpha'] += d_gamma.item()
                        
                        if enable_hardening:
                            Et = (curr_E * H) / (curr_E + H)
                        else:
                            Et = 1e-5 * curr_E # Perfect plastic
                
                elif enable_visco:
                     if dt > 0:
                         eps_prev = hist_trial[e]['eps_prev'] # this is actually eps_old from last converged step
                         # Wait, hist_trial is deepcopied from history (converged).
                         # eps_prev in history should be from t-1.
                         # correct.
                         rate = (eps - hist_trial[e]['eps_prev']) / dt 
                         sigma = curr_E * eps + eta * rate
                         Et = curr_E + eta/dt
                     else:
                         sigma = curr_E * eps
                         Et = curr_E
                         
                else:
                    # Hooke
                    sigma = curr_E * eps
                    Et = curr_E
                
                # STORE CURRENT EPS for next step (if converged)
                hist_trial[e]['eps_current_step'] = eps.item()
                
                elem_stresses.append(sigma.item())
                # Add strain tracking
                if e == 0: # Init list if needed, or just append to local list
                     pass
                # Better: collect strains for this step
                
            # --- End Element Loop (but inside Newton) ---    
            
            # We need to collect strains for ALL elements to save them later
            # But inside Newton loop we don't save history yet.
            # We'll save it after convergence below.

                
                # ---------------------------------------------

                # Kräfte & Steifigkeit
                n = dxc / L # direction
                force = sigma * curr_area
                
                f_local = torch.cat((-n*force, n*force))
                
                # Scatter indices
                gdof = []
                for n_idx in idx:
                    gdof.extend([n_idx*ndf, n_idx*ndf+1])
                
                # F_int assemblieren
                # Manuelle Zuweisung für f_int (da 2D tensor)
                for i_loc, i_glob in enumerate(gdof):
                    f_int[i_glob, 0] += f_local[i_loc]
                    
                # Steifigkeitsmatrix K
                n_vec = n.unsqueeze(1)
                P = torch.matmul(n_vec, n_vec.t())
                I = torch.eye(2, device=device, dtype=torch.double)
                
                Km = (Et * curr_area / L0) * P
                Kg = (force / L) * (I - P)
                Kt = Km + Kg
                
                # Ke construct (4x4)
                Ke = torch.zeros(4,4, device=device, dtype=torch.double)
                Ke[0:2, 0:2] = Kt
                Ke[0:2, 2:4] = -Kt
                Ke[2:4, 0:2] = -Kt
                Ke[2:4, 2:4] = Kt
                
                for r in range(4):
                    for c in range(4):
                        K_glob[gdof[r], gdof[c]] += Ke[r,c]

            # --- End Element Loop ---

            # Dynamisches Residuum
            if enable_dyn:
                if AUFGABE == 8: # Lagerdämpfung
                    c_bear = 1e4
                    for i in [0, 1]:
                        for d in [0, 1]:
                            dof = i*2+d
                            # Federsteifigkeit muss vor Rayleigh-Berechnung in K_glob sein
                            k_bear = 1e9
                            K_glob[dof, dof] += k_bear
                            f_int[dof, 0] += k_bear * u[dof, 0]
                            
                # Dämpfung (Rayleigh + Lager)
                # Rayleigh auf voller Steifigkeitsmatrix berechnen
                C_damp = alpha_ray * M + beta_ray * K_glob
                
                if AUFGABE == 8:
                    # Add explicit damper
                     for i in [0, 1]:
                        for d in [0, 1]:
                            dof = i*2+d
                            C_damp[dof, dof] += c_bear
                            
                a = c1 * (u - u_pred)
                v = v_pred + gamma_nm * dt * a
                
                f_inert = torch.matmul(M, a)
                f_damp = torch.matmul(C_damp, v)
                
                residual = f_ext - f_int - f_inert - f_damp
                K_eff = K_glob + c1*M + c2*C_damp
                
                K_solve = K_eff
                rhs = residual
            else:
                residual = f_ext - f_int
                K_solve = K_glob
                rhs = residual
                
            # Dirichlet Randbedingungen
            # Zeilen/Spalten nullen
            # Manually
            if len(drltDofs) > 0:
                 # Für Aufgabe 8: Lager sind elastisch (Federn), daher KEINE starren Dirichlet BCs an Node 0,1
                 if AUFGABE == 8:
                     # Filter raus: Nur BCs die NICHT Node 0 oder 1 sind
                     # Unsere drlt Liste hat [0,0], [0,1], [1,0], [1,1] -> Das sind ALLES Lager BCs.
                     # Also einfach komplett überspringen für Task 8
                     pass 
                 else:
                     rhs[drltDofs] = 0.0
                     K_solve[drltDofs, :] = 0.0
                     K_solve[:, drltDofs] = 0.0
                     K_solve[drltDofs, drltDofs] = 1.0
            
            # Regularisierung für Stabilität
            K_solve.diagonal().add_(1.0)
            
            norm = torch.norm(residual)
            if norm < tol:
                # Converged
                # Update history for next step
                for e in range(nel):
                    history[e] = hist_trial[e]
                    # Use the epsilon calculated during the stress update
                    # This ensures perfect consistency
                    if 'eps_current_step' in hist_trial[e]:
                        history[e]['eps_prev'] = hist_trial[e]['eps_current_step']
                    else:
                        # Fallback (e.g. initial step)
                        pass
                break
            
            try:
                du = torch.linalg.solve(K_solve, rhs)
            except:
                print("LGS Error")
                break
                
            u += du
        
        # End Newton
        
        # Finales Update für v und a basierend auf konvergiertem u
        if enable_dyn:
            a = c1 * (u - u_pred)
            v = v_pred + gamma_nm * dt * a
            
               
        max_u = torch.max(torch.abs(u)).item()
        results_u.append(max_u)
        results_t.append(t.item())
        
        # Collect stress/strain for history (from converged state)
        # We need to re-loop or capture them. 
        # Since we just finished Newton, we can recalc or better: 
        # use the values from the last iteration? 
        # elem_stresses was populated in the last iteration.
        results_stress_hist.append(elem_stresses)
        
        # Recalculate strains for history output (fast)
        step_strains = []
        for e in range(nel):
            idx = conn[e]
            x0 = x_nodes[idx].T
            ud = torch.zeros(ndm, nen, device=device, dtype=torch.double)
            for k in range(nen):
               g = idx[k]
               ud[0,k] = u[g*ndf]
               ud[1,k] = u[g*ndf+1]
            xc = x0 + ud
            l0 = torch.norm(x0[:,1]-x0[:,0])
            l = torch.norm(xc[:,1]-xc[:,0])
            ep = (l-l0)/l0
            step_strains.append(ep.item())
        results_strain_hist.append(step_strains)

        max_displacement = max(max_displacement, max_u)
        curr_max_stress = max([abs(s) for s in elem_stresses]) if len(elem_stresses)>0 else 0
        max_stress_global = max(max_stress_global, curr_max_stress)
        
        if t_idx % 5 == 0:
            print(f"Time {t:.2f} s | Max U: {max_u:.4f} m | Stress: {curr_max_stress/1e6:.1f} MPa")
            
            if plot_live:
                u_curr = u.cpu().detach().numpy().reshape(-1, 2)
                x_curr = x_nodes.cpu().numpy() + u_curr
                
                max_s_plot = 500e6 # Fixed color scale
                
                for i, ln in enumerate(plot_lines):
                    idx = conn[i].cpu().numpy()
                    p1 = x_curr[idx[0]]
                    p2 = x_curr[idx[1]]
                    ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    
                    val = elem_stresses[i]
                    # Map -max..max to 0..1
                    c_val = 0.5 + 0.5 * (val/max_s_plot)
                    ln.set_color(plt.cm.jet(c_val))
                
                sm.set_clim(-max_s_plot/1e6, max_s_plot/1e6)
                fig.canvas.flush_events()

    if plot_live:
        plt.ioff()
        # plt.show() # Don't block here, we plot below
        plt.close(fig) # Close live window
        
    return results_u, results_stress_hist, results_t, results_strain_hist


#===============================
#============ MAIN =============
#===============================

print(f"--- STARTING TASK {AUFGABE} ---")

if AUFGABE == 4:
    # 3 Iterationen für Sizing
    # 1. Run
    print("Iteration 1: Area = 20 cm2 (Start)")
    u_hist, s_hist, t_hist, str_hist = run_fem(0.002, 210e9)
    
    # Calculate Max Stress from history
    s_max = 0.0
    for step_stresses in s_hist:
        current_max = max([abs(s) for s in step_stresses]) if len(step_stresses) > 0 else 0
        s_max = max(s_max, current_max)
        
    print(f"Result 1: Stress = {s_max/1e6:.2f} MPa")
    
    # Resize
    target = 1000e6 
    req_area = 0.002 * (s_max / target)
    print(f"Required Area: {req_area*1e4:.2f} cm2")
    
    # 2. Run
    u_hist, s_hist, t_hist, str_hist = run_fem(req_area, 210e9)
    
    s_max_2 = 0.0
    for step_stresses in s_hist:
        current_max = max([abs(s) for s in step_stresses]) if len(step_stresses) > 0 else 0
        s_max_2 = max(s_max_2, current_max)
        
    print(f"Result 2: Stress = {s_max_2/1e6:.2f} MPa")
    
    # Plotten (Spannung)
    s_max_hist = []
    for step_stresses in s_hist:
         s_max_hist.append(max([abs(s) for s in step_stresses]) if len(step_stresses) > 0 else 0)
         
    plt.figure()
    plt.plot(t_hist, [s/1e6 for s in s_max_hist])
    plt.title("Aufgabe 4: Max Spannung vs Zeit")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Spannung [MPa]")
    plt.savefig("Task4_Stress.png")
    
else:
    # AUFGABE = 9 removed
    if AUFGABE in [5, 6, 7, 8]:
        u_vec, s_mat, t_vec, str_mat = run_fem(Area, E_modul, enable_dyn=dynamic, enable_plastic=plastic, enable_visco=visco, enable_hardening=hardening)
        
        if AUFGABE == 8:
            plt.figure()
            plt.plot(t_vec, u_vec)
            plt.title("Aufgabe 8: Dynamische Antwort")
            plt.xlabel("Zeit [s]")
            plt.ylabel("Verschiebung [m]")
            plt.grid(True)
            plt.savefig("Task8_Dynamics.png")
            print("Plot gespeichert: Task8_Dynamics.png")
            
    elif AUFGABE == 9:
        # Vergleichslauf
        # 3 Modelle: Elastisch, Lineares Hardening, (Optional) Perfekt Plastisch
        
        show_perf_plast = False # <--- SCHALTER HIER (True/False)
        
        results = {}
        
        print("--- Running Model 1: Elastic ---")
        # Elastic: plastic=False
        u1, s1, t1, str1 = run_fem(Area, E_modul, enable_dyn=False, enable_plastic=False, enable_visco=False, enable_hardening=False)
        results['Elastic'] = (u1, s1, str1)

        print("--- Running Model 2: Linear Hardening ---")
        # Hardening: plastic=True, hardening=True
        u2, s2, t2, str2 = run_fem(Area, E_modul, enable_dyn=False, enable_plastic=True, enable_visco=False, enable_hardening=True)
        results['Hardening'] = (u2, s2, str2)
        
        if show_perf_plast:
            print("--- Running Model 3: Perfect Plastic ---")
            # Perf Plastic: plastic=True, hardening=False
            u3, s3, t3, str3 = run_fem(Area, E_modul, enable_dyn=False, enable_plastic=True, enable_visco=False, enable_hardening=False)
            results['Perfect Plastic'] = (u3, s3, str3)
            
        
        # Plotting - 3 Subplots
        plt.figure(figsize=(18, 5))
        
        # 1. Spannung-Dehnung
        plt.subplot(1, 3, 1)
        # Element finden
        max_idx = 0
        max_val = 0
        for i, val in enumerate(str2[-1]): 
            if abs(val) > max_val: 
                max_val = abs(val)
                max_idx = i
        print(f"Plotting Element {max_idx}")
        
        colors = {'Elastic': 'green', 'Hardening': 'blue', 'Perfect Plastic': 'red'}
        
        for name, data in results.items():
            u_r, s_mat_r, str_mat_r = data
            strs = [step[max_idx]/1e6 for step in s_mat_r]
            strn = [step[max_idx] for step in str_mat_r]
            plt.plot(strn, strs, label=name, color=colors[name], linewidth=2)
            
        plt.title(f"Spannungs-Dehnungs-Diagramm (Elem {max_idx})")
        plt.xlabel("Dehnung [-]")
        plt.ylabel("Spannung [MPa]")
        plt.grid(True)
        plt.legend()
        
        # 2. Strain-Time
        plt.subplot(1, 3, 2)
        for name, data in results.items():
            u_r, s_mat_r, str_mat_r = data
            strn = [step[max_idx] for step in str_mat_r]
            plt.plot(t1, strn, label=name, color=colors[name], linewidth=2)
            
        plt.title("Dehnungs-Zeit-Verlauf")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Dehnung [-]")
        plt.grid(True)
        # plt.legend() # Legend only in first or all? Let's hide to save space or keep
        
        # 3. Stress-Time
        plt.subplot(1, 3, 3)
        for name, data in results.items():
            u_r, s_mat_r, str_mat_r = data
            strs = [step[max_idx]/1e6 for step in s_mat_r]
            plt.plot(t1, strs, label=name, color=colors[name], linewidth=2)
            
        plt.title("Spannungs-Zeit-Verlauf")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Spannung [MPa]")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("Task9_Comparison.png")
        print("Plot gespeichert: Task9_Comparison.png")

print("Fertig.")

