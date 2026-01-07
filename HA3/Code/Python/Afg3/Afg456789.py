
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# ============ CORE FEM CLASSES ============
# ==========================================

class MaterialModel:
    def __init__(self, E, Area):
        self.E = E
        self.Area = Area
        self.name = "Generic"

    def compute_stress_tangent(self, eps, deps, dt, history):
        raise NotImplementedError

class Hooke(MaterialModel):
    def __init__(self, E, Area):
        super().__init__(E, Area)
        self.name = "Hooke"

    def compute_stress_tangent(self, eps, deps, dt, history):
        sig = self.E * eps
        Et = self.E
        return sig, Et, history

class PerfectPlasticity(MaterialModel):
    def __init__(self, E, Area, Sy):
        super().__init__(E, Area)
        self.Sy = Sy
        self.name = "PerfectPlasticity"

    def compute_stress_tangent(self, eps, deps, dt, history):
        eps_p = history.get('eps_p', torch.tensor(0.0, device=device, dtype=torch.double))
        alpha = history.get('alpha', torch.tensor(0.0, device=device, dtype=torch.double))
        
        eps_trial = eps
        sig_trial = self.E * (eps_trial - eps_p)
        
        phi = torch.abs(sig_trial) - self.Sy
        
        if phi <= 0:
            sig = sig_trial
            Et = self.E
            new_history = {'eps_p': eps_p, 'alpha': alpha}
        else:
            d_gamma = phi / self.E
            sign_sig = torch.sign(sig_trial)
            sig = sig_trial - d_gamma * self.E * sign_sig
            eps_p_new = eps_p + d_gamma * sign_sig
            alpha_new = alpha + d_gamma
            Et = torch.tensor(1e-5 * self.E, device=device, dtype=torch.double)
            new_history = {'eps_p': eps_p_new, 'alpha': alpha_new}
            
        return sig, Et, new_history

class LinearHardening(MaterialModel):
    def __init__(self, E, Area, Sy, H):
        super().__init__(E, Area)
        self.Sy = Sy
        self.H = H
        self.name = "LinearHardening"

    def compute_stress_tangent(self, eps, deps, dt, history):
        eps_p = history.get('eps_p', torch.tensor(0.0, device=device, dtype=torch.double))
        alpha = history.get('alpha', torch.tensor(0.0, device=device, dtype=torch.double))
        
        eps_trial = eps
        sig_trial = self.E * (eps_trial - eps_p)
        flow_stress = self.Sy + self.H * alpha
        phi = torch.abs(sig_trial) - flow_stress
        
        if phi <= 0:
            sig = sig_trial
            Et = self.E
            new_history = {'eps_p': eps_p, 'alpha': alpha}
        else:
            d_gamma = phi / (self.E + self.H)
            sign_sig = torch.sign(sig_trial)
            sig = sig_trial - d_gamma * self.E * sign_sig
            eps_p_new = eps_p + d_gamma * sign_sig
            alpha_new = alpha + d_gamma
            Et = (self.E * self.H) / (self.E + self.H)
            new_history = {'eps_p': eps_p_new, 'alpha': alpha_new}
            
        return sig, Et, new_history

class ViscoElastic(MaterialModel):
    def __init__(self, E, Area, eta):
        super().__init__(E, Area)
        self.eta = eta
        self.name = "ViscoElastic"

    def compute_stress_tangent(self, eps, deps, dt, history):
        if dt > 0:
            rate = deps / dt
            viscous_term = self.eta * rate
            Et_viscous = self.eta / dt
        else:
            rate = 0.0
            viscous_term = 0.0
            Et_viscous = 0.0
        
        sig = self.E * eps + viscous_term
        Et = self.E + Et_viscous
        return sig, Et, history

def get_crane_geometry():
    x = torch.tensor([[0, 0], [1, 0], [2, 0], [4, 0], [6, 0], [8, 0], [9, 0],
                      [1, 0.4], [3, 0.4], [5, 0.4], [7, 0.4], [8, 0.4],
                      [9, -1], [6, -2], [12, -2], [6, -4.5], [12, -4.5]], 
                     device=device, dtype=torch.double)
    
    conn = torch.tensor([
        [0, 7], [1, 7], [1, 2], [2, 7], [2, 8], [2, 3], [3, 8], [3, 9], [3, 4], [4, 9],
        [4, 10], [4, 5], [5, 10], [5, 11], [5, 6], [6, 11], [7, 8], [8, 9], [9, 10],
        [10, 11], [6, 12], [12, 13], [12, 14], [13, 14], [13, 15], [14, 16]
    ], device=device)
    return x, conn

class FEMSolver:
    def __init__(self, nodes, conn, material_model, bearing_damping=False):
        self.nodes_initial = nodes.clone()
        self.conn = conn
        self.material = material_model
        
        self.ndm = 2
        self.ndf = 2
        self.nnp = nodes.shape[0]
        self.nel = conn.shape[0]
        
        self.u = torch.zeros(self.nnp * self.ndf, device=device, dtype=torch.double)
        self.v = torch.zeros_like(self.u)
        self.a = torch.zeros_like(self.u)
        
        self.history = [{} for _ in range(self.nel)]
        
        self.M = torch.zeros(self.nnp * self.ndf, self.nnp * self.ndf, device=device, dtype=torch.double)
        rho_steel = 7850.0 
        for e in range(self.nel):
            idx = conn[e]
            x_e = self.nodes_initial[idx]
            L = torch.norm(x_e[1] - x_e[0])
            m_elem = rho_steel * self.material.Area * L
            
            node1_dofs = [idx[0]*2, idx[0]*2+1]
            node2_dofs = [idx[1]*2, idx[1]*2+1]
            for d in node1_dofs: self.M[d, d] += m_elem / 2.0
            for d in node2_dofs: self.M[d, d] += m_elem / 2.0
            
        self.bearing_damping = bearing_damping
        self.damping_c = 1e4 
        self.bearing_k = 1e9 

        # Global Rayleigh Damping (alpha * M + beta * K)
        # alpha = 0.5 (mass prop), beta = 0.001 (stiffness prop - reduced to avoid stiffening)
        self.rayleigh_alpha = 0.5
        self.rayleigh_beta = 0.001 

    def solve(self, times, load_factors, drlt_bcs, neum_bcs, dynamic=False, visualize=False):
        results = {'time': [], 'max_stress': [], 'u_max': [], 'stress_history': [], 'strain_history': []}
        
        # Setup visualization if requested
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect('equal')
            # Determine bounds
            x_np = self.nodes_initial.cpu().numpy()
            margin = 2.0
            ax.set_xlim(x_np[:,0].min()-margin, x_np[:,0].max()+margin)
            ax.set_ylim(x_np[:,1].min()-margin, x_np[:,1].max()+margin)
            
            # Create plot objects
            plot_lines = []
            for i in range(self.nel):
                ln, = ax.plot([], [], 'k-', linewidth=2)
                plot_lines.append(ln)
            
            sc_nodes = ax.scatter(x_np[:,0], x_np[:,1], c='black', s=10, zorder=5)
            # Add text for max stress
            txt_status = ax.text(0.05, 0.95, "", transform=ax.transAxes, verticalalignment='top')
            
            # Add Colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Stress [MPa]')

        
        drltDofs = []
        for bc in drlt_bcs:
            dof = int(bc[0]*self.ndf + bc[1])
            drltDofs.append(dof)
        
        drltDofs = torch.tensor(drltDofs, device=device, dtype=torch.long)
        
        dt = times[1] - times[0]
        
        for t_idx, t in enumerate(times):
            f_ext = torch.zeros_like(self.u)
            lf = load_factors[t_idx] if t_idx < len(load_factors) else 0.0
            
            for bc in neum_bcs:
                node, dof, val = int(bc[0]), int(bc[1]), float(bc[2])
                f_ext[node*self.ndf + dof] = val * lf
            
            if dynamic:
                beta = 0.25
                gamma = 0.5
                u_pred = self.u + dt * self.v + (0.5 - beta) * dt**2 * self.a
                v_pred = self.v + (1 - gamma) * dt * self.a
                a_pred = torch.zeros_like(self.a)
                
                self.u = u_pred.clone()
                self.v = v_pred.clone()
                self.a = a_pred.clone()
            
            for iter in range(20):
                K_global = torch.zeros(self.nnp*self.ndf, self.nnp*self.ndf, device=device, dtype=torch.double)
                f_int = torch.zeros_like(self.u)
                
                element_stresses = []
                element_strains = []
                trial_history = copy.deepcopy(self.history)
                
                for e in range(self.nel):
                    node_indices = self.conn[e]
                    dof_indices = []
                    for n in node_indices:
                        dof_indices.extend([n*2, n*2+1])
                    dof_indices_t = torch.tensor(dof_indices, device=device, dtype=torch.long)
                    
                    ue = self.u[dof_indices_t]
                    x0 = self.nodes_initial[node_indices]
                    xc = x0 + ue.view(2, 2)
                    
                    dx0 = x0[1] - x0[0]
                    dxc = xc[1] - xc[0]
                    L0 = torch.norm(dx0)
                    L = torch.norm(dxc)
                    n = dxc / L
                    
                    eps = (L - L0) / L0
                    if 'eps_prev' not in trial_history[e]:
                        trial_history[e]['eps_prev'] = 0.0
                    eps_prev = trial_history[e]['eps_prev']
                    deps = eps - eps_prev
                    
                    sig, Et, new_hist_e = self.material.compute_stress_tangent(eps, deps, dt, trial_history[e])
                    element_stresses.append(sig.item())
                    element_strains.append(eps.item())
                    
                    force = sig * self.material.Area
                    f_local = torch.cat((-n * force, n * force))
                    
                    n_vec = n.unsqueeze(1)
                    P = torch.matmul(n_vec, n_vec.t())
                    I = torch.eye(2, device=device, dtype=torch.double)
                    
                    Km = (Et * self.material.Area / L0) * P
                    Kg = (force / L) * (I - P)
                    Kt = Km + Kg
                    Ke = torch.zeros(4, 4, device=device, dtype=torch.double)
                    Ke[0:2, 0:2] = Kt
                    Ke[0:2, 2:4] = -Kt
                    Ke[2:4, 0:2] = -Kt
                    Ke[2:4, 2:4] = Kt
                    
                    for i, gi in enumerate(dof_indices):
                        f_int[gi] += f_local[i]
                        for j, gj in enumerate(dof_indices):
                            K_global[gi, gj] += Ke[i, j]

                    trial_history[e].update(new_hist_e)

                # Add Bearing Spring Forces (Task 8)
                if dynamic and self.bearing_damping:
                     for node_idx in [0, 1]:
                         for d in [0, 1]:
                            dof = node_idx*2 + d
                            f_int[dof] += self.bearing_k * self.u[dof]
                            # Add to Stiffness
                            K_global[dof, dof] += self.bearing_k

                if dynamic:
                    c1 = 1.0 / (beta * dt**2)
                    c2 = gamma / (beta * dt)
                    
                    a = c1 * (self.u - u_pred)
                    v = v_pred + gamma * dt * a
                    
                    f_damp = torch.zeros_like(self.u)
                    C_damp_matrix = torch.zeros_like(self.M)
                    
                    if self.bearing_damping:
                        for node_idx in [0, 1]:
                            dxy = [node_idx*2, node_idx*2+1]
                            for d in dxy:
                                C_damp_matrix[d, d] += self.damping_c
                    
                    # Add Rayleigh Damping
                    C_damp_matrix += self.rayleigh_alpha * self.M + self.rayleigh_beta * K_global

                    f_damp = -torch.matmul(C_damp_matrix, v)
                    f_inert = -torch.matmul(self.M, a)
                    
                    rsd = f_ext + f_inert + f_damp - f_int
                    K_eff = K_global + c1 * self.M + c2 * C_damp_matrix
                    K_solve = K_eff
                    rhs = rsd
                else:
                    rsd = f_ext - f_int
                    K_solve = K_global
                    rhs = rsd

                active_drlt = []
                if self.bearing_damping and dynamic:
                    filter_mask = torch.ones(drltDofs.shape[0], dtype=torch.bool)
                    for i, d in enumerate(drltDofs):
                         if d <= 3: 
                             filter_mask[i] = False
                    active_drlt = drltDofs[filter_mask]
                else:
                    active_drlt = drltDofs

                if len(active_drlt) > 0:
                    rsd[active_drlt] = 0.0
                    K_solve[active_drlt, :] = 0.0
                    K_solve[:, active_drlt] = 0.0
                    K_solve[active_drlt, active_drlt] = 1.0
                    rhs[active_drlt] = 0.0
                
                # Debug Info
                # if iter == 0 and t_idx == 0:
                #    print(f"BC Count: {len(active_drlt)}")
                
                # Regularization
                K_solve.diagonal().add_(1.0)
                
                norm_rsd = torch.norm(rsd)
                if norm_rsd < 1e-3: 
                    for e in range(self.nel):
                        trial_history[e]['eps_prev'] = element_strains[e]
                        self.history[e] = copy.deepcopy(trial_history[e])
                    break
                
                try:
                    du = torch.linalg.solve(K_solve, rhs)
                except Exception as e:
                    print(f"Solver Error at t={t:.3f}: {e}")
                    # If singular, this is fatal usually. Break
                    break
                
                if torch.any(torch.isnan(du)):
                    print("Solver NaN")
                    break

                self.u += du
                
                if len(active_drlt) > 0:
                     self.u[active_drlt] = 0.0
            
            if dynamic:
                a = c1 * (self.u - u_pred)
                v = v_pred + gamma * dt * a
                self.v = v
                self.a = a

            results['time'].append(t)
            max_s = max([abs(s) for s in element_stresses])
            results['max_stress'].append(max_s)
            results['u_max'].append(torch.max(torch.abs(self.u)).item())
            results['stress_history'].append(element_stresses)
            results['strain_history'].append(element_strains)
            
            # Print Max Stress per step (Auftrag Aufgabe 4)
            print(f"  > Time {t:.2f}s | Max Stress: {max_s/1e6:.2f} MPa")

            # Update Visualization
            if visualize:
                # Update positions
                cur_u = self.u.detach().cpu().numpy()
                cur_nodes = self.nodes_initial.cpu().numpy() + cur_u.reshape(-1, 2)
                
                # Normalize stress for color
                max_stress_val = max(1e-6, max_s) # Avoid div by zero
                # Simple normalization - usually stresses are positive (tension) or negative (compression)
                # Let's map -Max to +Max to Blue-Red
                
                for i in range(self.nel):
                    n_idx = self.conn[i].cpu().numpy()
                    p1 = cur_nodes[n_idx[0]]
                    p2 = cur_nodes[n_idx[1]]
                    plot_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    
                    # Color map: normalized by CURRENT max stress (or global? Current is better for dynamic range)
                    s_val = element_stresses[i]
                    # Map -max_s..+max_s to 0..1
                    norm_val = 0.5 + 0.5 * (s_val / max_stress_val)
                    color = plt.cm.seismic(norm_val)
                    plot_lines[i].set_color(color)
                
                # Update colorbar limits dynamically
                if cbar:
                    sm.set_clim(-max_stress_val/1e6, max_stress_val/1e6)
                    cbar.update_normal(sm)

                sc_nodes.set_offsets(cur_nodes)
                txt_status.set_text(f"Time: {t:.2f} s\nMax Stress: {max_s/1e6:.2f} MPa")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                # plt.pause(0.001) # Small pause
        
        if visualize:
            plt.ioff()
            print("Simulation finished. Close the plot window to proceed to the next task.")
            plt.show()
            
        return results

def get_load_curve(scenario='ramp'):
    if scenario == 'ramp':
        times = torch.linspace(0, 5, 21, dtype=torch.double)
        factors = torch.linspace(0, 1, 21, dtype=torch.double)
    elif scenario == 'cycle':
        # More steps for stability in plasticity
        times = torch.linspace(0, 5, 201, dtype=torch.double)
        steps = 101
        f1 = torch.linspace(0, 1, steps)
        f2 = torch.linspace(1, -1, 2*steps - 1) # Full cycle to compression
        # Simple cycle 0 -> 1 -> 0
        f1 = torch.linspace(0, 1, 101)
        f2 = torch.linspace(1, 0, 101)
        factors = torch.cat((f1, f2[1:]))
    elif scenario == 'step':
        # Finer steps for dynamics
        times = torch.linspace(0, 2, 1001, dtype=torch.double) # dt = 0.002
        factors = torch.ones_like(times)
        factors[0] = 0 
    return times, factors

def run_all_tasks():
    x, conn = get_crane_geometry()
    
    drlt_bcs = [[0,0], [0,1], [1,0], [1,1]]
    load_val = -3090 * 9.81 
    neum_bcs = [[15, 1, load_val], [16, 1, load_val]]

    print("\n--- Running Task 4 (Hooke Sizing) ---")
    target_stress = 1000e6
    current_area = 0.002
    model_t4 = Hooke(E=210e9, Area=current_area)
    times, factors = get_load_curve('ramp')
    solver = FEMSolver(x, conn, model_t4)
    # Enable visualization for the first run!
    res = solver.solve(times, factors, drlt_bcs, neum_bcs, visualize=True)
    
    max_stress_reached = max(res['max_stress'])
    print(f"Initial Stress: {max_stress_reached/1e6:.2f} MPa")
    
    required_area = current_area * (max_stress_reached / target_stress)
    print(f"Resizing Area to {required_area*1e4:.2f} cm2")
    
    model_t4.Area = required_area
    solver = FEMSolver(x, conn, model_t4)
    # Visualize verification run
    res_t4 = solver.solve(times, factors, drlt_bcs, neum_bcs, visualize=True)
    print(f"New Stress: {max(res_t4['max_stress'])/1e6:.2f} MPa")
    
    plt.figure()
    plt.plot(res_t4['time'], np.array(res_t4['max_stress'])/1e6)
    plt.title("Task 4: Max Stress vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Stress [MPa]")
    plt.savefig("Task4_Stress.png")

    print("\n--- Running Task 5 (Perfect Plasticity) ---")
    model_t5 = PerfectPlasticity(E=210e9, Area=required_area, Sy=960e6)
    solver = FEMSolver(x, conn, model_t5)
    _ = solver.solve(times, factors, drlt_bcs, neum_bcs, visualize=True)
    
    print("\n--- Running Task 6 (Linear Hardening) ---")
    model_t6 = LinearHardening(E=210e9, Area=required_area, Sy=960e6, H=2e9)
    solver = FEMSolver(x, conn, model_t6)
    _ = solver.solve(times, factors, drlt_bcs, neum_bcs, visualize=True)

    print("\n--- Running Task 7 (Viscoelastic) ---")
    glue_E = 600e6
    glue_eta = 1e8
    glue_area = required_area * (210e9 / glue_E)
    model_t7 = ViscoElastic(E=glue_E, Area=glue_area, eta=glue_eta)
    solver = FEMSolver(x, conn, model_t7)
    res_t7 = solver.solve(times, factors, drlt_bcs, neum_bcs, visualize=True)
    print(f"Deflection: {max(res_t7['u_max']):.4f} m")

    print("\n--- Running Task 8 (Dynamics) ---")
    times_dyn, factors_dyn = get_load_curve('step')
    model_t8 = LinearHardening(E=210e9, Area=required_area, Sy=960e6, H=2e9)
    solver_dyn = FEMSolver(x, conn, model_t8, bearing_damping=True)
    res_t8 = solver_dyn.solve(times_dyn, factors_dyn, drlt_bcs, neum_bcs, dynamic=True, visualize=True)
    
    plt.figure()
    plt.plot(res_t8['time'], res_t8['u_max'])
    plt.title("Task 8: Dynamic Response with Damping")
    plt.xlabel("Time [s]")
    plt.ylabel("Max Displacement [m]")
    plt.savefig("Task8_Dynamics.png")
    
    print("\n--- Running Task 9 (Comparison) ---")
    times_cyc, factors_cyc = get_load_curve('cycle')
    models = [
        Hooke(E=210e9, Area=required_area),
        PerfectPlasticity(E=210e9, Area=required_area, Sy=960e6),
        LinearHardening(E=210e9, Area=required_area, Sy=960e6, H=2e9)
    ]
    results_comp = []
    
    for m in models:
        print(f"Solving for {m.name}...")
        solver = FEMSolver(x, conn, m)
        r = solver.solve(times_cyc, factors_cyc, drlt_bcs, neum_bcs)
        results_comp.append((m.name, r))
    
    if len(results_comp[0][1]['strain_history']) > 0:
        peak_idx = 25 
        strains_at_peak = results_comp[0][1]['strain_history'][peak_idx]
        max_e_idx = np.argmax(np.abs(strains_at_peak))
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        for name, res in results_comp:
            strs = [step[max_e_idx]/1e6 for step in res['stress_history']]
            strn = [step[max_e_idx] for step in res['strain_history']]
            plt.plot(strn, strs, label=name)
        plt.title("Stress-Strain")
        plt.ylabel("Stress [MPa]")
        
        plt.subplot(1, 3, 2)
        for name, res in results_comp:
            strn = [step[max_e_idx] for step in res['strain_history']]
            plt.plot(res['time'], strn, label=name)
        plt.title("Strain-Time")
        
        plt.subplot(1, 3, 3)
        for name, res in results_comp:
            strs = [step[max_e_idx]/1e6 for step in res['stress_history']]
            plt.plot(res['time'], strs, label=name)
        plt.title("Stress-Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Task9_Comparison.png")
    else:
        print("No results to plot for Task 9")

if __name__ == "__main__":
    run_all_tasks()
