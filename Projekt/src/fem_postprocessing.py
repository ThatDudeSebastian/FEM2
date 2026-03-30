import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import torch
import math

def calculate_von_mises_and_norm(nel, nqp, element_dofs, u, x, conn, nnp, ndf, nen, qpt, w8, thickness, nu, von_mises_fn, state_gp, get_shape_data_fn):
    """
    Computes Von Mises stress and displacement norm averaged per element.
    """
    element_results = {"u_norm": np.zeros(nel), "svm": np.zeros(nel)}
    
    for el in range(nel):
        indices = element_dofs[el]
        ue = u[indices].reshape(-1, 2).t()
        xe = x[conn[el]].t()
        svm_el, u_norm_el = 0.0, 0.0
        
        for q in range(nqp):
            N, Gsh = get_shape_data_fn(qpt[q], nen)
            Je = xe @ Gsh
            dv = torch.det(Je) * w8[q] * thickness
            G = torch.linalg.solve(Je.T, Gsh.T).T
            eps = 0.5 * (ue @ G + (ue @ G).t())
            
            # Using the specific von_mises_return from the script
            sig, _, _ = von_mises_fn(eps, state_gp[el][q])
            
            s11, s22, s12 = float(sig[0, 0]), float(sig[1, 1]), float(sig[0, 1])
            s33 = nu * (s11 + s22)
            tr = (s11 + s22 + s33) / 3.0
            # Deviatoric stress
            sd11, sd22, sd33, sd12 = s11-tr, s22-tr, s33-tr, s12
            
            # svm = sqrt(3/2 * s:s) = sqrt(3/2 * (sd11^2 + sd22^2 + sd33^2 + 2*sd12^2))
            svm_el += math.sqrt(1.5 * (sd11**2 + sd22**2 + sd33**2 + 2*sd12**2))
            
            u_vals = (ue @ N).detach().numpy()
            u_norm_el += np.linalg.norm(u_vals)
            
        element_results["svm"][el] = svm_el / nqp / 1e6 # MPa
        element_results["u_norm"][el] = (u_norm_el / nqp) * 1000 # mm
        
    return element_results

def plot_spatial_results(fig_title, x_def, conn, element_type, element_results, interactive=True, save_path=None):
    """
    Plots the final deformed mesh with Displacement and Stress maps.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle(fig_title, fontweight='bold')
    
    # Vertices for PolyCollection
    if element_type == "quad8":
        idx = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    else:
        idx = [0, 1, 2, 3, 0]
        
    verts = [x_def[conn[e][idx]] for e in range(len(conn))]
    
    # Displacement Plot
    pc_u = PolyCollection(verts, cmap='viridis', edgecolors='none', alpha=0.9)
    pc_u.set_array(element_results["u_norm"])
    ax1.add_collection(pc_u)
    plt.colorbar(pc_u, ax=ax1, label="Verschiebung [mm]")
    ax1.set_title("Verschiebungsbetrag [mm]")
    ax1.set_aspect('auto')
    ax1.autoscale_view()
    
    # Stress Plot
    pc_s = PolyCollection(verts, cmap='jet', edgecolors='none', alpha=0.9)
    pc_s.set_array(element_results["svm"])
    ax2.add_collection(pc_s)
    plt.colorbar(pc_s, ax=ax2, label="Spannung [MPa]")
    ax2.set_title("Von Mises Spannung [MPa]")
    ax2.set_aspect('auto')
    ax2.autoscale_view()
    
    if interactive:
        ann1 = ax1.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", 
                           bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
        ann2 = ax2.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points", 
                           bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
        ann1.set_visible(False); ann2.set_visible(False)
        
        def hover(event):
            for ax, ann, pc, key, unit in [(ax1, ann1, pc_u, "u_norm", "mm"), (ax2, ann2, pc_s, "svm", "MPa")]:
                if event.inaxes == ax:
                    cont, ind = pc.contains(event)
                    if cont:
                        i = ind["ind"][0]
                        b = pc.get_paths()[i].get_extents()
                        ann.xy = [(b.x0+b.x1)/2, (b.y0+b.y1)/2]
                        ann.set_text(f"Elem: {i}\n{element_results[key][i]:.2f} {unit}")
                        ann.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                ann.set_visible(False)
            fig.canvas.draw_idle()
            
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved spatial plot to {save_path}")

    return fig

def get_yield_surface(sigma_y, H, r, k, a):
    """
    Computes yield surface coordinates in principal stress space.
    """
    sig_eff = sigma_y + r * H * k
    theta = np.linspace(0, 2 * np.pi, 200)
    # Circle in s-space
    sx = sig_eff * np.cos(theta)
    sy = sig_eff * np.sin(theta)
    
    # Backstress shift
    factor = (2.0/3.0) * (1.0 - r) * H
    beta = factor * a
    
    # Shift principal stresses
    # s1 = sqrt(1.5)*s_x ... actually the mapping depends on the projection
    # For J2 plasticity in principal space:
    s1 = (sx - sy/math.sqrt(3)) + (beta[0,0] + beta[1,1]).item()
    s2 = (sx + sy/math.sqrt(3)) + (beta[1,1] - beta[0,0]).item()
    
    return s1/1e6, s2/1e6 # MPa

def plot_history_overview(disp_pl, load_hist, load_target_hist, eps_p_xx_hist, sig_yy_hist, eps_p_eq_hist, sig_eq_hist, sig_1_hist, sig_2_hist, k_hist, a_hist, sigma_y, H, r, save_path=None):
    """
    Plots the 2x2 history analysis matrix.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Simulation History Analysis", fontweight='bold')
    
    # 1. Force-Displacement
    plt.subplot(2, 2, 1)
    plt.plot(disp_pl, load_hist, 'r', label="Actual")
    plt.plot(disp_pl, load_target_hist, 'k--', alpha=0.5, label="Target")
    plt.title("Force-Displacement")
    plt.xlabel("U_y [m]")
    plt.ylabel("Force [N]")
    plt.grid(True)
    plt.legend()
    # Markers
    if len(disp_pl) > 0:
        plt.scatter(disp_pl[0], load_hist[0], c='g', marker='o', s=80, zorder=5, label="Start")
        plt.scatter(disp_pl[-1], load_hist[-1], c='r', marker='x', s=80, zorder=5, label="End")
    
    # 2. Hysteresis
    plt.subplot(2, 2, 2)
    plt.plot(eps_p_xx_hist, sig_yy_hist, 'b')
    plt.title("Hysterese: sigma_yy vs eps_p_xx")
    plt.xlabel("eps_p_xx [-]")
    plt.ylabel("sigma_yy [Pa]")
    plt.grid(True)
    if len(eps_p_xx_hist) > 0:
        plt.scatter(eps_p_xx_hist[0], sig_yy_hist[0], c='g', marker='o', s=80, zorder=5)
        plt.scatter(eps_p_xx_hist[-1], sig_yy_hist[-1], c='r', marker='x', s=80, zorder=5)
    
    # 3. Equivalent Hysteresis
    plt.subplot(2, 2, 3)
    plt.plot(eps_p_eq_hist, sig_eq_hist, 'g')
    plt.title("Equivalent Hysteresis")
    plt.xlabel("eps_p_eq [-]")
    plt.xlabel("eps_p_eq [-]")
    plt.ylabel("sigma_eq [Pa]")
    plt.grid(True)
    if len(eps_p_eq_hist) > 0:
        plt.scatter(eps_p_eq_hist[0], sig_eq_hist[0], c='g', marker='o', s=80, zorder=5)
        plt.scatter(eps_p_eq_hist[-1], sig_eq_hist[-1], c='r', marker='x', s=80, zorder=5)
    
    # 4. Yield Surface
    plt.subplot(2, 2, 4)
    if k_hist:
        x0, y0 = get_yield_surface(sigma_y, H, r, 0, torch.zeros(3,3))
        x1, y1 = get_yield_surface(sigma_y, H, r, k_hist[-1], a_hist[-1])
        
        plt.plot(x0, y0, 'k--', label="Initial")
        plt.plot(x1, y1, 'r', label="Hardened")
        plt.plot(sig_1_hist, sig_2_hist, 'b', alpha=0.6, label="Path")
        plt.scatter(sig_1_hist[-1], sig_2_hist[-1], color='blue')
        
        plt.title(f"Yield Surface (r={r})")
        plt.xlabel("sigma_1 [MPa]")
        plt.ylabel("sigma_2 [MPa]")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved history plot to {save_path}")
        
    return fig
