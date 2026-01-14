"""
vmises_iso_kin_rr.py

Small-strain J2 (von Mises) plasticity with linear isotropic + kinematic
hardening, mixed by r in [0,1]. Radial return mapping with a consistent
algorithmic tangent.

This is a direct Python translation of the provided MATLAB routine
`vmises_iso_kin_rr.m`, including the same state layout:

    state = [epsp(:); k; a(:)]   (length 19)

where
- epsp is the plastic strain tensor (deviatoric, 3x3)
- k    is accumulated plastic strain (scalar)
- a    is the backstress-like tensor (deviatoric, 3x3)

Packing/unpacking uses MATLAB column-major order.
"""

from __future__ import annotations
import numpy as np

from .tensor_utils import (
    dev,
    double_contract,
    outer2,
    I4sym_tensor,
    pack_matlab_9,
    unpack_matlab_9,
)


def vmises_iso_kin_rr(eps_in: np.ndarray, matparam: np.ndarray, state_old: np.ndarray):
    """
    Parameters
    ----------
    eps_in   : (2,2) or (3,3) symmetric strain tensor
    matparam : array-like [E, nu, sigy0, H, rho, r]
    state_old: length-19 array [epsp(:); k; a(:)] (MATLAB order)

    Returns
    -------
    sigma     : (3,3) Cauchy stress
    C4        : (3,3,3,3) consistent algorithmic tangent
    state_new : length-19 updated state vector
    is_plastic: bool
    """
    # --- Embed into 3D (plane strain: eps33 = 0) ---
    eps_in = np.asarray(eps_in, dtype=float)
    if eps_in.shape == (2, 2):
        eps = np.zeros((3, 3), dtype=float)
        eps[:2, :2] = eps_in
    elif eps_in.shape == (3, 3):
        eps = eps_in.copy()
    else:
        raise ValueError("eps_in must be 2x2 or 3x3.")
    eps = 0.5 * (eps + eps.T)

    # --- Material parameters ---
    mp = np.asarray(matparam, dtype=float).reshape(-1)
    E = float(mp[0])
    nu = float(mp[1])
    sigy0 = float(mp[2])
    H = float(mp[3])
    r = float(mp[5]) if mp.size >= 6 else 0.5

    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    # --- State unpack ---
    if state_old is None or (isinstance(state_old, np.ndarray) and state_old.size == 0):
        state_old = np.zeros(19, dtype=float)
    state_old = np.asarray(state_old, dtype=float).reshape(-1)
    if state_old.size != 19:
        raise ValueError("state_old must have length 19: [epsp(:); k; a(:)].")

    epsp_n = unpack_matlab_9(state_old[:9])
    k_n = float(state_old[9])
    a_n = unpack_matlab_9(state_old[10:])

    # enforce symmetry + deviatoric
    epsp_n = dev(0.5 * (epsp_n + epsp_n.T))
    a_n = dev(0.5 * (a_n + a_n.T))

    # --- Trial state ---
    ee_tr = 0.5 * ((eps - epsp_n) + (eps - epsp_n).T)
    ee_tr_dev = dev(ee_tr)
    s_tr = 2.0 * G * ee_tr_dev

    kappa_tr = -r * H * k_n
    alpha_tr = -(2.0 / 3.0) * (1.0 - r) * H * a_n

    s_red_tr = s_tr + alpha_tr
    norm_sred = np.sqrt(max(0.0, double_contract(s_red_tr, s_red_tr)))

    Phi_tr = norm_sred - np.sqrt(2.0 / 3.0) * (sigy0 - kappa_tr)

    # --- Elastic tangent blocks ---
    I = np.eye(3)
    I_outer = outer2(I, I)
    I4sym = I4sym_tensor()
    I4dev = I4sym - (1.0 / 3.0) * I_outer

    tol = 1e-14 * max(1.0, sigy0)

    if (Phi_tr <= tol) or (norm_sred < 1e-30):
        # Elastic step
        sigma_vol = K * np.trace(ee_tr) * I
        sigma = sigma_vol + s_tr
        C4 = K * I_outer + 2.0 * G * I4dev
        return sigma, C4, state_old.copy(), False

    # --- Plastic step (radial return) ---
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
    sigma_vol = K * np.trace(ee_np1) * I
    sigma = sigma_vol + s_np1

    # Consistent algorithmic tangent
    c1 = 2.0 * G * (1.0 - (A * dlam) / norm_sred)
    c2 = 2.0 * G * A * ((dlam / norm_sred) - (1.0 / denom))

    C4 = K * I_outer + c1 * I4dev + c2 * outer2(nu_dir, nu_dir)

    state_new = np.zeros(19, dtype=float)
    state_new[:9] = pack_matlab_9(epsp_np1)
    state_new[9] = k_np1
    state_new[10:] = pack_matlab_9(a_np1)
    return sigma, C4, state_new, True
