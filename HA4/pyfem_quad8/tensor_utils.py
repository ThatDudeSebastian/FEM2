"""
tensor_utils.py

Small, explicit tensor helpers for 3D small-strain J2 plasticity and
4th-order isotropic elasticity.

Conventions
- All tensors are 3x3 (embedded 3D). Plane strain is handled by eps33 = 0.
- When packing/unpacking to vectors for state variables, we use MATLAB-
  compatible column-major (Fortran) order.
"""

from __future__ import annotations
import numpy as np


def dev(A: np.ndarray) -> np.ndarray:
    """Deviatoric part of a 3x3 tensor: A - tr(A)/3 * I."""
    A = np.asarray(A, dtype=float)
    I = np.eye(3)
    return A - (np.trace(A) / 3.0) * I


def double_contract(A: np.ndarray, B: np.ndarray) -> float:
    """Double contraction A:B = sum_ij A_ij B_ij."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(np.sum(A * B))


def outer2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Outer product (A ⊗ B)_{ijkl} = A_{ij} B_{kl}."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.einsum("ij,kl->ijkl", A, B)


def I4sym_tensor() -> np.ndarray:
    """Symmetric 4th-order identity: I4sym_{ijkl} = 0.5*(δik δjl + δil δjk)."""
    I = np.eye(3)
    return 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))


def isotropic_elastic_tangent(E: float, nu: float) -> np.ndarray:
    """
    3D isotropic elastic tangent (4th order):
      C = K (I ⊗ I) + 2G I4dev
    where I4dev = I4sym - 1/3 (I ⊗ I).
    """
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    I = np.eye(3)
    I_outer = outer2(I, I)
    I4sym = I4sym_tensor()
    I4dev = I4sym - (1.0 / 3.0) * I_outer
    return K * I_outer + 2.0 * G * I4dev


def pack_matlab_9(A: np.ndarray) -> np.ndarray:
    """
    Pack 3x3 tensor to length-9 vector in MATLAB column-major order.
    """
    A = np.asarray(A, dtype=float).reshape((3, 3))
    return A.reshape((9,), order="F")


def unpack_matlab_9(v: np.ndarray) -> np.ndarray:
    """
    Unpack length-9 vector (MATLAB column-major) to 3x3 tensor.
    """
    v = np.asarray(v, dtype=float).reshape((9,))
    return v.reshape((3, 3), order="F")
