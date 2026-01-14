"""
t1_dot_t4_dot_t1.py

Contraction used in the MATLAB code:

  T2(j,k) = sum_{i,l} G_A(i) * C4(i,j,k,l) * G_B(l)

where G_A and G_B are length-ndm gradients (ndm=2 in this project),
and C4 is the 4th-order tangent restricted to in-plane indices.
"""

from __future__ import annotations
import numpy as np


def t1_dot_t4_dot_t1(G_A: np.ndarray, C4: np.ndarray, G_B: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    G_A, G_B : (ndm,) arrays
    C4       : (ndm, ndm, ndm, ndm) array

    Returns
    -------
    T2 : (ndm, ndm) array
    """
    G_A = np.asarray(G_A, dtype=float).reshape(-1)
    G_B = np.asarray(G_B, dtype=float).reshape(-1)
    C4 = np.asarray(C4, dtype=float)
    n = C4.shape[0]

    T2 = np.zeros((n, n), dtype=float)
    for i in range(n):
        Gi = G_A[i]
        for l in range(n):
            T2 += (Gi * G_B[l]) * C4[i, :, :, l]
    return T2
