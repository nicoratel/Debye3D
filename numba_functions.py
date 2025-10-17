import math
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from xraydb import XrayDB
import psutil

from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange



# -----------------------------------------------------
# Numba JIT function for intensity calculation
# -----------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_intensity_numba(positions, f_atom, Q):
    """
    Compute scattering intensity for many Q vectors using Numba.
    The computed quantity is |sum_a exp(i Q · r_a)|^2 * f(Q)^2 for each Q.

    Parameters
    ----------
    positions : numpy.ndarray, shape (N_atoms, 3)
        Cartesian coordinates of atoms in Å.
    f_atom : numpy.ndarray, shape (N_q,)
        Atomic form factor f(Q) evaluated at each Q magnitude.
    Q : numpy.ndarray, shape (N_q, 3)
        Q-vectors for which intensity is computed (Å⁻¹).

    Returns
    -------
    I : numpy.ndarray, shape (N_q,)
        Intensity values for each Q vector (arbitrary units).
    """
    n_q = Q.shape[0]
    n_atoms = positions.shape[0]
    I = np.empty(n_q, dtype=np.float64)

    for iq in prange(n_q):
        qx, qy, qz = Q[iq]
        Re, Im = 0.0, 0.0
        for ia in range(n_atoms):
            phase = qx * positions[ia, 0] + qy * positions[ia, 1] + qz * positions[ia, 2]
            Re += np.cos(phase)
            Im += np.sin(phase)
        I[iq] = (Re ** 2 + Im ** 2) * (f_atom[iq] ** 2)
    return I


# ------------------------------
# Orientation averaging utilities
# ------------------------------
def fibonacci_sphere(n_orient):
    """
    Generate approximately uniformly distributed directions on the unit sphere
    using the Fibonacci sphere algorithm.

    Parameters
    ----------
    n_orient : int
        Number of orientation directions to generate.

    Returns
    -------
    dirs : numpy.ndarray, shape (n_orient, 3)
        Unit vectors uniformly distributed on the sphere.
    """
    indices = np.arange(n_orient)
    phi = (1 + np.sqrt(5.0)) / 2.0  # golden ratio
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - 2.0 * (indices + 0.5) / n_orient
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return dirs


@njit(parallel=True, fastmath=True)
def compute_intensity_fibonacci(positions, f_q, q_vals, dirs):
    """
    Compute isotropic intensity via spherical quadrature using Fibonacci directions.

    This function evaluates the scattering amplitude for each sampled
    direction on the sphere and performs an average to obtain an isotropic I(q).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N_atoms, 3)
        Atomic positions.
    f_q : numpy.ndarray, shape (N_q,)
        Atomic form factor values at the q magnitudes.
    q_vals : numpy.ndarray, shape (N_q,)
        Radial q values corresponding to f_q.
    dirs : numpy.ndarray, shape (N_orient, 3)
        Unit direction vectors on the sphere.

    Returns
    -------
    Iq : numpy.ndarray, shape (N_q,)
        Isotropic intensity I(q) for each q in q_vals.
    """
    n_atoms = positions.shape[0]
    n_q = len(q_vals)
    n_orient = dirs.shape[0]
    Iq = np.zeros(n_q, dtype=np.float64)

    for iq in prange(n_q):
        q = q_vals[iq]
        f = f_q[iq]
        I_sum = 0.0

        for io in range(n_orient):
            qx, qy, qz = q * dirs[io, :]
            Re = 0.0
            Im = 0.0
            for ia in range(n_atoms):
                phase = qx * positions[ia, 0] + qy * positions[ia, 1] + qz * positions[ia, 2]
                Re += np.cos(phase)
                Im += np.sin(phase)
            I_sum += (Re ** 2 + Im ** 2)

        Iq[iq] = (I_sum / n_orient) * (f ** 2)

    return Iq


def compute_intensity_isotropic_fibonacci(positions, xdb, element,
                                          q_min, q_max, n_q=500, n_orient=1000):
    """
    Compute isotropic I(q) for a specific element using Fibonacci quadrature.

    Parameters
    ----------
    positions : numpy.ndarray, shape (N_atoms, 3)
        Atomic positions.
    xdb : xraydb.XrayDB
        X-ray database instance to provide atomic form factors.
    element : str
        Chemical symbol of the scattering atom (e.g. "Au", "Fe").
    q_min : float
        Minimum q-value (Å⁻¹).
    q_max : float
        Maximum q-value (Å⁻¹).
    n_q : int, optional
        Number of radial q points (default: 500).
    n_orient : int, optional
        Number of Fibonacci directions (default: 1000).

    Returns
    -------
    q_vals : numpy.ndarray, shape (n_q,)
        Radial q grid used for the calculation.
    Iq : numpy.ndarray, shape (n_q,)
        Isotropic intensity computed on q_vals.
    """
    q_vals = np.linspace(q_min, q_max, n_q)
    f_q = xdb.f0(element, q_vals)
    dirs = fibonacci_sphere(n_orient)
    Iq = compute_intensity_fibonacci(positions, f_q, q_vals, dirs)
    return q_vals, Iq

# -----------------------------------------------------
# Methods accountig for wavelength dispersion
# -----------------------------------------------------

@njit(parallel=True, fastmath=True)
def _compute_intensity_numba_dispersion_vectorized(positions, f_atom, Q,
                                                   dlambda_over_lambda=0.01,
                                                   nsamples=5):
    """
    Numba JIT vectorized function: intensity with wavelength dispersion.
    Vectorisation sur les échantillons Monte-Carlo.
    """
    n_q = Q.shape[0]
    n_atoms = positions.shape[0]
    I = np.zeros(n_q, dtype=np.float64)

    rng = np.random.rand(n_q, nsamples)  # valeurs uniformes [0,1)
    # transformer en gaussienne standard
    rand_gauss = np.sqrt(-2.0 * np.log(rng)) * np.cos(2.0 * np.pi * rng)  # Box-Muller approx
    rand_gauss = rand_gauss * dlambda_over_lambda  # échelle Δλ/λ

    for iq in prange(n_q):
        qvec = Q[iq]
        f = f_atom[iq]

        # initialisation pour accumulation
        I_sum = 0.0

        # Boucle sur les samples mais vectorisation du calcul cos/sin
        for isamp in range(nsamples):
            factor = 1.0 + rand_gauss[iq, isamp]
            qvec_sample = qvec * factor

            # phases : array shape (N_atoms,)
            phases = positions @ qvec_sample  # produit matriciel vectorisé

            Re = np.cos(phases).sum()
            Im = np.sin(phases).sum()

            I_sum += (Re**2 + Im**2)

        I[iq] = (I_sum / nsamples) * (f**2)

    return I

# --------------------------------------------------------------
# Method for uniaxial ODF calculation
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _compute_intensity_uniaxial_numba(positions, f_atom, Q,
                                      theta_y, theta_z,
                                      verbose=False):
    """
    Version Numba parallèle du calcul Monte Carlo d'orientation uniaxiale.
    """
    n_samples = len(theta_y)
    n_q = Q.shape[0]
    n_atoms = positions.shape[0]
    I_accum = np.zeros(n_q, dtype=np.float64)

    for isamp in prange(n_samples):
        ty = theta_y[isamp]
        tz = theta_z[isamp]

        # matrices de rotation (Rz * Ry)
        cy, sy = np.cos(ty), np.sin(ty)
        cz, sz = np.cos(tz), np.sin(tz)

        R = np.empty((3, 3))
        R[0, 0] = cz * cy
        R[0, 1] = -sz
        R[0, 2] = cz * sy
        R[1, 0] = sz * cy
        R[1, 1] = cz
        R[1, 2] = sz * sy
        R[2, 0] = -sy
        R[2, 1] = 0.0
        R[2, 2] = cy

        # rotation des positions
        pos_rot = np.zeros_like(positions)
        for i in range(n_atoms):
            x, y, z = positions[i]
            pos_rot[i, 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
            pos_rot[i, 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
            pos_rot[i, 2] = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

        # accumulation de l’intensité pour cette orientation
        for iq in range(n_q):
            qx, qy, qz = Q[iq]
            f = f_atom[iq]
            Re, Im = 0.0, 0.0
            for ia in range(n_atoms):
                phase = qx * pos_rot[ia, 0] + qy * pos_rot[ia, 1] + qz * pos_rot[ia, 2]
                Re += math.cos(phase)
                Im += math.sin(phase)
            I_accum[iq] += (Re ** 2 + Im ** 2) * (f ** 2)

    return I_accum / n_samples