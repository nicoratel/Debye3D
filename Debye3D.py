# NPScattering2D.py
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from xraydb import XrayDB
import psutil
from tqdm import tqdm
from scipy.interpolate import interp1d
from numba import njit, prange
# DebyeCalculator and pyFAI are optional runtime dependencies (see README)
try:
    from debyecalculator import DebyeCalculator
except Exception:
    DebyeCalculator = None
try:
    from pyFAI import AzimuthalIntegrator
    from pyFAI.detectors import Detector
except Exception:
    AzimuthalIntegrator = None
    Detector = None


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
# Main class: nanoparticle scattering in 2D detector plane
# -----------------------------------------------------
class Debye3D:
    """
    Class to simulate 2D scattering patterns from a nanoparticle structure.

    The class reads a structure (ASE-readable file), prepares a detector
    Q-grid and offers methods to compute 2D intensity maps and isotropic
    radial averages using different numerical strategies.

    Parameters
    ----------
    structure_file : str
        Path to a structure file readable by ASE (e.g., .xyz, .cif).
    npix : int, optional
        Number of pixels along one detector axis (detector is npix x npix).
        Default is 250.
    wl : float, optional
        X-ray wavelength in Å (default: 1.0 Å).
    Distance : float, optional
        Sample-to-detector distance in meters (default: 0.5 m).
    pixel_size : float, optional
        Pixel size in meters (default: 1e-4 m).
    """

    def __init__(self, structure_file, npix=250,
                 wl=1.0, Distance=0.5, pixel_size=0.0001):
        # Read structure and basic setup
        self.atoms = read(structure_file)
        self.file = structure_file
        self.positions = self.atoms.get_positions()
        self.elements = self.atoms.get_chemical_symbols()
        self.nb_atoms = len(self.positions)
        self.npix = int(npix)
        self.wl = float(wl)
        self.D = float(Distance)
        self.pixel_size = float(pixel_size)
        self.xdb = XrayDB()

        # Prepare detector Q-vectors (Qx, Qy, Qz)
        i_vals = np.arange(-self.npix // 2, self.npix // 2)
        j_vals = np.arange(-self.npix // 2, self.npix // 2)
        I, J = np.meshgrid(i_vals, j_vals, indexing='xy')

        delta_i = I * pixel_size
        delta_j = J * pixel_size
        denom = np.sqrt(self.D ** 2 + delta_i ** 2 + delta_j ** 2)
        a = 2.0 * np.pi / self.wl

        self.Qx = (a / denom) * delta_i
        self.Qz = (a / denom) * delta_j
        # Qy is defined such that Qy points along the beam direction offset
        self.Qy = (a / denom) * (self.D - denom)

        self.qvecs = np.stack([self.Qx.ravel(), self.Qy.ravel(), self.Qz.ravel()], axis=1)

        Q_magnitude = np.linalg.norm(self.qvecs, axis=1)
        self.q_min = float(Q_magnitude.min())
        self.q_max = float(Q_magnitude.max())

        # Print detector configuration summary
        print("----------------------------------------------------")
        print(" Detector configuration / accessible Q-range")
        print("----------------------------------------------------")
        print(f" Wavelength λ = {self.wl:.4f} Å")
        print(f" Sample-detector distance = {self.D * 1000.0:.2f} mm")
        print(f" Pixel size = {self.pixel_size * 1e3:.3f} mm")
        print(f" Number of pixels = {self.npix} x {self.npix}")
        print("")
        print(f" Qx range : {self.Qx.min():.4f} → {self.Qx.max():.4f} Å⁻¹")
        print(f" Qy range : {self.Qy.min():.4f} → {self.Qy.max():.4f} Å⁻¹")
        print(f" Qz range : {self.Qz.min():.4f} → {self.Qz.max():.4f} Å⁻¹")
        print(f" |Q| range : {self.q_min:.4f} → {self.q_max:.4f} Å⁻¹")
        print("----------------------------------------------------\n")

        # Warm up Numba to avoid long first-call compilation time
        self._warmup_numba()

    def _warmup_numba(self):
        """
        Execute a tiny dummy Numba computation to trigger compilation.

        This reduces latency for the first real calculation by compiling the
        compute_intensity_numba function in advance.
        """
        dummy_pos = np.zeros((2, 3), dtype=np.float64)
        dummy_q = np.zeros((2, 3), dtype=np.float64)
        dummy_f = np.ones(2, dtype=np.float64)
        _ = compute_intensity_numba(dummy_pos, dummy_f, dummy_q)

    def view_structure(self):
        """
        Write a temporary XYZ file and launch an external viewer (jmol).

        Notes
        -----
        This function calls an external 'jmol' binary using os.system and
        requires jmol to be installed and accessible in the PATH.
        """
        atoms = Atoms(symbols=self.elements, positions=self.positions)
        write('./file.xyz', atoms)
        os.system('jmol file.xyz')
        os.remove('file.xyz')

    def compute_Iq_debyecalc(self):
        """
        Compute I(q) using the external DebyeCalculator (if available).

        The method builds a q-grid consistent with the detector Q-range and
        invokes DebyeCalculator.iq to compute a Debye scattering curve.

        Returns
        -------
        q_dc : numpy.ndarray
            q-grid returned by the DebyeCalculator (Å⁻¹).
        i_dc_scaled : numpy.ndarray
            Intensity returned by DebyeCalculator, scaled to be comparable to
            other computations (arbitrary units). If DebyeCalculator is not
            available, raises RuntimeError.
        """
        if DebyeCalculator is None:
            raise RuntimeError("DebyeCalculator is not available. Install the optional dependency 'debyecalculator'.")

        n = self.npix / 2.0
        qstep = (self.q_max - self.q_min) / n
        calc = DebyeCalculator(qmin=self.q_min, qmax=self.q_max, qstep=qstep, biso=0, device='cuda')

        q_dc, i_dc = calc.iq(self.file)

        # Simple rescaling to match form-factor normalization used elsewhere
        f0 = self.xray_f0(self.elements[0], q_dc)
        # avoid divide-by-zero by using max
        K = 2.0 * f0 ** 2 / np.maximum(np.max(f0 ** 2), 1e-12)
        return q_dc, i_dc * K

    # -----------------------------
    # Atomic form factor
    # -----------------------------
    def xray_f0(self, element, q):
        """
        Query the X-ray atomic form factor f0 for a given element.

        Parameters
        ----------
        element : str
            Chemical symbol (e.g. "Au").
        q : array_like
            q values (Å⁻¹) where f0 is evaluated.

        Returns
        -------
        f0 : numpy.ndarray
            Atomic form factor values corresponding to q.
        """
        return self.xdb.f0(element, q)

    # -----------------------------
    # Intensity calculation (Numba)
    # -----------------------------
    def compute_intensity(self, positions=None):
        """
        Compute full 2D intensity map on the detector using Numba-accelerated sum.

        Parameters
        ----------
        positions : numpy.ndarray, shape (N_atoms, 3), optional
            If provided, compute with these positions; otherwise use structure positions.

        Returns
        -------
        I_map : numpy.ndarray, shape (npix, npix)
            2D intensity map arranged in detector coordinates [Qx x Qz].
        """
        if positions is None:
            positions = self.positions

        Q = self.qvecs.reshape(-1, 3)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)

        # Call the Numba-parallel routine
        I = compute_intensity_numba(positions, f_atom, Q)
        return I.reshape(self.Qx.shape)

    
    def compute_isotropic_intensity_fibonacci(self, q_vals=None, n_orient=1000):
        """
        Compute isotropic I(q) using Fibonacci quadrature with explicit direction set.

        Parameters
        ----------
        q_vals : array_like, optional
            Radial q grid (if None a default between self.q_min and self.q_max is used).
        n_orient : int, optional
            Number of Fibonacci directions to sample on the unit sphere.

        Returns
        -------
        q_vals : numpy.ndarray
            Radial q grid used for the calculation.
        Iq : numpy.ndarray
            Isotropic intensity computed from Fibonacci sampling.
        """
        if q_vals is None:
            q_vals = np.linspace(self.q_min, self.q_max, 500)

        f_q = self.xray_f0(self.elements[0], q_vals)
        dirs = fibonacci_sphere(n_orient)
        Iq = compute_intensity_fibonacci(self.positions, f_q, q_vals, dirs)
        return q_vals, Iq

    # -----------------------------
    # Plotting helpers
    # -----------------------------
    def plot_intensity(self, I_map, log=True, vmin=-6, vmax=0, qmax=None, interpolation='bicubic'):
        """
        Plot a 2D intensity map on the detector Q-plane.

        Parameters
        ----------
        I_map : numpy.ndarray, shape (npix, npix)
            Intensity map to plot.
        log : bool, optional
            If True, plot on a logarithmic color scale (default True).
        vmin : float, optional
            Lower bound exponent (10**vmin) for LogNorm (default -6).
        vmax : float, optional
            Upper bound exponent (10**vmax) for LogNorm (default 0).
        qmax : float or None, optional
            If provided, restrict the upper extent of the displayed Qx/Qz axes to qmax.
        interpolation : str, optional
            Matplotlib interpolation method for imshow (default 'bicubic').

        Returns
        -------
        None
        """
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10.0 ** vmin, vmax=10.0 ** vmax)
        else:
            norm = None

        if qmax is None:
            extent = [self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent = [self.Qx.min(), qmax, self.Qz.min(), qmax]

        im = ax.imshow(
            I_map,
            extent=extent,
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation=interpolation
        )

        plt.colorbar(im, ax=ax, label="Intensity (log)" if log else "Intensity")
        ax.set_xlabel("$q_x$ (Å⁻¹)")
        ax.set_ylabel("$q_z$ (Å⁻¹)")

        def format_coord(x, y):
            """
            Status-bar coordinate formatter that prints q, real-space d and angle.
            """
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.degrees(np.arctan2(y, x))
            if r > 0:
                d = 2.0 * np.pi / (10.0 * r)
                return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, θ={theta:.1f}°"
            else:
                return f"q={r:.4f} Å⁻¹, θ={theta:.1f}°"
        ax.format_coord = format_coord

        plt.show()

    # -----------------------------
    # Euler rotations
    # -----------------------------
    @staticmethod
    def euler_rotation_matrix(alpha, beta, gamma):
        """
        Return a rotation matrix for Z-Y-Z Euler angles (degrees).

        Parameters
        ----------
        alpha : float
            Rotation angle (degrees) about Z for the first rotation.
        beta : float
            Rotation angle (degrees) about Y (in the middle).
        gamma : float
            Rotation angle (degrees) about Z for the last rotation.

        Returns
        -------
        R : numpy.ndarray, shape (3, 3)
            Rotation matrix corresponding to the Z-Y-Z Euler sequence.
        """
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)
        ca, cb, cg = np.cos([alpha, beta, gamma])
        sa, sb, sg = np.sin([alpha, beta, gamma])
        Rz_alpha = np.array([[ca, -sa, 0],
                             [sa, ca, 0],
                             [0, 0, 1]])
        Ry_beta = np.array([[cb, 0, sb],
                            [0, 1, 0],
                            [-sb, 0, cb]])
        Rz_gamma = np.array([[cg, -sg, 0],
                             [sg, cg, 0],
                             [0, 0, 1]])
        return Rz_alpha @ Ry_beta @ Rz_gamma

    def rotate_positions(self, alpha, beta, gamma):
        """
        Rotate the stored atomic positions in-place using Euler angles.

        Parameters
        ----------
        alpha, beta, gamma : float
            Euler angles in degrees (ZYZ convention).

        Returns
        -------
        rotated_positions : numpy.ndarray, shape (N_atoms, 3)
            The rotated positions (also stored in self.positions).
        """
        R = self.euler_rotation_matrix(alpha, beta, gamma)
        self.positions = self.positions @ R.T
        return self.positions

    # -----------------------------
    # pyFAI convenience
    # -----------------------------
    def ai(self):
        """
        Create and return a pyFAI AzimuthalIntegrator configured for the detector.

        Returns
        -------
        ai : pyFAI.AzimuthalIntegrator
            Configured integrator instance.

        Notes
        -----
        pyFAI must be installed for this method to work.
        """
        if AzimuthalIntegrator is None or Detector is None:
            raise RuntimeError("pyFAI is not available. Install 'pyFAI' to use integrate functions.")
        detector = Detector(pixel1=self.pixel_size, pixel2=self.pixel_size)
        ai = AzimuthalIntegrator(dist=self.D, detector=detector)
        ai.setFit2D(self.D * 1000.0, self.npix / 2.0, self.npix / 2.0, wavelength=self.wl)
        return ai

    def integrate_with_pyfai(self, I, plot=False):
        """
        Integrate a 2D intensity map radially (1D integration) using pyFAI.

        Parameters
        ----------
        I : numpy.ndarray, shape (npix, npix)
            2D intensity map in detector coordinates.
        plot : bool, optional
            If True, plot the integrated 1D I(q).

        Returns
        -------
        q : numpy.ndarray
            1D q-grid returned by pyFAI (Å⁻¹).
        i : numpy.ndarray
            Radially integrated intensity values.
        """
        ai = self.ai()
        q, i = ai.integrate1d(I, npt=1000, unit="q_A^-1")
        if plot:
            plt.figure()
            plt.loglog(q, i, 'k-')
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("I(Q) (a.u.)")
            plt.xlim(self.q_min, self.q_max)
            plt.title("Radial integration in Q-space")
            plt.grid(True)
            plt.show()
        return q, i

    # ============================================================
    # Gaussian orientation distribution (3D) → 2D intensity map
    # ============================================================
    @staticmethod
    def _rotation_y(theta):
        """
        Return a rotation matrix for a rotation about Y by theta (radians).
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    @staticmethod
    def _rotation_z(phi):
        """
        Return a rotation matrix for a rotation about Z by phi (radians).
        """
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    @staticmethod
    def _rotation_yz(theta_y, theta_z):
        """
        Rotation matrix R = Rz(theta_z) @ Ry(theta_y).

        Parameters
        ----------
        theta_y : float
            Rotation around Y in radians.
        theta_z : float
            Rotation around Z in radians.

        Returns
        -------
        R : numpy.ndarray, shape (3,3)
            Combined rotation matrix.
        """
        return NanoparticleScattering2D._rotation_z(theta_z) @ NanoparticleScattering2D._rotation_y(theta_y)

    @staticmethod
    def _gaussian_orientation_grid(n_y=9, n_z=9,
                                   sigma_y=np.deg2rad(5),
                                   sigma_z=np.deg2rad(5),
                                   n_sigma=3):
        """
        Generate a 2D angular sampling grid for a Gaussian orientation distribution.

        The grid covers [-n_sigma * sigma, +n_sigma * sigma] in both axes,
        and returns normalized weights for each sampled orientation.

        Parameters
        ----------
        n_y, n_z : int
            Number of grid points along the y and z angular axes.
        sigma_y, sigma_z : float
            Standard deviations of the Gaussian distribution (radians).
        n_sigma : float
            Number of sigma to span (positive number, e.g. 3).

        Returns
        -------
        orientations : numpy.ndarray, shape (n_y * n_z, 2)
            Array of (theta_y, theta_z) pairs (radians).
        weights : numpy.ndarray, shape (n_y * n_z,)
            Corresponding normalized weights that sum to 1.
        """
        thetas_y = np.linspace(-n_sigma * sigma_y, n_sigma * sigma_y, n_y)
        thetas_z = np.linspace(-n_sigma * sigma_z, n_sigma * sigma_z, n_z)
        ThetaY, ThetaZ = np.meshgrid(thetas_y, thetas_z, indexing='ij')
        W = np.exp(-0.5 * ((ThetaY / sigma_y) ** 2 + (ThetaZ / sigma_z) ** 2))
        W /= np.sum(W)
        orientations = np.stack([ThetaY.ravel(), ThetaZ.ravel()], axis=1)
        weights = W.ravel()
        return orientations, weights

    def compute_intensity_gaussian_3D(self,
                                      sigma_y=np.deg2rad(5),
                                      sigma_z=np.deg2rad(5),
                                      n_y=9, n_z=9,
                                      n_sigma=3,
                                      show_progress=True,
                                      use_friedel_sym=False):
        """
        Compute a 2D intensity map for a Gaussian distribution of orientations.

        The method assumes a population of identical nanoparticles whose
        orientations are described by independent Gaussian spreads around
        the x-axis (rotations about y and z). For each sampled orientation,
        the 2D intensity map is computed and weighted by the orientation probability.

        Parameters
        ----------
        sigma_y : float, optional
            Standard deviation of orientation distribution around the y axis (radians).
        sigma_z : float, optional
            Standard deviation of orientation distribution around the z axis (radians).
        n_y, n_z : int, optional
            Number of sampling points along the y and z angular axes.
        n_sigma : int or float, optional
            Extent of the grid in units of sigma (e.g. 3 covers ±3σ).
        show_progress : bool, optional
            If True, show a tqdm progress bar over orientation samples.
        use_friedel_sym : bool, optional
            If True, exploit Friedel symmetry I(-Q) = I(Q) and compute only half
            detector (Qz >= 0) to save cost; the full map is then reconstructed.

        Returns
        -------
        I_total : numpy.ndarray, shape (npix, npix)
            2D intensity map integrated over the Gaussian orientation distribution.
        """
        orientations, weights = self._gaussian_orientation_grid(
            n_y=n_y, n_z=n_z, sigma_y=sigma_y, sigma_z=sigma_z, n_sigma=n_sigma
        )

        # When using Friedel symmetry, restrict computations to half detector
        if use_friedel_sym:
            mask_half = self.Qz >= 0
            Qx_half = self.Qx[mask_half]
            Qz_half = self.Qz[mask_half]
            qvecs_half = np.stack([Qx_half, np.zeros_like(Qx_half), Qz_half], axis=1)
        else:
            qvecs_half = self.qvecs

        I_total_half = np.zeros(qvecs_half.shape[0], dtype=np.float64)

        iterable = tqdm(enumerate(zip(orientations, weights)),
                        total=len(weights),
                        disable=not show_progress,
                        desc="Orientation averaging")

        for _, ((theta_y, theta_z), w) in iterable:
            # Rotate the structure for this orientation
            R = self._rotation_yz(theta_y, theta_z)
            rotated_positions = self.positions @ R.T

            # Compute intensity for rotated structure on the half-grid
            f_vals = self.xray_f0(self.elements[0], np.linalg.norm(qvecs_half, axis=1))
            I_rot_half = compute_intensity_numba(rotated_positions, f_vals, qvecs_half)
            I_total_half += w * I_rot_half

        # Reconstruct the full detector map if Friedel symmetry was used
        if use_friedel_sym:
            I_total = np.zeros_like(self.Qx)
            I_total[mask_half] = I_total_half
            # naive reconstruction for the other half: mirror the data
            # This requires mapping indices; we approximate by sorting approach
            # to keep the original behavior consistent with the initial code.
            Qz_masked = self.Qz[mask_half]
            order = np.argsort(Qz_masked)
            reversed_order = order[::-1]
            # Fill the other half by reversed ordering (best-effort)
            I_total[~mask_half] = I_total_half[reversed_order]
        else:
            I_total = I_total_half.reshape(self.Qx.shape)

        return I_total.reshape(self.Qx.shape)
