import math
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from xraydb import XrayDB
import psutil
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
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
from numba_functions import *
from numba_functions import _compute_intensity_uniaxial_numba




class Experiment:
    def __init__(self, npix=250, wl=1.0, distance=0.5, 
                 pixel_size=0.0001, verbose=True):

        self.npix = int(npix)
        self.wl = float(wl)
        self.distance = float(distance)
        self.D=self.distance
        self.pixel_size = float(pixel_size)

        # Prepare detector Q-vectors (Qx, Qy, Qz)
        i_vals = np.arange(-self.npix // 2, self.npix // 2)
        j_vals = np.arange(-self.npix // 2, self.npix // 2)
        I, J = np.meshgrid(i_vals, j_vals, indexing='xy')

        delta_i = I * self.pixel_size
        delta_j = J * self.pixel_size
        denom = np.sqrt(self.D**2 + delta_i**2 + delta_j**2)
        a = 2.0 * np.pi / self.wl

        self.Qx = (a / denom) * delta_i
        self.Qz = (a / denom) * delta_j
        self.Qy = (a / denom) * (self.D - denom)

        self.qvecs = np.stack([self.Qx.ravel(), self.Qy.ravel(), self.Qz.ravel()], axis=1)

        Q_magnitude = np.linalg.norm(self.qvecs, axis=1)
        self.q_min = float(Q_magnitude.min())
        self.q_max = float(Q_magnitude.max())

        if verbose:
            print("----------------------------------------------------")
            print(" Detector configuration / accessible Q-range")
            print("----------------------------------------------------")
            print(f" Wavelength λ = {self.wl:.4f} Å")
            print(f" Sample-detector distance = {self.D*1e3:.2f} mm")
            print(f" Pixel size = {self.pixel_size*1e3:.3f} mm")
            print(f" Number of pixels = {self.npix} x {self.npix}")
            print(f" Qx range : {self.Qx.min():.4f} → {self.Qx.max():.4f} Å⁻¹")
            print(f" Qy range : {self.Qy.min():.4f} → {self.Qy.max():.4f} Å⁻¹")
            print(f" Qz range : {self.Qz.min():.4f} → {self.Qz.max():.4f} Å⁻¹")
            print(f" |Q| range : {self.q_min:.4f} → {self.q_max:.4f} Å⁻¹")
            print("----------------------------------------------------\n")
            





# -----------------------------------------------------
# Main class: nanoparticle scattering in 2D detector plane
# -----------------------------------------------------
class Debye3D(Experiment):
    def __init__(self, structure_file,
                 npix=250, wl=1.0, distance=0.5,
                 pixel_size=0.0001, verbose=True):

        # Initialise les attributs expérimentaux
        super().__init__(npix=npix, wl=wl, distance=distance,
                         pixel_size=pixel_size, verbose=verbose)

        # Read structure and basic setup
        self.atoms = read(structure_file)
        self.file = structure_file
        self.positions = self.atoms.get_positions()
        self.elements = self.atoms.get_chemical_symbols()
        self.nb_atoms = len(self.positions)
        self.xdb = XrayDB()
        if verbose:
            print(f'\n Structure contains {self.nb_atoms} atoms.\n')
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

        Q = self.qvecs.reshape(-1, 3) # each line is a (Q,xQy,Qz vector)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)

        # Call the Numba-parallel routine
        I = compute_intensity_numba(positions, f_atom, Q)
        return I # (npix*npix) each value is I(Qx,Qy,Qz)

    

    
    def compute_isotropic_intensity_fibonacci(self, q_vals=None, npoints=1000):
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
            q_vals = np.linspace(self.q_min, self.q_max, 1000)

        f_q = self.xray_f0(self.elements[0], q_vals)
        dirs = fibonacci_sphere(npoints)
        Iq = compute_intensity_fibonacci(self.positions, f_q, q_vals, dirs)
        return q_vals, Iq

     

    def compute_intensity_wl_dispersion(self,
     positions=None,
     dlambda_over_lambda=0.01,
     nsamples=5):
        """
        Compute 2D intensity map including wavelength dispersion (vectorized version).

        Parameters
        ----------
        positions : (N_atoms,3) array, optional
            If None, use self.positions.
        dlambda_over_lambda : float
            Relative wavelength spread Δλ/λ (default 0.01 = 1%).
        nsamples : int
            Number of Monte-Carlo samples per pixel (default 5).

        Returns
        -------
        I_map : (npix, npix) array
            2D intensity map including wavelength dispersion.
        """
        if positions is None:
            positions = self.positions

        Q = self.qvecs.reshape(-1, 3)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)

        # Appel à la fonction Numba vectorisée
        I = _compute_intensity_numba_dispersion_vectorized(positions, f_atom, Q,
                                                        dlambda_over_lambda, nsamples)
        return I.reshape(self.Qx.shape)

    # -----------------------------
    # Plotting helpers
    # -----------------------------
    def plot_intensity(self, I_flat, log=True, vmin=-6, vmax=0, qmax=None, interpolation='bicubic'):
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
        I_map = I_flat.reshape(self.Qx.shape) # reshape I_flat to match detector shape
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

    def integrate_with_pyfai(self, I_flat, plot=False):
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
        I = I_flat.reshape(self.Qx.shape)
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

    def compute_intensity_uniaxial_ODF(self,
                                      n_samples=200,
                                      sigma_y=2,
                                      sigma_z=2,
                                      seed=None,
                                      verbose=True):
        """
        Compute the mean scattered intensity for a population of particles
        oriented uniaxially along the X-axis, with a Gaussian angular dispersion
        around the Y and Z axes (Monte Carlo + Numba parallel implementation).

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo orientations to sample.
        sigma_y, sigma_z : float
            Standard deviations (in radians) of the rotations around Y and Z.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            If True, display progress information.

        Returns
        -------
        I_map_mean : ndarray of shape (npix, npix)
            2D averaged intensity map on the detector.
        """
        import math
        from tqdm import tqdm
        rng = np.random.default_rng(seed)
        sigma_y=np.deg2rad(sigma_y)
        sigma_z=np.deg2rad(sigma_z)
        positions = self.positions.astype(np.float64)
        Q = self.qvecs.astype(np.float64)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags).astype(np.float64)

        # Draw random rotation angles (Gaussian distribution)
        theta_y = rng.normal(0, sigma_y, n_samples)
        theta_z = rng.normal(0, sigma_z, n_samples)

        # Convert to float64 for Numba compatibility
        theta_y = theta_y.astype(np.float64)
        theta_z = theta_z.astype(np.float64)

        # Call the compiled Numba function for parallel computation
        I_mean = _compute_intensity_uniaxial_numba(positions, f_atom, Q,
                                                   theta_y, theta_z,
                                                   verbose)
        return I_mean.reshape(self.Qx.shape)
    # ------------------------------------------------
    # Structure factor computatio
    # ------------------------------------------------
    def compute_structure_factor(self,N,Z):
        return N*Z**2*self.compute_intensity()
