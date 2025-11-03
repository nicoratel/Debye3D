import math
import time
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read,write
from xraydb import XrayDB
from tqdm import tqdm
from ase import Atoms
import psutil
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
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

# --- Torch (GPU/CPU backend) ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# --- Numba (CPU-parallel backend) ---
try:
    from numba import njit, prange
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False


# ===============================================================
# Utility: fibonacci sphere directions
# ===============================================================
def fibonacci_sphere(n_orient):
    indices = np.arange(n_orient)
    phi = (1 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - 2.0 * (indices + 0.5) / n_orient
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return dirs


# ===============================================================
# Experiment
# ===============================================================
class Experiment:
    def __init__(self, npix=250, wl=1.0, distance=0.5, pixel_size=0.0001, verbose=True):
        self.npix = int(npix)
        self.wl = float(wl)
        self.distance = float(distance)
        self.pixel_size = float(pixel_size)
        self.D = self.distance

        # Detector grid
        i_vals = np.arange(-self.npix // 2, self.npix // 2)
        j_vals = np.arange(-self.npix // 2, self.npix // 2)
        I, J = np.meshgrid(i_vals, j_vals, indexing="xy")
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
            print(f" |Q| range : {self.q_min:.4f} → {self.q_max:.4f} Å⁻¹")
            print("----------------------------------------------------\n")


# ===============================================================
# Debye3D hybrid class
# ===============================================================
class Debye3D(Experiment):
    def __init__(self, structure_file, npix=250, wl=1.0, distance=0.5, pixel_size=0.0001,
                 verbose=True, torch_device=None):
        super().__init__(npix=npix, wl=wl, distance=distance, pixel_size=pixel_size, verbose=verbose)
        self.atoms = read(structure_file)
        self.file = structure_file
        self.positions = self.atoms.get_positions()
        self.elements = self.atoms.get_chemical_symbols()
        self.nb_atoms = len(self.positions)
        self.xdb = XrayDB()
        if verbose:
            print(f"\n Structure contains {self.nb_atoms} atoms.\n")

        # Torch device
        if TORCH_AVAILABLE:
            if torch_device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(torch_device)
        else:
            self.device = None

        # Warm-up
        if self.device is not None and self.device.type == "cuda":
            _ = torch.zeros(1, device=self.device)
            del _
            print(f"[Debye3D] Use of GPU: {torch.cuda.get_device_name(self.device)}")

    
    # ==============================================================
    # Utility: view_structure
    # ==============================================================
    def view_structure(self):
        """
        Write a temporary XYZ file and launch an external viewer (jmol).

        Notes
        -----
        This function calls an external 'jmol' binary using os.system and
        requires jmol to be installed and accessible in the PATH.
        """
        self.save_structure_as_xyz('./file.xyz')
        os.system('jmol file.xyz')
        os.remove('file.xyz')

    def save_structure_as_xyz(self, filename):
        natoms=self.positions.shape[0]
        line2write=f'{natoms}\n\n'
        for i in range(len(self.elements)):
            line2write+=f'{self.elements[i]}\t{self.positions[i,0]:.8f}\t{self.positions[i,1]:.8f}\t{self.positions[i,2]:.8f}\n'
        with open(filename,'w') as f:
            f.write(line2write)

    def update_structure(self,coords,element):
        """ 
        Update strcutre associated to the classe given a list of coordinates and a single element
        Parameters
        ----------
        coords: tuple, ndarray
            List of atomic coordinates
        element: string
        """
        self.positions = coords
        self.nb_atoms= self.positions.shape[0]
        self.elements = np.full(self.nb_atoms,element)


    
    # ===============================================================
    # Auto batch size based on available VRAM
    # ===============================================================
    def auto_batch_size(self, target_fraction=0.8, reference=200_000, reference_mem_gb=8.0):
        """
        Automatic batch size determination adapted to available VRAM size.
        """
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            if self.device:
                print(f"[auto_batch_size] No detected GPU {self.device}. Default_batch_size = 200_000")
            return 200_000

        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9
        batch_size = int(reference * (free_gb / reference_mem_gb) * target_fraction)
        print(f"[auto_batch_size] free VRAM : {free_gb:.1f} GB → batch ≈ {batch_size}")
        return max(50_000, min(batch_size, 1_000_000))

    # ===============================================================
    # Atomic form factor
    # ===============================================================
    def xray_f0(self, element, q):
        return self.xdb.f0(element, q)

    # ===============================================================
    # GPU version (PyTorch)
    # ===============================================================
    def _to_torch(self, arr, dtype=torch.float32):
        if not TORCH_AVAILABLE or self.device is None:
            raise RuntimeError("PyTorch is required for GPU mode.")
        return torch.from_numpy(np.asarray(arr)).to(device=self.device, dtype=dtype)

       # ===============================================================
    # Intensity calculation
    # ===============================================================
    def compute_intensity(self, use_gpu=True, batch_size=None):
        qvecs = self.qvecs
        q_mags = np.linalg.norm(qvecs, axis=1)
        f_atom = self.xdb.f0(self.elements[0], q_mags)

        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            if batch_size is None:
                batch_size = self.auto_batch_size()
            return self._compute_intensity_torch(batch_size, f_atom)
        else:
            return compute_intensity_numba(self.positions, f_atom, qvecs)

    # ===============================================================
    # GPU (PyTorch)
    # ===============================================================
    def _compute_intensity_torch(self, batch_size=None, f_atom=None, atom_chunk=None, verbose=True):
        """
        High-performance GPU version for computing X-ray scattering intensity.
        Automatically adjusts batch and atom chunk sizes to prevent CUDA OOM errors.

        Parameters
        ----------
        batch_size : int or None
            Number of q-vectors processed per batch. Auto-scaled if None.
        f_atom : array-like or None
            Atomic scattering factors (precomputed). If None, computed internally.
        atom_chunk : int or None
            Number of atoms processed per sub-batch. Auto-scaled if None.
        verbose : bool
            If True, prints progress and GPU memory diagnostics.
        """

        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            raise RuntimeError("GPU not available or PyTorch not compiled with CUDA.")

        t0 = time.time()
        device = self.device
        Q = self.qvecs.astype(np.float32)
        Nq = Q.shape[0]
        Nat = self.nb_atoms

        # --- GPU memory diagnostics ---
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
        except RuntimeError:
            raise RuntimeError("Unable to query GPU memory. Check NVML / driver installation.")
        free_gb = free_mem / 1e9

        # --- adaptive batch and chunk sizes ---
        if batch_size is None:
            batch_size = int(200_000 * (free_gb / 40.0))
            batch_size = max(20_000, min(batch_size, 400_000))

        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # --- Prevent excessive matrix sizes (batch_size × atom_chunk)
        # Allow up to 40% of free VRAM for the main phases tensor
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_size * atom_chunk > max_elements:
            scale = (max_elements / (batch_size * atom_chunk)) ** 0.5
            batch_size = int(batch_size * scale)
            atom_chunk = int(atom_chunk * scale)
            if verbose:
                print(f"[⚠️] Adjusted batch sizes to avoid OOM: "
                    f"batch_q={batch_size}, chunk_atoms={atom_chunk}")

        if verbose:
            print(f"[Debye3D GPU] {Nat} atoms, {Nq} q-vectors")
            print(f" → batch_q = {batch_size}, chunk_atoms = {atom_chunk}, free VRAM ≈ {free_gb:.1f} GB")

        # --- Move data to GPU ---
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)

        if f_atom is None:
            q_mags = np.linalg.norm(Q, axis=1)
            f_atom = self.xdb.f0(self.elements[0], q_mags)
        f_atom_t = torch.tensor(f_atom, dtype=torch.float32, device=device)

        I_acc = torch.zeros(Nq, dtype=torch.float32, device=device)

        # --- Main computation ---
        with torch.no_grad():
            try:
                for q_start in tqdm(range(0, Nq, batch_size), desc="Computing intensity", disable=not verbose):
                    q_stop = min(Nq, q_start + batch_size)
                    Qbatch = torch.tensor(Q[q_start:q_stop], dtype=torch.float32, device=device)

                    Re_acc = torch.zeros(Qbatch.shape[0], dtype=torch.float32, device=device)
                    Im_acc = torch.zeros_like(Re_acc)

                    for a_start in range(0, Nat, atom_chunk):
                        a_stop = min(Nat, a_start + atom_chunk)
                        pos_chunk = positions_t[a_start:a_stop]

                        # Large dot product: (batch_q × atom_chunk)
                        phases = torch.matmul(Qbatch, pos_chunk.T)
                        Re_acc += torch.cos(phases).sum(dim=1)
                        Im_acc += torch.sin(phases).sum(dim=1)

                        del pos_chunk, phases
                        torch.cuda.empty_cache()

                    I_batch = (Re_acc**2 + Im_acc**2) * (f_atom_t[q_start:q_stop] ** 2)
                    I_acc[q_start:q_stop] = I_batch

                    del Qbatch, Re_acc, Im_acc, I_batch
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if verbose:
                        print("[⚠️] CUDA OOM detected — reducing batch size and retrying on CPU.")
                    torch.cuda.empty_cache()
                    return self._compute_intensity_cpu(verbose=verbose)
                else:
                    raise e

        if verbose:
            dt = time.time() - t0
            print(f"[✓] GPU computation completed in {dt/60:.2f} minutes")

        return I_acc.cpu().numpy()


    # ===============================================================
    # Isotropic (Fibonacci)
    # ===============================================================
    def compute_isotropic_intensity_fibonacci(
        self, n_q=500,
        n_orient=1000,
        use_gpu=True,
        batch_orient=None,
        atom_chunk=None,verbose=True):
        """
        Isotropic intensity computation using Fibonacci sphere sampling.
        Fully GPU-optimized with adaptive double batching (orientation × atom).

        Parameters
        ----------
        n_q : int
            Number of q magnitudes.
        n_orient : int
            Number of orientations sampled on the sphere.
        use_gpu : bool
            Whether to use GPU (if available).
        batch_orient : int or None
            Number of orientations processed per GPU batch (auto-scaled if None).
        atom_chunk : int or None
            Number of atoms processed per sub-batch (auto-scaled if None).
        verbose : bool
            If True, print GPU diagnostics and progress.
        """

        import torch, math, gc, time
        import numpy as np
        from tqdm import tqdm

        q_vals = np.linspace(self.q_min, self.q_max, n_q)
        f_q = self.xdb.f0(self.elements[0], q_vals)
        dirs = fibonacci_sphere(n_orient)
        Nat = self.nb_atoms

        if not (use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            if verbose:
                print("→ Isotropic average (CPU) over orientations")
            return q_vals, compute_intensity_fibonacci_numba(self.positions, f_q, q_vals, dirs)

        # --- GPU memory info ---
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9

        # --- Adaptive batching ---
        if batch_orient is None:
            batch_orient = int(10_000 * (free_gb / 40.0))
            batch_orient = max(500, min(batch_orient, 50_000))
        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # --- Cap to avoid excessive tensor sizes (batch_orient × atom_chunk) ---
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_orient * atom_chunk > max_elements:
            scale = (max_elements / (batch_orient * atom_chunk)) ** 0.5
            batch_orient = int(batch_orient * scale)
            atom_chunk = int(atom_chunk * scale)
            if verbose:
                print(f"[⚠️] Adjusted batches to fit VRAM: orient={batch_orient}, atoms={atom_chunk}")

        if verbose:
            print(f"[Debye3D GPU - Isotropic] {Nat} atoms, {n_orient} orientations, {n_q} q-points")
            print(f" → batch_orient = {batch_orient}, chunk_atoms = {atom_chunk}, free ≈ {free_gb:.1f} GB")

        device = self.device
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)
        dirs_t = torch.tensor(dirs.astype(np.float32), dtype=torch.float32, device=device)
        q_vals_t = torch.tensor(q_vals.astype(np.float32), dtype=torch.float32, device=device)

        Iq = torch.zeros_like(q_vals_t)
        t0 = time.time()

        with torch.no_grad():
            for iq in tqdm(range(n_q), disable=not verbose):
                q = q_vals_t[iq]
                f = f_q[iq]
                Q_dirs = q * dirs_t

                I_partial_sum = 0.0
                count = 0

                # --- batch over orientations ---
                for o_start in range(0, n_orient, batch_orient):
                    o_stop = min(n_orient, o_start + batch_orient)
                    Q_batch = Q_dirs[o_start:o_stop]

                    Re_acc = torch.zeros(Q_batch.shape[0], device=device)
                    Im_acc = torch.zeros_like(Re_acc)

                    # --- sub-batch over atoms ---
                    for a_start in range(0, Nat, atom_chunk):
                        a_stop = min(Nat, a_start + atom_chunk)
                        pos_chunk = positions_t[a_start:a_stop]
                        phases = torch.matmul(Q_batch, pos_chunk.T)
                        Re_acc += torch.cos(phases).sum(dim=1)
                        Im_acc += torch.sin(phases).sum(dim=1)

                        del pos_chunk, phases
                        torch.cuda.empty_cache()

                    I_batch = (Re_acc**2 + Im_acc**2)
                    I_partial_sum += I_batch.sum().item()
                    count += I_batch.shape[0]

                    del Q_batch, Re_acc, Im_acc, I_batch
                    torch.cuda.empty_cache()
                    gc.collect()

                I_mean = (I_partial_sum / count) * (f**2)
                Iq[iq] = I_mean

        if verbose:
            dt = time.time() - t0
            print(f"[✓] Completed isotropic average in {dt/60:.2f} minutes")

        return q_vals, Iq.cpu().numpy()


    # ===============================================================
    # Uniaxial
    # ===============================================================
   
    def compute_intensity_uniaxial_ODF(
        self,n_samples=200,
        sigma_y=2.0,sigma_z=2.0,
        use_gpu=True,
        batch_q=None,atom_chunk=None,
        verbose=True):
        """
        Uniaxial ODF intensity computation with full double batching (q × atom).
        Prevents CUDA OOM by adaptively scaling both batch dimensions.

        Parameters
        ----------
        n_samples : int
            Number of random orientation samples.
        sigma_y, sigma_z : float
            Angular standard deviations (degrees).
        use_gpu : bool
            Whether to use GPU acceleration.
        batch_q : int or None
            Number of q-vectors per batch (auto-scaled if None).
        atom_chunk : int or None
            Number of atoms per sub-batch (auto-scaled if None).
        verbose : bool
            If True, print progress and GPU diagnostics.
        """

        import torch, math, gc, time
        import numpy as np
        from tqdm import tqdm

        sigma_y_rad = np.deg2rad(sigma_y)
        sigma_z_rad = np.deg2rad(sigma_z)
        rng = np.random.default_rng()
        theta_y = rng.normal(0, sigma_y_rad, n_samples)
        theta_z = rng.normal(0, sigma_z_rad, n_samples)

        qvecs = self.qvecs
        q_mags = np.linalg.norm(qvecs, axis=1)
        f_atom = self.xdb.f0(self.elements[0], q_mags)
        Nat = self.nb_atoms

        if not (use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            if verbose:
                print("→ Uniaxial average (CPU)")
            return compute_intensity_uniaxial_numba(self.positions, f_atom, qvecs, theta_y, theta_z)

        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9

        if batch_q is None:
            batch_q = int(100_000 * (free_gb / 40.0))
            batch_q = max(10_000, min(batch_q, 400_000))
        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # Avoid huge tensors (batch_q × atom_chunk)
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_q * atom_chunk > max_elements:
            scale = (max_elements / (batch_q * atom_chunk)) ** 0.5
            batch_q = int(batch_q * scale)
            atom_chunk = int(atom_chunk * scale)
            if verbose:
                print(f"[⚠️] Adjusted batches to fit VRAM: q={batch_q}, atoms={atom_chunk}")

        if verbose:
            print(f"[Debye3D GPU - Uniaxial] {Nat} atoms, {len(qvecs)} q-points, {n_samples} orientations")
            print(f" → batch_q = {batch_q}, chunk_atoms = {atom_chunk}, free ≈ {free_gb:.1f} GB")

        device = self.device
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)
        Q_t_full = torch.tensor(qvecs.astype(np.float32), dtype=torch.float32, device=device)
        f_t_full = torch.tensor(f_atom.astype(np.float32), dtype=torch.float32, device=device)
        Nq = Q_t_full.shape[0]
        I_acc = torch.zeros(Nq, dtype=torch.float32, device=device)

        t0 = time.time()
        with torch.no_grad():
            for i in tqdm(range(n_samples), disable=not verbose):
                ty, tz = float(theta_y[i]), float(theta_z[i])
                cy, sy = math.cos(ty), math.sin(ty)
                cz, sz = math.cos(tz), math.sin(tz)
                R = torch.tensor(
                    [
                        [cz * cy, -sz, cz * sy],
                        [sz * cy,  cz, sz * sy],
                        [-sy,      0.,     cy],
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                # Rotate positions once per sample
                pos_rot = positions_t @ R.T

                # --- batch over q-vectors ---
                for q_start in range(0, Nq, batch_q):
                    q_stop = min(Nq, q_start + batch_q)
                    Q_batch = Q_t_full[q_start:q_stop]
                    f_batch = f_t_full[q_start:q_stop]

                    Re_acc = torch.zeros(Q_batch.shape[0], device=device)
                    Im_acc = torch.zeros_like(Re_acc)

                    # --- sub-batch over atoms ---
                    for a_start in range(0, Nat, atom_chunk):
                        a_stop = min(Nat, a_start + atom_chunk)
                        pos_chunk = pos_rot[a_start:a_stop]
                        phases = torch.matmul(Q_batch, pos_chunk.T)
                        Re_acc += torch.cos(phases).sum(dim=1)
                        Im_acc += torch.sin(phases).sum(dim=1)

                        del pos_chunk, phases
                        torch.cuda.empty_cache()

                    I_batch = (Re_acc**2 + Im_acc**2) * (f_batch**2)
                    I_acc[q_start:q_stop] += I_batch

                    del Q_batch, f_batch, Re_acc, Im_acc, I_batch
                    torch.cuda.empty_cache()
                    gc.collect()

        I_mean = I_acc / float(n_samples)
        if verbose:
            dt = time.time() - t0
            print(f"[✓] Completed uniaxial average in {dt/60:.2f} minutes")

        return I_mean.cpu().numpy()

    # ===============================================================
    # Miscellaneous
    # ===============================================================

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

    # ===============================================================
    # Plot
    # ===============================================================
    def plot_intensity(self, I_flat, log=True, vmin=-6, vmax=0, qmax=None, interpolation='bicubic',filename = None):
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
        filename: str, optional
            Full destination path to save the plot

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

        if filename:
            plt.savefig(filename)

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

    def shake_positions(self, frac_a, frac_b, frac_c, reference_length=None, seed=None):
        """
        Apply small random displacements ("shake") to atomic positions,
        using fractions of a characteristic interatomic distance.

        Parameters
        ----------
        frac_a, frac_b, frac_c : float
            Fractional amplitudes of random displacement along x, y, z.
            Each atom is shifted by a random amount within:
            [-frac_a * ref, +frac_a * ref] along x,
            [-frac_b * ref, +frac_b * ref] along y,
            [-frac_c * ref, +frac_c * ref] along z.
        reference_length : float, optional
            Characteristic distance used as reference (e.g. lattice constant).
            If None, it is estimated as the mean nearest-neighbor distance.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        shaken_positions : numpy.ndarray, shape (N_atoms, 3)
            The shaken (disordered) atomic positions (also stored in self.positions).
        """
        import numpy as np
        from scipy.spatial import cKDTree

        if seed is not None:
            np.random.seed(seed)

        coords = self.positions

        # Estimate reference length if not given: mean nearest-neighbor distance
        if reference_length is None:
            tree = cKDTree(coords)
            dists, _ = tree.query(coords, k=2)  # k=1 is self (distance 0), k=2 is nearest neighbor
            reference_length = np.mean(dists[:, 1])
        print(f'Reference length is {reference_length:.2f}')
        # Compute maximum displacements per axis
        dx_max = frac_a * reference_length
        dy_max = frac_b * reference_length
        dz_max = frac_c * reference_length

        # Random displacements for each atom
        dx = np.random.uniform(-dx_max, dx_max, size=len(coords))
        dy = np.random.uniform(-dy_max, dy_max, size=len(coords))
        dz = np.random.uniform(-dz_max, dz_max, size=len(coords))

        # Apply displacements
        displacements = np.column_stack((dx, dy, dz))
        self.positions = coords + displacements

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

    # ===========================================================
    # Structure factor computation
    # -==========================================================
    def compute_structure_factor(self,N,Z,use_gpu=True):
        """ 
        N:int
            number of atoms in particle
        Z: int
            atomic number of atoms in particle        
        """
        return N*Z**2*self.compute_intensity(use_gpu=use_gpu)


# ===============================================================
# Numba core
# ===============================================================

if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)
    def compute_intensity_numba(positions, f_atom, Q):
        n_q = Q.shape[0]
        n_atoms = positions.shape[0]
        I = np.empty(n_q, dtype=np.float64)
        for iq in prange(n_q):
            qx, qy, qz = Q[iq]
            Re, Im = 0.0, 0.0
            for ia in range(n_atoms):
                phase = qx * positions[ia, 0] + qy * positions[ia, 1] + qz * positions[ia, 2]
                Re += math.cos(phase)
                Im += math.sin(phase)
            I[iq] = (Re ** 2 + Im ** 2) * (f_atom[iq] ** 2)
        return I

    @njit(parallel=True, fastmath=True)
    def compute_intensity_fibonacci_numba(positions, f_q, q_vals, dirs):
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
                    Re += math.cos(phase)
                    Im += math.sin(phase)
                I_sum += (Re ** 2 + Im ** 2)
            Iq[iq] = (I_sum / n_orient) * (f ** 2)
        return Iq

    @njit(parallel=True, fastmath=True)
    def compute_intensity_uniaxial_numba(positions, f_atom, Q, theta_y, theta_z):
        n_samples = len(theta_y)
        n_q = Q.shape[0]
        n_atoms = positions.shape[0]
        I_accum = np.zeros(n_q, dtype=np.float64)

        for isamp in prange(n_samples):
            ty = theta_y[isamp]
            tz = theta_z[isamp]
            cy, sy = math.cos(ty), math.sin(ty)
            cz, sz = math.cos(tz), math.sin(tz)
            R = np.array([
                [cz * cy, -sz, cz * sy],
                [sz * cy,  cz, sz * sy],
                [-sy,      0.0, cy]
            ])
            pos_rot = np.zeros_like(positions)
            for i in range(n_atoms):
                x, y, z = positions[i]
                pos_rot[i, 0] = R[0, 0]*x + R[0, 1]*y + R[0, 2]*z
                pos_rot[i, 1] = R[1, 0]*x + R[1, 1]*y + R[1, 2]*z
                pos_rot[i, 2] = R[2, 0]*x + R[2, 1]*y + R[2, 2]*z

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


# ===============================================================
# Utility: Fibonacci sphere
# ===============================================================
def fibonacci_sphere(n_orient):
    indices = np.arange(n_orient)
    phi = (1 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - 2.0 * (indices + 0.5) / n_orient
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return dirs


# ===============================================================
# Script entry point
# ===============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debye3D hybrid CPU/GPU with orientation averaging")
    parser.add_argument("structure", help="Structure file (XYZ, CIF, ... supported by ASE)")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cuda:0 or cpu)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--npix", type=int, default=250, help="Detector pixel count")
    parser.add_argument("--wl", type=float, default=1.0, help="Wavelength (Å)")
    parser.add_argument("--distance", type=float, default=0.5, help="Sample-detector distance (m)")
    parser.add_argument("--pixel", type=float, default=0.0001, help="Pixel size (m)")
    parser.add_argument("--fib", type=int, default=0, help="Average over N orientations using Fibonacci grid (0 = disabled)")
    parser.add_argument("--uniax", type=int, default=0, help="Average over N uniaxial orientations (0 = disabled)")
    parser.add_argument("--sigma_y", type=float, default=2.0, help="σy for uniaxial distribution (deg)")
    parser.add_argument("--sigma_z", type=float, default=2.0, help="σz for uniaxial distribution (deg)")
    args = parser.parse_args()

    # 1️⃣ Initialisation de l’expérience
    exp = Experiment(npix=args.npix, wl=args.wl, distance=args.distance,
                     pixel_size=args.pixel, verbose=True)

    # 2️⃣ Création du modèle Debye3D
    model = Debye3D(structure_file=args.structure,
                    npix=exp.npix,
                    wl=exp.wl,
                    distance=exp.distance,
                    pixel_size=exp.pixel_size,
                    verbose=True,
                    torch_device=args.device)

    print("\n==============================================================")
    print(f" Using device : {model.device}")
    print(f" GPU available : {torch.cuda.is_available() if TORCH_AVAILABLE else False}")
    print("==============================================================\n")

    # 3️⃣ Choix automatique du batch
    batch = model.auto_batch_size()

    # 4️⃣ Calcul de l’intensité
    if args.fib > 0:
        print(f"\n>>> Calcul orientationnel (Fibonacci, {args.fib} directions)...\n")
        I = model.compute_intensity_fibonacci(n_orient=args.fib, use_gpu=args.use_gpu, batch_size=batch)
    elif args.uniax > 0:
        print(f"\n>>> Calcul uniaxial ({args.uniax} échantillons, σy={args.sigma_y}°, σz={args.sigma_z}°)...\n")
        I = model.compute_intensity_uniaxial(n_samples=args.uniax,
                                             sigma_y=args.sigma_y,
                                             sigma_z=args.sigma_z,
                                             use_gpu=args.use_gpu,
                                             batch_size=batch)
    else:
        print("\n>>> Calcul d’intensité simple (aucune moyenne orientationnelle)...\n")
        I = model.compute_intensity(use_gpu=args.use_gpu, batch_size=batch)

    # 5️⃣ Affichage de la carte d’intensité
    model.plot_intensity(I)
    print("\n✅ Calcul terminé.")
