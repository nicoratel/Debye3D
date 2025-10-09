import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from xraydb import XrayDB
import os
import psutil
from tqdm import tqdm
from scipy.interpolate import interp1d
from numba import njit, prange

# -----------------------------------------------------
# Fonction JIT Numba pour le calcul de l'intensité
# -----------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_intensity_numba(positions, f_atom, Q):
    """
    Calcul intensité |Σ exp(i·Q·r)|² * f(Q)² pour chaque vecteur Q.
    Version optimisée avec Numba (parallélisme interne).
    """
    n_q = Q.shape[0]
    n_atoms = positions.shape[0]
    I = np.empty(n_q, dtype=np.float64)

    for iq in prange(n_q):
        qx, qy, qz = Q[iq]
        Re, Im = 0.0, 0.0
        for ia in range(n_atoms):
            phase = qx*positions[ia,0] + qy*positions[ia,1] + qz*positions[ia,2]
            Re += np.cos(phase)
            Im += np.sin(phase)
        I[iq] = (Re**2 + Im**2) * (f_atom[iq]**2)
    return I


# -----------------------------------------------------
# Classe principale de diffusion nanoparticulaire
# -----------------------------------------------------
class NanoparticleScattering2D:
    def __init__(self, structure_file, npix=250,
                 wl=1.0, Distance=0.5, pixel_size=0.0001):
        self.atoms = read(structure_file)
        self.positions = self.atoms.get_positions()
        self.elements = self.atoms.get_chemical_symbols()
        self.nb_atoms = len(self.positions)
        self.npix = npix
        self.wl = wl
        self.D = Distance
        self.pixel_size = pixel_size
        self.xdb = XrayDB()

        i_vals = np.arange(-npix//2, npix//2)
        j_vals = np.arange(-npix//2, npix//2)
        I, J = np.meshgrid(i_vals, j_vals)

        delta_i = I * pixel_size
        delta_j = J * pixel_size
        denom = np.sqrt(Distance**2 + delta_i**2 + delta_j**2)
        a = 2 * np.pi / wl

        self.Qx = (a / denom) * delta_i
        self.Qz = (a / denom) * delta_j
        self.Qy = (a / denom) * (Distance - denom)

        self.qvecs = np.stack([self.Qx.ravel(), self.Qy.ravel(), self.Qz.ravel()], axis=1)

        Q_magnitude = np.linalg.norm(self.qvecs, axis=1)
        self.q_min = Q_magnitude.min()
        self.q_max = Q_magnitude.max()

        print("----------------------------------------------------")
        print(" Configuration du détecteur / gamme Q accessible")
        print("----------------------------------------------------")
        print(f" Longueur d'onde λ = {wl:.4f} Å")
        print(f" Distance échantillon-détecteur = {Distance*1000:.2f} mm")
        print(f" Taille pixel = {pixel_size*1e3:.3f} mm")
        print(f" Nombre de pixels = {npix} x {npix}")
        print("")
        print(f" Gamme Qx : {self.Qx.min():.4f} → {self.Qx.max():.4f} Å⁻¹")
        print(f" Gamme Qy : {self.Qy.min():.4f} → {self.Qy.max():.4f} Å⁻¹")
        print(f" Gamme Qz : {self.Qz.min():.4f} → {self.Qz.max():.4f} Å⁻¹")
        print(f" Module |Q| : {self.q_min:.4f} → {self.q_max:.4f} Å⁻¹")
        print("----------------------------------------------------\n")
        # 🔥 Amorçage Numba pour éviter la compilation lente au premier calcul
        self._warmup_numba()

    def _warmup_numba(self):
        """Exécute un petit calcul factice pour compiler Numba à l'avance."""
        dummy_pos = np.zeros((2, 3), dtype=np.float64)
        dummy_q = np.zeros((2, 3), dtype=np.float64)
        dummy_f = np.ones(2, dtype=np.float64)
        _ = compute_intensity_numba(dummy_pos, dummy_f, dummy_q)
        

    # -----------------------------
    # Facteur atomique
    # -----------------------------
    def xray_f0(self, element, q):
        return self.xdb.f0(element, q)

    # -----------------------------
    # Calcul intensité (optimisé Numba)
    # -----------------------------
    def compute_intensity(self, positions=None):
        if positions is None:
            positions = self.positions

        
        Q = self.qvecs.reshape(-1, 3)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)

        # Calcul JIT parallèle
        I = compute_intensity_numba(positions, f_atom, Q)
        return I.reshape(self.Qx.shape)

    # -----------------------------
    # Moyenne radiale isotrope
    # -----------------------------
    def make_isotropic(self, I_aniso):
        ny, nx = I_aniso.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.indices(I_aniso.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_int = r.astype(int)
        sums = np.bincount(r_int.ravel(), weights=I_aniso.ravel())
        counts = np.bincount(r_int.ravel())
        I_iso_1d = sums / counts
        I_iso = I_iso_1d[r_int]
        return I_iso

    # -----------------------------
    # Affichage
    # -----------------------------
    def plot_intensity_old(self, I_map, log=True, vmin=-4, vmax=0, qmax=None):
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        plt.figure(figsize=(6,5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
        else:
            norm = None

        if qmax is None:
            extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent=[self.Qx.min(), qmax, self.Qz.min(), qmax]

        plt.imshow(
            I_map,
            extent=extent,
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation='bicubic'
        )
        plt.colorbar(label="Intensité (log)" if log else "Intensité")
        plt.xlabel("$q_x$ (Å⁻¹)")
        plt.ylabel("$q_z$ (Å⁻¹)")
        plt.show()

    def plot_intensity(self, I_map, log=True, vmin=-4, vmax=0, qmax=None):
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        fig, ax = plt.subplots(figsize=(6,5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
        else:
            norm = None

        if qmax is None:
            extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent=[self.Qx.min(), qmax, self.Qz.min(), qmax]

        im = ax.imshow(
            I_map,
            extent=extent,
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation='bicubic'
        )

        plt.colorbar(im, ax=ax, label="Intensité (log)" if log else "Intensité")
        ax.set_xlabel("$q_x$ (Å⁻¹)")
        ax.set_ylabel("$q_z$ (Å⁻¹)")
        ax.set_title("Carte d’intensité simulée (Numba optimisée)")

        # ---- Ajout : afficher coordonnées polaires dans la barre de statut ----
        def format_coord(x, y):
            r = np.sqrt(x**2 + y**2)
            theta = np.degrees(np.arctan2(y, x))
            if r > 0:
                d = 2 * np.pi / (10 * r)
                return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, θ={theta:.1f}°"
            else:
                return f"q={r:.4f} Å⁻¹, θ={theta:.1f}°"
        ax.format_coord = format_coord

        plt.show()


    # -----------------------------
    # Rotation selon Euler
    # -----------------------------
    @staticmethod
    def euler_rotation_matrix(alpha, beta, gamma):
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)
        ca, cb, cg = np.cos([alpha, beta, gamma])
        sa, sb, sg = np.sin([alpha, beta, gamma])
        Rz_alpha = np.array([[ca, -sa, 0],
                             [sa,  ca, 0],
                             [ 0,   0, 1]])
        Ry_beta = np.array([[ cb, 0, sb],
                            [  0, 1,  0],
                            [-sb, 0, cb]])
        Rz_gamma = np.array([[cg, -sg, 0],
                             [sg,  cg, 0],
                             [ 0,   0, 1]])
        return Rz_alpha @ Ry_beta @ Rz_gamma

    def rotate_positions(self, alpha, beta, gamma):
        R = self.euler_rotation_matrix(alpha, beta, gamma)
        return self.positions @ R.T

    # -----------------------------
    # Intégration radiale directe
    # -----------------------------
    def integrate_intensity(self, I_map, npt=1000, q_range=None, plot=False):
        Qx_flat = self.Qx.ravel()
        Qz_flat = self.Qz.ravel()
        I_flat = I_map.ravel()

        q = np.sqrt(Qx_flat**2 + Qz_flat**2)

        if q_range is None:
            q_min, q_max = q.min(), q.max()
        else:
            q_min, q_max = q_range

        q_bins = np.linspace(q_min, q_max, npt+1)
        q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
        Iq = np.zeros_like(q_centers)

        inds = np.digitize(q, q_bins)
        for i in range(1, len(q_bins)):
            mask = inds == i
            if np.any(mask):
                Iq[i-1] = np.mean(I_flat[mask])

        if plot:
            plt.figure()
            plt.loglog(q_centers, Iq, 'k-')
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("I(Q) (a.u.)")
            plt.xlim(self.q_min, self.q_max)
            plt.title("Intégration radiale directe dans l’espace Q")
            plt.grid(True)
            plt.show()

        return q_centers, Iq
