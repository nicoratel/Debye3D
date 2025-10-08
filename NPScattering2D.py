import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from xraydb import XrayDB
from multiprocessing import Pool, cpu_count
import os
import psutil
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from tqdm import tqdm
from scipy.interpolate import interp1d

# Nombre de cœurs logiques (hyper-threading inclus)
logical_cores = os.cpu_count()

# Nombre de cœurs physiques
physical_cores = psutil.cpu_count(logical=False)

# -----------------------------
# Fonction de traitement vectorisée pour un bloc de Q
# -----------------------------
def process_block_wrapper(args):
    positions, f_block, q_block = args
    # Produit scalaire positions · q_block (résultat : n_atoms x n_q)
    phases = np.exp(1j * positions.dot(q_block.T))
    # Somme sur les atomes (axis=0)
    F = np.sum(phases, axis=0) * f_block
    return np.abs(F)**2


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

    # -----------------------------
    # Facteur atomique
    # -----------------------------
    def xray_f0(self, element, q):
        return self.xdb.f0(element, q)

    # -----------------------------
    # Calcul vectorisé (monoatomique)
    # -----------------------------
    def compute_intensity(self, positions=None):
        if positions is None:
            positions = self.positions
        q_mags = np.linalg.norm(self.qvecs, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)
        phases = np.exp(1j * positions.dot(self.qvecs.T))
        F = np.sum(phases, axis=0) * f_atom
        I_map = np.abs(F)**2
        return I_map.reshape((self.npix, self.npix))

    # -----------------------------
    # Calcul parallèle vectorisé
    # -----------------------------
    def compute_intensity_parallel(self, positions=None, nproc=None, block_size=500):
        if positions is None:
            positions = self.positions
        if nproc is None:
            nproc = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        print(f"→ Utilisation de {nproc} processus")

        Q = self.qvecs.reshape(-1, 3)
        q_mags = np.linalg.norm(Q, axis=1)
        f_atom = self.xray_f0(self.elements[0], q_mags)

        blocks = [Q[i:i+block_size] for i in range(0, len(Q), block_size)]
        f_blocks = [f_atom[i:i+block_size] for i in range(0, len(f_atom), block_size)]
        args = [(positions, f_blk, Q_blk) for f_blk, Q_blk in zip(f_blocks, blocks)]

        with Pool(processes=nproc) as pool:
            results = pool.map(process_block_wrapper, args)

        I = np.concatenate(results).reshape(self.Qx.shape)
        return I

    # -----------------------------
    # Affichage
    # -----------------------------
    def plot_intensity(self, I_map, log=True, vmin=-4, vmax=0):
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        plt.figure(figsize=(6,5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
        else:
            norm = None
        plt.imshow(
            I_map,
            extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()],
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation='bicubic'
        )
        plt.colorbar(label="Intensité (log)" if log else "Intensité")
        plt.xlabel("$q_x$ (Å⁻¹)")
        plt.ylabel("$q_z$ (Å⁻¹)")
        plt.title("Carte d’intensité simulée (monoatomique)")
        plt.show()


    @staticmethod
    def euler_rotation_matrix(alpha, beta, gamma):
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians (gamma)
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
        """
        rotation de la particule suivant les angles d'Euler (ZYZ convention, degrés)
        """
        R = self.euler_rotation_matrix(alpha, beta, gamma)
        #self.positions = self.positions @ R.T
        return self.positions @ R.T

    def plot_intensity(self, I_map, log=True, vmin=-4, vmax=0):
        # Normalisation et sécurité
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)

        plt.figure(figsize=(6,5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
        else:
            norm = None

        plt.imshow(
            I_map,
            extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()],
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation='bicubic'
        )

        plt.colorbar(label="Intensité" + (" (log)" if log else ""))
        plt.xlabel("$q_x$ (Å⁻¹)")
        plt.ylabel("$q_z$ (Å⁻¹)")
        plt.show()

    
    
    def integrate_with_pyFAI(self, I_map, npt=1000, mask=None, method="numpy",plot=True):
        """
        Intègre une image brute simulée avec pyFAI pour obtenir I(Q).
        """
        ai = AzimuthalIntegrator(
            dist=self.D,
            wavelength=self.wl * 1e-10,  # Å → m
            pixel1=self.pixel_size,
            pixel2=self.pixel_size,
            poni1=0.0,  # centre détecteur (optionnel)
            poni2=0.0
        )

        q, Iq = ai.integrate1d(
            I_map,
            npt,
            mask=mask,
            unit="q_A^-1",
            method=method
        )
        if plot:
            plt.figure()
            plt.loglog(q, Iq, 'k-')
            plt.xlim(self.q_min,self.q_max)
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("I(Q) (a.u.)")
            plt.title("Profil I(Q) intégré avec pyFAI (faisceau le long de y)")
            plt.grid(True)
            plt.show()

        return q, Iq

    def integrate_intensity(self, I_map, npt=500, q_range=None,plot=False):
        """
        Intégration radiale directe (I(Qx,Qz) → I(|Q|)) sans pyFAI.

        q_range : (qmin, qmax) en Å⁻¹, sinon auto-détecté
        """
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

        # Binning radial
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
            plt.xlim(self.q_min,self.q_max)
            plt.title("Intégration radiale directe dans l’espace Q")
            plt.grid(True)
            plt.show()

        return q_centers, Iq







  


