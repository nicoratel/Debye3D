import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read,write
from xraydb import XrayDB
import os
import psutil
from tqdm import tqdm
from scipy.interpolate import interp1d
from numba import njit, prange
from debyecalculator import DebyeCalculator
from pyFAI import AzimuthalIntegrator
from pyFAI.detectors import Detector

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
# ------------------------------
# Moyenne orientationelle
# ------------------------------
# Génération des directions de Fibonacci
# ------------------------------
def fibonacci_sphere(n_orient):
    """
    Génère n_orient directions uniformément réparties sur la sphère
    selon la méthode de Fibonacci.
    """
    indices = np.arange(n_orient)
    phi = (1 + np.sqrt(5)) / 2  # nombre d'or
    theta = 2 * np.pi * indices / phi
    z = 1 - 2 * (indices + 0.5) / n_orient
    r = np.sqrt(1 - z * z)
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return dirs  # shape (n_orient, 3)

# ------------------------------
# Calcul Numba parallèle
# ------------------------------
@njit(parallel=True, fastmath=True)
def compute_intensity_fibonacci(positions, f_q, q_vals, dirs):
    """
    Calcule l'intensité isotrope via quadrature de Fibonacci sur la sphère.
    
    positions : (N_atoms, 3)
    f_q       : (N_q,)
    q_vals    : (N_q,)
    dirs      : (N_orient, 3)
    """
    n_atoms = positions.shape[0]
    n_q = len(q_vals)
    n_orient = dirs.shape[0]
    Iq = np.zeros(n_q)

    for iq in prange(n_q):
        q = q_vals[iq]
        f = f_q[iq]
        I_sum = 0.0
        
        for io in range(n_orient):
            qx, qy, qz = q * dirs[io, :]
            Re = 0.0
            Im = 0.0
            for ia in range(n_atoms):
                phase = qx*positions[ia,0] + qy*positions[ia,1] + qz*positions[ia,2]
                Re += np.cos(phase)
                Im += np.sin(phase)
            I_sum += (Re**2 + Im**2)
        
        Iq[iq] = (I_sum / n_orient) * (f**2)
    
    return Iq

def compute_intensity_isotropic_fibonacci(positions, xdb, element,
                                          q_min, q_max, n_q=500, n_orient=1000):
    """
    Calcule I(q) isotrope par quadrature de Fibonacci.
    """
    q_vals = np.linspace(q_min, q_max, n_q)
    f_q = xdb.f0(element, q_vals)
    dirs = fibonacci_sphere(n_orient)
    
    Iq = compute_intensity_fibonacci_numba(positions, f_q, q_vals, dirs)
    return q_vals, Iq
# -----------------------------------------------------
# Classe principale de diffusion nanoparticulaire
# -----------------------------------------------------
class NanoparticleScattering2D:
    def __init__(self, structure_file, npix=250,
                 wl=1.0, Distance=0.5, pixel_size=0.0001):
        self.atoms = read(structure_file)
        self.file = structure_file
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
        
    def view_structure(self):
        atoms=Atoms(symbols=self.elements,positions=self.positions)
        write('./file.xyz',atoms)
        os.system('jmol file.xyz')
        os.remove('file.xyz')

    
    def compute_Iq_debyecalc(self):
        n=self.npix/2
        
        qstep = (self.q_max-self.q_min) / n
        calc= DebyeCalculator(qmin=self.q_min,qmax=self.q_max,qstep=qstep,biso=0, device='cuda')
        
        q_dc, i_dc = calc.iq(self.file)

        # scale to match our calculation
        f0 = self.xray_f0(self.elements[0],q_dc )
        K=2*f0**2/np.max(f0**2) # 2f²/Z²
        return q_dc, i_dc*K
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
    def compute_isotropic_intensity(self, q_vals=None, n_orient=3600):
        """
        Calcule I(Q) isotrope (moyenne sphérique 3D, équivalent à Debye)
        """
        if q_vals is None:
            q_vals = np.linspace(self.q_min, self.q_max, 500)
        
        f_q = self.xray_f0(self.elements[0], q_vals)
        Iq = compute_intensity_isotropic(self.positions, f_q, q_vals, n_orient=n_orient)
        return q_vals, Iq

    def compute_isotropic_intensity_fibonacci(self, q_vals=None, n_orient=1000):
        """
        Calcule I(Q) isotrope (moyenne sphérique via quadrature de Fibonacci)
        """
        if q_vals is None:
            q_vals = np.linspace(self.q_min, self.q_max, 500)

        f_q = self.xray_f0(self.elements[0], q_vals)
        dirs = fibonacci_sphere(n_orient)
        Iq = compute_intensity_fibonacci(self.positions, f_q, q_vals, dirs)
        return q_vals, Iq
    
    # -----------------------------
    # Affichage
    # -----------------------------
    
    def plot_intensity(self, I_map, log=True, vmin=-6, vmax=0, qmax=None,interpolation='bicubic'):
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
            interpolation=interpolation
        )

        plt.colorbar(im, ax=ax, label="Intensité (log)" if log else "Intensité")
        ax.set_xlabel("$q_x$ (Å⁻¹)")
        ax.set_ylabel("$q_z$ (Å⁻¹)")
        
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
        """
        ZYZ convention
        """
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
        self.positions = self.positions @ R.T

        return self.positions

    def ai(self):        
        detector = Detector(pixel1=self.pixel_size, pixel2=self.pixel_size)
        ai = AzimuthalIntegrator(dist=self.D, detector=detector) 
        ai.setFit2D(self.D *1000, self.npix/2, self.npix/2, wavelength=self.wl)
        return ai

    def integrate_with_pyfai(self,I,plot=False):
        # Create integrator
        ai = self.ai()
        q, i = ai.integrate1d(I,npt=1000,unit="q_A^-1")
        if plot:
            plt.figure()
            plt.loglog(q, i, 'k-')
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("I(Q) (a.u.)")
            plt.xlim(self.q_min, self.q_max)
            plt.title("Intégration radiale directe dans l’espace Q")
            plt.grid(True)
            plt.show()
        return q,i

    # ============================================================
    #  Distribution d’orientation gaussienne 3D → intensité 2D
    # ============================================================
    @staticmethod
    def _rotation_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    @staticmethod
    def _rotation_z(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])

    @staticmethod
    def _rotation_yz(theta_y, theta_z):
        """Rotation R = Rz(theta_z) * Ry(theta_y)"""
        return NanoparticleScattering2D._rotation_z(theta_z) @ NanoparticleScattering2D._rotation_y(theta_y)

    @staticmethod
    def _gaussian_orientation_grid(n_y=9, n_z=9,
                                   sigma_y=np.deg2rad(5),
                                   sigma_z=np.deg2rad(5),
                                   n_sigma=3):
        """
        Génère une grille 2D d’angles (θ_y, θ_z) pondérée selon une distribution gaussienne.
        """
        thetas_y = np.linspace(-n_sigma*sigma_y, n_sigma*sigma_y, n_y)
        thetas_z = np.linspace(-n_sigma*sigma_z, n_sigma*sigma_z, n_z)
        ThetaY, ThetaZ = np.meshgrid(thetas_y, thetas_z, indexing='ij')
        W = np.exp(-0.5*((ThetaY/sigma_y)**2 + (ThetaZ/sigma_z)**2))
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
        Calcul de l’intensité de diffusion simulée pour une distribution d’orientation
        gaussienne de nanoparticules alignées globalement selon l’axe x.

        Chaque particule est supposée subir une rotation aléatoire autour des axes y et z,
        suivant une loi normale centrée sur 0 avec des écarts-types sigma_y et sigma_z.
        L’intensité totale correspond à la somme pondérée des intensités calculées pour
        chaque orientation échantillonnée sur une grille angulaire.

        Paramètres
        ----------
        sigma_y : float (radians)
            Écart-type (σ) de la distribution d’orientation autour de l’axe y.
            Contrôle la dispersion angulaire selon y (rotation hors du plan x–z).
        sigma_z : float (radians)
            Écart-type (σ) de la distribution d’orientation autour de l’axe z.
            Contrôle la dispersion angulaire dans le plan x–y.
        n_y : int
            Nombre de points d’échantillonnage pour les rotations autour de y.
        n_z : int
            Nombre de points d’échantillonnage pour les rotations autour de z.
        n_sigma : float
            Étendue angulaire couverte en unités d’écart-type.
            Par exemple, n_sigma=3 couvre environ ±3σ (≈99,7 % de la distribution).
        show_progress : bool
            Si True, affiche une barre de progression `tqdm` pendant l’intégration
            sur les orientations.
        use_friedel_sym : bool
            Si True, exploite la symétrie de Friedel I(-Q) = I(Q) pour ne calculer
            qu’une moitié du détecteur (Qz ≥ 0) et reconstruire la carte par symétrie.

        Retour
        ------
        I_total : ndarray (npix × npix)
            Carte d’intensité totale projetée sur le plan du détecteur (Qx, Qz),
            intégrée sur la distribution d’orientations gaussiennes.

        
        """
        from tqdm import tqdm

        orientations, weights = self._gaussian_orientation_grid(
            n_y=n_y, n_z=n_z, sigma_y=sigma_y, sigma_z=sigma_z, n_sigma=n_sigma
        )

        # Masque pour Qz >= 0 (si symétrie de Friedel utilisée)
        if use_friedel_sym:
            mask_half = self.Qz >= 0
            Qx_half = self.Qx[mask_half]
            Qz_half = self.Qz[mask_half]
            qvecs_half = np.stack([Qx_half, np.zeros_like(Qx_half), Qz_half], axis=1)
        else:
            qvecs_half = self.qvecs

        I_total_half = np.zeros_like(qvecs_half[:, 0])

        iterable = tqdm(enumerate(zip(orientations, weights)),
                        total=len(weights),
                        disable=not show_progress,
                        desc="Orientation averaging")

        for _, ((theta_y, theta_z), w) in iterable:
            # Rotation de la particule
            R = self._rotation_yz(theta_y, theta_z)
            rotated_positions = self.positions @ R.T

            # Calcul intensité pour cette orientation (sur demi-espace)
            I_rot_half = compute_intensity_numba(
                rotated_positions,
                self.xray_f0(self.elements[0], np.linalg.norm(qvecs_half, axis=1)),
                qvecs_half
            )
            I_total_half += w * I_rot_half

        # Reconstruction du plan complet si symétrie de Friedel activée
        if use_friedel_sym:
            I_total = np.zeros_like(self.Qx)
            I_total[mask_half] = I_total_half
            I_total[~mask_half] = I_total_half[self.Qz[mask_half][::-1].argsort()]
        else:
            I_total = I_total_half.reshape(self.Qx.shape)

        return I_total.reshape(self.Qx.shape)




