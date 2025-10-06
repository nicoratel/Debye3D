import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from xraydb import XrayDB

class NanoparticleScattering2D:
    def __init__(self, structure_file, qmax=15.0, npix=250):
        """
        structure_file: xyz file
        qmax: q_max pour la grille (Å^-1)
        npix: nombre de pixels pour la grille Qx-Qz
        """
        self.atoms = read(structure_file)
        self.positions = self.atoms.get_positions()  # (N_atoms,3)
        self.elements = self.atoms.get_chemical_symbols()
        
        self.qmax = qmax
        self.npix = npix
        
        # Initialisation XrayDB
        self.xdb = XrayDB()
                
        # Grille q
        self.qx = np.linspace(-qmax, qmax, npix)
        self.qz = np.linspace(-qmax, qmax, npix)
        self.Qx, self.Qz = np.meshgrid(self.qx, self.qz)
        self.qvecs = np.stack([self.Qx.ravel(), np.zeros(self.Qx.size), self.Qz.ravel()], axis=1)
        
    # -----------------------------
    # Fonction facteur atomique
    # -----------------------------
    def xray_f0(self, element, q):
        return self.xdb.f0(element, q)
    
    # -----------------------------
    # Matrice rotation Euler
    # -----------------------------
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
    
    # -----------------------------
    # Calcul vectorisé F(q) et intensité
    # -----------------------------
    def compute_intensity(self, positions=None):
        if positions is None:
            positions = self.positions
        # Phases
        phase = np.exp(1j * positions.dot(self.qvecs.T))  # (N_atoms, N_pixels)
        # Facteurs atomiques
        q_mags = np.linalg.norm(self.qvecs, axis=1)  # (N_pixels,)
        f_array = np.array([self.xray_f0(el, q_mags) for el in self.elements])  # (N_atoms, N_pixels)
        # Somme et intensité
        F = np.sum(f_array * phase, axis=0)
        I_map = np.abs(F)**2
        return I_map.reshape((self.npix, self.npix))
    
    # -----------------------------
    # Moyennage sur plusieurs orientations
    # -----------------------------
    def compute_orientational_average(self, euler_angles_list):
        I_avg = np.zeros((self.npix, self.npix))
        for alpha, beta, gamma in euler_angles_list:
            pos_rot = self.rotate_positions(alpha, beta, gamma)
            I_map = self.compute_intensity(pos_rot)
            I_avg += I_map
        I_avg /= len(euler_angles_list)
        return I_avg
    
    # -----------------------------
    # Affichage
    # -----------------------------
    def plot_intensity(self, I_map, log=True):
        plt.figure(figsize=(6,5))
        if log:
            plt.imshow(I_map, extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()],
                       origin='lower', cmap='jet', norm=plt.matplotlib.colors.LogNorm())
        else:
            plt.imshow(I_map, extent=[self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()],
                       origin='lower', cmap='jet')
        plt.colorbar(label="Intensité" + (" (log)" if log else ""))
        plt.xlabel("$q_x$ (Å⁻¹)")
        plt.ylabel("$q_z$ (Å⁻¹)")
        plt.show()
