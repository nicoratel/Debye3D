from ase.spacegroup import crystal
from ase.io import write
import numpy as np

def generate_supercell(cellpar, spacegroup = 225, supercell_size=(10, 10, 10), output_file="supercell.xyz"):
    """
    Génère une supermaille monoatomique à partir d'un groupe d'espace et de paramètres de maille.

    Paramètres
    ----------
    cellpar : list ou tuple de 6 floats
        Paramètres de maille [a, b, c, alpha, beta, gamma] en Å et degrés
    spacegroup : int
        Numéro du groupe d'espace (1 à 230)
    supercell_size : tuple de 3 int, optionnel
        Facteur de répétition dans les directions (nx, ny, nz)
    output_file : str, optionnel
        Nom du fichier de sortie (.xyz)
    """

    # Création de la maille élémentaire monoatomique
    atoms = crystal('Au',
                    basis=[(0, 0, 0)],
                    spacegroup=spacegroup,
                    cellpar=cellpar)

    # Création de la supermaille
    supercell = atoms.repeat(supercell_size)

    # Sauvegarde au format XYZ
    write(output_file, supercell)
    print(f"Supermaille générée : {output_file} ({len(supercell)} atomes)")
    return output_file

import numpy as np

def honeycomb(a, n_y, n_z, n_layers, dx, yz_noise=0.0, seed=None):
    """
    Génère un empilement de lamelles hexagonales 2D le long de x,
    avec des positions (y,z) légèrement aléatoires d'une lamelle à l'autre.

    Parameters
    ----------
    a : float
        Paramètre de maille hexagonale (distance entre voisins dans la lamelle).
    n_y, n_z : int
        Nombre de points le long de y et z dans chaque lamelle.
    n_layers : int
        Nombre de lamelles le long de x.
    dx : float
        Distance entre les lamelles le long de x.
    yz_noise : float
        Amplitude du bruit aléatoire ajouté à y et z pour chaque lamelle.
    seed : int, optional
        Seed pour reproductibilité.

    Returns
    -------
    coords : numpy.ndarray, shape (N_atoms, 3)
        Coordonnées (x,y,z) de tous les points du réseau.
    """
    if seed is not None:
        np.random.seed(seed)

    coords = []

    # vecteurs de base du réseau hexagonal 2D (y,z)
    a1 = np.array([a, 0])
    a2 = np.array([a/2, a*np.sqrt(3)/2])

    for k in range(n_layers):
        x_offset = k * dx
        # génération d'un petit décalage aléatoire pour cette lamelle
        delta_yz = np.random.uniform(-yz_noise, yz_noise, size=2)

        for i in range(n_y):
            for j in range(n_z):
                y, z = i*a1[0] + j*a2[0], i*a1[1] + j*a2[1]
                # on ajoute le petit décalage aléatoire spécifique à cette lamelle
                coords.append([x_offset, y + delta_yz[0], z + delta_yz[1]])

    return np.array(coords)



