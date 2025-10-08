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

def generate_lattice_hex2d(a=1.0, nx=3, ny=3, nz=3, dz=0.8165):
    """
    Génère un réseau hexagonal empilé (HEX2D) avec un espacement vertical ajustable.
    
    Params:
    - a: distance inter-particules dans le plan xy
    - nx, ny: nombre de particules dans le plan
    - nz: nombre de couches empilées
    - dz: espacement vertical entre les plans (modifiable)
    """
    positions = []
    dy = np.sqrt(3)/2 * a  # distance verticale entre lignes dans le plan

    for k in range(nz):
        z = k * dz * a  # espacement ajustable
        for j in range(ny):
            for i in range(nx):
                x = i*a + (0.5*a if j % 2 else 0)  # décalage en x pour la ligne
                y = j*dy
                # Décalage pour les plans B (empilement AB)
                if k % 2 == 1:
                    x += a/2
                    y += dy/3
                positions.append([x, y, z])
    return np.array(positions)