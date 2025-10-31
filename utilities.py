import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.build import make_supercell
import os
import math
from scipy.interpolate import griddata

def make_cylinder_from_cif(
    cif_file: str,
    R: float,
    L: float,
    axis: str = 'z',
    output_file: str = None) -> str:
    """
    Extract a cylindrical region from a CIF structure and save it as an XYZ file.
    The function automatically builds a sufficiently large supercell
    to fully contain the cylinder.

    Parameters
    ----------
    cif_file : str
        Path to the CIF file.
    R : float
        Cylinder radius (Å).
    L : float
        Cylinder length (Å).
    axis : str, optional
        Cylinder axis ('x', 'y', or 'z'), default is 'z'.
    output_file : str, optional
        Output XYZ filename. If None, it is generated automatically.

    Returns
    -------
    str
        Absolute path to the generated XYZ file.
    """

    # === 1. Load CIF ===
    atoms = read(cif_file)
    cell = atoms.get_cell_lengths_and_angles()[:3]  # (a, b, c)
    a_len, b_len, c_len = cell

    # === 2. Determine replication factors based on R and L ===
    # For example, if R = 20 Å and a = 5 Å → need ceil(2*R / a) ≈ 8 repeats in x,y
    if axis == 'x':
        nx = math.ceil(L / a_len) + 2  # axial direction (length)
        ny = math.ceil(2 * R / b_len) + 2  # radial directions
        nz = math.ceil(2 * R / c_len) + 2
    elif axis == 'y':
        nx = math.ceil(2 * R / a_len) + 2
        ny = math.ceil(L / b_len) + 2
        nz = math.ceil(2 * R / c_len) + 2
    elif axis == 'z':
        nx = math.ceil(2 * R / a_len) + 2
        ny = math.ceil(2 * R / b_len) + 2
        nz = math.ceil(L / c_len) + 2
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # === 3. Build the adaptive supercell ===
    P = np.diag([nx, ny, nz])
    atoms_super = make_supercell(atoms, P)

    # === 4. Center the atomic positions around their geometric center ===
    positions = atoms_super.get_positions()
    geom_center = positions.mean(axis=0)
    pos_rel = positions - geom_center

    # === 5. Define the cylindrical region ===
    if axis == 'x':
        axial = pos_rel[:, 0]
        radial_sq = pos_rel[:, 1]**2 + pos_rel[:, 2]**2
    elif axis == 'y':
        axial = pos_rel[:, 1]
        radial_sq = pos_rel[:, 0]**2 + pos_rel[:, 2]**2
    else:  # z
        axial = pos_rel[:, 2]
        radial_sq = pos_rel[:, 0]**2 + pos_rel[:, 1]**2

    mask = (radial_sq <= R**2) & (np.abs(axial) <= L / 2)
    atoms_cylinder = atoms_super[mask]

    if len(atoms_cylinder) == 0:
        raise RuntimeError("No atoms found in the defined cylindrical region. "
                           "Try increasing R or L.")

    # === 6. Recenter the resulting structure ===
    atoms_cylinder.positions -= atoms_cylinder.get_positions().mean(axis=0)
    atoms_cylinder.center(vacuum=0.0)

    # === 7. Determine output filename ===
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(cif_file))[0]
        output_file = f"{base_name}_cylinder_{axis}_R{R}_L{L}.xyz"
    
        
    output_path = os.path.abspath(output_file)

    # === 8. Save as XYZ ===
    write(output_path, atoms_cylinder)

    

    return output_path


# -----------------------------------------------------
# Fonction pour sauvegarder l'image et les axes Q
# -----------------------------------------------------
def save_intensity_npz(filename, I_flat, Qx, Qz):
    """
    Reshape I_flat to detector shape, and save Qx, Qz values.

    Parameters
    ----------
    filename : str
       Name of .npz 
    I_flat : np.ndarray
        computed intensity by Debye3D (flattened, npix*npix).
    Qx, Qz : np.ndarray
        Detector axes corresponding to image.
    """
    I_map = I_flat.reshape(Qx.shape)
    np.savez_compressed(filename, I_map=I_map, Qx=Qx, Qz=Qz)
    print(f"Saved intensity map to '{filename}'")

# -----------------------------------------------------
# Fonction pour afficher l'image depuis le .npz
# -----------------------------------------------------
def plot_from_npz(filename, log=True, vmin=-6, vmax=0, interpolation='bicubic',qmin=0,qmax=None, grid_size = 1000):
    """
    Charger une intensité depuis un fichier .npz et afficher.

    Parameters
    ----------
    filename : str
        Chemin vers le fichier .npz.
    log : bool, optional
        Afficher en échelle logarithmique (default True).
    vmin, vmax : float
        Bornes pour LogNorm si log=True.
    interpolation : str
        Méthode d'interpolation matplotlib pour imshow.
    """
    data = np.load(filename)
    I_map = data['I_map']
    Qx = data['Qx']
    Qz = data['Qz']

    # mask using qmin and qmax
    q = np.sqrt(Qx**2+Qz**2)
    
    if qmax is not None:
        mask = (q>qmin) & (q<qmax)
    else:
        mask = (q>qmin)
        print('mask=(qmin only)')

    # Recreate Qx_masked, Qz_masked, I_masked
    qx_masked, qz_masked, I_masked = Qx[mask], Qz[mask], I_map[mask]

    # Normalize intensity
    intensity = np.clip(I_masked/np.max(I_masked),1e-15,1)


    qx_lin = np.linspace(qx_masked.min(), qx_masked.max(), grid_size)
    qz_lin = np.linspace(qz_masked.min(), qz_masked.max(), grid_size)
    QX, QZ = np.meshgrid(qx_lin, qz_lin)
    Z = griddata((qx_masked, qz_masked), intensity, (QX, QZ), method='linear')

    fig, ax = plt.subplots(figsize=(6,5))
    if log:
        norm = plt.matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
    else:
        norm = None

    extent = [QX.min(), QX.max(), QZ.min(), QZ.max()]

    im = ax.imshow(Z, extent=extent, origin='lower', cmap='jet',
                   norm=norm, interpolation=interpolation)
    plt.colorbar(im, ax=ax, label="Intensity (log)" if log else "Intensity")
    ax.set_xlabel("$q_x$ (Å⁻¹)")
    ax.set_ylabel("$q_z$ (Å⁻¹)")
    #ax.set_title(f"Intensity map from '{filename}'")
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



# --------------------------------------------------------
# Polydispersity functions (for memory)
# should be adapted depending on the scientific case
# The example shown here is to consider polydispersity function on cylinder
# Polydispersity on 2 parameters: cylinder radius r and length L
# Gaussian (normal) distribution defined by mu and PD_ratio (sigma = mu * PD_ratio)
# stored here for memory
# --------------------------------------------------------

def compute_cylinder_intensity_uniaxial_ODF(r,L,n_samples_uniaxial=20, sigma_y =2,sigma_z=2):
    xyz_file = make_cylinder_from_cif(cif_file, R = r, L= L)
    d = Debye3D(xyz_file,distance= exp.distance,npix=exp.npix,wl=exp.wl,pixel_size=exp.pixel_size, verbose = False)
    # align cylinder along x axis
    d.rotate_positions(alpha=0,beta=90,gamma=0)
    # compute scattering (unique object _ form factor)
    Pq = d.compute_intensity_uniaxial_ODF(n_samples = n_samples_uniaxial, sigma_y=sigma_y,sigma_z=sigma_z)
    os.remove(xyz_file)
    return Pq

def monte_carlo_polydisp_intensity_uniaxiali_ODF(
    compute_func, 
    mu_r, 
    PD_ratio_r, 
    mu_L, 
    PD_ratio_L, 
    N=50, 
    plot=True,
    npix=256,
    n_samples_uniaxial=20,
    sigma_y=2,
    sigma_z=2):
    """
    Estime l'intensité moyenne et l'incertitude d'un cylindre en utilisant Monte-Carlo.

    Parameters
    ----------
    compute_func : function
        Function to cumpute intensity forgiven size parameters
        
    mu_r : float
        .
    sigma_r : float
        Écart-type de r.
    mu_L : float
        Moyenne de L.
    sigma_L : float
        Écart-type de L.
    N : int
        Nombre d'échantillons Monte-Carlo pour la distribution de taille
    plot : bool
        Si True, affiche les images de moyenne et d'écart-type.
    n_samples_uniaxial: int
        Nombre d'échantillons Monte-Carlo pour la distribution d'orientation
    sigma_y: float
        Ecart type de la distribution d'orientation gaussienne autour de y
    sigma_z: float
        Ecart type de la distribution d'orientation gaussienne autour de z

    Returns
    -------
    Pq_mean : ndarray
        Image moyenne sur N échantillons.
    
    """
    sigma_r =PD_ratio_r * mu_r
    sigma_L =PD_ratio_L * mu_L
    # Échantillons r et L
    r_samples = np.random.normal(mu_r, sigma_r, N)
    L_samples = np.random.normal(mu_L, sigma_L, N)

    # Calcul test pour connaître la forme de Pq
    Pq_test = compute_func(r_samples[0], L_samples[0],n_samples_uniaxial,sigma_y,sigma_z)
    height, width = Pq_test.shape

    # Stockage des images Monte-Carlo
    Pq_all = np.zeros((N, height, width))

    # Monte-Carlo loop
    for i in range(N):
        Pq = compute_func(r_samples[i], L_samples[i],n_samples_uniaxial,sigma_y,sigma_z)
        Pq=Pq.reshape([npix,npix])
        Pq_all[i, :, :] = Pq
        print(f'Iteration {i}/{N}')

    # Average
    Pq_mean = np.mean(Pq_all, axis=0)
    

    d.plot_intensity(Pq_mean)
    return Pq_mean