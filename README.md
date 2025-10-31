# Debye3D – 3D X-ray Scattering Intensity (CPU/GPU Hybrid)

**Debye3D** is a hybrid **CPU/GPU** implementation of 3D X-ray scattering intensity calculations using the **Debye formula**.
It supports isotropic and uniaxial orientation averaging, GPU acceleration via PyTorch, and high-performance CPU fallback via Numba.

---

## Installation

```bash
git clone https://github.com/<your-username>/debye3d.git
cd debye3d
pip install -r requirements.txt
```

### Main dependencies

* `numpy`, `scipy`, `matplotlib`
* `torch` (for GPU acceleration, optional)
* `numba` (for CPU acceleration)
* `ase` (for reading atomic structures)
* `pyFAI` (for optional radial integration)
* `xraydb` (for atomic form factors)

---

## Main Classes

### `Experiment`

Represents the **experimental detector geometry** and defines the **accessible Q-space**.

#### **Attributes**

* `npix`: number of detector pixels per side
* `wl`: wavelength (Å)
* `distance`: sample-to-detector distance (m)
* `pixel_size`: pixel size (m)
* `Qx`, `Qy`, `Qz`: detector-space Q-grid arrays
* `q_min`, `q_max`: minimum and maximum accessible |Q| values

#### **Methods**

| Method                                                   | Description                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| `__init__(npix, wl, distance, pixel_size, verbose=True)` | Initializes the detector grid and computes Q-space coverage. |
| *(verbose output)*                                       | Prints detector configuration and accessible Q-range.        |

---

### `Debye3D(structure_file, ...)`

Inherits from `Experiment` and adds **atomic structure** handling and **scattering computation** capabilities.

#### **Main Attributes**

* `atoms`, `positions`, `elements`, `nb_atoms`: ASE atomic structure and properties
* `xdb`: X-ray database (from `xraydb`)
* `device`: torch device (CPU or CUDA)
* Inherits all `Experiment` attributes

---

## Key Methods

| Method                                                             | Description                                                                             |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `view_structure()`                                                 | Writes a temporary XYZ file and opens it in *Jmol*. Requires `jmol` in PATH.            |
| `save_structure_as_xyz(filename)`                                  | Saves positions as xyz file. May be used for structures built with external funcions.   |
| `auto_batch_size()`                                                | Automatically determines an optimal GPU batch size based on available VRAM.             |
| `xray_f0(element, q)`                                              | Returns the atomic form factor *f₀(q)* using XrayDB.                                    |
| `compute_intensity(use_gpu=True, batch_size=None)`                 | Computes 2D scattering intensity (Debye equation) using GPU or CPU fallback.            |
| `_compute_intensity_torch(...)`                                    | High-performance GPU backend for intensity computation with adaptive batching.          |
| `compute_isotropic_intensity_fibonacci(n_q, n_orient, ...)`        | Computes isotropic average intensity using Fibonacci-sphere orientation sampling.       |
| `compute_intensity_uniaxial_ODF(n_samples, sigma_y, sigma_z, ...)` | Computes uniaxial orientation distribution (ODF) averages with Gaussian angular spread. |
| `compute_Iq_debyecalc()`                                           | Interfaces with an external `DebyeCalculator` (if installed) for I(q) computation.      |
| `plot_intensity(I_flat, log=True, ...)`                            | Displays a 2D Q-map of intensity (log or linear scale).                                 |
| `euler_rotation_matrix(alpha, beta, gamma)`                        | Returns the Z–Y–Z Euler rotation matrix for given angles (degrees).                     |
| `rotate_positions(alpha, beta, gamma)`                             | Rotates stored atomic coordinates according to ZYZ Euler convention.                    |
| `shake_positions(frac_a, frac_b, frac_c, ...)`                     | Applies random atomic displacements (“shake”) based on fractional amplitudes.           |
| `ai()`                                                             | Creates a `pyFAI.AzimuthalIntegrator` configured with current detector settings.        |
| `integrate_with_pyfai(I_flat, plot=False)`                         | Performs 1D radial integration of a 2D intensity map using pyFAI.                       |
| `compute_structure_factor(N, Z, use_gpu=True)`                     | Computes structure factor approximation: *S(q) = N·Z²·I(q)*.                            |

---

## CPU-Accelerated (Numba) Functions

If `numba` is installed, the following parallelized versions are automatically available:

* `compute_intensity_numba()` – standard Debye equation
* `compute_intensity_fibonacci_numba()` – isotropic averaging
* `compute_intensity_uniaxial_numba()` – uniaxial ODF averaging

These functions run efficiently on multicore CPUs using parallel JIT compilation.

---

## Command-Line Usage

```bash
python debye3d.py structure.xyz --use_gpu --npix 300 --fib 1000
```

### Arguments

| Option                   | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| `structure`              | Input atomic structure file (XYZ, CIF, etc.)            |
| `--use_gpu`              | Enables GPU acceleration (if CUDA available)            |
| `--fib N`                | Perform isotropic averaging over N Fibonacci directions |
| `--uniax N`              | Perform uniaxial averaging over N orientation samples   |
| `--sigma_y`, `--sigma_z` | Angular standard deviations for uniaxial ODF (degrees)  |
| `--npix`                 | Detector pixel grid size                                |
| `--wl`                   | X-ray wavelength (Å)                                    |
| `--distance`             | Sample-to-detector distance (m)                         |
| `--pixel`                | Pixel size (m)                                          |

---

## Example (Python API)

```python
from debye3d import Debye3D

# Initialize from atomic structure
model = Debye3D("structure.xyz", wl=1.0, distance=0.5)

# Compute intensity (GPU if available)
I = model.compute_intensity(use_gpu=True)

# Plot the resulting intensity map
model.plot_intensity(I, log=True)
```



---
