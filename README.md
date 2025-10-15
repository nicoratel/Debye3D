# Debye3D

**Debye3D** is a Python module for simulating 2D X-ray scattering patterns from atomic-scale nanoparticle structures.  
It implements direct Debye-like summation, orientation averaging (Fibonacci quadrature and Gaussian orientation distributions), and utilities for radial integration and visualization.

---

## Overview

The core objective of this code is to compute scattering intensities either as:

- a full 2D detector-plane intensity map `I(Q_x, Q_z)` computed from the atomic positions via a direct coherent sum, or
- isotropic radial averages `I(q)` produced by spherical averaging over orientations (useful for comparison with powder or solution scattering).

The numerical kernels are implemented with Numba for efficient parallel execution on CPU. Optionally, an external `DebyeCalculator` (if available) supports GPU-accelerated Debye-sum computations.
In the code, the direct beam is aligned with y axis, and the detector frame lies in the (x,z) plane.

---

## Key features

- Direct evaluation of `|∑_a exp(i Q·r_a)|^2 * f(Q)^2` on a user-defined detector geometry.
- Isotropic radial averaging via:
  - Fibonacci sphere quadrature (uniform direction sampling).
  - Explicit Gaussian orientation distributions (3D orientation spreading).
- pyFAI helpers for fast azimuthal integration when pyFAI is installed.
- Numba acceleration for computationally intensive summations.
- Minimal dependencies beyond scientific Python stack (ASE for structure I/O).
- structures given as input can be rotated using a dedicated method based on ZYZ Euler convention

---

## Installation

It is recommended to use a virtual environment.

```bash
# create & activate virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows (PowerShell)

# install core dependencies
pip install numpy scipy matplotlib numba tqdm psutil ase xraydb
