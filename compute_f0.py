#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Computes atomic form factors f0 and neutron scattering lengths from YAML data (contains the Cromer-Mann coefficients of
different compounds). 
YAML data source : https://github.com/FrederikLizakJohansen/DebyeCalculator/tree/main/debyecalculator/utility/elements_info.yaml
Usage:
    1) Load YAML table using load_elements_yaml(path)
    2) Call f0_from_Q(Q, element, yaml_table) or f0_from_k(k, element, yaml_table) to get f0 values.
    3) Optionally, call neutron_scattering_length(element, yaml_table) to get neutron scattering length.
"""

import numpy as np
import yaml
from typing import Union

def load_elements_yaml(path: str):
    """Loads YAML and returns a dict."""
    with open(path, 'r', encoding='utf-8') as f:
        table = yaml.safe_load(f)
    return table

def _extract_coeffs(entry):
    """
    Two common formats in YAML:
    - a list/array [a1,a2,a3,a4,a5,c,b1,b2,b3,b4,b5,neutron_val]
    - a dict with named fields ('a1'..'a5','c','b1'..'b5','neutron')
    Returns tuple (a: np.array(5), c: float, b: np.array(5), neutron_val: float or None)
    """
    if entry is None:
        raise KeyError("Entry is None")
    # if list-like
    try:
        arr = np.asarray(entry, dtype=float)
        if arr.size >= 12:
            a = arr[0:5].astype(float)
            c = float(arr[5])
            b = arr[6:11].astype(float)
            neutron = float(arr[11]) if arr.size > 11 else None
            return a, c, b, neutron
    except Exception:
        pass
    # if dict-like
    if isinstance(entry, dict):
        # get a1..a5, b1..b5, c, neutron 
        a = np.array([float(entry.get(f'a{i}', entry.get(f'A{i}', 0.0))) for i in range(1,6)], dtype=float)
        b = np.array([float(entry.get(f'b{i}', entry.get(f'B{i}', 0.0))) for i in range(1,6)], dtype=float)
        c = float(entry.get('c', entry.get('C', 0.0)))
        neutron = entry.get('neutron', entry.get('n', entry.get('neutron_val', None)))
        neutron = float(neutron) if neutron is not None else None
        return a, c, b, neutron
    raise ValueError("Unsupported entry format for element coefficients")

def f0_from_Q(Q: Union[float, np.ndarray], element: str, yaml_table: dict) -> np.ndarray:
    """
    Compute f0(k) using Q input (Q = 4*pi*sin(theta)/lambda in 1/Å).
    Returns array shaped like Q.
    """
    if element not in yaml_table:
        raise KeyError(f"Element '{element}' not found in YAML table")
    a, c, b, neutron = _extract_coeffs(yaml_table[element])
    Q_arr = np.atleast_1d(np.asarray(Q, dtype=float))
    # convert Q -> k used in formula (k = sin(theta)/lambda = Q/(4*pi))
    k = Q_arr / (4.0 * np.pi)
    k2 = k**2  # shape (N,)
    # exponentials: shape (N,5)
    expo = np.exp(- np.outer(k2, b))  # outer(k2, b) -> shape (N,5)
    f0 = c + np.sum(a * expo, axis=1)
    return f0 if f0.shape[0] > 1 else float(f0[0])


def f0_from_k(k: Union[float, np.ndarray], element: str, yaml_table: dict) -> np.ndarray:
    """
    Compute f0(k) using k = sin(theta)/lambda directly.
    """
    if element not in yaml_table:
        raise KeyError(f"Element '{element}' not found in YAML table")
    a, c, b, neutron = _extract_coeffs(yaml_table[element])
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    k2 = k_arr**2
    expo = np.exp(- np.outer(k2, b))
    f0 = c + np.sum(a * expo, axis=1)
    return f0 if f0.shape[0] > 1 else float(f0[0])

# Option: neutron constant (useful for neutron scattering calculations in SasView)
def neutron_scattering_length(element: str, yaml_table: dict) -> float:
    a, c, b, neutron = _extract_coeffs(yaml_table[element])
    return neutron

# Example usage 
if __name__ == "__main__":
    # change path to where you copied elements_info.yaml
    path = "elements_info.yaml"
    table = load_elements_yaml(path)
    element = "Au"    
    # create Q-grid in 1/Å (0..20 Å^-1)
    # Q = np.linspace(0.0, 20.0, 1001)
    q_min = 0.001
    q_max = 1
    num_q = 20000
    Q = np.logspace(np.log10(q_min), np.log10(q_max), num_q)
    Q = np.linspace(0.0, 20.0, 20000)
    print("len(Q)", len(Q))
    f0_Q = f0_from_Q(Q, element, table)
    print(f"Length of f0(Q) for {element}: {len(f0_Q)}")
    print(f"f0({element}) at Q[0] (Q=0): {f0_Q[0]:.6g}")
    print(f"f0({element}) at Q[-1] (Q={Q[-1]:.3g} 1/Å): {f0_Q[-1]:.6g}")








