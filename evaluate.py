
import numpy as np

def total_variation_distance(p1: np.ndarray, p2: np.ndarray):
    p1 = np.array(p1)
    p2 = np.array(p2)
    if not len(p1.shape) == 1:
        raise NotImplementedError()
    if p1.shape != p2.shape:
        raise ValueError(f"Shape mismatch: p1.shape={p1.shape}, p2.shape={p2.shape}")
    return (np.abs(p1 - p2) / 2).sum()

def kl_divergence(p1: np.ndarray, p2: np.ndarray):
    p1 = np.array(p1)
    p2 = np.array(p2)
    mask = p1 == 0
    p1[mask] = 1e-10
    mask = p2 == 0
    p2[mask] = 1e-10

    if not len(p1.shape) == 1:
        raise NotImplementedError()
    if p1.shape != p2.shape:
        raise ValueError(f"Shape mismatch: p1.shape={p1.shape}, p2.shape={p2.shape}")
    m = (p1 > 0) # mask out cases when p1 = 0 
    p1 = p1[m] 
    p2 = p2[m]
    return (p1 * (np.log(p1) - np.log(p2))).sum()

