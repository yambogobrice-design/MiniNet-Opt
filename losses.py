"""Perte binaire (binary cross-entropy) et son gradient."""
import numpy as np

EPS = 1e-12


def binary_cross_entropy(p, y):
    """l(p, y) = -y log(p) - (1-y) log(1-p), moyennée sur le lot."""
    p = np.clip(p, EPS, 1.0 - EPS)
    y = y.reshape(-1, 1)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def binary_cross_entropy_grad(p, y):
    """d l / d p = (p - y) / (m * p * (1-p))."""
    m = y.shape[0]
    p = np.clip(p, EPS, 1.0 - EPS)
    y = y.reshape(-1, 1)
    return (p - y) / (m * (p * (1.0 - p) + EPS))
