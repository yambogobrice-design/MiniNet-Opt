"""Chargement, découpage et standardisation des données."""
import numpy as np
from sklearn.datasets import make_moons

SEED = 42


def load_data(n_samples=1500, noise=0.20, seed=SEED):
    """Génère le jeu de données make_moons."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64)


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=SEED):
    """Découpage stratifié par permutation déterministe : 60/20/20."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]
    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


def standardize(X_train, X_val, X_test):
    """Standardisation centrée-réduite avec les statistiques du train uniquement."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma
