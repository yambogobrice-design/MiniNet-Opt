"""Outils de régularisation : pénalité L2 et early stopping."""
import numpy as np


def l2_penalty(params):
    """(1/2) * somme_l ||W^(l)||_F^2 : les biais ne sont pas pénalisés."""
    return 0.5 * sum(np.sum(v ** 2) for k, v in params.items() if k.startswith('W'))


def l2_weight_grad(params, lmbda):
    """Gradient de la pénalité : lambda * W pour les poids, 0 pour les biais."""
    grads = {}
    for k, v in params.items():
        grads[k] = lmbda * v if k.startswith('W') else np.zeros_like(v)
    return grads


class EarlyStopping:
    """Arrêt précoce avec patience P : sauvegarde et restaure les meilleurs paramètres."""

    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.best_params = None
        self.best_epoch = 0
        self.counter = 0

    def update(self, val_loss, params, epoch):
        """Retourne True si l'entraînement doit s'arrêter."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_params = {k: v.copy() for k, v in params.items()}
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
