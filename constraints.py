"""Optimisation sous contraintes : norme des poids, projection sur la boule L2,
gradient projeté et résidu de faisabilité."""
import numpy as np


def param_norm(params):
    """||theta||_2^2 = somme_l ||W^(l)||_F^2 (les biais sont exclus)."""
    return np.sqrt(sum(np.sum(v ** 2) for k, v in params.items() if k.startswith('W')))


def project_l2_ball(params, R):
    """Projection euclidienne sur C = {theta : ||theta||_2 <= R}, modifie en place.

    P_C(z) = z si ||z|| <= R, sinon R * z / ||z||.
    """
    norm = param_norm(params)
    if norm > R:
        scale = R / norm
        for k in params:
            if k.startswith('W'):
                params[k] *= scale
    return params


class ProjectedOptimizer:
    """Gradient projeté : theta_{k+1} = P_C(theta_k - lr * g_k).

    Enveloppe un optimiseur de base (SGD, Momentum ou Adam) et projette
    les poids après chaque mise à jour.
    """

    def __init__(self, base_optimizer, R):
        self.base = base_optimizer
        self.R = R
        self.epoch_excess = 0.0
        self.n_projections = 0

    def step(self, params, grads):
        self.base.step(params, grads)
        excess = max(0.0, param_norm(params) - self.R)
        self.epoch_excess = max(self.epoch_excess, excess)
        if excess > 0.0:
            self.n_projections += 1
        project_l2_ball(params, self.R)

    def feasibility_residual(self, params):
        """r_feas = max(0, ||theta||_2 - R)."""
        return max(0.0, param_norm(params) - self.R)

    @staticmethod
    def is_active(params, R, tol=1e-3):
        """Contrainte active si | ||theta||_2 - R | <= tol."""
        return abs(param_norm(params) - R) <= tol
