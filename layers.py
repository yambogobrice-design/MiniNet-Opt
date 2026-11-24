"""Couches élémentaires du réseau : Linear, ReLU, Sigmoid, Dropout."""
import numpy as np


class Linear:
    """Couche affine z = x W^T + b. Initialisation He (recommandée avec ReLU)."""

    def __init__(self, in_features, out_features, rng):
        bound = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, bound, size=(out_features, in_features))
        self.b = np.zeros(out_features)
        self.grads = {}
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        self.grads['W'] = dout.T @ self._x
        self.grads['b'] = dout.sum(axis=0)
        return dout @ self.W


class ReLU:
    """Activation ReLU(z) = max(0, z)."""

    def __init__(self):
        self._mask = None

    def forward(self, z):
        self._mask = z > 0.0
        return z * self._mask

    def backward(self, dout):
        return dout * self._mask


class Sigmoid:
    """Activation sigmoïde sigma(z) = 1 / (1 + e^{-z})."""

    def __init__(self):
        self._out = None

    def forward(self, z):
        z = np.clip(z, -60.0, 60.0)
        self._out = 1.0 / (1.0 + np.exp(-z))
        return self._out

    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)


class Dropout:
    """Dropout inversé : h_tilde = (M / p) * h, M_j ~ Bernoulli(p)."""

    def __init__(self, keep_prob, rng):
        self.keep_prob = keep_prob
        self.training = True
        self.rng = rng
        self._mask = None

    def forward(self, h):
        if self.training:
            self._mask = self.rng.binomial(1, self.keep_prob, size=h.shape) / self.keep_prob
            return h * self._mask
        return h

    def backward(self, dout):
        if self.training:
            return dout * self._mask
        return dout
