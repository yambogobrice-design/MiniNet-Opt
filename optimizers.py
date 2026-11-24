"""Optimiseurs de premier ordre : SGD, Momentum, Adam."""
import numpy as np


class SGD:
    """theta_{k+1} = theta_k - lr * g_k."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Momentum:
    """v_{k+1} = beta*v_k + (1-beta)*g_k ; theta_{k+1} = theta_k - lr*v_{k+1}."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])
            self.v[k] = self.beta * self.v[k] + (1.0 - self.beta) * grads[k]
            params[k] -= self.lr * self.v[k]


class Adam:
    """Adam avec correction de biais : m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
