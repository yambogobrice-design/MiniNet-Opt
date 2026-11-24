"""Assemblage du réseau multicouche d -> n1 -> n2 -> 1 (ReLU + sigmoïde)."""
import numpy as np
from layers import Linear, ReLU, Sigmoid, Dropout


class MiniNet:
    """Réseau de classification binaire : couches cachées ReLU, sortie sigmoïde.

    Les paramètres sont référencés par un dictionnaire {'W1','b1','W2','b2',...}
    afin que les optimiseurs opèrent de manière générique.
    """

    def __init__(self, input_dim=2, hidden_dims=(16, 8), dropout_keep=None, seed=42):
        self.rng = np.random.default_rng(seed)
        dims = [input_dim] + list(hidden_dims) + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], self.rng))
            if i < len(dims) - 2:
                self.layers.append(ReLU())
                if dropout_keep is not None:
                    self.layers.append(Dropout(dropout_keep, self.rng))
        self.layers.append(Sigmoid())
        self.dropouts = [l for l in self.layers if isinstance(l, Dropout)]
        self._linear = [l for l in self.layers if isinstance(l, Linear)]

    def train_mode(self):
        for d in self.dropouts:
            d.training = True

    def eval_mode(self):
        for d in self.dropouts:
            d.training = False

    def forward(self, X, training=True):
        if training:
            self.train_mode()
        else:
            self.eval_mode()
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params(self):
        return {f'W{i+1}': l.W for i, l in enumerate(self._linear)} | \
               {f'b{i+1}': l.b for i, l in enumerate(self._linear)}

    def get_grads(self):
        return {f'W{i+1}': l.grads['W'] for i, l in enumerate(self._linear)} | \
               {f'b{i+1}': l.grads['b'] for i, l in enumerate(self._linear)}

    def copy_params(self):
        return {k: v.copy() for k, v in self.get_params().items()}

    def load_params(self, params):
        for i, l in enumerate(self._linear):
            l.W = params[f'W{i+1}']
            l.b = params[f'b{i+1}']

    def predict(self, X):
        p = self.forward(X, training=False)
        return (p.ravel() > 0.5).astype(np.float64)
