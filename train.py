"""Boucle d'entraînement commune à toutes les expériences."""
import numpy as np
from losses import binary_cross_entropy, binary_cross_entropy_grad
from constraints import param_norm
from regularization import l2_penalty, l2_weight_grad, EarlyStopping


def make_batches(n, batch_size, rng):
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def evaluate(model, X, y):
    p = model.forward(X, training=False)
    loss = binary_cross_entropy(p, y)
    acc = np.mean((p.ravel() > 0.5).astype(np.float64) == y)
    return loss, acc


def full_grad_norm(model, X, y):
    """Norme du gradient de la BCE (sans pénalité) sur l'ensemble du train."""
    p = model.forward(X, training=False)
    model.backward(binary_cross_entropy_grad(p, y))
    grads = model.get_grads()
    return np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))


def train(model, optimizer, data, epochs=300, batch_size=32, seed=42, l2=0.0,
          early_stopping=None, record_grad_norm=True, record_norm=True):
    """Entraîne le modèle et renvoie l'historique des métriques par époque.

    loss_train / loss_val sont les pertes de données (BCE sans pénalité),
    évaluées sans dropout pour être comparables entre les méthodes.
    """
    (X_train, y_train), (X_val, y_val), _ = data
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    history = dict(loss_train=[], loss_val=[], loss_train_reg=[],
                   acc_train=[], acc_val=[], grad_norm=[], param_norm=[],
                   rfeas=[])
    R = getattr(optimizer, 'R', None)

    for epoch in range(1, epochs + 1):
        if R is not None:
            optimizer.epoch_excess = 0.0
        for batch in make_batches(n, batch_size, rng):
            p = model.forward(X_train[batch], training=True)
            model.backward(binary_cross_entropy_grad(p, y_train[batch]))
            grads = model.get_grads()
            if l2 > 0.0:
                pen_grads = l2_weight_grad(model.get_params(), l2)
                for k in grads:
                    grads[k] = grads[k] + pen_grads[k]
            optimizer.step(model.get_params(), grads)

        loss_tr, acc_tr = evaluate(model, X_train, y_train)
        loss_va, acc_va = evaluate(model, X_val, y_val)
        history['loss_train'].append(loss_tr)
        history['loss_val'].append(loss_va)
        history['loss_train_reg'].append(loss_tr + l2 * l2_penalty(model.get_params()))
        history['acc_train'].append(acc_tr)
        history['acc_val'].append(acc_va)
        if record_grad_norm:
            history['grad_norm'].append(full_grad_norm(model, X_train, y_train))
        if record_norm:
            history['param_norm'].append(param_norm(model.get_params()))
        if R is not None:
            history['rfeas'].append(optimizer.epoch_excess)

        if early_stopping is not None:
            if early_stopping.update(loss_va, model.get_params(), epoch):
                model.load_params(early_stopping.best_params)
                break

    return history
