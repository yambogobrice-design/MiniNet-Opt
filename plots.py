"""Visualisations : courbes d'entraînement et tracés de données."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
})

OPT_COLORS = {'SGD': '#1f77b4', 'Momentum': '#ff7f0e', 'Adam': '#2ca02c'}


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    png = os.path.join(FIG_DIR, name.replace('.pdf', '.png'))
    fig.savefig(png, bbox_inches='tight', dpi=110)
    plt.close(fig)
    return path


def plot_histories(histories, labels, colors, name, ylabel='Loss (BCE)', ylog=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, lab, col in zip(histories, labels, colors):
        ax.plot(hist['loss_train'], color=col, linestyle='-', label=f'{lab} — train')
        ax.plot(hist['loss_val'], color=col, linestyle='--', label=f'{lab} — validation')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel('Époque')
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=8.5)
    return savefig(fig, name)


def plot_accuracy_curves(histories, labels, colors, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, lab, col in zip(histories, labels, colors):
        ax.plot(hist['acc_train'], color=col, linestyle='-', label=f'{lab} — train')
        ax.plot(hist['acc_val'], color=col, linestyle='--', label=f'{lab} — validation')
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel('Époque')
    ax.set_ylabel('Accuracy')
    ax.legend(ncol=2, fontsize=8.5)
    return savefig(fig, name)


def plot_grad_norms(histories, labels, colors, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, lab, col in zip(histories, labels, colors):
        ax.plot(hist['grad_norm'], color=col, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('Époque')
    ax.set_ylabel(r'$\|\nabla F_{\mathrm{train}}(\theta_k)\|$')
    ax.legend(fontsize=9)
    return savefig(fig, name)


def plot_data(data, name='a1_data.pdf'):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (X, y, title) in zip(axes,
                                 [(Xtr, ytr, 'Entraînement (60 %)'),
                                  (Xva, yva, 'Validation (20 %)'),
                                  (Xte, yte, 'Test (20 %)')]):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, alpha=0.7, label='Classe 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, alpha=0.7, label='Classe 1')
        ax.set_title(f'{title} (n = {len(X)})', fontsize=10)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_xlabel(r'$x_1$')
    axes[0].set_ylabel(r'$x_2$')
    return savefig(fig, name)


def plot_train_val(hist, name, ylog=False, vline=None, vline_label=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(hist['loss_train']) + 1)
    ax.plot(epochs, hist['loss_train'], label='Loss train')
    ax.plot(epochs, hist['loss_val'], label='Loss validation')
    if ylog:
        ax.set_yscale('log')
    if vline is not None:
        ax.axvline(vline, color='red', linestyle=':', label=vline_label)
        ax.legend()
    ax.set_xlabel('Époque')
    ax.set_ylabel('Loss (BCE)')
    ax.legend()
    return savefig(fig, name)


def plot_norms(histories, labels, colors, name, ylabel=r'$\|\theta_k\|_2$'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, lab, col in zip(histories, labels, colors):
        ax.plot(hist['param_norm'], color=col, label=lab)
    ax.set_xlabel('Époque')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    return savefig(fig, name)


def plot_rfeas(histories, labels, colors, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, lab, col in zip(histories, labels, colors):
        ax.plot(hist['rfeas'], color=col, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('Époque')
    ax.set_ylabel(r'$r_{\mathrm{feas}}(k) = \max(0, \|\theta_k\|_2 - R)$')
    ax.legend(fontsize=9)
    return savefig(fig, name)
