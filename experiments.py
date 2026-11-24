"""Protocoles expérimentaux des parties A à G.

Chaque expérience est reproductible : graines fixées, split unique,
architecture fixe. Les figures sont écrites dans figures/, les tableaux
LaTeX dans results/tables.tex et les indicateurs clés dans
results/summary.json.
"""
import os
import time
import json
import numpy as np

from data import load_data, train_val_test_split, standardize
from model import MiniNet
from losses import binary_cross_entropy, binary_cross_entropy_grad
from optimizers import SGD, Momentum, Adam
from regularization import EarlyStopping
from constraints import ProjectedOptimizer, param_norm, project_l2_ball
from train import train, evaluate
from plots import OPT_COLORS, plot_histories, plot_accuracy_curves, plot_grad_norms, \
    plot_data, plot_train_val, plot_norms, plot_rfeas, savefig, plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 300
BATCH = 32
SEED = 42
SEEDS_C = (42, 7, 123)

summary = {}
tables = []
l2_hists_cache = {}


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return f'{x:.{nd}f}'


def fmt_sci(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    if x == 0.0:
        return '0'
    mant, exp = f'{x:.{nd}e}'.split('e')
    return f'${mant}\\times 10^{{{int(exp)}}}$'


def lam_fmt(l):
    return f'{l:.5f}'.rstrip('0').rstrip('.')


def register_table(tex):
    tables.append(tex)


def get_data():
    X, y = load_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(X, y)
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)


def new_model(seed=SEED, **kw):
    return MiniNet(input_dim=2, hidden_dims=(16, 8), seed=seed, **kw)


def first_epoch_acc(hist, threshold=0.95):
    for i, a in enumerate(hist['acc_val'], start=1):
        if a >= threshold:
            return i
    return np.nan


# ---------------------------------------------------------------- Partie A

def exp_a(data):
    plot_data(data, 'a1_data.pdf')
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    summary['dims'] = dict(X_shape=list(Xtr.shape), y_shape=list(ytr.shape),
                           n_train=int(len(Xtr)), n_val=int(len(Xva)), n_test=int(len(Xte)),
                           n_features=int(Xtr.shape[1]))
    summary['class_balance_train'] = float(np.mean(ytr))


# --------------------------------------------------- J1 : gradient checking

def exp_grad_check(data):
    (Xtr, ytr), _, _ = data
    X = Xtr[:5]
    y = ytr[:5]
    model = new_model(seed=SEED)
    eps = 1e-5

    p = model.forward(X, training=False)
    model.backward(binary_cross_entropy_grad(p, y))
    grads = model.get_grads()
    params = model.get_params()

    errs = {}
    for k in params:
        g = grads[k]
        P = params[k]
        num = np.zeros_like(g)
        for i in range(g.size):
            idx = np.unravel_index(i, g.shape)
            old = P[idx]
            P[idx] = old + eps
            fp = binary_cross_entropy(model.forward(X, training=False), y)
            P[idx] = old - eps
            fm = binary_cross_entropy(model.forward(X, training=False), y)
            P[idx] = old
            num[idx] = (fp - fm) / (2.0 * eps)
        denom = np.linalg.norm(g) + np.linalg.norm(num) + eps
        errs[k] = float(np.linalg.norm(g - num) / denom)
        summary[f'gradcheck_{k}'] = errs[k]

    max_err = max(errs.values())
    summary['gradcheck_max'] = max_err
    row = ' & '.join(fmt_sci(errs[k]) for k in sorted(errs)) + ' & ' + fmt_sci(max_err) + r' \\'
    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Erreurs relatives du gradient checking (différences finies centrées, $\varepsilon = 10^{-5}$).}
\label{tab:gradcheck}
\begin{tabular}{lccccccc}
\toprule
 & $W^{(1)}$ & $b^{(1)}$ & $W^{(2)}$ & $b^{(2)}$ & $W^{(3)}$ & $b^{(3)}$ & Maximum \\
\midrule
""" + f'Erreur relative & {row}\n' +
        r"""\bottomrule
\end{tabular}
\end{table}
""")


def exp_j1_small_fit(data):
    (Xtr, ytr), _, _ = data
    sel = np.concatenate([np.where(ytr == 0)[0][:4], np.where(ytr == 1)[0][:4]])
    small = (Xtr[sel], ytr[sel]), (Xtr[:8], ytr[:8]), (Xtr[:8], ytr[:8])
    model = new_model(seed=SEED)
    hist = train(model, SGD(lr=0.5), small, epochs=300, batch_size=8, seed=SEED,
                 record_grad_norm=False)
    plot_train_val(hist, 'j1_small_fit.pdf')
    summary['j1_small_final_loss'] = float(hist['loss_train'][-1])
    summary['j1_small_final_acc'] = float(hist['acc_train'][-1])


# ------------------------------------------------------------- Partie C

OPT_FACTORIES = [
    ('SGD', lambda: SGD(lr=0.1)),
    ('Momentum', lambda: Momentum(lr=0.1, beta=0.9)),
    ('Adam', lambda: Adam(lr=1e-3)),
]


def exp_c(data):
    histories = {name: [] for name, _ in OPT_FACTORIES}
    rows = []
    for name, factory in OPT_FACTORIES:
        hyper = {'SGD': r'$\alpha = 0{,}1$',
                 'Momentum': r'$\alpha = 0{,}1$, $\beta = 0{,}9$',
                 'Adam': r'$\alpha = 10^{-3}$'}[name]
        finals = dict(loss_train=[], loss_val=[], acc_val=[], acc_train=[],
                      epoch95=[], time=[], loss1=[], loss5=[])
        for s in SEEDS_C:
            model = new_model(seed=s)
            t0 = time.perf_counter()
            hist = train(model, factory(), data, EPOCHS, BATCH, seed=SEED)
            finals['time'].append(time.perf_counter() - t0)
            finals['loss_train'].append(hist['loss_train'][-1])
            finals['loss_val'].append(hist['loss_val'][-1])
            finals['acc_train'].append(hist['acc_train'][-1])
            finals['acc_val'].append(hist['acc_val'][-1])
            finals['epoch95'].append(first_epoch_acc(hist))
            finals['loss1'].append(hist['loss_train'][0])
            finals['loss5'].append(hist['loss_train'][4])
            histories[name].append(hist)

        def ms(v):
            v = np.array(v)
            return np.nanmean(v), np.nanstd(v)

        lt, lt_s = ms(finals['loss_train'])
        lv, lv_s = ms(finals['loss_val'])
        av, av_s = ms(finals['acc_val'])
        e95, e95_s = ms(finals['epoch95'])
        t, t_s = ms(finals['time'])
        rows.append(
            f"{name} & {hyper} & {fmt(lt)} $\\pm$ {fmt(lt_s)} & {fmt(lv)} $\\pm$ {fmt(lv_s)} & {fmt(av, 3)} $\\pm$ {fmt(av_s, 3)}"
            f" & {fmt(e95, 1)} $\\pm$ {fmt(e95_s, 1)} & {fmt(t, 2)} $\\pm$ {fmt(t_s, 2)} \\\\")
        summary[f'c_{name}'] = dict(loss_train=float(lt), loss_val=float(lv), acc_val=float(av),
                                    epoch95=float(e95), time=float(t),
                                    loss_epoch1=float(np.mean(finals['loss1'])),
                                    loss_epoch5=float(np.mean(finals['loss5'])))

    plot_histories([h[0] for h in histories.values()],
                   [n for n, _ in OPT_FACTORIES],
                   [OPT_COLORS[n] for n, _ in OPT_FACTORIES], 'c_loss.pdf')
    plot_grad_norms([h[0] for h in histories.values()],
                    [n for n, _ in OPT_FACTORIES],
                    [OPT_COLORS[n] for n, _ in OPT_FACTORIES], 'c_grad.pdf')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, _ in OPT_FACTORIES:
        accs = np.array([h['acc_val'] for h in histories[name]])
        m, s = accs.mean(axis=0), accs.std(axis=0)
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, color=OPT_COLORS[name], label=name)
        ax.fill_between(ep, m - s, m + s, color=OPT_COLORS[name], alpha=0.2)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel('Époque')
    ax.set_ylabel('Accuracy validation')
    ax.legend()
    savefig(fig, 'c_acc.pdf')

    table = r"""\begin{table}[htbp]
\centering
\caption{Comparaison SGD / Momentum / Adam (moyenne $\pm$ écart-type sur 3 graines, 300 époques, batch 32).}
\label{tab:c}
\begin{tabular}{llccccc}
\toprule
Optimiseur & Hyperparamètres & \multicolumn{1}{c}{Loss train} & \multicolumn{1}{c}{Loss val.} & \multicolumn{1}{c}{Acc. val.} & \multicolumn{1}{c}{1\textsuperscript{re} époque acc $\ge 0{,}95$} & \multicolumn{1}{c}{Temps (s)} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    register_table(table)
    return histories


def exp_c_lr(data):
    lr_grids = [
        ('SGD', SGD, [1e-3, 1e-2, 1e-1, 1.0], 'c_lr_sgd.pdf'),
        ('Adam', Adam, [1e-4, 1e-3, 1e-2, 1e-1], 'c_lr_adam.pdf'),
    ]
    rows = []
    for name, cls, lrs, fname in lr_grids:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        lr_hists = {}
        for lr in lrs:
            model = new_model(seed=SEED)
            hist = train(model, cls(lr=lr), data, EPOCHS, BATCH, seed=SEED)
            lr_hists[lr] = hist
            rows.append(f'{name} ($\\alpha = {lr:g}$) & {fmt(hist["loss_train"][-1])} & '
                        f'{fmt(hist["loss_val"][-1])} & {fmt(hist["acc_val"][-1], 3)} & '
                        f'{fmt(hist["param_norm"][-1], 3)} \\\\')
            summary[f'c_lr_{name}_{lr:g}'] = dict(
                loss_train=float(hist['loss_train'][-1]),
                loss_val=float(hist['loss_val'][-1]),
                acc_val=float(hist['acc_val'][-1]))
        for ax, key in zip(axes, ['loss_train', 'loss_val']):
            for lr in lrs:
                vals = np.array(lr_hists[lr][key], dtype=float)
                ax.plot(vals, label=f'$\\alpha = {lr:g}$')
            ax.set_xlabel('Époque')
            ax.set_ylabel('Loss train' if key == 'loss_train' else 'Loss validation')
            ax.legend(fontsize=9, title=name)
        savefig(fig, fname)

    beta_rows = []
    for beta in [0.5, 0.9, 0.99]:
        model = new_model(seed=SEED)
        hist = train(model, Momentum(lr=0.1, beta=beta), data, EPOCHS, BATCH, seed=SEED)
        beta_rows.append(f'Momentum ($\\beta = {beta}$) & {fmt(hist["loss_train"][-1])} & '
                         f'{fmt(hist["loss_val"][-1])} & {fmt(hist["acc_val"][-1], 3)} & '
                         f'{fmt(hist["param_norm"][-1], 3)} \\\\')
        summary[f'c_beta_{beta}'] = dict(loss_val=float(hist['loss_val'][-1]),
                                         acc_val=float(hist['acc_val'][-1]))

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Sensibilité au pas d'apprentissage $\alpha$ et au paramètre $\beta$ du momentum (300 époques, batch 32).}
\label{tab:c_lr}
\begin{tabular}{lcccc}
\toprule
Configuration & Loss train & Loss val. & Acc. val. & $\|\theta^\star\|_2$ \\
\midrule
""" + '\n'.join(rows + beta_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------- Partie D

L2_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DROPOUT_KEEP = [0.9, 0.8, 0.7, 0.5]
PATIENCE = 20


def exp_d(data):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data

    baseline = new_model(seed=SEED)
    hist_base = train(baseline, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED)
    summary['d_base'] = dict(loss_train=float(hist_base['loss_train'][-1]),
                             loss_val=float(hist_base['loss_val'][-1]),
                             acc_val=float(hist_base['acc_val'][-1]),
                             norm=float(hist_base['param_norm'][-1]))

    l2_hists, l2_rows = {}, []
    for lmbda in L2_GRID:
        model = new_model(seed=SEED)
        hist = train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED, l2=lmbda)
        l2_hists[lmbda] = hist
        gap = hist['loss_val'][-1] - hist['loss_train'][-1]
        l2_rows.append(f'$\\lambda = {lam_fmt(lmbda)}$ & {fmt(hist["loss_train"][-1])} & '
                       f'{fmt(hist["loss_val"][-1])} & {fmt(hist["acc_val"][-1], 3)} & '
                       f'{fmt(hist["param_norm"][-1], 3)} & {fmt(gap, 3)} \\\\')
        summary[f'd_l2_{lmbda:g}'] = dict(loss_train=float(hist['loss_train'][-1]),
                                          loss_val=float(hist['loss_val'][-1]),
                                          acc_val=float(hist['acc_val'][-1]),
                                          norm=float(hist['param_norm'][-1]))
    best_l2 = min((l for l in L2_GRID if l > 0), key=lambda l: l2_hists[l]['loss_val'][-1])
    summary['d_best_l2'] = float(best_l2)
    global l2_hists_cache
    l2_hists_cache = l2_hists

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ['loss_train', 'loss_val']):
        for lmbda in L2_GRID:
            ax.plot(l2_hists[lmbda][key], label=f'$\\lambda = {lam_fmt(lmbda)}$')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Loss train' if key == 'loss_train' else 'Loss validation')
        ax.legend(fontsize=8, title='Pénalité $L_2$')
    savefig(fig, 'd_l2_loss.pdf')
    plot_norms([l2_hists[l] for l in L2_GRID], [f'$\\lambda = {l:g}$' for l in L2_GRID],
               plt.cm.viridis(np.linspace(0.15, 0.9, len(L2_GRID))), 'd_l2_norm.pdf')

    do_hists, do_rows = {}, []
    for keep in DROPOUT_KEEP:
        model = new_model(seed=SEED, dropout_keep=keep)
        hist = train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED)
        do_hists[keep] = hist
        gap = hist['loss_val'][-1] - hist['loss_train'][-1]
        do_rows.append(f'$p = {keep}$ & {fmt(hist["loss_train"][-1])} & '
                       f'{fmt(hist["loss_val"][-1])} & {fmt(hist["acc_val"][-1], 3)} & '
                       f'{fmt(gap, 3)} \\\\')
        summary[f'd_dropout_{keep}'] = dict(loss_train=float(hist['loss_train'][-1]),
                                            loss_val=float(hist['loss_val'][-1]),
                                            acc_val=float(hist['acc_val'][-1]))
    best_p = min(DROPOUT_KEEP, key=lambda p: do_hists[p]['loss_val'][-1])
    summary['d_best_dropout'] = float(best_p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ['loss_train', 'loss_val']):
        for keep in DROPOUT_KEEP:
            ax.plot(do_hists[keep][key], label=f'$p = {keep}$ (taux $\\delta = {1 - keep}$)')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Loss train' if key == 'loss_train' else 'Loss validation')
        ax.legend(fontsize=8)
    savefig(fig, 'd_dropout.pdf')

    model = new_model(seed=SEED)
    es = EarlyStopping(patience=PATIENCE)
    hist_es = train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED, early_stopping=es)
    loss_tr_es, acc_tr_es = evaluate(model, Xtr, ytr)
    loss_va_es, acc_va_es = evaluate(model, Xva, yva)
    loss_te_es, acc_te_es = evaluate(model, Xte, yte)
    plot_train_val(hist_es, 'd_early.pdf', vline=es.best_epoch,
                   vline_label=f'Meilleure époque ({es.best_epoch})')
    summary['d_es'] = dict(best_epoch=int(es.best_epoch), loss_train=float(loss_tr_es),
                           loss_val=float(loss_va_es), acc_val=float(acc_va_es),
                           loss_test=float(loss_te_es), acc_test=float(acc_te_es),
                           norm=float(param_norm(model.get_params())))

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Effet de la pénalité $L_2$ (Adam, 300 époques) : les biais ne sont pas pénalisés.}
\label{tab:d_l2}
\begin{tabular}{lccccc}
\toprule
Pénalité & Loss train & Loss val. & Acc. val. & $\|\theta^\star\|_2$ & Écart val. $-$ train \\
\midrule
""" + '\n'.join(l2_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Effet du dropout (taux $\delta = 1-p$), Adam, 300 époques. Le dropout est désactivé en validation.}
\label{tab:d_dropout}
\begin{tabular}{lcccc}
\toprule
Conservation $p$ & Loss train & Loss val. & Acc. val. & Écart val. $-$ train \\
\midrule
""" + '\n'.join(do_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    over_model = new_model(seed=SEED)
    hist_over = train(over_model, Adam(lr=1e-3), data, 2000, BATCH, seed=SEED)
    argmin = int(np.argmin(hist_over['loss_val'])) + 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = np.arange(1, len(hist_over['loss_train']) + 1)
    ax.plot(ep, hist_over['loss_train'], label='Loss train')
    ax.plot(ep, hist_over['loss_val'], label='Loss validation')
    ax.axvline(argmin, color='red', linestyle=':', label=f'Min. val. : époque {argmin}')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Loss (BCE)')
    ax.legend()
    savefig(fig, 'd_overfit.pdf')
    summary['d_overfit'] = dict(min_val_epoch=int(argmin),
                                min_val_loss=float(hist_over['loss_val'][argmin - 1]),
                                final_train=float(hist_over['loss_train'][-1]),
                                final_val=float(hist_over['loss_val'][-1]))

    loss_te_base, acc_te_base = evaluate(baseline, Xte, yte)
    model = new_model(seed=SEED)
    train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED, l2=best_l2)
    loss_te_l2, acc_te_l2 = evaluate(model, Xte, yte)
    model = new_model(seed=SEED, dropout_keep=best_p)
    train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED)
    loss_te_do, acc_te_do = evaluate(model, Xte, yte)

    summary['d_test'] = dict(
        base=dict(loss=float(loss_te_base), acc=float(acc_te_base)),
        l2=dict(loss=float(loss_te_l2), acc=float(acc_te_l2), lambda_=float(best_l2)),
        dropout=dict(loss=float(loss_te_do), acc=float(acc_te_do), p=float(best_p)),
        es=dict(loss=float(loss_te_es), acc=float(acc_te_es)))

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Évaluation finale unique sur le jeu de test des modèles sélectionnés par validation.}
\label{tab:d_test}
\begin{tabular}{lccccc}
\toprule
Modèle sélectionné & Hyperparamètre & Loss train & Loss val. & Loss test & Acc. test \\
\midrule
""" +
        f"Sans régularisation & -- & {fmt(hist_base['loss_train'][-1])} & {fmt(hist_base['loss_val'][-1])} & {fmt(loss_te_base)} & {fmt(acc_te_base, 3)} \\\\\n" +
        f"Pénalité $L_2$ & $\\lambda = {best_l2:g}$ & {fmt(l2_hists[best_l2]['loss_train'][-1])} & {fmt(l2_hists[best_l2]['loss_val'][-1])} & {fmt(loss_te_l2)} & {fmt(acc_te_l2, 3)} \\\\\n" +
        f"Dropout & $p = {best_p}$ & {fmt(do_hists[best_p]['loss_train'][-1])} & {fmt(do_hists[best_p]['loss_val'][-1])} & {fmt(loss_te_do)} & {fmt(acc_te_do, 3)} \\\\\n" +
        f"Early stopping & patience $= {PATIENCE}$ & {fmt(loss_tr_es)} & {fmt(loss_va_es)} & {fmt(loss_te_es)} & {fmt(acc_te_es, 3)} \\\\\n" +
        r"""\bottomrule
\end{tabular}
\end{table}
""")

    return baseline, best_l2, best_p


# ------------------------------------------------------------- Partie E

R_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]


def exp_e(data, baseline, R0=None):
    if R0 is None:
        R0 = float(param_norm(baseline.get_params()))
    summary['e_R0'] = R0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(R0, color='black', linestyle='--', label=f'$R_0 = {R0:.3f}$ (non contraint)')
    rows = []
    rfeas_histories, rfeas_labels = [], []
    for f in R_FACTORS:
        R = f * R0
        model = new_model(seed=SEED)
        opt = ProjectedOptimizer(Adam(lr=1e-3), R)
        hist = train(model, opt, data, EPOCHS, BATCH, seed=SEED)
        norm_star = float(hist['param_norm'][-1])
        max_rfeas = float(np.max(hist['rfeas'])) if hist['rfeas'] else 0.0
        active = opt.is_active(model.get_params(), R)
        ax.plot(hist['param_norm'], label=f'$R = {f:g}R_0$')
        rfeas_histories.append(hist)
        rfeas_labels.append(f'$R = {f:g}R_0$')
        rows.append(f'$R = {f:g}R_0$ ({fmt(R, 3)}) & {fmt(hist["loss_train"][-1])} & '
                    f'{fmt(hist["loss_val"][-1])} & {fmt(hist["acc_val"][-1], 3)} & '
                    f'{fmt(norm_star, 3)} & {fmt(max_rfeas, 4)} & '
                    f'{("oui" if active else "non")} \\\\')
        summary[f'e_R_{f:g}'] = dict(R=float(R), loss_train=float(hist['loss_train'][-1]),
                                     loss_val=float(hist['loss_val'][-1]),
                                     acc_val=float(hist['acc_val'][-1]),
                                     norm=norm_star, max_rfeas=max_rfeas, active=bool(active),
                                     n_projections=int(opt.n_projections))
    ax.set_xlabel('Époque')
    ax.set_ylabel(r'$\|\theta_k\|_2$')
    ax.legend(fontsize=8)
    savefig(fig, 'e_norm.pdf')
    plot_rfeas(rfeas_histories, rfeas_labels,
               plt.cm.viridis(np.linspace(0.15, 0.9, len(rfeas_histories))), 'e_rfeas.pdf')

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Gradient projeté (Adam projeté sur la boule $\|\theta\|_2 \le R$). $R_0$ est la norme de la solution non contrainte de référence. $r_{\mathrm{feas}}$ est le maximum par époque de la violation avant projection.}
\label{tab:e}
\begin{tabular}{lccccccl}
\toprule
Contrainte & Loss train & Loss val. & Acc. val. & $\|\theta^\star\|_2$ & $r_{\mathrm{feas}}$ & Contrainte \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    return R0


# ------------------------------------------------------------- Partie F

F_LAMBDAS = [1e-4, 1e-3, 1e-2]


def exp_f(data, l2_hists, baseline):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    rows = []
    for lmbda in F_LAMBDAS:
        R = float(l2_hists[lmbda]['param_norm'][-1])
        pen_vals, con_vals = [], []
        pen_norms, con_norms = [], []
        pen_test, con_test = [], []
        for s in SEEDS_C:
            model = new_model(seed=s)
            train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED, l2=lmbda)
            pen_vals.append(evaluate(model, Xva, yva)[0])
            pen_norms.append(param_norm(model.get_params()))
            pen_test.append(evaluate(model, Xte, yte)[1])
            model = new_model(seed=s)
            train(model, ProjectedOptimizer(Adam(lr=1e-3), R), data, EPOCHS, BATCH, seed=SEED)
            con_vals.append(evaluate(model, Xva, yva)[0])
            con_norms.append(param_norm(model.get_params()))
            con_test.append(evaluate(model, Xte, yte)[1])
        rows.append(
            f'$\\lambda = {lam_fmt(lmbda)}$ & $R = {fmt(R, 3)}$ & '
            f'{fmt(np.mean(pen_vals))} $\\pm$ {fmt(np.std(pen_vals))} & '
            f'{fmt(np.mean(con_vals))} $\\pm$ {fmt(np.std(con_vals))} & '
            f'{fmt(np.mean(pen_test), 3)} $\\pm$ {fmt(np.std(pen_test), 3)} & '
            f'{fmt(np.mean(con_test), 3)} $\\pm$ {fmt(np.std(con_test), 3)} & '
            f'{fmt(np.mean(pen_norms), 3)} & {fmt(np.mean(con_norms), 3)} \\\\')
        summary[f'f_pair_{lmbda:g}'] = dict(R=R,
                                            pen=dict(loss_val=float(np.mean(pen_vals)),
                                                     acc_test=float(np.mean(pen_test)),
                                                     norm=float(np.mean(pen_norms))),
                                            con=dict(loss_val=float(np.mean(con_vals)),
                                                     acc_test=float(np.mean(con_test)),
                                                     norm=float(np.mean(con_norms))))

    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Comparaison pénalisation $L_2$ contre contrainte explicite $\|\theta\|_2 \le R$ à normes finales comparables (moyenne $\pm$ écart-type sur 3 graines).}
\label{tab:f}
\begin{tabular}{lccccccc}
\toprule
Pénalité & Rayon $R$ & \multicolumn{1}{c}{Loss val. (pén.)} & \multicolumn{1}{c}{Loss val. (cont.)} & \multicolumn{1}{c}{Acc. test (pén.)} & \multicolumn{1}{c}{Acc. test (cont.)} & \multicolumn{1}{c}{$\|\theta^\star\|$ (pén.)} & \multicolumn{1}{c}{$\|\theta^\star\|$ (cont.)} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    lmbda = F_LAMBDAS[1]
    R = float(l2_hists[lmbda]['param_norm'][-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    model = new_model(seed=SEED)
    h_pen = train(model, Adam(lr=1e-3), data, EPOCHS, BATCH, seed=SEED, l2=lmbda)
    model = new_model(seed=SEED)
    h_con = train(model, ProjectedOptimizer(Adam(lr=1e-3), R), data, EPOCHS, BATCH, seed=SEED)
    for ax, key in zip(axes, ['loss_val', 'param_norm']):
        ax.plot(h_pen[key], label=f'Pénalisation $\\lambda = {lam_fmt(lmbda)}$')
        ax.plot(h_con[key], label=f'Contrainte $R = {fmt(R, 3)}$')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Loss validation' if key == 'loss_val' else r'$\|\theta_k\|_2$')
        ax.legend(fontsize=9)
    savefig(fig, 'f_pair.pdf')


# ---------------------------------------------------- Partie G (PyTorch)

def exp_g(data, histories_c=None):
    from torch_verify import compare_torch
    res = compare_torch(data, histories_c)
    summary['g'] = res
    register_table(
        r"""\begin{table}[htbp]
\centering
\caption{Vérification PyTorch : mêmes initialisations, mêmes lots, mêmes hyperparamètres (float64).}
\label{tab:g}
\begin{tabular}{lcccccc}
\toprule
Optimiseur & Cadre & Loss train & Loss val. & Acc. val. & Acc. test & Max $|\Delta$loss$|$ \\
\midrule
""" +
        '\n'.join(
            f"{r['name']} & NumPy & {fmt(r['numpy_loss_train'])} & {fmt(r['numpy_loss_val'])} & "
            f"{fmt(r['numpy_acc_val'], 3)} & {fmt(r['numpy_acc_test'], 3)} & -- \\\\\n"
            f"{r['name']} & PyTorch & {fmt(r['torch_loss_train'])} & {fmt(r['torch_loss_val'])} & "
            f"{fmt(r['torch_acc_val'], 3)} & {fmt(r['torch_acc_test'], 3)} & {fmt_sci(r['max_diff'])} \\\\\n"
            for r in res) +
        r"""\bottomrule
\end{tabular}
\end{table}
""")
    return res


# ------------------------------------------------------- Frontière de décision

def exp_boundary(data, model, name='d_boundary.pdf', title=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    all_x = np.vstack([Xtr, Xva, Xte])
    x_min, x_max = all_x[:, 0].min() - 0.4, all_x[:, 0].max() + 0.4
    y_min, y_max = all_x[:, 1].min() - 0.4, all_x[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=['#ffd9b3', '#b3d9ff'], alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.2)
    ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=12, label='Test classe 0')
    ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=12, label='Test classe 1')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    if title:
        ax.set_title(title)
    ax.legend()
    return savefig(fig, name)


# ---------------------------------------------------------------- Sauvegarde

def save_results():
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, 'tables.tex'), 'w', encoding='utf-8') as f:
        f.write('% Tableaux générés automatiquement par experiments.py\n')
        f.write('% Ne pas éditer à la main.\n\n')
        joined = '\n\n'.join(tables)
        joined = joined.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
        joined = joined.replace(r'\end{tabular}', r'\end{tabular}}')
        f.write(joined)
        f.write('\n')
