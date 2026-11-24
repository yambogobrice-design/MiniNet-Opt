"""Vérification des implémentations NumPy avec PyTorch (partie G).

Construit une architecture équivalente sous PyTorch, recopie exactement les
mêmes poids initiaux, utilise les mêmes lots (indices générés par le même
RNG NumPy) et les mêmes hyperparamètres, puis compare SGD et Adam.
"""
import numpy as np
import torch

from model import MiniNet
from optimizers import SGD, Adam
from train import train as train_numpy, evaluate, make_batches
from plots import savefig, plt, OPT_COLORS

torch.set_default_dtype(torch.float64)
EPOCHS = 300
BATCH = 32
SEED = 42


class TorchNet(torch.nn.Module):
    """Architecture équivalente à MiniNet : 2 -> 16 -> 8 -> 1, ReLU + sigmoïde."""

    def __init__(self, params):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(params['W1']))
            self.fc1.bias.copy_(torch.from_numpy(params['b1']))
            self.fc2.weight.copy_(torch.from_numpy(params['W2']))
            self.fc2.bias.copy_(torch.from_numpy(params['b2']))
            self.fc3.weight.copy_(torch.from_numpy(params['W3']))
            self.fc3.bias.copy_(torch.from_numpy(params['b3']))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def train_torch(net, optimizer, data, epochs=EPOCHS, batch_size=BATCH, seed=SEED):
    (Xtr, ytr), (Xva, yva), _ = data
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr).view(-1, 1)
    Xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(yva).view(-1, 1)
    rng = np.random.default_rng(seed)
    loss_fn = torch.nn.BCELoss()
    hist = {'loss_train': [], 'loss_val': []}
    for _ in range(epochs):
        net.train()
        for batch in make_batches(len(Xt), batch_size, rng):
            optimizer.zero_grad()
            loss = loss_fn(net(Xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            hist['loss_train'].append(loss_fn(net(Xt), yt).item())
            hist['loss_val'].append(loss_fn(net(Xv), yv).item())
    return hist


def compare_torch(data, histories_c=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    ref = MiniNet(input_dim=2, hidden_dims=(16, 8), seed=SEED)
    init = ref.copy_params()

    configs = [
        ('SGD', lambda: SGD(lr=0.1), lambda: torch.optim.SGD),
        ('Adam', lambda: Adam(lr=1e-3), lambda: torch.optim.Adam),
    ]
    results = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, np_opt, torch_opt_cls) in zip(axes, configs):
        model_np = MiniNet(input_dim=2, hidden_dims=(16, 8), seed=SEED)
        hist_np = train_numpy(model_np, np_opt(), data, EPOCHS, BATCH, seed=SEED)

        net = TorchNet(init)
        opt = torch_opt_cls()(net.parameters(), lr=0.1 if name == 'SGD' else 1e-3)
        hist_t = train_torch(net, opt, data)

        loss_va_np, acc_va_np = evaluate(model_np, Xva, yva)
        loss_te_np, acc_te_np = evaluate(model_np, Xte, yte)
        net.eval()
        with torch.no_grad():
            pv = net(torch.from_numpy(Xva)).numpy().ravel()
            pt = net(torch.from_numpy(Xte)).numpy().ravel()
        loss_va_t = float(-np.mean(yva * np.log(np.clip(pv, 1e-12, 1)) +
                                   (1 - yva) * np.log(np.clip(1 - pv, 1e-12, 1))))
        loss_te_t = float(-np.mean(yte * np.log(np.clip(pt, 1e-12, 1)) +
                                   (1 - yte) * np.log(np.clip(1 - pt, 1e-12, 1))))
        acc_va_t = float(np.mean((pv > 0.5) == yva))
        acc_te_t = float(np.mean((pt > 0.5) == yte))

        diff = np.max(np.abs(np.array(hist_np['loss_train']) - np.array(hist_t['loss_train'])))
        results.append(dict(name=name,
                            numpy_loss_train=float(hist_np['loss_train'][-1]),
                            numpy_loss_val=loss_va_np, numpy_acc_val=acc_va_np,
                            numpy_acc_test=acc_te_np,
                            torch_loss_train=float(hist_t['loss_train'][-1]),
                            torch_loss_val=loss_va_t, torch_acc_val=acc_va_t,
                            torch_acc_test=acc_te_t, max_diff=diff))

        ep = np.arange(1, EPOCHS + 1)
        ax.plot(hist_np['loss_train'], color=OPT_COLORS[name], label=f'{name} NumPy — train')
        ax.plot(hist_np['loss_val'], color=OPT_COLORS[name], linestyle='--',
                label=f'{name} NumPy — val.')
        ax.plot(hist_t['loss_train'], color='black', linestyle=':',
                label=f'{name} PyTorch — train')
        ax.plot(hist_t['loss_val'], color='black', linestyle='-.',
                label=f'{name} PyTorch — val.')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Loss (BCE)')
        ax.legend(fontsize=8)
    savefig(fig, 'g_torch.pdf')
    return results
