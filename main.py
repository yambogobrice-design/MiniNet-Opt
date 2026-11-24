"""Point d'entrée : exécute toutes les expériences (parties A à G).

Usage : python main.py
Les figures sont écrites dans figures/, les résultats dans results/.
"""
import time

import experiments as ex
from experiments import get_data, new_model
from optimizers import Adam
from train import train


def main():
    t0 = time.perf_counter()
    print('MiniNet-Opt : exécution du protocole expérimental complet\n')
    data = get_data()

    print('A. Données et visualisation...')
    ex.exp_a(data)

    print('J1. Gradient checking...')
    ex.exp_grad_check(data)
    ex.exp_j1_small_fit(data)

    print('C. Comparaison SGD / Momentum / Adam...')
    ex.exp_c(data)
    ex.exp_c_lr(data)

    print('D. Régularisation et généralisation...')
    baseline, best_l2, best_p = ex.exp_d(data)

    print('E. Gradient projeté sous contrainte...')
    ex.exp_e(data, baseline)

    print('F. Pénalisation vs contrainte explicite...')
    ex.exp_f(data, ex.l2_hists_cache, baseline)

    print('G. Vérification PyTorch...')
    ex.exp_g(data)

    print('Frontière de décision du meilleur modèle validé...')
    model = new_model(seed=ex.SEED, dropout_keep=best_p)
    train(model, Adam(lr=1e-3), data, ex.EPOCHS, ex.BATCH, seed=ex.SEED)
    ex.exp_boundary(data, model, title='Frontière de décision (modèle dropout sélectionné)')

    ex.save_results()
    print(f'\nTerminé en {time.perf_counter() - t0:.1f} s.')
    print('Figures : figures/  |  Résultats : results/')


if __name__ == '__main__':
    main()
