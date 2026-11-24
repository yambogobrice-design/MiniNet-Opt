# MiniNet-Opt

Conception, entraînement, régularisation et optimisation sous contraintes d'un réseau
de neurones de classification binaire, entièrement implémenté en **NumPy** (sans
autograd), avec vérification par **PyTorch**.

Projet intégrateur — Optimisation pour le Machine Learning (S7).

## Contenu du dépôt

```
project/
|-- data.py            # génération (make_moons), split 60/20/20, standardisation
|-- layers.py          # Linear, ReLU, Sigmoid, Dropout
|-- model.py           # assemblage du réseau 2 -> 16 -> 8 -> 1 (MiniNet)
|-- losses.py          # binary cross-entropy et gradient
|-- optimizers.py      # SGD, Momentum, Adam
|-- regularization.py  # pénalité L2, EarlyStopping
|-- constraints.py     # norme, projection sur la boule L2, ProjectedOptimizer
|-- train.py           # boucle d'entraînement commune
|-- experiments.py     # protocoles expérimentaux des parties A à G
|-- torch_verify.py    # vérification NumPy <-> PyTorch
|-- plots.py           # visualisations
|-- main.py            # point d'entrée
|-- figures/           # figures générées (PDF + PNG)
|-- results/           # summary.json, tables.tex (générés)
|-- rapport.tex        # rapport LaTeX
`-- rapport.pdf        # rapport compilé
```

## Installation

Python 3.9+ et les paquets suivants :

```
pip install numpy scikit-learn matplotlib
pip install torch   # optionnel, uniquement pour la vérification (partie G)
```

## Exécution

```
python main.py
```

Rejoue l'intégralité du protocole expérimental (≈ 1,5–6 min selon la machine) :

1. **A** — données, découpage, standardisation ;
2. **J1** — gradient checking par différences finies et entraînement sur petit jeu ;
3. **C** — comparaison SGD / Momentum / Adam + sensibilité au learning rate ;
4. **D** — régularisation L2, dropout, early stopping, sur-apprentissage, test final ;
5. **E** — gradient projeté sous la contrainte ‖θ‖₂ ≤ R (grille de rayons) ;
6. **F** — pénalisation L2 contre contrainte explicite à normes comparables ;
7. **G** — vérification PyTorch (SGD et Adam).

Les figures sont écrites dans `figures/`, les métriques dans `results/summary.json`
et les tableaux LaTeX du rapport dans `results/tables.tex`.

## Compilation du rapport

```
pdflatex rapport.tex
pdflatex rapport.tex
```

(penser à renseigner `\NomEtudiant` dans `rapport.tex`).

## Reproductibilité

Toutes les sources d'aléa sont contrôlées : graines fixées pour les données (42),
le découpage (42), l'initialisation des poids (42, 7, 123) et le tirage des
mini-batchs (42). Le jeu de test n'est évalué qu'une seule fois, après sélection
des hyperparamètres sur la validation.

## Validation numérique

- Gradient checking (différences finies centrées, ε = 10⁻⁵) : erreur relative maximale ≈ 5·10⁻¹¹.
- Comparaison NumPy / PyTorch (mêmes poids initiaux, mêmes lots, float64) : écart maximal
  sur les pertes ≈ 5,7·10⁻¹¹ (SGD) et 1,1·10⁻¹² (Adam).
