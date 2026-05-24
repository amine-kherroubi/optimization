# Explication détaillée — `5_hybrid_ml_metaheuristics`

> **Date :** 2026-05-24  
> **Module :** `5_hybrid_ml_metaheuristics/hybrid_alns/`  
> **Problème traité :** Bin Packing Problem 1D (1D-BPP)

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Problème : le Bin Packing 1D](#2-problème--le-bin-packing-1d)
3. [Architecture du module](#3-architecture-du-module)
4. [Initialisation : Best-Fit Decreasing (BFD)](#4-initialisation--best-fit-decreasing-bfd)
5. [Cadre métaheuristique : ALNS](#5-cadre-métaheuristique--alns)
   - 5.1 [Pourquoi pas une recherche locale classique ?](#51-pourquoi-pas-une-recherche-locale-classique-)
   - 5.2 [LNS — Large Neighborhood Search](#52-lns--large-neighborhood-search)
   - 5.3 [ALNS — opérateurs de destruction](#53-alns--opérateurs-de-destruction)
6. [Critère d'acceptation : Simulated Annealing avec plateau](#6-critère-dacceptation--simulated-annealing-avec-plateau)
7. [Composante ML 1 — Thompson Sampling (sélection d'opérateur)](#7-composante-ml-1--thompson-sampling-sélection-dopérateur)
8. [Composante ML 2 — Réparation apprise (ML Repair)](#8-composante-ml-2--réparation-apprise-ml-repair)
   - 8.1 [Principe : imitation de BFD](#81-principe--imitation-de-bfd)
   - 8.2 [Représentation en features (11 dimensions)](#82-représentation-en-features-11-dimensions)
   - 8.3 [Modèle : GradientBoostingClassifier](#83-modèle--gradientboostingclassifier)
   - 8.4 [Inférence dans le solveur](#84-inférence-dans-le-solveur)
9. [Post-repair : consolidation locale](#9-post-repair--consolidation-locale)
10. [Pipeline d'entraînement (`train_repair_model.py`)](#10-pipeline-dentraînement-train_repair_modelpy)
11. [Collecte de données ALNS réelles (`collect_alns_states.py`) — DAgger-lite](#11-collecte-de-données-alns-réelles-collect_alns_statespy--dagger-lite)
12. [Contrat de features et versioning](#12-contrat-de-features-et-versioning)
13. [Flux d'exécution complet (résumé)](#13-flux-dexécution-complet-résumé)
14. [Structure des fichiers](#14-structure-des-fichiers)
15. [Comment utiliser le module](#15-comment-utiliser-le-module)

---

## 1. Vue d'ensemble

La partie `5_hybrid_ml_metaheuristics` implémente un **solveur hybride ML + métaheuristique** pour le problème du bin packing 1D. L'idée centrale est de combiner deux approches :

| Famille | Ce qu'elle apporte |
|---|---|
| **Métaheuristique (ALNS)** | Exploration large de l'espace de recherche, sortie des optima locaux |
| **Machine Learning** | Décisions guidées par des patterns appris, sans règles codées en dur |

Le module contient **deux composantes ML distinctes** :

1. **Thompson Sampling** — bandit adaptatif qui choisit quel opérateur de destruction utiliser à chaque itération.
2. **Réparation apprise** — un classifieur gradient boosting qui prédit, pour chaque item déplacé, le meilleur bin où le réinsérer.

---

## 2. Problème : le Bin Packing 1D

### Définition formelle

On dispose de :
- $n$ items d'indices $\mathcal{I} = \{1, \ldots, n\}$, chacun de taille $s_i \in \mathbb{Z}_{>0}$
- Des bins identiques de capacité $C \in \mathbb{Z}_{>0}$ (avec $s_i \le C$ pour tout $i$)

Une **solution feasible** est une partition de $\mathcal{I}$ en $m$ bins $B_1, \ldots, B_m$ vérifiant :

$$
\sum_{i \in B_j} s_i \le C \quad \forall j
$$

**Objectif :** minimiser $m$ (le nombre de bins utilisés).

### Borne inférieure

La borne continue classique :

$$
\mathrm{LB}_1 = \left\lceil \frac{\sum_{i} s_i}{C} \right\rceil
$$

Elle sert d'indicateur de progression dans le solveur.

### Complexité

Le 1D-BPP est **NP-difficile au sens fort** (réduction depuis 3-Partition). Cela justifie l'usage de métaheuristiques pour les instances de taille réelle.

---

## 3. Architecture du module

```
5_hybrid_ml_metaheuristics/
└── hybrid_alns/
    ├── solver.py                          # Solveur ALNS principal (runtime)
    ├── approach_explanation.md            # Documentation formelle
    ├── README.md                          # Usage rapide
    ├── models/
    │   ├── repair_model_v1.pkl            # Artefact modèle v1
    │   ├── repair_model_v2.pkl            # Artefact modèle v2
    │   └── alns_states_v1.pkl             # États ALNS collectés
    ├── repair_model_training/
    │   ├── features.py                    # Contrat de features (partagé)
    │   ├── train_repair_model.py          # Script d'entraînement
    │   ├── collect_alns_states.py         # Collecte de données on-policy
    │   └── README.md
    └── parameter_tuning/
        └── parameter_tuning.ipynb        # Notebook de tuning
```

**Principe clé :** `features.py` est importé **à la fois** par `solver.py` (inférence) et `train_repair_model.py` (entraînement), ce qui garantit que les features sont identiques dans les deux contextes.

---

## 4. Initialisation : Best-Fit Decreasing (BFD)

Avant de lancer ALNS, le solveur construit une solution initiale déterministe avec **BFD** :

**Algorithme :**
1. Trier les items par taille décroissante (ties: par indice).
2. Pour chaque item, parcourir tous les bins ouverts qui peuvent l'accueillir.
3. Placer l'item dans le bin avec le **slack post-placement minimal** (fit le plus serré).
4. Si aucun bin ne peut l'accueillir, ouvrir un nouveau bin.

**Pourquoi BFD et pas FFD ?** BFD produit une solution initiale plus serrée, ce qui donne à ALNS un meilleur point de départ et réduit le temps pour atteindre une bonne solution.

> **Note dans le code :** la méthode s'appelle `_build_ffd_start_solution` pour des raisons de compatibilité ascendante, mais l'implémentation est bien BFD.

---

## 5. Cadre métaheuristique : ALNS

### 5.1 Pourquoi pas une recherche locale classique ?

La recherche locale classique déplace un seul item à la fois. Elle converge vers des **optima locaux** — des solutions depuis lesquelles aucun déplacement unitaire n'améliore le coût. Sur le BPP, ces optima locaux sont nombreux et structurés, et la recherche s'y bloque facilement.

### 5.2 LNS — Large Neighborhood Search

LNS (Shaw, 1998) contourne ce problème en travaillant sur de **grands voisinages implicites** :

1. **Destroy** : retirer $k$ items de la solution courante → solution partielle + ensemble $D$ d'items déplacés.
2. **Repair** : réinsérer tous les items de $D$ → nouvelle solution complète.

La taille du voisinage est **exponentiellement grande** en $|D|$ car la réparation peut produire n'importe quelle completion feasible.

### 5.3 ALNS — opérateurs de destruction

ALNS (Ropke & Pisinger, 2006) maintient un **portefeuille d'opérateurs** et adapte leur probabilité de sélection selon leurs performances passées.

Le solveur dispose de **3 opérateurs** :

| Bras | Opérateur | Mécanisme |
|---|---|---|
| 0 | **Random** | Choisit des bins dans un ordre aléatoire jusqu'à déplacer au moins $k$ items |
| 1 | **Worst-load** | Choisit en priorité les bins les moins chargés (avec petite perturbation aléatoire pour éviter les égalités déterministes) |
| 2 | **Related-item** | Choisit un item "seed" aléatoire, puis déplace les $k-1$ items dont la taille est la plus proche |

**Taille de destruction adaptative :** $k$ est tiré uniformément dans $[k_\min, k_\max]$ avec :

$$
k_\min = \max(1, \lfloor 0.05 \cdot n \rfloor), \quad k_\max = \max(k_\min + 1, \lfloor 0.25 \cdot n \rfloor)
$$

Soit entre **5% et 25%** des items déplacés par itération. De plus, $k_\max$ **s'élargit progressivement** au fur et à mesure de la stagnation :

```python
stagnation_ratio = min(1.0, iterations_since_improvement / max_iterations)
k_adaptive_max = k_min + int(stagnation_ratio * (k_max - k_min))
```

Cela augmente le rayon de destruction quand la recherche stagne — une heuristique d'intensification progressive.

**Garantie d'intégrité :** chaque opérateur s'assure qu'au moins un bin reste intact après destruction (évite de vider complètement la solution).

---

## 6. Critère d'acceptation : Simulated Annealing avec plateau

### Acceptation SA de base

Soit $\Delta = f(x') - f(x)$ la différence de coût (nombre de bins) entre candidat et courant.

La solution candidate $x'$ est acceptée si :

$$
\Delta \le 0 \quad \text{ou} \quad U < \exp(-\Delta / T)
$$

où $U \sim \text{Uniform}(0,1)$ et $T$ est la température courante. Cela permet d'accepter des solutions légèrement dégradées avec une probabilité décroissante.

### Cooling + Reheat

- **Refroidissement géométrique :** $T \leftarrow \alpha \cdot T$ avec $\alpha = 0.9995$ par défaut.
- **Soft reheat :** si la recherche stagne pendant un certain nombre d'itérations, on applique :

$$
T \leftarrow \max(T, 0.35 \cdot T_0)
$$

Le reheat restaure une exploration modérée sans repartir de zéro.

### Early stop

Si le nombre d'itérations sans amélioration dépasse `no_improve_limit = max(250, max_iterations // 20)`, la recherche s'arrête. Cela évite de gaspiller du temps sur un plateau plat.

---

## 7. Composante ML 1 — Thompson Sampling (sélection d'opérateur)

À chaque itération, le solveur doit choisir l'un des 3 opérateurs de destruction. Ce choix est modélisé comme un **problème de bandit à 3 bras**.

### Thompson Sampling (Beta-Bernoulli)

Chaque bras $k$ maintient une distribution Beta postérieure $\mathrm{Beta}(\alpha_k, \beta_k)$, initialisée à $\mathrm{Beta}(1,1)$ (prior uniforme).

**Procédure par itération :**
1. Échantillonner $\tilde{\theta}_k \sim \mathrm{Beta}(\alpha_k, \beta_k)$ pour chaque bras $k$.
2. Choisir $k^* = \arg\max_k \tilde{\theta}_k$.
3. Appliquer l'opérateur $k^*$, exécuter la réparation, évaluer.
4. Calculer la récompense : nouveau meilleur → `1.0`, accepté non-améliorant → `0.5`, rejeté → `0.0`.
5. Convertir en feedback Bernoulli : tirer $\tilde{r} \sim \mathrm{Bernoulli}(r)$, puis :
   - $\tilde{r} = 1$ : $\alpha_{k^*} \leftarrow \alpha_{k^*} + 1$
   - $\tilde{r} = 0$ : $\beta_{k^*} \leftarrow \beta_{k^*} + 1$

**Pourquoi cette conversion ?** Thompson Sampling attend un feedback binaire. La récompense à 3 niveaux est convertie probabilistiquement pour conserver l'espérance tout en restant compatible avec la posterior Beta.

**Propriété :** au début, tous les opérateurs sont également probables. Progressivement, les opérateurs qui donnent de bons résultats voient leur $\alpha$ augmenter et sont sélectionnés plus souvent.

---

## 8. Composante ML 2 — Réparation apprise (ML Repair)

### 8.1 Principe : imitation de BFD

Après destruction, chaque item déplacé $i$ doit être réinséré. L'ensemble des bins feasibles est :

$$
\mathcal{F}(i) = \{j : \text{capacité résiduelle du bin } j \ge s_i\}
$$

Au lieu d'une heuristique fixe (comme BFD), le solveur utilise un **classifieur binaire** entraîné à imiter BFD : pour chaque paire $(i, j)$, le modèle prédit si $j$ est le "bon" bin (celui que BFD aurait choisi).

À l'inférence, le solveur score tous les bins feasibles et choisit celui avec la **probabilité prédite maximale**.

Les items sont réinsérés dans **l'ordre de taille décroissante** — les plus grands d'abord, car ils ont moins de choix feasibles.

### 8.2 Représentation en features (11 dimensions)

Chaque paire $(i, j)$ est encodée en un vecteur de 11 features, toutes **normalisées par la capacité $C$** :

| Index | Nom | Formule | Rôle |
|---|---|---|---|
| 0 | `item_size` | $s_i / C$ | Taille normalisée de l'item |
| 1 | `item_size_sq` | $(s_i / C)^2$ | Effet non-linéaire du remplissage |
| 2 | `size_rank` | $\mathrm{rank}(i) / n$ | Position de l'item dans le tri par taille |
| 3 | `remaining` | items restants / $n$ | Signal de progression de l'insertion |
| 4 | `bin_load` | $\ell_j / C$ | Taux d'utilisation courant du bin |
| 5 | `bin_rem` | $(C - \ell_j) / C$ | Capacité résiduelle du bin |
| 6 | `slack_after` | $(C - \ell_j - s_i) / C$ | Slack après placement |
| 7 | `bin_count` | $\|B_j\| / n$ | Nombre d'items dans le bin (normalisé) |
| 8 | `bin_largest` | $\max_{k \in B_j} s_k / C$ | Plus grand item déjà dans le bin |
| 9 | `bin_smallest` | $\min_{k \in B_j} s_k / C$ | Plus petit item déjà dans le bin |
| 10 | `fill_ratio` | $s_i / (C - \ell_j)$ | Ratio de remplissage (tightness) |

**Pourquoi normaliser par $C$ ?** Cela permet d'entraîner le modèle avec `capacity = 1.0` (pratique) et de l'utiliser avec n'importe quelle capacité entière, en obtenant des distributions de features identiques.

### 8.3 Modèle : GradientBoostingClassifier

Le classifieur est un `GradientBoostingClassifier` de scikit-learn (arbres de décision boostés en gradient). Un `StandardScaler` est ajusté et stocké avec le modèle pour normaliser les features avant l'inférence.

**Paramètres par défaut (v2) :**
```
n_estimators      = 500
max_depth         = 6
learning_rate     = 0.03
subsample         = 0.75
min_samples_split = 20
min_samples_leaf  = 10
early stopping    : validation_fraction=0.1, n_iter_no_change=50
```

**Poids de classe :** `compute_sample_weight("balanced")` est utilisé pour contrebalancer le déséquilibre de classes (il y a toujours plus de négatifs que de positifs).

**Métrique principale : ROC-AUC** — car le solveur consomme les scores de probabilité pour un ranking, pas une classification binaire.

### 8.4 Inférence dans le solveur

```python
# Pour chaque item déplacé (dans l'ordre décroissant de taille) :
for item in sorted(displaced, key=lambda i: -sizes[i]):
    # Construire les features pour toutes les paires (item, bin_feasible)
    feats_arr = make_features_batch_py(...)  # shape: (n_feasible_bins, 11)
    
    # Normaliser et scorer
    feats_arr = scaler.transform(feats_arr)
    scores = model.predict_proba(feats_arr)[:, 1]  # probabilité classe positive
    
    # Choisir le bin avec le score maximal
    best_j = idxs[scores.argmax()]
    sol.bins[best_j].append(item)
```

**Optimisation :** si `numba` est installé, la fonction `make_features_batch_jit` (version JIT compilée) est utilisée à la place de `make_features_batch_py` pour un calcul de features plus rapide.

---

## 9. Post-repair : consolidation locale

Après chaque réparation, le solveur applique une **recherche locale de consolidation** (`_consolidate`) :

- Itérer sur les bins du moins chargé au plus chargé.
- Pour chaque item d'un bin léger, essayer de le déplacer dans un autre bin qui a la place.
- Si un déplacement réussit, le bin source peut devenir vide → on le supprime.

Ce post-traitement permet de **fusionner des bins sous-utilisés** et de réduire le coût sans passer par une itération ALNS complète. Une seule passe `rebuild_item_to_bin()` est appelée à la fin (et non à chaque déplacement) pour éviter les scans O(n) redondants.

---

## 10. Pipeline d'entraînement (`train_repair_model.py`)

Le script génère un dataset synthétique, entraîne le modèle, évalue ses performances et sauvegarde un **bundle** pickle.

### Génération du dataset

Pour chaque instance synthétique :
1. Tirer $n \in [\text{n\_min}, \text{n\_max}]$ items avec des tailles tirées selon 3 distributions possibles (uniforme, bimodale, normale).
2. Construire une solution BFD.
3. Pour chaque item, identifier `best_bin` (choix BFD) → **exemple positif**.
4. Échantillonner jusqu'à `max_negatives` autres bins feasibles → **exemples négatifs**.

### Phases d'entraînement

```
Phase 1 : Génération dataset (synthétique + éventuellement augmentation ALNS)
Phase 2 : Split train/test (85/15, stratifié)
Phase 3 : Entraînement GradientBoosting (optionnel : GridSearchCV)
Phase 4 : Cross-validation k-fold (ROC-AUC)
Phase 5 : Évaluation sur test set (AUC, F1, Precision, Recall, matrice de confusion)
Phase 6 : Visualisations (courbe ROC, importance des features, learning curves)
Phase 7 : Sauvegarde du bundle pkl
```

### Bundle sauvegardé

```python
{
    "model":           GradientBoostingClassifier,  # estimateur ajusté
    "scaler":          StandardScaler,              # scaler ajusté
    "feature_version": int,                        # FEATURE_VERSION
    "n_features":      int,                        # N_FEATURES = 11
    "metrics":         dict,                       # toutes les métriques
    "cv_scores":       list[float],                # scores CV
}
```

Le solveur **valide** ce bundle à chaque chargement : il vérifie `feature_version` et `n_features` pour rejeter les modèles incompatibles.

---

## 11. Collecte de données ALNS réelles (`collect_alns_states.py`) — DAgger-lite

### Problème : covariate shift

Le modèle est entraîné sur des états générés par BFD (imitation learning). Mais pendant ALNS, le solveur visite des états partiellement détruits que BFD ne produit jamais : des bins avec des charges arbitraires. Ce **covariate shift** dégrade la qualité de la réparation sur ces états.

### Solution : DAgger-lite

L'algorithme DAgger (Dataset Aggregation) résout ce problème en boucle : collecter des données on-policy, réentraîner, répéter. Ici, on fait une **version simplifiée** :

1. Exécuter l'ALNS actuel sur 500 instances synthétiques.
2. Intercepter chaque appel de réparation **avant** que le modèle ne décide.
3. Labelliser l'état capturé avec l'**oracle BFD** (quel bin BFD aurait choisi ?).
4. Sauvegarder ces `(X_alns, y_alns)` dans un fichier `.pkl`.
5. Fusionner avec le dataset BFD et réentraîner.

```bash
# Étape 1 : Collecter les états ALNS
python collect_alns_states.py \
    --model-path models/repair_model_v1.pkl \
    --instances 500 \
    --output models/alns_states_v1.pkl

# Étape 2 : Réentraîner avec augmentation
python train_repair_model.py \
    --instances 5000 \
    --augment-with models/alns_states_v1.pkl \
    --output models/repair_model_v2.pkl
```

"DAgger-lite" car on fait **une seule passe** offline, sans boucle d'itération.

---

## 12. Contrat de features et versioning

`features.py` définit le **contrat partagé** entre entraînement et inférence :

```python
FEATURE_VERSION = 2   # Incrémenter à chaque changement du vecteur de features
N_FEATURES = 11       # Taille fixe du vecteur
```

**Pourquoi ce mécanisme ?** Si quelqu'un modifie les features (ajoute une colonne, change une normalisation), le modèle entraîné avant la modification produira des scores incorrects sans avertissement. Le versioning détecte cette incohérence au chargement :

```python
if version != _EXPECTED_FEATURE_VERSION:
    raise ValueError(f"Model feature_version={version} does not match ...")
```

**Règle :** toute modification de `features.py` doit incrémenter `FEATURE_VERSION` et nécessite un réentraînement du modèle.

---

## 13. Flux d'exécution complet (résumé)

```
solve(model_path=..., max_iterations=5000)
│
├── _load_model()           Charger + valider le bundle pkl
│
├── _build_ffd_start_solution()
│   └── BFD : trier items par taille desc → placer dans le bin le plus serré
│
└── Boucle ALNS (max_iterations)
    │
    ├── ThompsonSampling.select_arm()
    │   └── Choisir opérateur [Random | Worst-load | Related-item]
    │
    ├── _destroy_*(candidate, k_items)
    │   └── Retirer k items → liste displaced
    │
    ├── _repair_learned(candidate, displaced)
    │   └── Pour chaque item (ordre décroissant de taille) :
    │       ├── Construire features pour (item, tous bins feasibles)
    │       ├── scaler.transform()
    │       ├── model.predict_proba()
    │       └── Placer dans le bin au score max
    │
    ├── _consolidate(candidate)
    │   └── Essayer de vider les bins sous-chargés
    │
    ├── Acceptation SA : delta ≤ 0 ou exp(-delta/T) > U
    │
    ├── Mise à jour best solution si amélioration
    │
    ├── ThompsonSampling.update(arm, reward)
    │
    ├── Soft reheat si plateau
    │
    └── Early stop si no_improve_limit dépassé
```

---

## 14. Structure des fichiers

| Fichier | Rôle |
|---|---|
| [solver.py](../../5_hybrid_ml_metaheuristics/hybrid_alns/solver.py) | Solveur runtime — seul fichier à appeler pour résoudre |
| [features.py](../../5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/features.py) | Contrat de features partagé (entraînement + inférence) |
| [train_repair_model.py](../../5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/train_repair_model.py) | Entraîner un nouveau modèle de réparation |
| [collect_alns_states.py](../../5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/collect_alns_states.py) | Collecter des états on-policy pour réduire le covariate shift |
| `models/repair_model_v2.pkl` | Modèle pré-entraîné (version courante recommandée) |
| `models/alns_states_v1.pkl` | États ALNS collectés pour l'augmentation |

---

## 15. Comment utiliser le module

### Lancer un benchmark

```bash
# Depuis la racine du projet
python utilities/benchmarking.py \
    --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \
    --dataset falkenauer-u \
    --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v2.pkl,max_iterations=5000"
```

### Utiliser le solveur en Python

```python
from hybrid_alns.solver import BinPackingSolver

solver = BinPackingSolver(item_sizes=[30, 50, 20, 80, 10], bin_capacity=100)
solver.solve(model_path="models/repair_model_v2.pkl", max_iterations=5000)
solution = solver.get_solution()
print(f"Bins utilisés : {solution.total_bins_used}")
```

### Entraîner un nouveau modèle

```bash
# Entraînement basique
python 5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/train_repair_model.py \
    --instances 5000 \
    --output 5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v3.pkl

# Avec augmentation DAgger-lite (recommandé)
python 5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/collect_alns_states.py \
    --model-path 5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v2.pkl \
    --instances 500 \
    --output /tmp/alns_states.pkl

python 5_hybrid_ml_metaheuristics/hybrid_alns/repair_model_training/train_repair_model.py \
    --instances 5000 \
    --augment-with /tmp/alns_states.pkl \
    --output 5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v3.pkl
```

### Paramètres du solveur

| Paramètre | Défaut | Description |
|---|---|---|
| `model_path` | *requis* | Chemin vers le bundle `.pkl` du modèle |
| `max_iterations` | `5000` | Nombre maximal d'itérations ALNS |
| `initial_temperature` | `1/ln(2) ≈ 1.44` | Température initiale SA |
| `alpha_cool` | `0.9995` | Taux de refroidissement géométrique |
| `time_limit_seconds` | `None` | Limite de temps optionnelle |
