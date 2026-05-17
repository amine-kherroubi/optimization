# Design — Corrections hybrid_alns (13 problèmes)

**Date** : 2026-05-18  
**Projet** : `5_hybrid_ml_metaheuristics/hybrid_alns`  
**Objectif** : corriger les 13 problèmes identifiés dans `REVIEW_DEBUTANT.md`, dans un ordre qui respecte les dépendances entre corrections.  
**Approche** : par couche — fondations d'abord, quick wins en dernier.

---

## Contexte

Le solveur hybride ALNS combine :
- Thompson Sampling Beta-Bernoulli pour choisir un destroy operator
- Simulated Annealing pour l'acceptation
- Logistic Regression entraînée par imitation de BFD pour la réparation

13 problèmes ont été identifiés (2 critiques, 4 importants, 7 mineurs). Ils sont corrigés en 5 couches successives.

---

## Couche 1 — Fondations

**Problèmes** : P2, P13  
**Fichiers** : `features.py` (nouveau), `solver.py`, `train_repair_model.py`

### P2 — Créer `features.py` partagé

**Problème** : `_make_repair_features` (solver.py) et `_make_features` (train_repair_model.py) sont deux copies identiques du même calcul de features. Un changement dans l'une qui n'est pas répercuté dans l'autre produit un mismatch silencieux.

**Design** :
- Créer `5_hybrid_ml_metaheuristics/hybrid_alns/features.py`
- Y déplacer la logique de calcul des features, les constantes `N_FEATURES` et `FEATURE_VERSION`
- `solver.py` et `train_repair_model.py` importent depuis `features.py`
- La signature unifiée prend `capacity: float` explicitement (le training utilisait `capacity=1.0` implicitement, le solver divise par `bin_capacity`)

```python
# features.py
FEATURE_VERSION = 2
N_FEATURES = 11

def make_features(
    item: int,
    item_size: float,
    bin_items: list[int],
    bin_load: float,
    capacity: float,
    sizes: list[float],
    n_total: int,
    size_rank: dict[int, int],
    remaining_ratio: float,
) -> list[float]:
    ...
```

**Critère de succès** : comportement identique à avant (mêmes features produites), aucune duplication restante.

### P13 — Seed RNG configurable

**Problème** : `self._rng = np.random.default_rng(42)` est hardcodé. Impossible de faire plusieurs runs indépendants pour mesurer la variance.

**Design** :
- Ajouter `seed: int | None = 42` au constructeur `BinPackingSolver.__init__`
- `self._rng = np.random.default_rng(seed)`
- Comportement par défaut inchangé (seed=42)

---

## Couche 2 — Cœur ALNS

**Problèmes** : P1, P4, P5  
**Fichiers** : `solver.py`

### P5 — Time limit

**Problème** : le solver tourne exactement `max_iterations` fois sans notion de budget temps.

**Design** :
- Ajouter `time_limit_seconds: float | None = None` dans `solve()`
- En début de boucle : `if deadline and time.perf_counter() > deadline: break`
- Les deux contraintes coexistent : la boucle s'arrête quand la première est atteinte

### P1 — Reward bandit 3 niveaux

**Problème** : reward=1 seulement si `accepted AND improved_best`, reward=0 sinon. Un move accepté mais non-meilleur est traité comme un move rejeté.

**Design** :
Scoring à 3 niveaux :
- Nouveau meilleur global → `reward = 1.0`
- Accepté mais pas meilleur → `reward = 0.5`
- Rejeté → `reward = 0.0`

Thompson Sampling utilise des rewards binaires. Conversion probabiliste :
```python
reward = 1.0 if improved else (0.5 if accepted else 0.0)
bandit.update(arm, 1 if self._rng.random() < reward else 0)
```

Cela préserve l'espérance : l'arm "nouveau meilleur" converge toujours vers alpha=1, les autres se différencient par leur reward moyen.

### P4 — k adaptatif selon stagnation

**Problème** : `k_min` et `k_max` ne changent jamais. En cas de stagnation, le solver ne diversifie pas.

**Design** :
- Tracker `iterations_since_improvement` (remis à 0 à chaque amélioration du best)
- `stagnation_ratio = iterations_since_improvement / max_iterations`
- `k_effective_max = k_min + int(stagnation_ratio * (k_max - k_min))`
- `k_items = rng.integers(k_min, max(k_min + 1, k_effective_max + 1))`

Au début : k reste petit (intensification). Après stagnation prolongée : k monte vers k_max (diversification).

---

## Couche 3 — Pipeline ML

**Problèmes** : P9, P8, P3  
**Fichiers** : `train_repair_model.py`

### P9 — destroy_fraction variable

**Problème** : `destroy_fraction=0.20` fixe. L'ALNS détruit entre 5% et 25% des items. 20% des bins ≠ 20% des items quand les bins ont des tailles variées.

**Design** :
- Dans `build_dataset`, tirer `destroy_fraction = rng.uniform(0.05, 0.40)` pour chaque instance
- Passer cette valeur à `extract_repair_examples`
- Le paramètre CLI `--destroy-fraction` devient la valeur centrale (conservé pour compatibilité mais non utilisé dans la boucle de génération)

### P8 — Distributions d'entraînement mixtes

**Problème** : `rng.uniform(0.1, 0.9)` uniquement. Manque les instances bimodales, gaussiennes, etc.

**Design** :
- Modifier `generate_instance` pour choisir aléatoirement parmi 3 distributions :
  - **Uniforme** [0.1, 0.9] — cas actuel, 40% du temps
  - **Bimodale** — moitié dans [0.1, 0.35], moitié dans [0.6, 0.9], 40% du temps
  - **Gaussienne** — N(0.5, 0.2) clampée dans [0.05, 0.95], 20% du temps
- Distributions normalisées pour rester dans (0, 1) avec capacity implicite = 1.0

### P3 — LR → GradientBoostingClassifier

**Problème** : Logistic Regression ne capture pas les interactions non-linéaires entre features (ex: item_size × remaining_capacity).

**Design** :
- Remplacer `LogisticRegressionCV` par `GradientBoostingClassifier`
- Hyperparamètres : `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8`
- Conserver `StandardScaler` dans le Pipeline (utile pour GBM aussi)
- Conserver `class_weight` equivalent via `sample_weight` dans `fit()`
- Scoring toujours ROC-AUC
- Le bundle sauvegardé est identique (`model`, `scaler`, `feature_version`, `n_features`)

**Note** : GBM n'a pas de `class_weight` natif dans sklearn. Alternative : calculer les poids avec `compute_sample_weight("balanced", y)` et les passer à `fit()`.

---

## Couche 4 — Biais d'imitation

**Problèmes** : P6  
**Fichiers** : `train_repair_model.py`, nouveau `collect_alns_states.py`

### P6 — Collecte d'états ALNS réels

**Problème** : le modèle apprend à imiter BFD sur des états BFD. Pendant ALNS, le solver visite des états que BFD ne produit jamais. Covariate shift.

**Design** (DAgger simplifié — sans oracle interactif) :
1. Faire tourner le solver (après couches 1-3) sur N instances synthétiques
2. À chaque appel de `_repair_learned`, capturer les repair states réels (bins actuels + items déplacés)
3. Étiqueter ces états avec BFD (meilleur bin selon minimum slack)
4. Ajouter ces exemples au dataset d'entraînement existant (ratio ~30% états ALNS, 70% états BFD)
5. Ré-entraîner le modèle sur le dataset augmenté

**Nouveau fichier** `collect_alns_states.py` :
- Accepte le modèle existant et génère des états ALNS
- Produit un fichier `alns_states.pkl` avec (X, y) supplémentaires
- `train_repair_model.py` accepte `--augment-with alns_states.pkl` pour fusionner les datasets

**Critère de succès** : qualité de solution égale ou meilleure sur des instances bimodales (non vues en training BFD pur).

---

## Couche 5 — Quick wins

**Problèmes** : P11, P7, P12, P10  
**Fichiers** : `solver.py`

### P11 — Warning si `method` non-None

```python
import warnings
if method is not None:
    warnings.warn(
        f"method={method!r} is ignored; BinPackingSolver runs a single ALNS pipeline.",
        stacklevel=2,
    )
```

### P7 — swap-with-last O(1) dans `_destroy_related`

Remplacer `sol.bins[j].remove(item)` (O(n)) par :
```python
lst = sol.bins[j]
idx = lst.index(item)
lst[idx] = lst[-1]
lst.pop()
```

### P12 — FFD avec heapq

Remplacer le scan linéaire des bins par un tas (min-heap) sur la capacité restante :
- `heapq` stocke `(-remaining_capacity, bin_index)` (max-heap simulé)
- Pour chaque item : pop le bin avec le plus de capacité restante, vérifier le fit, push la mise à jour
- Complexité : O(n log b) au lieu de O(n × b) où b = nombre de bins

### P10 — Local search post-réparation (consolidation)

Après `_repair_learned`, appeler `_consolidate` :
- Parcourir les bins du moins chargé au plus chargé
- Tenter de déplacer chaque item vers un autre bin avec assez de place
- Supprimer les bins devenus vides
- Appeler `rebuild_item_to_bin()` une seule fois en fin de méthode (pas à chaque déplacement)

---

## Ordre de commits suggéré

```
commit 1  : P2 — créer features.py, adapter solver.py et train_repair_model.py
commit 2  : P13 — seed configurable dans BinPackingSolver
commit 3  : P5 — time_limit_seconds dans solve()
commit 4  : P1 — reward bandit 3 niveaux
commit 5  : P4 — k adaptatif selon stagnation
commit 6  : P9 — destroy_fraction variable dans build_dataset
commit 7  : P8 — distributions mixtes dans generate_instance
commit 8  : P3 — LR → GradientBoostingClassifier (+ réentraînement modèle)
commit 9  : P6 — collect_alns_states.py + augmentation dataset (+ réentraînement)
commit 10 : P11 — warning method parameter
commit 11 : P7 — swap-with-last dans _destroy_related
commit 12 : P12 — FFD avec heapq
commit 13 : P10 — local search post-réparation (_consolidate)
```

---

## Critères de succès globaux

- `solver.py` et `train_repair_model.py` n'ont plus de feature engineering dupliqué
- Le solver accepte `seed`, `time_limit_seconds` en paramètres
- AUC du modèle de réparation > 0.85 (vs ~0.77 avec LR)
- La qualité de solution (nb bins) est égale ou meilleure qu'avant sur les benchmarks Falkenauer
- Tous les commits sont atomiques et documentés
