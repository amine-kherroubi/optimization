# LinUCB Contextual Bandit — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer `ThompsonSamplingBandit` (Beta-Bernoulli, sans état) par `LinUCBBandit` (bandit contextuel disjoint, Chu et al. ICML 2011) dans `hybrid_alns_solver.py`.

**Architecture:** `LinUCBBandit` maintient par bras une matrice A (d×d) et un vecteur b (d), mis à jour à chaque itération ALNS avec le vecteur de contexte et la récompense normalisée. Le contexte 5D encode l'état complet de la recherche (phase SA, stagnation, gap optimum, rayon destruction, progression). La récompense est normalisée par le gap au lower bound, récompensant davantage les améliorations proches de l'optimum.

**Tech Stack:** Python 3.10+, NumPy (calcul matriciel UCB), pytest (TDD). Aucune dépendance nouvelle.

**Fichier unique :** `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py`

**Référence :** Chu et al., "Contextual Bandits with Linear Payoff Functions", ICML 2011.

**Note sur le credit assignment :** La récompense est attribuée à l'opérateur de destruction mais le réparateur GBT contribue aussi au résultat — biais inévitable dans ALNS hybride, acceptable pour publication (même limitation dans la littérature ALNS + RL).

---

## Pourquoi LinUCB > Thompson Sampling ici

| Dimension | Thompson Sampling (actuel) | LinUCB (nouveau) |
|---|---|---|
| État pris en compte | Aucun | T/T0, stagnation, gap LB, k/n, progression |
| Paradoxe stagnation | `stagnation_ratio` adapte `k_adaptive_max` mais **n'informe pas** le choix de bras | La stagnation est une feature directe → le bandit apprend quand chaque opérateur est utile |
| Récompense | Binaire (tirage aléatoire) | Continue normalisée par gap → signal plus informatif |
| Exploration | Aléatoire (variance Beta) | UCB structuré : visite les bras sous-estimés en contexte similaire |
| Hyperparamètres | 0 (priors α=β=1) | 1 : `alpha` (poids UCB, défaut=1.0) |

---

## Récapitulatif des changements dans le fichier

```
hybrid_alns_solver.py
│
├── ligne 4-5     → docstring module : TS → LinUCB
├── ligne 89      → docstring classe BinPackingSolver : TS → LinUCB
├── lignes 69–85  → REMPLACER ThompsonSamplingBandit → LinUCBBandit + N_CONTEXT_FEATURES
├── ligne 156     → bandit = LinUCBBandit(...) + lower_bound = ceil(...)
├── ligne 164     → _ctx = _build_context(...) ; arm = bandit.select_arm(_ctx)
├── lignes 202–208 → récompense LinUCB + bandit.update(arm, _ctx, _reward)
└── fin de classe → AJOUTER _build_context() + _linucb_reward()
```

---

## Tâche 0 — Préparer la branche de travail

- [ ] **Créer et checkout la branche feature**

```bash
cd /home/ziadi/optimization
git checkout -b improve/online-rl-linucb
```

Expected output : `Switched to a new branch 'improve/online-rl-linucb'`
(Si la branche existe déjà : `git checkout improve/online-rl-linucb`)

---

## Tâche 1 — Écrire les tests unitaires (TDD : rouge d'abord)

**Files:**
- Create: `tests/hybrid_alns/test_linucb_bandit.py`

- [ ] **Créer le dossier tests**

```bash
mkdir -p tests/hybrid_alns
touch tests/__init__.py tests/hybrid_alns/__init__.py
```

- [ ] **Écrire les tests pour LinUCBBandit et les méthodes statiques**

Contenu de `tests/hybrid_alns/test_linucb_bandit.py` :

```python
"""Tests TDD pour LinUCBBandit, _build_context, _linucb_reward."""

import math
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

# Ces imports échouent avant l'implémentation — c'est voulu (rouge).
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import (
    LinUCBBandit,
    N_CONTEXT_FEATURES,
    BinPackingSolver,
)


# ───────────────────────────── N_CONTEXT_FEATURES ──────────────────────────

class TestConstant:
    def test_n_context_features_is_5(self):
        assert N_CONTEXT_FEATURES == 5


# ───────────────────────────── LinUCBBandit ────────────────────────────────

class TestLinUCBBanditInit:
    def test_init_creates_identity_matrices(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        assert len(bandit._A) == 3
        for A in bandit._A:
            np.testing.assert_array_equal(A, np.eye(5))

    def test_init_creates_zero_vectors(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        for b in bandit._b:
            np.testing.assert_array_equal(b, np.zeros(5))

    def test_alpha_stored(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5, alpha=2.5)
        assert bandit.alpha == 2.5


class TestLinUCBBanditSelectArm:
    def test_returns_valid_arm_index(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.ones(5)
        arm = bandit.select_arm(ctx)
        assert arm in {0, 1, 2}

    def test_returns_int(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        arm = bandit.select_arm(np.ones(5))
        assert isinstance(arm, int)

    def test_uniform_context_selects_arm_0_initially(self):
        # Toutes les matrices A sont identiques (eye), tous les b=0 → theta=0
        # Les scores UCB sont identiques → argmax retourne 0.
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        arm = bandit.select_arm(np.ones(5))
        assert arm == 0

    def test_biased_update_changes_selection(self):
        """Après mise à jour répétée du bras 2 avec récompense élevée, il doit être préféré."""
        bandit = LinUCBBandit(n_arms=3, n_features=5, alpha=0.1)
        ctx = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            bandit.update(2, ctx, 1.0)
        arm = bandit.select_arm(ctx)
        assert arm == 2


class TestLinUCBBanditUpdate:
    def test_update_modifies_A(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.ones(5)
        A_before = bandit._A[1].copy()
        bandit.update(1, ctx, 0.5)
        assert not np.allclose(bandit._A[1], A_before)

    def test_update_modifies_b(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.ones(5)
        bandit.update(0, ctx, 1.0)
        np.testing.assert_array_almost_equal(bandit._b[0], ctx)

    def test_update_correct_arm_only(self):
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.ones(5)
        bandit.update(1, ctx, 1.0)
        # Bras 0 et 2 ne doivent pas changer
        np.testing.assert_array_equal(bandit._A[0], np.eye(5))
        np.testing.assert_array_equal(bandit._A[2], np.eye(5))
        np.testing.assert_array_equal(bandit._b[0], np.zeros(5))
        np.testing.assert_array_equal(bandit._b[2], np.zeros(5))

    def test_A_update_formula(self):
        """A_k doit être augmentée de x @ x^T."""
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        bandit.update(0, ctx, 0.5)
        expected_A = np.eye(5) + np.outer(ctx, ctx)
        np.testing.assert_array_almost_equal(bandit._A[0], expected_A)

    def test_b_update_formula(self):
        """b_k doit être augmentée de r * x."""
        bandit = LinUCBBandit(n_arms=3, n_features=5)
        ctx = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        reward = 0.7
        bandit.update(0, ctx, reward)
        np.testing.assert_array_almost_equal(bandit._b[0], reward * ctx)


# ───────────────────────────── _build_context ──────────────────────────────

class TestBuildContext:
    def test_returns_ndarray(self):
        ctx = BinPackingSolver._build_context(
            temperature=0.5, t0=1.0,
            iterations_since_improvement=10, no_improve_limit=100,
            current_cost=8, lower_bound=6,
            k_items=3, n=20,
            iteration=50, max_iterations=500,
        )
        assert isinstance(ctx, np.ndarray)

    def test_output_shape(self):
        ctx = BinPackingSolver._build_context(
            temperature=0.5, t0=1.0,
            iterations_since_improvement=10, no_improve_limit=100,
            current_cost=8, lower_bound=6,
            k_items=3, n=20,
            iteration=50, max_iterations=500,
        )
        assert ctx.shape == (5,)

    def test_output_dtype_float64(self):
        ctx = BinPackingSolver._build_context(
            temperature=0.5, t0=1.0,
            iterations_since_improvement=10, no_improve_limit=100,
            current_cost=8, lower_bound=6,
            k_items=3, n=20,
            iteration=50, max_iterations=500,
        )
        assert ctx.dtype == np.float64

    def test_feature_0_temperature_ratio(self):
        """Feature 0 = T/T0."""
        ctx = BinPackingSolver._build_context(
            temperature=0.25, t0=1.0,
            iterations_since_improvement=0, no_improve_limit=100,
            current_cost=6, lower_bound=6,
            k_items=1, n=10,
            iteration=0, max_iterations=100,
        )
        assert pytest.approx(ctx[0]) == 0.25

    def test_feature_1_stagnation_ratio(self):
        """Feature 1 = iter_no_improve / limit."""
        ctx = BinPackingSolver._build_context(
            temperature=1.0, t0=1.0,
            iterations_since_improvement=50, no_improve_limit=100,
            current_cost=6, lower_bound=6,
            k_items=1, n=10,
            iteration=0, max_iterations=100,
        )
        assert pytest.approx(ctx[1]) == 0.5

    def test_feature_2_cost_over_lb(self):
        """Feature 2 = current_cost / lower_bound."""
        ctx = BinPackingSolver._build_context(
            temperature=1.0, t0=1.0,
            iterations_since_improvement=0, no_improve_limit=100,
            current_cost=9, lower_bound=6,
            k_items=1, n=10,
            iteration=0, max_iterations=100,
        )
        assert pytest.approx(ctx[2]) == 1.5

    def test_feature_3_destruction_radius(self):
        """Feature 3 = k_items / n."""
        ctx = BinPackingSolver._build_context(
            temperature=1.0, t0=1.0,
            iterations_since_improvement=0, no_improve_limit=100,
            current_cost=6, lower_bound=6,
            k_items=2, n=20,
            iteration=0, max_iterations=100,
        )
        assert pytest.approx(ctx[3]) == 0.1

    def test_feature_4_search_progress(self):
        """Feature 4 = iteration / max_iterations."""
        ctx = BinPackingSolver._build_context(
            temperature=1.0, t0=1.0,
            iterations_since_improvement=0, no_improve_limit=100,
            current_cost=6, lower_bound=6,
            k_items=1, n=10,
            iteration=250, max_iterations=500,
        )
        assert pytest.approx(ctx[4]) == 0.5

    def test_no_division_by_zero_all_ones(self):
        """Dénominateurs nuls → aucune erreur grâce aux max(..., eps)."""
        ctx = BinPackingSolver._build_context(
            temperature=0.0, t0=0.0,
            iterations_since_improvement=0, no_improve_limit=0,
            current_cost=0, lower_bound=0,
            k_items=0, n=0,
            iteration=0, max_iterations=0,
        )
        assert np.all(np.isfinite(ctx))


# ───────────────────────────── _linucb_reward ──────────────────────────────

class TestLinUCBReward:
    def test_rejected_returns_zero(self):
        r = BinPackingSolver._linucb_reward(
            delta=0, accepted=False, improved=False,
            current_cost=8, lower_bound=6,
        )
        assert r == 0.0

    def test_accepted_no_improvement_returns_02(self):
        r = BinPackingSolver._linucb_reward(
            delta=0, accepted=True, improved=False,
            current_cost=8, lower_bound=6,
        )
        assert pytest.approx(r) == 0.2

    def test_improvement_normalised_by_gap(self):
        """Sauver 1 bin avec gap=2 → r = 1/2 = 0.5."""
        # delta = -1 (cost improved by 1 bin)
        r = BinPackingSolver._linucb_reward(
            delta=-1, accepted=True, improved=True,
            current_cost=8, lower_bound=6,
        )
        # gap = current_cost - lower_bound = 8 - 6 = 2, bins_saved = 1
        assert pytest.approx(r) == 0.5

    def test_improvement_capped_at_1(self):
        """bins_saved > gap → récompense plafonnée à 1.0."""
        r = BinPackingSolver._linucb_reward(
            delta=-5, accepted=True, improved=True,
            current_cost=8, lower_bound=6,
        )
        assert pytest.approx(r) == 1.0

    def test_near_optimal_higher_reward(self):
        """À gap=1, sauver 1 bin → r=1.0. Même gain loin (gap=10) → r=0.1."""
        r_near = BinPackingSolver._linucb_reward(
            delta=-1, accepted=True, improved=True,
            current_cost=7, lower_bound=6,
        )
        r_far = BinPackingSolver._linucb_reward(
            delta=-1, accepted=True, improved=True,
            current_cost=16, lower_bound=6,
        )
        assert r_near > r_far

    def test_reward_in_0_1(self):
        for delta in [-3, -1, 0, 1]:
            for accepted in [True, False]:
                for improved in [True, False]:
                    r = BinPackingSolver._linucb_reward(
                        delta=delta, accepted=accepted, improved=improved,
                        current_cost=10, lower_bound=6,
                    )
                    assert 0.0 <= r <= 1.0
```

- [ ] **Vérifier que les tests échouent (rouge)**

```bash
cd /home/ziadi/optimization
python3 -m pytest tests/hybrid_alns/test_linucb_bandit.py -v 2>&1 | head -40
```

Expected : `ImportError: cannot import name 'LinUCBBandit'` ou `ModuleNotFoundError`

---

## Tâche 2 — Remplacer `ThompsonSamplingBandit` par `LinUCBBandit`

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py:69-85`

- [ ] **Remplacer les lignes 69–85**

Supprimer :
```python
class ThompsonSamplingBandit:
    """Beta-Bernoulli Thompson Sampling for destroy-operator selection."""

    __slots__ = ("alpha", "beta")

    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms, dtype=np.float64)
        self.beta = np.ones(n_arms, dtype=np.float64)

    def select_arm(self, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: int) -> None:
        if reward == 1:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
```

Mettre à la place :
```python
N_CONTEXT_FEATURES = 5  # [T/T0, stagnation/limit, cost/LB, k/n, iter/max_iter]


class LinUCBBandit:
    """Disjoint LinUCB contextual bandit for destroy-operator selection.

    Each arm k maintains A_k (d×d covariance) and b_k (d reward vector).
    Selection: argmax_k [ θ_k^T x + α √(x^T A_k^{-1} x) ]
    Update:    A_k += x x^T,  b_k += r * x

    Reference: Chu et al., ICML 2011.
    """

    __slots__ = ("alpha", "_A", "_b", "_n_arms")

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._n_arms = n_arms
        self._A: list[np.ndarray] = [np.eye(n_features) for _ in range(n_arms)]
        self._b: list[np.ndarray] = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Return arm with highest UCB score for the given context vector."""
        scores = np.empty(self._n_arms)
        for k in range(self._n_arms):
            A_inv = np.linalg.inv(self._A[k])
            theta = A_inv @ self._b[k]
            scores[k] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update arm k with observed (context, reward)."""
        self._A[arm] += np.outer(context, context)
        self._b[arm] += reward * context
```

- [ ] **Vérifier l'import de base**

```bash
cd /home/ziadi/optimization
python3 -c "
import sys; sys.path.insert(0, '.')
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import LinUCBBandit, N_CONTEXT_FEATURES
print('LinUCBBandit importé OK')
print('N_CONTEXT_FEATURES =', N_CONTEXT_FEATURES)
"
```

Expected :
```
LinUCBBandit importé OK
N_CONTEXT_FEATURES = 5
```

- [ ] **Lancer les tests LinUCBBandit (vert)**

```bash
python3 -m pytest tests/hybrid_alns/test_linucb_bandit.py -k "TestConstant or TestLinUCB" -v
```

Expected : tous les tests `TestConstant`, `TestLinUCBBanditInit`, `TestLinUCBBanditSelectArm`, `TestLinUCBBanditUpdate` → PASSED

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git add tests/
git commit -m "refactor: replace ThompsonSamplingBandit with LinUCBBandit"
```

---

## Tâche 3 — Mettre à jour `solve()` : initialisation

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py:156`

La ligne 156 est dans `solve()`, juste après `k_max = ...` :

```python
# AVANT (ligne 156)
bandit = ThompsonSamplingBandit(n_arms=3)
```

- [ ] **Remplacer par**

```python
lower_bound = math.ceil(sum(self._item_sizes) / self._bin_capacity)
bandit = LinUCBBandit(n_arms=3, n_features=N_CONTEXT_FEATURES, alpha=1.0)
```

> `lower_bound` (LB1) = ceil(Σsize / capacity) — borne inférieure continue sur le nombre minimal de bins. Utilisée dans le reward et dans la feature `cost/LB`.

- [ ] **Vérifier la syntaxe**

```bash
cd /home/ziadi/optimization
python3 -c "
import sys; sys.path.insert(0, '.')
import ast, pathlib
src = pathlib.Path('bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py').read_text()
ast.parse(src)
print('syntaxe OK')
"
```

Expected : `syntaxe OK`

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git commit -m "refactor: instantiate LinUCBBandit with lower_bound in solve()"
```

---

## Tâche 4 — Mettre à jour `solve()` : sélection de bras

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py:164`

La ligne 164 actuelle (dans la boucle for iteration) :
```python
arm = bandit.select_arm(self._rng)
```

- [ ] **Remplacer par**

```python
_ctx = self._build_context(
    temperature, t0,
    iterations_since_improvement, no_improve_limit,
    current.cost(), lower_bound,
    k_min, n,
    iteration, max_iterations,
)
arm = bandit.select_arm(_ctx)
```

> **Pourquoi `k_min` et non `k_items` ?** `k_items` est tiré *après* la sélection du bras (ligne suivante). On utilise donc `k_min` comme approximation du rayon de destruction — le contexte décrit l'état *avant* destruction, ce qui est correct pour un bandit en ligne.

> `_ctx` est stocké comme variable locale car on en a besoin dans `bandit.update()` plus loin dans la même itération (Tâche 5).

- [ ] **Vérifier la syntaxe**

```bash
cd /home/ziadi/optimization
python3 -c "
import ast, pathlib
src = pathlib.Path('bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py').read_text()
ast.parse(src)
print('syntaxe OK')
"
```

Expected : `syntaxe OK`

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git commit -m "refactor: LinUCB arm selection with context vector"
```

---

## Tâche 5 — Mettre à jour `solve()` : reward et update

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py:202-208`

Lignes 202–208 actuelles :
```python
# 3-level reward: new best (1.0) > accepted (0.5) > rejected (0.0).
# Thompson Sampling expects binary feedback, so we convert the
# continuous reward probabilistically: update with 1 if a uniform
# draw falls below the reward, 0 otherwise. This preserves the
# expected value while keeping the Beta posterior well-calibrated.
reward = 1.0 if improved else (0.5 if accepted else 0.0)
bandit.update(arm, 1 if self._rng.random() < reward else 0)
```

- [ ] **Remplacer par**

```python
# LinUCB reward: improvement normalised by gap to lower bound.
# Saving bins near the optimum (gap small) earns more than saving
# the same number when far away (gap large). Accepted-without-saving
# earns a small signal (0.2) to keep exploration; rejected earns 0.
_reward = self._linucb_reward(
    delta, accepted, improved, current.cost(), lower_bound
)
bandit.update(arm, _ctx, _reward)
```

> **Note :** `_ctx` a été construit juste avant `arm = bandit.select_arm(_ctx)` dans la même itération — il est disponible ici.
> **Note :** `current.cost()` est évalué *après* l'acceptation éventuelle du candidat, donc reflète bien l'état courant au moment du feedback.

- [ ] **Vérifier la syntaxe**

```bash
cd /home/ziadi/optimization
python3 -c "
import ast, pathlib
src = pathlib.Path('bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py').read_text()
ast.parse(src)
print('syntaxe OK')
"
```

Expected : `syntaxe OK`

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git commit -m "refactor: LinUCB normalised reward and bandit.update"
```

---

## Tâche 6 — Ajouter les 2 méthodes statiques à `BinPackingSolver`

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py` (fin de classe, juste avant `_validate_model_components`)

- [ ] **Insérer juste avant la méthode `_validate_model_components` (ligne ~493)**

```python
    @staticmethod
    def _build_context(
        temperature: float,
        t0: float,
        iterations_since_improvement: int,
        no_improve_limit: int,
        current_cost: int,
        lower_bound: int,
        k_items: int,
        n: int,
        iteration: int,
        max_iterations: int,
    ) -> np.ndarray:
        """Build the 5-D context vector for LinUCB arm selection.

        Features (all normalised to comparable ranges):
          0  T / T0                        — SA phase (1→0, hot→cold)
                                             Hot = exploration phase, cold = exploitation
          1  iter_no_improve / limit       — stagnation (0→1)
                                             Signals when diversification is needed
          2  current_cost / LB            — gap to optimum (≥1)
                                             1.0 = optimal; >1 = room for improvement
          3  k_items / n                  — destruction radius (0.05–0.25)
                                             Captures the current neighbourhood size
          4  iteration / max_iterations   — search progress (0→1)
                                             Early vs late stage of the search
        """
        return np.array(
            [
                temperature / max(t0, 1e-12),
                iterations_since_improvement / max(no_improve_limit, 1),
                current_cost / max(lower_bound, 1),
                k_items / max(n, 1),
                iteration / max(max_iterations, 1),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _linucb_reward(
        delta: int,
        accepted: bool,
        improved: bool,
        current_cost: int,
        lower_bound: int,
    ) -> float:
        """Normalised reward in [0, 1] for LinUCB update.

        Improvement is divided by the gap to LB so that gains near the
        optimum are rewarded more than the same gain when far away.

          r = min(1, bins_saved / gap)   if bins were saved (improved=True)
          r = 0.2                        if accepted without saving bins
          r = 0.0                        if rejected

        Note: credit assignment — reward is attributed to the destroy operator
        but the GBT repair model also contributes. This is an accepted
        limitation in hybrid ALNS + RL literature.
        """
        gap = max(1, current_cost - lower_bound)
        bins_saved = max(0, -delta)
        if bins_saved > 0:
            return min(1.0, bins_saved / gap)
        return 0.2 if accepted else 0.0
```

- [ ] **Lancer tous les tests (vert)**

```bash
cd /home/ziadi/optimization
python3 -m pytest tests/hybrid_alns/test_linucb_bandit.py -v
```

Expected : **tous les tests PASSED**

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git commit -m "feat: add _build_context and _linucb_reward to BinPackingSolver"
```

---

## Tâche 7 — Mettre à jour les docstrings

**Fichier modifié :**
- Modify: `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py:4-5,89`

- [ ] **Ligne 4-5 — docstring module**

Changer :
```python
- Thompson Sampling to choose destroy operators
```
En :
```python
- LinUCB contextual bandit to choose destroy operators (Chu et al., ICML 2011)
```

- [ ] **Ligne 89 — docstring de classe BinPackingSolver**

Changer :
```python
"""Single-path hybrid ALNS solver: Thompson bandit + learned repair."""
```
En :
```python
"""Single-path hybrid ALNS solver: LinUCB contextual bandit + learned repair."""
```

- [ ] **Vérifier la syntaxe**

```bash
cd /home/ziadi/optimization
python3 -c "
import ast, pathlib
src = pathlib.Path('bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py').read_text()
ast.parse(src)
print('syntaxe OK')
"
```

Expected : `syntaxe OK`

- [ ] **Commit**

```bash
git add bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
git commit -m "docs: update module and class docstrings TS → LinUCB"
```

---

## Tâche 8 — Test de fumée (intégration complète)

Le test de fumée utilise un **modèle mock** car les fichiers `.pkl` ont une incompatibilité NumPy 1.x/2.x dans l'environnement. L'API correcte du solver est `model_bundle=dict` (plus de `model_path=`).

- [ ] **Lancer le test de fumée**

```bash
cd /home/ziadi/optimization
python3 - <<'EOF'
import sys, math
sys.path.insert(0, ".")
import numpy as np
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import BinPackingSolver

# Mock model bundle : feature_version=2, n_features=11 (cf. features.py)
class _MockModel:
    def predict_proba(self, X):
        rng = np.random.default_rng(0)
        probs = rng.random((len(X), 2))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

mock_bundle = {
    "model": _MockModel(),
    "scaler": None,
    "feature_version": 2,
    "n_features": 11,
}

sizes    = [50, 40, 30, 20, 10, 60, 70, 80, 90, 15, 25, 35, 45, 55, 65]
capacity = 100

solver = BinPackingSolver(sizes, capacity, seed=42)
solver.solve(
    model_bundle=mock_bundle,
    max_iterations=300,
)
sol = solver.get_solution()
lb = math.ceil(sum(sizes) / capacity)
print(f"LB      : {lb}")
print(f"Bins    : {sol.total_bins_used}")
assert sol.total_bins_used >= lb, f"Infeasible: {sol.total_bins_used} < LB={lb}"
print("OK ✅")
EOF
```

Expected output :
```
LB      : 7
Bins    : <nombre >= 7>
OK ✅
```

- [ ] **Vérifier que ThompsonSamplingBandit n'existe plus**

```bash
grep -n "ThompsonSampling" bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/hybrid_alns_solver.py
```

Expected : aucune ligne (sortie vide)

- [ ] **Commit final et push**

```bash
git add .
git commit -m "test: LinUCB smoke test passed"
git push origin improve/online-rl-linucb
```

---

## Récapitulatif des changements

```
hybrid_alns_solver.py
│
├── ligne 4-5      → docstring module : TS → LinUCB
├── ligne 89       → docstring classe : TS → LinUCB
├── lignes 69–85   → REMPLACER ThompsonSamplingBandit → N_CONTEXT_FEATURES + LinUCBBandit
├── ligne 156      → lower_bound = ceil(...) + bandit = LinUCBBandit(...)
├── ligne 164      → _ctx = _build_context(...) ; arm = bandit.select_arm(_ctx)
├── lignes 202–208 → _reward = _linucb_reward(...) ; bandit.update(arm, _ctx, _reward)
└── fin de classe  → AJOUTER _build_context() + _linucb_reward()

tests/hybrid_alns/test_linucb_bandit.py
└── 6 classes, ~35 tests : N_CONTEXT_FEATURES, LinUCBBandit (init/select/update),
                           _build_context (shape/dtype/chaque feature),
                           _linucb_reward (tous les cas)
```

## Vecteur de contexte — justification des 5 features

| idx | Feature | Motivation |
|-----|---------|-----------|
| 0 | `T / T0` | Phase SA. Chaud = exploration (tous opérateurs utiles), froid = exploitation (related-item pertinent). Le TS ignorait ce signal. |
| 1 | `stagnation / limit` | Plateau détecté. Quand ce ratio → 1, random-destroy doit être favorisé pour diversifier. Le TS ignorait la stagnation malgré l'utilisation de `stagnation_ratio` pour `k_adaptive_max`. |
| 2 | `cost / LB` | Gap à l'optimum. Près de la borne (ratio ≈ 1), worst-load-destroy est plus précis ; loin, on peut se permettre une destruction plus large. |
| 3 | `k_min / n` | Rayon de destruction relatif. Encode si on opère sur un problème "petit" ou "grand" relativement. |
| 4 | `iteration / max_iterations` | Progression globale. Début = exploration, fin = exploitation fine. Complémentaire à T/T0 (le reheat thermique décorèle les deux). |
