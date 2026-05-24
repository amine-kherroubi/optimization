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
