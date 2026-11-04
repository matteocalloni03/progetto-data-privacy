"""
Unit tests for Differential Privacy and Privacy Budget Manager (FR-06, FR-07).
"""

import pytest
import numpy as np
from src.core.differential_privacy import LaplaceMechanism
from src.gateway.budget_manager import PrivacyBudgetManager


def test_laplace_count_query():
    dp = LaplaceMechanism(seed=42)
    true_count = 100
    noisy_count, bounds = dp.private_count(true_count, epsilon=1.0)
    assert abs(noisy_count - true_count) < 20  # high probability bound
    assert bounds["scale_b"] == 1.0
    assert bounds["theoretical_mae"] == 1.0
    assert bounds["theoretical_mse"] == 2.0


def test_laplace_mean_query_sensitivity():
    dp = LaplaceMechanism(seed=42)
    values = np.array([120.0, 130.0, 140.0, 150.0])
    lower = 90.0
    upper = 210.0
    n = 4
    # Sensitivity = (210 - 90) / 4 = 120 / 4 = 30.0
    # For epsilon = 1.0 -> b = 30.0, MAE = 30.0, MSE = 2 * 900 = 1800.0
    noisy_mean, bounds = dp.private_mean(values, lower, upper, epsilon=1.0, known_n=n)
    assert bounds["scale_b"] == 30.0
    assert bounds["theoretical_mae"] == 30.0
    assert bounds["theoretical_mse"] == 1800.0


def test_privacy_budget_manager():
    mgr = PrivacyBudgetManager(default_budget=2.0)
    user = "USR-TEST-001"
    mgr.register_user(user, allocated_budget=2.0)

    # First query: eps = 0.5 -> remaining 1.5
    ok, msg, rem = mgr.check_and_consume_budget(user, 0.5)
    assert ok is True
    assert round(rem, 4) == 1.5

    # Second query: eps = 1.0 -> remaining 0.5
    ok, msg, rem = mgr.check_and_consume_budget(user, 1.0)
    assert ok is True
    assert round(rem, 4) == 0.5

    # Third query: eps = 0.8 -> exceeds 0.5 -> must be rejected
    ok, msg, rem = mgr.check_and_consume_budget(user, 0.8)
    assert ok is False
    assert "budget exhausted" in msg.lower()
    assert round(rem, 4) == 0.5

    # Fourth query: eps = 0.5 -> exactly consumes remaining
    ok, msg, rem = mgr.check_and_consume_budget(user, 0.5)
    assert ok is True
    assert round(rem, 4) == 0.0
