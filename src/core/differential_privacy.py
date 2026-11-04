"""
Differential Privacy Module (Laplace Mechanism).
Implements calibrated noise addition, global sensitivity derivation,
analytical error bounds (MSE = 2b^2, MAE = b), and query execution engines.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import SYSTOLIC_BP_LOWER, SYSTOLIC_BP_UPPER


class LaplaceMechanism:
    """
    Standard Laplace Mechanism for numerical queries under Differential Privacy.
    Satisfies epsilon-differential privacy: Pr[M(D1) in S] <= exp(epsilon) * Pr[M(D2) in S].
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def sample_noise(self, scale: float, size: int = 1) -> np.ndarray:
        """
        Sample noise from Laplace(0, scale = Delta_f / epsilon).
        """
        if scale <= 0:
            return np.zeros(size)
        return self.rng.laplace(loc=0.0, scale=scale, size=size)

    @staticmethod
    def theoretical_bounds(sensitivity: float, epsilon: float) -> Dict[str, float]:
        """
        Compute theoretical noise scale b, MAE, and MSE:
            b = Delta_f / epsilon
            MAE = b
            MSE = 2 * b^2
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive (> 0).")
        b = sensitivity / epsilon
        mae = b
        mse = 2.0 * (b ** 2)
        return {
            "scale_b": round(b, 6),
            "theoretical_mae": round(mae, 6),
            "theoretical_mse": round(mse, 6)
        }

    def private_count(
        self,
        count: int,
        epsilon: float,
        clip_non_negative: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute epsilon-differentially private count query.
        Global sensitivity Delta_f = 1 (one individual changes the count by at most 1).
        """
        sensitivity = 1.0
        b = sensitivity / epsilon
        noise = float(self.sample_noise(scale=b, size=1)[0])
        noisy_count = float(count) + noise
        if clip_non_negative:
            noisy_count = max(0.0, noisy_count)

        bounds = self.theoretical_bounds(sensitivity, epsilon)
        return round(noisy_count, 2), bounds

    def private_mean(
        self,
        values: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        epsilon: float,
        known_n: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute epsilon-differentially private mean for bounded values.
        Global sensitivity:
            Delta_f = (upper_bound - lower_bound) / N
        (Assuming N is public and one record replacement).
        """
        arr = np.asarray(values, dtype=float)
        n = known_n if known_n is not None else len(arr)
        if n == 0:
            raise ValueError("Dataset cannot be empty for mean query.")

        # Clip values to guaranteed public bounds
        clipped = np.clip(arr, lower_bound, upper_bound)
        true_mean = float(np.mean(clipped))

        sensitivity = (upper_bound - lower_bound) / float(n)
        b = sensitivity / epsilon
        noise = float(self.sample_noise(scale=b, size=1)[0])
        noisy_mean = true_mean + noise

        bounds = self.theoretical_bounds(sensitivity, epsilon)
        return round(noisy_mean, 4), bounds

    def private_bounded_sum(
        self,
        values: np.ndarray,
        upper_bound: float,
        epsilon: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute epsilon-differentially private sum for values bounded in [0, upper_bound].
        Global sensitivity Delta_f = upper_bound.
        """
        arr = np.clip(np.asarray(values, dtype=float), 0.0, upper_bound)
        true_sum = float(np.sum(arr))

        sensitivity = float(upper_bound)
        b = sensitivity / epsilon
        noise = float(self.sample_noise(scale=b, size=1)[0])
        noisy_sum = true_sum + noise

        bounds = self.theoretical_bounds(sensitivity, epsilon)
        return round(noisy_sum, 2), bounds
