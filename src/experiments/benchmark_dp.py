"""
Benchmark 1: Differential Privacy Mean Blood Pressure Error vs Epsilon.
Replicates Table 12 and Figure 2(a) from the study.
Evaluates empirical vs theoretical MAE and MSE across epsilon values
for both the full patient cohort (N=1000) and the small oncology sub-group (N=64).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.config import SYSTOLIC_BP_LOWER, SYSTOLIC_BP_UPPER
from src.core.synthetic_data import load_or_generate_dataset
from src.core.differential_privacy import LaplaceMechanism


def run_dp_mean_benchmark(
    epsilons: List[float] = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    n_repetitions: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Execute Monte Carlo simulation for DP mean systolic blood pressure.
    """
    df = load_or_generate_dataset()
    dp = LaplaceMechanism(seed=seed)

    all_values = df["blood_pressure_systolic"].to_numpy()
    onco_values = df[df["is_oncology"] == True]["blood_pressure_systolic"].to_numpy()

    n_all = len(all_values)
    n_onco = len(onco_values)

    true_mean_all = float(np.mean(all_values))
    true_mean_onco = float(np.mean(onco_values))

    results = []

    for eps in epsilons:
        # Full cohort sensitivity & theoretical bounds
        sens_all = (SYSTOLIC_BP_UPPER - SYSTOLIC_BP_LOWER) / float(n_all)
        bounds_all = dp.theoretical_bounds(sens_all, eps)

        # Oncology sub-group sensitivity & theoretical bounds
        sens_onco = (SYSTOLIC_BP_UPPER - SYSTOLIC_BP_LOWER) / float(n_onco)
        bounds_onco = dp.theoretical_bounds(sens_onco, eps)

        # Monte Carlo repetitions for All Patients
        noisy_all_samples = np.array([
            dp.private_mean(all_values, SYSTOLIC_BP_LOWER, SYSTOLIC_BP_UPPER, eps, n_all)[0]
            for _ in range(n_repetitions)
        ])
        diff_all = noisy_all_samples - true_mean_all
        empirical_mae_all = float(np.mean(np.abs(diff_all)))
        empirical_mse_all = float(np.mean(diff_all ** 2))

        # Monte Carlo repetitions for Oncology Group
        noisy_onco_samples = np.array([
            dp.private_mean(onco_values, SYSTOLIC_BP_LOWER, SYSTOLIC_BP_UPPER, eps, n_onco)[0]
            for _ in range(n_repetitions)
        ])
        diff_onco = noisy_onco_samples - true_mean_onco
        empirical_mae_onco = float(np.mean(np.abs(diff_onco)))
        empirical_mse_onco = float(np.mean(diff_onco ** 2))

        results.append({
            "epsilon": eps,
            "N_all": n_all,
            "scale_b_all": bounds_all["scale_b"],
            "theo_mae_all": bounds_all["theoretical_mae"],
            "theo_mse_all": bounds_all["theoretical_mse"],
            "emp_mae_all": round(empirical_mae_all, 4),
            "emp_mse_all": round(empirical_mse_all, 4),
            "N_onco": n_onco,
            "scale_b_onco": bounds_onco["scale_b"],
            "theo_mae_onco": bounds_onco["theoretical_mae"],
            "theo_mse_onco": bounds_onco["theoretical_mse"],
            "emp_mae_onco": round(empirical_mae_onco, 4),
            "emp_mse_onco": round(empirical_mse_onco, 4),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_dp_mean_benchmark()
    print("=== Table 12: Differential Privacy Mean Blood Pressure Error ===")
    print(df_res[["epsilon", "scale_b_all", "emp_mae_all", "emp_mse_all", "scale_b_onco", "emp_mae_onco", "emp_mse_onco"]].to_string(index=False))
