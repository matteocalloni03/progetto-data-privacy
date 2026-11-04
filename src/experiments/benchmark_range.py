"""
Benchmark 3: Range-Count Query Sequence Comparison (Figure 3).
Compares error behavior between Differential Privacy and k-Anonymity
as the number of queries Q increases.
DP error grows linearly with Q because the total privacy budget must be shared (epsilon_i = epsilon_total / Q),
whereas k-anonymity error remains constant regardless of the number of queries.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.core.synthetic_data import load_or_generate_dataset
from src.core.crypto import pseudonymize_dataframe
from src.core.k_anonymity import MondrianAnonymizer
from src.core.differential_privacy import LaplaceMechanism


def run_range_queries_benchmark(
    query_counts: List[int] = [1, 5, 10, 15, 20, 25, 30, 40, 50],
    total_epsilon: float = 1.0,
    k_anon: int = 5,
    n_repetitions: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate a sequence of Q range-count queries under DP vs k-Anonymity.
    """
    rng = np.random.default_rng(seed)
    df_raw = load_or_generate_dataset()
    dp = LaplaceMechanism(seed=seed)

    # Prepare k-anonymized dataset (k=5)
    df_pseudo = pseudonymize_dataframe(df_raw)
    anonymizer = MondrianAnonymizer(k=k_anon)
    df_kanon, _ = anonymizer.anonymize(df_pseudo)

    # Generate synthetic age range queries, e.g. age between min_age and max_age
    query_ranges = []
    for _ in range(max(query_counts)):
        low = rng.integers(20, 65)
        high = low + rng.integers(10, 25)
        query_ranges.append((low, high))

    results = []

    # Pre-evaluate k-anonymity average error over the queries
    kanon_errors = []
    for low, high in query_ranges:
        # True count on raw data
        true_count = len(df_raw[(df_raw["age"] >= low) & (df_raw["age"] <= high)])

        # Estimated count on k-anonymized data (using interval overlap)
        est_count = 0.0
        for val in df_kanon["age"]:
            if "-" in str(val):
                p_low, p_high = map(float, str(val).split("-"))
                overlap = max(0.0, min(high, p_high) - max(low, p_low))
                total_span = max(1.0, p_high - p_low)
                est_count += (overlap / total_span)
            else:
                try:
                    v = float(val)
                    if low <= v <= high:
                        est_count += 1.0
                except ValueError:
                    pass
        kanon_errors.append(abs(est_count - true_count))

    base_kanon_mae = float(np.mean(kanon_errors))

    for Q in query_counts:
        # Differential Privacy: budget split evenly across Q queries
        eps_per_query = total_epsilon / float(Q)
        # For a count query, sensitivity Delta_f = 1 -> noise scale b = 1 / eps_per_query = Q / total_epsilon
        dp_theo_mae = 1.0 / eps_per_query

        # Empirical DP simulation over n_repetitions
        dp_errors = []
        for i in range(min(Q, len(query_ranges))):
            low, high = query_ranges[i]
            true_count = len(df_raw[(df_raw["age"] >= low) & (df_raw["age"] <= high)])
            # Simulate noisy answers
            noisy_counts = [
                dp.private_count(true_count, eps_per_query)[0]
                for _ in range(n_repetitions)
            ]
            dp_errors.append(np.mean(np.abs(np.array(noisy_counts) - true_count)))

        dp_emp_mae = float(np.mean(dp_errors))

        results.append({
            "num_queries_Q": Q,
            "total_epsilon": total_epsilon,
            "epsilon_per_query": round(eps_per_query, 4),
            "dp_noise_scale_b": round(dp_theo_mae, 4),
            "dp_empirical_mae": round(dp_emp_mae, 4),
            "kanon_constant_mae": round(base_kanon_mae, 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_range_queries_benchmark()
    print("=== Figure 3: Error of Range-Count Queries (DP vs k-Anonymity) ===")
    print(df_res.to_string(index=False))
