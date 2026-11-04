"""
Metrics for Anonymization Quality, Information Loss, and Privacy Risk.
Implements Normalized Certainty Penalty (NCP), Re-identification Linkage Risk,
Homogeneity vulnerability analysis, and Differential Privacy error metrics.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def compute_normalized_certainty_penalty(
    original_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
    quasi_identifiers: List[str]
) -> float:
    """
    Compute Normalized Certainty Penalty (NCP) / Global Loss Metric (GLM).
    Measures information loss caused by generalization of quasi-identifiers.
    NCP is normalized in [0, 1], where 0 represents zero loss and 1 represents complete loss.

    For a numeric attribute interval [low, high]:
        NCP(interval) = (high - low) / (global_max - global_min)
    For a categorical attribute generalized to wildcard/set:
        NCP(cat) = (|generalized_set| - 1) / (|domain| - 1)
    """
    total_penalty = 0.0
    n_records = len(original_df)
    n_qis = len(quasi_identifiers)

    if n_records == 0 or n_qis == 0:
        return 0.0

    for qi in quasi_identifiers:
        if qi not in original_df.columns or qi not in anonymized_df.columns:
            continue

        orig_col = original_df[qi]
        anon_col = anonymized_df[qi]

        # Numeric attribute handling
        if pd.api.types.is_numeric_dtype(orig_col):
            global_min = float(orig_col.min())
            global_max = float(orig_col.max())
            global_range = global_max - global_min
            if global_range == 0:
                continue

            for val in anon_col:
                if isinstance(val, (int, float)):
                    span = 0.0
                elif isinstance(val, tuple) and len(val) == 2:
                    span = float(val[1]) - float(val[0])
                elif isinstance(val, str) and "-" in val:
                    parts = val.split("-")
                    try:
                        span = float(parts[1]) - float(parts[0])
                    except (ValueError, IndexError):
                        span = 0.0
                else:
                    span = 0.0
                total_penalty += span / global_range

        # Categorical attribute handling (e.g. gender)
        else:
            unique_vals = set(orig_col.dropna().unique())
            domain_size = len(unique_vals)
            if domain_size <= 1:
                continue

            for val in anon_col:
                if isinstance(val, (set, list, tuple)):
                    cat_size = len(val)
                elif isinstance(val, str) and (val == "*" or "/" in val or "," in val):
                    cat_size = domain_size
                else:
                    cat_size = 1
                total_penalty += (cat_size - 1) / (domain_size - 1)

    return total_penalty / (n_records * n_qis)


def evaluate_equivalence_classes(
    anonymized_df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_column: str = "diagnosis"
) -> Dict[str, Any]:
    """
    Evaluate equivalence class distribution, linkage risk, and homogeneity vulnerability.

    Returns:
        dict with:
        - num_classes: count of equivalence classes
        - min_class_size: size of smallest class (must be >= k for k-anonymity)
        - avg_class_size: mean size of classes
        - max_linkage_risk: max 1/|EC| (worst-case re-identification probability)
        - avg_linkage_risk: average re-identification probability
        - homogeneous_classes_count: count of classes with zero sensitive diversity
        - l_diversity_min: minimum distinct sensitive values across all classes
    """
    if len(anonymized_df) == 0:
        return {
            "num_classes": 0,
            "min_class_size": 0,
            "avg_class_size": 0.0,
            "max_linkage_risk": 0.0,
            "avg_linkage_risk": 0.0,
            "homogeneous_classes_count": 0,
            "l_diversity_min": 0
        }

    # Group by stringified quasi-identifier values
    qi_tuples = anonymized_df[quasi_identifiers].astype(str).apply(tuple, axis=1)
    grouped = anonymized_df.groupby(qi_tuples)

    class_sizes = grouped.size()
    num_classes = len(class_sizes)
    min_size = int(class_sizes.min())
    avg_size = float(class_sizes.mean())

    # Linkage risk: 1 / |EC|
    max_linkage_risk = 1.0 / min_size if min_size > 0 else 1.0
    # Average re-identification risk across individuals: sum(|EC| * (1/|EC|)) / N = num_classes / N
    avg_linkage_risk = num_classes / len(anonymized_df)

    # Sensitive diversity & homogeneity
    homogeneous_count = 0
    l_diversity_values = []

    if sensitive_column in anonymized_df.columns:
        for _, group in grouped:
            distinct_sens = group[sensitive_column].nunique()
            l_diversity_values.append(distinct_sens)
            if distinct_sens == 1 and len(group) > 1:
                homogeneous_count += 1

        l_diversity_min = int(min(l_diversity_values)) if l_diversity_values else 0
    else:
        l_diversity_min = 0

    return {
        "num_classes": num_classes,
        "min_class_size": min_size,
        "avg_class_size": round(avg_size, 2),
        "max_linkage_risk": round(max_linkage_risk, 4),
        "avg_linkage_risk": round(avg_linkage_risk, 4),
        "homogeneous_classes_count": homogeneous_count,
        "l_diversity_min": l_diversity_min
    }


def compute_dp_error_metrics(
    true_values: np.ndarray,
    noisy_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute empirical Mean Absolute Error (MAE) and Mean Squared Error (MSE).
    """
    diff = noisy_values - true_values
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4)
    }
