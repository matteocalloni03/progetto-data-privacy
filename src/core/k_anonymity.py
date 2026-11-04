"""
Mondrian Multidimensional k-Anonymity Algorithm.
Implements top-down greedy partitioning across numerical and categorical
quasi-identifiers to guarantee that every equivalence class contains at least k records (|EC| >= k).
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from src.config import QUASI_IDENTIFIERS
from src.core.metrics import compute_normalized_certainty_penalty, evaluate_equivalence_classes


class MondrianAnonymizer:
    """
    Multidimensional Mondrian partitioning for k-anonymity.
    """

    def __init__(
        self,
        k: int = 5,
        quasi_identifiers: Optional[List[str]] = None,
        suppress_outliers: bool = False
    ):
        if k < 2:
            raise ValueError("k must be an integer >= 2 for meaningful anonymity.")
        self.k = k
        self.quasi_identifiers = quasi_identifiers or list(QUASI_IDENTIFIERS)
        self.suppress_outliers = suppress_outliers
        self.equivalence_classes: List[pd.DataFrame] = []

    def _is_numeric(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    def _get_span(self, df: pd.DataFrame, col: str) -> float:
        """
        Compute normalized span of an attribute within a partition.
        """
        series = df[col]
        if self._is_numeric(series):
            span = float(series.max()) - float(series.min())
            global_span = float(self.global_ranges.get(col, 1.0))
            return span / global_span if global_span > 0 else 0.0
        else:
            # Categorical diversity
            n_unique = series.nunique()
            global_unique = self.global_ranges.get(col, 1.0)
            return (n_unique - 1) / (global_unique - 1) if global_unique > 1 else 0.0

    def _partition(self, partition_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Recursively split partition along the quasi-identifier with the largest normalized span.
        """
        if len(partition_df) < 2 * self.k:
            return [partition_df]

        # Order QIs by normalized span descending
        spans = [(qi, self._get_span(partition_df, qi)) for qi in self.quasi_identifiers]
        spans.sort(key=lambda x: x[1], reverse=True)

        for best_dim, span in spans:
            if span == 0:
                continue

            series = partition_df[best_dim]
            if self._is_numeric(series):
                median_val = float(series.median())
                left = partition_df[series <= median_val]
                right = partition_df[series > median_val]

                # Edge case: if median split concentrates on one side
                if len(left) < self.k or len(right) < self.k:
                    # Try splitting strictly by index split at median position
                    sorted_part = partition_df.sort_values(by=best_dim)
                    mid = len(sorted_part) // 2
                    left = sorted_part.iloc[:mid]
                    right = sorted_part.iloc[mid:]

                if len(left) >= self.k and len(right) >= self.k:
                    return self._partition(left) + self._partition(right)
            else:
                # Categorical split: divide frequent categories
                counts = series.value_counts()
                if len(counts) > 1:
                    first_cat = counts.index[0]
                    left = partition_df[series == first_cat]
                    right = partition_df[series != first_cat]
                    if len(left) >= self.k and len(right) >= self.k:
                        return self._partition(left) + self._partition(right)

        # No valid split that respects |EC| >= k
        return [partition_df]

    def anonymize(
        self,
        df: pd.DataFrame,
        format_intervals: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute Mondrian k-anonymity partitioning.

        Args:
            df: Input patient DataFrame.
            format_intervals: If True, formats ranges as strings 'low-high';
                              otherwise preserves tuple (low, high).

        Returns:
            Tuple of (anonymized_df, summary_metrics_dict).
        """
        data = df.copy()

        # Cache global ranges for normalization
        self.global_ranges = {}
        for qi in self.quasi_identifiers:
            if qi in data.columns:
                if self._is_numeric(data[qi]):
                    self.global_ranges[qi] = float(data[qi].max()) - float(data[qi].min())
                else:
                    self.global_ranges[qi] = float(data[qi].nunique())

        partitions = self._partition(data)
        self.equivalence_classes = partitions

        anonymized_parts = []
        for class_id, part in enumerate(partitions):
            p = part.copy()
            p["equivalence_class_id"] = class_id

            # Generalize quasi-identifiers across the equivalence class
            for qi in self.quasi_identifiers:
                if qi not in p.columns:
                    continue
                series = p[qi]
                if self._is_numeric(series):
                    min_val = series.min()
                    max_val = series.max()
                    if min_val == max_val:
                        p[qi] = str(min_val) if format_intervals else (min_val, min_val)
                    else:
                        if format_intervals:
                            if isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)):
                                p[qi] = f"{int(min_val)}-{int(max_val)}"
                            else:
                                p[qi] = f"{float(min_val):.1f}-{float(max_val):.1f}"
                        else:
                            p[qi] = [(min_val, max_val)] * len(p)
                else:
                    unique_vals = sorted(series.unique())
                    if len(unique_vals) == 1:
                        p[qi] = str(unique_vals[0])
                    else:
                        p[qi] = "*" if format_intervals else [set(unique_vals)] * len(p)

            anonymized_parts.append(p)

        anonymized_df = pd.concat(anonymized_parts, ignore_index=True)

        # Compute information loss and risk metrics
        ncp = compute_normalized_certainty_penalty(df, anonymized_df, self.quasi_identifiers)
        eq_metrics = evaluate_equivalence_classes(anonymized_df, self.quasi_identifiers)

        summary = {
            "k": self.k,
            "total_records": len(anonymized_df),
            "information_loss_ncp": round(ncp, 4),
            **eq_metrics
        }

        return anonymized_df, summary
