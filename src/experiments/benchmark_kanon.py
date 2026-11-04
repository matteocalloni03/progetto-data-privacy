"""
Benchmark 2: k-Anonymity Trade-off vs k (Mondrian Partitioning).
Replicates Table 13 and Figure 2(b) from the study.
Measures Information Loss (NCP) and Maximum Linkage Risk across k values.
"""

from typing import List, Dict, Any
import pandas as pd

from src.config import QUASI_IDENTIFIERS
from src.core.synthetic_data import load_or_generate_dataset
from src.core.crypto import pseudonymize_dataframe
from src.core.k_anonymity import MondrianAnonymizer


def run_kanon_benchmark(
    k_values: List[int] = [3, 5, 10, 20, 50],
    quasi_identifiers: List[str] = QUASI_IDENTIFIERS
) -> pd.DataFrame:
    """
    Execute Mondrian anonymization across multiple k values and compute metrics.
    """
    df_raw = load_or_generate_dataset()
    df_pseudo = pseudonymize_dataframe(df_raw)

    records = []

    for k in k_values:
        anonymizer = MondrianAnonymizer(k=k, quasi_identifiers=quasi_identifiers)
        anon_df, summary = anonymizer.anonymize(df_pseudo)

        records.append({
            "k": k,
            "num_records": summary["total_records"],
            "num_classes": summary["num_classes"],
            "min_class_size": summary["min_class_size"],
            "avg_class_size": summary["avg_class_size"],
            "info_loss_ncp": summary["information_loss_ncp"],
            "max_linkage_risk": summary["max_linkage_risk"],
            "avg_linkage_risk": summary["avg_linkage_risk"],
            "homogeneous_classes": summary["homogeneous_classes_count"],
            "min_l_diversity": summary["l_diversity_min"]
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df_res = run_kanon_benchmark()
    print("=== Table 13: k-Anonymity Results for Quasi-Identifiers (Age, Gender, Postal_area, BMI) ===")
    print(df_res.to_string(index=False))
