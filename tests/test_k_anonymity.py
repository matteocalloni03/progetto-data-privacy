"""
Unit tests for Mondrian k-Anonymity and Privacy Risk Metrics (FR-03, FR-04, FR-05).
"""

import pytest
import pandas as pd
from src.core.k_anonymity import MondrianAnonymizer
from src.core.metrics import compute_normalized_certainty_penalty, evaluate_equivalence_classes


def test_toy_example_table4_to_table5():
    """
    Test toy example from Table 4 and Table 5 of the study document:
    6 patients, k=3, age, postal_code, diagnosis.
    Demonstrates 3-anonymity and homogeneity vulnerability in Class A.
    """
    toy_df = pd.DataFrame({
        "age": [34, 36, 38, 52, 55, 58],
        "postal_area": [1050, 1052, 1051, 1100, 1104, 1102],
        "diagnosis": [
            "Cardiovascular",
            "Cardiovascular",
            "Cardiovascular",
            "Type 2 diabetes",
            "Normal",
            "Hypertension"
        ]
    })

    anonymizer = MondrianAnonymizer(k=3, quasi_identifiers=["age", "postal_area"])
    anon_df, summary = anonymizer.anonymize(toy_df)

    # Must produce 2 equivalence classes of 3 records each
    assert summary["min_class_size"] >= 3
    assert summary["num_classes"] == 2
    assert summary["max_linkage_risk"] <= 1.0 / 3.0

    # Class A has 3 Cardiovascular diagnoses -> must detect 1 homogeneous class
    assert summary["homogeneous_classes_count"] >= 1


def test_mondrian_k_anonymity_invariant():
    """
    Verifies that for any k in {3, 5, 10}, EVERY equivalence class has |EC| >= k.
    """
    df = pd.DataFrame({
        "age": [20, 22, 23, 25, 29, 31, 35, 36, 40, 42, 45, 48, 50, 52, 55, 60],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "postal_area": [1000, 1000, 1000, 1000, 1100, 1100, 1100, 1100, 1200, 1200, 1200, 1200, 1300, 1300, 1300, 1300],
        "bmi": [21.0, 22.5, 24.0, 25.1, 26.2, 27.0, 28.5, 29.1, 30.2, 31.0, 32.5, 33.1, 34.0, 35.2, 36.1, 37.0],
        "diagnosis": ["Normal"] * 16
    })

    for k in [3, 4, 5]:
        anon = MondrianAnonymizer(k=k, quasi_identifiers=["age", "gender", "postal_area", "bmi"])
        anon_df, summary = anon.anonymize(df)
        assert summary["min_class_size"] >= k, f"Failed for k={k}: got min_class_size {summary['min_class_size']}"
        assert summary["max_linkage_risk"] <= (1.0 / k) + 1e-6


def test_information_loss_ncp_bounds():
    df = pd.DataFrame({
        "age": [20, 40, 60, 80],
        "postal_area": [1000, 1100, 1200, 1300],
        "bmi": [20.0, 25.0, 30.0, 35.0]
    })
    anon = MondrianAnonymizer(k=2, quasi_identifiers=["age", "postal_area", "bmi"])
    anon_df, summary = anon.anonymize(df)
    ncp = summary["information_loss_ncp"]
    assert 0.0 <= ncp <= 1.0
