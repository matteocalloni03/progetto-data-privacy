"""
Unit tests for Synthetic Healthcare Dataset Generation (FR-01).
"""

import pytest
import pandas as pd
from src.core.synthetic_data import generate_synthetic_dataset
from src.config import (
    SYSTOLIC_BP_LOWER,
    SYSTOLIC_BP_UPPER,
    DIASTOLIC_BP_LOWER,
    DIASTOLIC_BP_UPPER,
    DIRECT_IDENTIFIERS,
    QUASI_IDENTIFIERS,
    SENSITIVE_ATTRIBUTES
)


def test_dataset_generation_dimensions():
    n_patients = 1000
    df = generate_synthetic_dataset(n_patients=n_patients, seed=123, oncology_count=64)
    assert len(df) == n_patients
    assert "patient_id" in df.columns
    assert "diagnosis" in df.columns


def test_direct_identifiers_present():
    df = generate_synthetic_dataset(n_patients=100, seed=123)
    for col in DIRECT_IDENTIFIERS:
        assert col in df.columns
    # Official government direct identifiers must be strictly unique per patient
    assert df["patient_id"].nunique() == 100
    assert df["national_id"].nunique() == 100
    assert df["full_name"].nunique() >= 90


def test_quasi_identifiers_present_and_valid():
    df = generate_synthetic_dataset(n_patients=100, seed=123)
    for col in QUASI_IDENTIFIERS:
        assert col in df.columns

    assert df["age"].min() >= 18
    assert df["age"].max() <= 90
    assert set(df["gender"].unique()).issubset({"M", "F"})
    assert df["bmi"].min() >= 15.0
    assert df["bmi"].max() <= 50.0


def test_oncology_subcohort_size():
    df = generate_synthetic_dataset(n_patients=1000, seed=42, oncology_count=64)
    onco_patients = df[df["is_oncology"] == True]
    assert len(onco_patients) == 64
    assert (onco_patients["diagnosis"] == "Oncology").all()


def test_clinical_bounds():
    df = generate_synthetic_dataset(n_patients=500, seed=42)
    assert df["blood_pressure_systolic"].min() >= SYSTOLIC_BP_LOWER
    assert df["blood_pressure_systolic"].max() <= SYSTOLIC_BP_UPPER
    assert df["blood_pressure_diastolic"].min() >= DIASTOLIC_BP_LOWER
    assert df["blood_pressure_diastolic"].max() <= DIASTOLIC_BP_UPPER
