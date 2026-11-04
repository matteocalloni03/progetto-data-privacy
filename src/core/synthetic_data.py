"""
Synthetic Healthcare Dataset Generator.
Generates a realistic clinical cohort with direct identifiers, quasi-identifiers,
and sensitive medical attributes under GDPR Art. 9 special category requirements.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    SYSTOLIC_BP_LOWER,
    SYSTOLIC_BP_UPPER,
    DIASTOLIC_BP_LOWER,
    DIASTOLIC_BP_UPPER,
)

# Reference first names and surnames for Portuguese synthetic records
FIRST_NAMES_M = ["João", "Manuel", "António", "José", "Francisco", "Luís", "Pedro", "Miguel", "Tiago", "Rui"]
FIRST_NAMES_F = ["Maria", "Ana", "Sofia", "Inês", "Beatriz", "Mariana", "Catarina", "Rita", "Teresa", "Helena"]
SURNAMES = ["Silva", "Santos", "Ferreira", "Pereira", "Oliveira", "Costa", "Rodrigues", "Martins", "Jesus", "Sousa"]

POSTAL_AREAS = [1000, 1050, 1100, 1200, 1300, 1500, 1700, 1900, 2600, 2700, 2800]
DIAGNOSES = ["Normal", "Hypertension", "Cardiovascular", "Type 2 diabetes", "Oncology"]


def generate_synthetic_dataset(
    n_patients: int = 1000,
    seed: int = 42,
    oncology_count: int = 64,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate synthetic health dataset conforming to Table 7 of the specification.

    Args:
        n_patients: Total number of patient records (default: 1000).
        seed: Random seed for exact reproducibility.
        oncology_count: Exact size for the small oncology sub-cohort (default: 64).
        save_path: Optional file path to persist the CSV.

    Returns:
        pd.DataFrame containing the generated patient records.
    """
    rng = np.random.default_rng(seed)

    # 1. Direct Identifiers
    patient_ids = [f"PAT-{i+1:05d}" for i in range(n_patients)]
    genders = rng.choice(["M", "F"], size=n_patients, p=[0.48, 0.52])

    names = []
    for g in genders:
        first = rng.choice(FIRST_NAMES_M if g == "M" else FIRST_NAMES_F)
        s1 = rng.choice(SURNAMES)
        s2 = rng.choice(SURNAMES)
        names.append(f"{first} {s1} {s2}")

    # Portuguese NIF: 9 digits, starting with 1, 2, or 3 (strictly unique per patient)
    base_nifs = rng.choice(np.arange(10_000_000, 99_999_999), size=n_patients, replace=False)
    prefix_digits = rng.choice([1, 2, 3], size=n_patients)
    national_ids = [f"{prefix_digits[i]}{base_nifs[i]}" for i in range(n_patients)]

    # 2. Quasi-Identifiers
    # Age: skewed towards older adult population (clinical demographic)
    ages = np.clip(np.round(rng.normal(loc=54.0, scale=16.0, size=n_patients)), 18, 88).astype(int)

    # Postal district area (4-digit code)
    postal_areas = rng.choice(POSTAL_AREAS, size=n_patients)

    # BMI: Body Mass Index, log-normal distribution centered around 26.5
    bmi_raw = rng.lognormal(mean=3.27, sigma=0.18, size=n_patients)
    bmis = np.clip(np.round(bmi_raw, 1), 16.0, 45.0)

    # 3. Diagnoses and Medical Cohorts
    # Set exactly oncology_count records as Oncology
    diagnoses = np.empty(n_patients, dtype=object)
    indices = np.arange(n_patients)
    oncology_idx = rng.choice(indices, size=oncology_count, replace=False)
    diagnoses[oncology_idx] = "Oncology"

    non_onco_idx = np.setdiff1d(indices, oncology_idx)
    non_onco_diagnoses = rng.choice(
        ["Normal", "Hypertension", "Cardiovascular", "Type 2 diabetes"],
        size=len(non_onco_idx),
        p=[0.35, 0.30, 0.20, 0.15]
    )
    diagnoses[non_onco_idx] = non_onco_diagnoses

    # 4. Sensitive Health Attributes (Correlated with Age, BMI, and Diagnosis)
    systolic_bp = np.zeros(n_patients, dtype=float)
    diastolic_bp = np.zeros(n_patients, dtype=float)
    glucose = np.zeros(n_patients, dtype=float)
    treatment_costs = np.zeros(n_patients, dtype=float)

    for i in range(n_patients):
        age_factor = (ages[i] - 18) / 70.0
        bmi_factor = (bmis[i] - 18.0) / 25.0
        diag = diagnoses[i]

        # Systolic BP baseline ~ 118, with clinical elevation
        sys_mean = 118.0 + 20.0 * age_factor + 10.0 * bmi_factor
        if diag == "Hypertension":
            sys_mean += 24.0
        elif diag == "Cardiovascular":
            sys_mean += 20.0
        elif diag == "Oncology":
            sys_mean += 8.0

        sbp = rng.normal(loc=sys_mean, scale=9.5)
        systolic_bp[i] = np.clip(np.round(sbp, 1), SYSTOLIC_BP_LOWER, SYSTOLIC_BP_UPPER)

        # Diastolic BP correlated with systolic
        dbp = 0.55 * systolic_bp[i] + rng.normal(loc=12.0, scale=5.0)
        diastolic_bp[i] = np.clip(np.round(dbp, 1), DIASTOLIC_BP_LOWER, DIASTOLIC_BP_UPPER)

        # Fasting Glucose (mg/dL)
        gluc_mean = 92.0 + 8.0 * bmi_factor
        if diag == "Type 2 diabetes":
            gluc_mean += 55.0
        glucose[i] = np.clip(np.round(rng.normal(loc=gluc_mean, scale=14.0), 1), 70.0, 250.0)

        # Treatment Cost in EUR
        if diag == "Oncology":
            cost = rng.uniform(4500.0, 14800.0)
        elif diag == "Cardiovascular":
            cost = rng.uniform(1500.0, 6800.0)
        elif diag == "Type 2 diabetes":
            cost = rng.uniform(800.0, 3200.0)
        elif diag == "Hypertension":
            cost = rng.uniform(300.0, 1200.0)
        else:
            cost = rng.uniform(100.0, 450.0)
        treatment_costs[i] = np.round(cost, 2)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "full_name": names,
        "national_id": national_ids,
        "age": ages,
        "gender": genders,
        "postal_area": postal_areas,
        "bmi": bmis,
        "diagnosis": diagnoses,
        "blood_pressure_systolic": systolic_bp,
        "blood_pressure_diastolic": diastolic_bp,
        "fasting_glucose": glucose,
        "treatment_cost": treatment_costs,
        "is_oncology": diagnoses == "Oncology"
    })

    if save_path is None:
        save_path = RAW_DATA_DIR / "patients.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df


def load_or_generate_dataset(
    n_patients: int = 1000,
    seed: int = 42,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Load the synthetic cohort from disk if present, or generate a fresh one.
    """
    target = RAW_DATA_DIR / "patients.csv"
    if target.exists() and not force_regenerate:
        df = pd.read_csv(target)
        if len(df) == n_patients:
            return df
    return generate_synthetic_dataset(n_patients=n_patients, seed=seed, save_path=target)
