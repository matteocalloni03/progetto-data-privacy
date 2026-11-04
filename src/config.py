"""
Configuration settings for the Health Data Privacy Gateway.
Provides central configuration with environment-variable fallbacks.
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
ANONYMIZED_DATA_DIR = DATA_DIR / "anonymized"
LOGS_DIR = DATA_DIR / "logs"
INCIDENTS_DIR = DATA_DIR / "incidents"
FIGURES_DIR = DATA_DIR / "figures"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, ANONYMIZED_DATA_DIR, LOGS_DIR, INCIDENTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Cryptographic secret for HMAC-SHA256 pseudonymization (GDPR Art. 4(5))
PSEUDONYMIZATION_SECRET_KEY = os.getenv(
    "PSEUDONYMIZATION_SECRET_KEY",
    "iscte-privacidade-dados-2025-2026-secret-salt-key-secure"
)

# Audit log file (NIS2 Art. 21(2)(h))
AUDIT_LOG_FILE = LOGS_DIR / "audit_log.jsonl"

# Privacy Gateway Security Parameters (Table 8)
DEFAULT_EPSILON_BUDGET = float(os.getenv("GATEWAY_DEFAULT_EPSILON_BUDGET", "5.0"))
DEFAULT_K_ANONYMITY = int(os.getenv("GATEWAY_DEFAULT_K_ANONYMITY", "5"))
GATEWAY_MAX_QUERIES_PER_MINUTE = int(os.getenv("GATEWAY_MAX_QUERIES_PER_MINUTE", "10"))
GATEWAY_SMALL_CELL_THRESHOLD = int(os.getenv("GATEWAY_SMALL_CELL_THRESHOLD", "5"))

# Clinical Bounds for Blood Pressure (Table 11: systolic bounds in mmHg)
SYSTOLIC_BP_LOWER = 90.0
SYSTOLIC_BP_UPPER = 210.0
DIASTOLIC_BP_LOWER = 60.0
DIASTOLIC_BP_UPPER = 120.0

# Quasi-Identifiers and Sensitive Attributes
QUASI_IDENTIFIERS = ["age", "gender", "postal_area", "bmi"]
DIRECT_IDENTIFIERS = ["patient_id", "full_name", "national_id"]
SENSITIVE_ATTRIBUTES = [
    "diagnosis",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "fasting_glucose",
    "treatment_cost"
]

# Web Server Configuration
HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
PORT = int(os.getenv("GATEWAY_PORT", "8000"))
