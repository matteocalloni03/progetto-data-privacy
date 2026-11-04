"""
Unit tests for Cryptographic Pseudonymization and Integrity (FR-02, SEC-01).
"""

import pytest
import pandas as pd
from src.core.crypto import (
    hmac_sha256_pseudonymize,
    sha256_hash,
    pseudonymize_dataframe
)


def test_hmac_deterministic():
    key = "test-secret-salt-123"
    p1 = hmac_sha256_pseudonymize("PAT-00001", key)
    p2 = hmac_sha256_pseudonymize("PAT-00001", key)
    assert p1 == p2
    assert p1.startswith("PSEUDO-")


def test_hmac_different_keys():
    p1 = hmac_sha256_pseudonymize("PAT-00001", "key-a")
    p2 = hmac_sha256_pseudonymize("PAT-00001", "key-b")
    assert p1 != p2


def test_pseudonymize_dataframe():
    df = pd.DataFrame({
        "patient_id": ["PAT-001", "PAT-002"],
        "full_name": ["John Doe", "Jane Smith"],
        "national_id": ["123456789", "987654321"],
        "age": [45, 52],
        "diagnosis": ["Normal", "Oncology"]
    })

    pseudo_df = pseudonymize_dataframe(df)

    # Check direct identifiers are dropped
    assert "patient_id" not in pseudo_df.columns
    assert "full_name" not in pseudo_df.columns
    assert "national_id" not in pseudo_df.columns

    # Check pseudonym column added
    assert "pseudonym_id" in pseudo_df.columns
    assert pseudo_df["pseudonym_id"].nunique() == 2

    # Check health attributes preserved
    assert list(pseudo_df["diagnosis"]) == ["Normal", "Oncology"]


def test_sha256_hash():
    h1 = sha256_hash("hello-world")
    h2 = sha256_hash("hello-world")
    h3 = sha256_hash("hello-world-2")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64
