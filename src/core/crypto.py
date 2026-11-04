"""
Cryptographic Operations and Pseudonymization Module.
Implements HMAC-SHA256 direct identifier pseudonymization as mandated by GDPR Art. 4(5)
and NIS2 Art. 21(2)(h) for secure processing of sensitive health data.
"""

import hmac
import hashlib
from typing import List, Union
import pandas as pd

from src.config import PSEUDONYMIZATION_SECRET_KEY, DIRECT_IDENTIFIERS


def hmac_sha256_pseudonymize(identifier: str, secret_key: str = PSEUDONYMIZATION_SECRET_KEY) -> str:
    """
    Generate an HMAC-SHA256 pseudonym from a direct identifier.

    Args:
        identifier: The raw identifier string (e.g., patient ID, national ID, full name).
        secret_key: Secret cryptographic key/salt.

    Returns:
        Hexadecimal string digest (64 characters).
    """
    key_bytes = secret_key.encode("utf-8")
    msg_bytes = str(identifier).strip().encode("utf-8")
    digest = hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()
    # Format as structured pseudonym: PSEUDO-<first 16 hex chars>
    return f"PSEUDO-{digest[:16].upper()}"


def sha256_hash(data: Union[str, bytes]) -> str:
    """
    Compute raw SHA-256 digest of arbitrary string or bytes.
    Used for tamper-evident hash chaining and data integrity verification.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def pseudonymize_dataframe(
    df: pd.DataFrame,
    direct_id_columns: List[str] = DIRECT_IDENTIFIERS,
    secret_key: str = PSEUDONYMIZATION_SECRET_KEY,
    drop_unmasked_direct: bool = True
) -> pd.DataFrame:
    """
    Apply cryptographic pseudonymization to all direct identifiers in a DataFrame.

    Replaces the primary direct identifier with a pseudonymized token
    and removes all auxiliary direct identifiers (names, national IDs)
    to prevent singling out as per GDPR Art. 4(5).
    """
    processed = df.copy()

    # Create pseudonymized patient token based on patient_id
    if "patient_id" in processed.columns:
        processed["pseudonym_id"] = processed["patient_id"].apply(
            lambda x: hmac_sha256_pseudonymize(str(x), secret_key)
        )

    if drop_unmasked_direct:
        cols_to_drop = [c for c in direct_id_columns if c in processed.columns]
        processed.drop(columns=cols_to_drop, inplace=True)

    return processed
