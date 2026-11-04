"""
Unit tests for Cryptographic Audit Logging & Tamper-Evident Hash Chaining (SEC-01).
"""

import pytest
import json
from pathlib import Path
from src.gateway.audit_logger import AuditLogger, GENESIS_HASH
from src.gateway.models import ActionType


def test_audit_hash_chain_validity(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    # Log 5 transactions
    for i in range(5):
        logger.log(
            user_id=f"USR-{i}",
            user_role="RESEARCHER",
            action=ActionType.EXECUTE_DP_QUERY,
            details={"query_idx": i, "epsilon": 0.5},
            status="SUCCESS"
        )

    is_valid, count, errors = logger.verify_chain()
    assert is_valid is True
    assert count == 5
    assert len(errors) == 0


def test_audit_hash_chain_tamper_detection(tmp_path):
    log_file = tmp_path / "tamper_audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    # Log 4 transactions
    for i in range(4):
        logger.log(
            user_id=f"USR-{i}",
            user_role="RESEARCHER",
            action=ActionType.EXECUTE_DP_QUERY,
            details={"param": i},
            status="SUCCESS"
        )

    # Confirm valid initially
    is_valid_initial, count_initial, _ = logger.verify_chain()
    assert is_valid_initial is True
    assert count_initial == 4

    # Tamper with block #2 (line index 1)
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    block2 = json.loads(lines[1])
    # Malicious actor changes query details
    block2["details"]["param"] = 9999
    lines[1] = json.dumps(block2)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Re-verify: must fail on Block 1 with hash mismatch
    is_valid, count, errors = logger.verify_chain()
    assert is_valid is False
    assert len(errors) > 0
    assert "Tampering detected" in errors[0]
