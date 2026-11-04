"""
Tamper-Evident Cryptographic Audit Logger.
Implements NIS2 Art. 21(2)(h) compliance via a SHA-256 cryptographic hash chain.
Every gateway event links back to the previous entry, providing mathematical proof
against post-hoc alteration or deletion of access records.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import AUDIT_LOG_FILE
from src.gateway.models import ActionType, AuditLogRecord
from src.core.crypto import sha256_hash

GENESIS_HASH = "0" * 64


class AuditLogger:
    """
    Append-only tamper-evident audit logger backed by a SHA-256 hash chain.
    """

    def __init__(self, log_path: Path = AUDIT_LOG_FILE):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._get_latest_hash()

    def _get_latest_hash(self) -> str:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH

        last_hash = GENESIS_HASH
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        last_hash = record.get("current_hash", GENESIS_HASH)
                    except json.JSONDecodeError:
                        pass
        return last_hash

    def _get_next_index(self) -> int:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return 0
        count = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def log(
        self,
        user_id: str,
        user_role: str,
        action: ActionType,
        details: Dict[str, Any],
        status: str = "SUCCESS"
    ) -> AuditLogRecord:
        """
        Append a new signed entry into the tamper-evident log chain.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        index = self._get_next_index()
        prev_hash = self._last_hash

        # Canonical string for cryptographic hashing
        payload_to_hash = (
            f"{index}|{now_iso}|{user_id}|{user_role}|{action.value}|"
            f"{json.dumps(details, sort_keys=True)}|{status}|{prev_hash}"
        )
        current_hash = sha256_hash(payload_to_hash)

        record = AuditLogRecord(
            index=index,
            timestamp=now_iso,
            user_id=user_id,
            user_role=user_role,
            action=action,
            details=details,
            status=status,
            previous_hash=prev_hash,
            current_hash=current_hash
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        self._last_hash = current_hash
        return record

    def verify_chain(self) -> Tuple[bool, int, List[str]]:
        """
        Audit the entire hash chain from genesis to head.

        Returns:
            Tuple of (is_valid: bool, records_checked: int, errors: List[str])
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return True, 0, []

        expected_prev = GENESIS_HASH
        records_checked = 0
        errors = []

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                records_checked += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_idx}: invalid JSON syntax ({e})")
                    return False, records_checked, errors

                idx = data.get("index")
                timestamp = data.get("timestamp")
                user_id = data.get("user_id")
                user_role = data.get("user_role")
                action = data.get("action")
                details = data.get("details")
                status = data.get("status")
                prev_hash = data.get("previous_hash")
                curr_hash = data.get("current_hash")

                # Verify pointer to previous block
                if prev_hash != expected_prev:
                    errors.append(
                        f"Block {idx}: Broken hash link! Expected prev_hash {expected_prev[:12]}..., got {prev_hash[:12]}..."
                    )
                    return False, records_checked, errors

                # Recompute and verify current hash
                recomputed_payload = (
                    f"{idx}|{timestamp}|{user_id}|{user_role}|{action}|"
                    f"{json.dumps(details, sort_keys=True)}|{status}|{prev_hash}"
                )
                recomputed_hash = sha256_hash(recomputed_payload)

                if recomputed_hash != curr_hash:
                    errors.append(
                        f"Block {idx}: Tampering detected! Hash mismatch. Computed {recomputed_hash[:12]}..., recorded {curr_hash[:12]}..."
                    )
                    return False, records_checked, errors

                expected_prev = curr_hash

        return True, records_checked, errors

    def get_recent_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent log entries in reverse chronological order.
        """
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries[-limit:][::-1]
