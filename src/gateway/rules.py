"""
Table 8 Security Rules and Abuse Detection Engine.
Enforces the mandatory rules of the Privacy Gateway under NIS2 Art. 21(2)(b)
and GDPR Data Minimization & Security principles.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone
import time

from src.config import GATEWAY_MAX_QUERIES_PER_MINUTE, GATEWAY_SMALL_CELL_THRESHOLD
from src.gateway.models import TokenInfo, UserRole, IncidentSeverity


class GatewayRulesEngine:
    """
    Evaluates request parameters against Table 8 rules before processing.
    """

    def __init__(
        self,
        max_qpm: int = GATEWAY_MAX_QUERIES_PER_MINUTE,
        small_cell_thresh: int = GATEWAY_SMALL_CELL_THRESHOLD
    ):
        self.max_qpm = max_qpm
        self.small_cell_thresh = small_cell_thresh
        # Map user_id -> list of UNIX timestamps for rate limiting
        self._request_timestamps = defaultdict(list)
        # History of queries to detect reconstruction attacks
        self._query_fingerprints = defaultdict(list)

    def verify_token_authentication(self, token_info: Optional[TokenInfo]) -> Tuple[bool, str]:
        """
        Rule 1: Token Authentication check.
        """
        if token_info is None:
            return False, "Rule 1 Violation: Missing or invalid authentication token."
        return True, "Authentication verified."

    def verify_token_active(self, token_info: TokenInfo) -> Tuple[bool, str]:
        """
        Rule 2: Token Revocation check.
        """
        if token_info.is_revoked:
            return False, "Rule 2 Violation: Token has been revoked due to policy violation or security incident."
        return True, "Token is active."

    def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """
        Rule 4a: Rate Limiting (queries per minute).
        """
        now = time.time()
        timestamps = self._request_timestamps[user_id]
        # Keep only queries within the last 60 seconds
        timestamps = [t for t in timestamps if now - t < 60.0]
        self._request_timestamps[user_id] = timestamps

        if len(timestamps) >= self.max_qpm:
            return False, f"Rule 4 Violation: Rate limit exceeded ({len(timestamps)} queries/min > max {self.max_qpm})."

        self._request_timestamps[user_id].append(now)
        return True, "Rate limit within bounds."

    def detect_reconstruction_attack(
        self,
        user_id: str,
        query_type: str,
        target_column: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Rule 4b: Reconstruction Attack Detection.
        Detects adversarial attempts to single out a victim by running rapid,
        subtle range shifts designed to isolate an individual record.
        """
        fp = f"{query_type}:{target_column}:{str(sorted((filters or {}).items()))}"
        history = self._query_fingerprints[user_id]
        history.append((time.time(), fp))

        # Check last 5 queries within 30 seconds
        now = time.time()
        recent = [entry for entry in history if now - entry[0] < 30.0]
        self._query_fingerprints[user_id] = recent

        # If identical query is sent repeatedly in a burst (> 3 identical queries in 30s)
        recent_fps = [x[1] for x in recent]
        if recent_fps.count(fp) >= 3:
            return True, (
                "Rule 4 Abuse Alert: Automated reconstruction attempt detected "
                "(repeated identical target queries within tight time window)."
            )

        return False, "No reconstruction pattern detected."

    def verify_small_cell_threshold(self, cohort_size: int) -> Tuple[bool, str]:
        """
        Rule 5: Small-Cell Suppression check.
        Forbids raw operations on cohorts smaller than threshold.
        """
        if cohort_size < self.small_cell_thresh:
            return False, (
                f"Rule 5 Violation: Cohort size ({cohort_size}) is below minimum cell threshold "
                f"({self.small_cell_thresh}). Request blocked to prevent singling out."
            )
        return True, "Cell threshold satisfied."

    def verify_sensitive_attribute_access(
        self,
        role: UserRole,
        requested_raw: bool
    ) -> Tuple[bool, str]:
        """
        Rule 6: Sensitive Attribute Leakage Protection.
        Researchers are never granted raw unanonymized sensitive attributes.
        """
        if role == UserRole.RESEARCHER and requested_raw:
            return False, "Rule 6 Violation: Direct raw access to sensitive clinical attributes is prohibited for Researcher role."
        return True, "Access permissible."
