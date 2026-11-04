"""
Privacy Budget Manager for Differential Privacy.
Tracks cumulative privacy loss under Sequential and Parallel Composition theorems,
preventing budget depletion and privacy leakage over multiple queries.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone


class PrivacyBudgetManager:
    """
    Manages epsilon privacy budgets allocated to researchers.
    Enforces sequential composition (epsilon_total = sum(epsilon_i)).
    """

    def __init__(self, default_budget: float = 5.0):
        self.default_budget = default_budget
        # Maps token/user_id to dict: {allocated, remaining, query_count, history}
        self._budgets: Dict[str, Dict[str, Any]] = {}

    def register_user(self, user_id: str, allocated_budget: Optional[float] = None) -> None:
        """
        Allocate a privacy budget to a user/token.
        """
        budget = allocated_budget if allocated_budget is not None else self.default_budget
        if user_id not in self._budgets:
            self._budgets[user_id] = {
                "allocated": budget,
                "remaining": budget,
                "query_count": 0,
                "history": []
            }

    def get_budget_status(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self._budgets:
            self.register_user(user_id)
        b = self._budgets[user_id]
        return {
            "user_id": user_id,
            "allocated": round(b["allocated"], 4),
            "remaining": round(b["remaining"], 4),
            "used": round(b["allocated"] - b["remaining"], 4),
            "query_count": b["query_count"]
        }

    def check_and_consume_budget(
        self,
        user_id: str,
        requested_epsilon: float,
        query_type: str = "generic"
    ) -> Tuple[bool, str, float]:
        """
        Evaluate if query is admissible under remaining privacy budget.

        Returns:
            Tuple of (admissible: bool, reason: str, remaining_budget: float)
        """
        if requested_epsilon <= 0:
            return False, "Requested epsilon must be strictly positive (> 0).", 0.0

        if user_id not in self._budgets:
            self.register_user(user_id)

        user_budget = self._budgets[user_id]
        remaining = user_budget["remaining"]

        if requested_epsilon > remaining:
            msg = (
                f"Privacy budget exhausted! Requested epsilon {requested_epsilon:.4f} "
                f"exceeds remaining budget {remaining:.4f}."
            )
            return False, msg, remaining

        # Deduct budget under sequential composition
        user_budget["remaining"] -= requested_epsilon
        user_budget["query_count"] += 1
        user_budget["history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_type": query_type,
            "epsilon_spent": requested_epsilon,
            "remaining_after": user_budget["remaining"]
        })

        return True, "Budget successfully consumed.", user_budget["remaining"]

    def reset_budget(self, user_id: str, new_budget: Optional[float] = None) -> None:
        """
        Admin reset of user budget.
        """
        b = new_budget if new_budget is not None else self.default_budget
        self._budgets[user_id] = {
            "allocated": b,
            "remaining": b,
            "query_count": 0,
            "history": []
        }
