"""
Pydantic Data Models and Enums for the Privacy Gateway.
Defines contracts for tokens, requests, audit logging, and regulatory reports.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    ADMIN = "ADMIN"                         # Privacy Officer
    HEALTHCARE_WORKER = "HEALTHCARE_WORKER" # Data Entry
    RESEARCHER = "RESEARCHER"               # Scientific Data Consumer


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    AUTHENTICATE = "AUTHENTICATE"
    REVOKE_TOKEN = "REVOKE_TOKEN"
    INGEST_RAW_DATA = "INGEST_RAW_DATA"
    REQUEST_KANON_DATASET = "REQUEST_KANON_DATASET"
    APPROVE_DATASET = "APPROVE_DATASET"
    DOWNLOAD_DATASET = "DOWNLOAD_DATASET"
    EXECUTE_DP_QUERY = "EXECUTE_DP_QUERY"
    AUDIT_VERIFY = "AUDIT_VERIFY"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"


class TokenInfo(BaseModel):
    token: str
    user_id: str
    username: str
    role: UserRole
    is_revoked: bool = False
    created_at: str
    allocated_epsilon: float = 5.0
    used_epsilon: float = 0.0


class AuditLogRecord(BaseModel):
    index: int
    timestamp: str
    user_id: str
    user_role: str
    action: ActionType
    details: Dict[str, Any]
    status: str  # SUCCESS, REJECTED, ALERT
    previous_hash: str
    current_hash: str


class DPQueryRequest(BaseModel):
    query_type: str = Field(..., description="mean, count, or range_count")
    target_column: str = "blood_pressure_systolic"
    filter_cohort: Optional[str] = None  # None (all), "oncology"
    range_condition: Optional[Dict[str, Any]] = None
    epsilon: float = Field(0.5, gt=0.0, le=10.0, description="Privacy budget consumed by query")


class DPQueryResponse(BaseModel):
    query_type: str
    noisy_result: float
    true_result: Optional[float] = None
    epsilon_spent: float
    remaining_budget: float
    noise_scale: float
    theoretical_mae: float
    theoretical_mse: float
    status: str = "SUCCESS"


class DatasetApprovalRequest(BaseModel):
    k: int = Field(5, ge=2, le=50)
    dataset_name: str = "research_health_cohort_k5"
    justification: str = "Cardiovascular research study on elderly cohort"
