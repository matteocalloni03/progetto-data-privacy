"""
Central Privacy Gateway Controller.
Coordinates the enforcement of GDPR and NIS2 requirements, acting as the sole
mediating gateway between scientific researchers and sensitive health records.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import secrets
import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_EPSILON_BUDGET,
    DEFAULT_K_ANONYMITY,
    SYSTOLIC_BP_LOWER,
    SYSTOLIC_BP_UPPER,
    ANONYMIZED_DATA_DIR,
    RAW_DATA_DIR,
    QUASI_IDENTIFIERS
)
from src.gateway.models import (
    UserRole,
    TokenInfo,
    ActionType,
    IncidentSeverity,
    DPQueryRequest,
    DPQueryResponse
)
from src.gateway.audit_logger import AuditLogger
from src.gateway.budget_manager import PrivacyBudgetManager
from src.gateway.rules import GatewayRulesEngine
from src.gateway.incident_reporter import IncidentReporter
from src.core.synthetic_data import load_or_generate_dataset
from src.core.crypto import pseudonymize_dataframe
from src.core.k_anonymity import MondrianAnonymizer
from src.core.differential_privacy import LaplaceMechanism


class PrivacyGateway:
    """
    Main Privacy Gateway implementation conforming to Figure 1, Figure 5, and Table 8.
    """

    def __init__(
        self,
        default_epsilon: float = DEFAULT_EPSILON_BUDGET,
        default_k: int = DEFAULT_K_ANONYMITY,
        audit_logger: Optional[AuditLogger] = None,
        incident_reporter: Optional[IncidentReporter] = None
    ):
        self.default_epsilon = default_epsilon
        self.default_k = default_k
        self.audit_logger = audit_logger or AuditLogger()
        self.incident_reporter = incident_reporter or IncidentReporter()
        self.rules_engine = GatewayRulesEngine()
        self.budget_manager = PrivacyBudgetManager(default_budget=default_epsilon)
        self.dp_engine = LaplaceMechanism(seed=42)

        # In-memory Token Registry: token -> TokenInfo
        self.tokens: Dict[str, TokenInfo] = {}

        # Approved datasets registry: dataset_id -> metadata
        self.approved_datasets: Dict[str, Dict[str, Any]] = {}

        # Load raw dataset into memory
        self._raw_dataset: pd.DataFrame = load_or_generate_dataset()

        # Seed initial standard tokens
        self._initialize_default_tokens()

    def _initialize_default_tokens(self) -> None:
        """
        Create default persona tokens for testing and interactive use.
        """
        # Admin / Privacy Officer
        self.register_token(
            token="admin-token-secret-iscte-2026",
            username="Dr. Leithardt (Privacy Officer)",
            role=UserRole.ADMIN,
            allocated_epsilon=100.0
        )
        # Healthcare Worker (Data Entry)
        self.register_token(
            token="worker-token-hospital-lisbon",
            username="Nurse Manuel (Data Entry)",
            role=UserRole.HEALTHCARE_WORKER,
            allocated_epsilon=0.0
        )
        # Scientific Researcher (Data Consumer)
        self.register_token(
            token="researcher-token-oncology-lab",
            username="Researcher Matteo (Cardio & Oncology Lab)",
            role=UserRole.RESEARCHER,
            allocated_epsilon=self.default_epsilon
        )

    def register_token(
        self,
        token: str,
        username: str,
        role: UserRole,
        allocated_epsilon: float = 5.0
    ) -> TokenInfo:
        info = TokenInfo(
            token=token,
            user_id=f"USR-{secrets.token_hex(4).upper()}",
            username=username,
            role=role,
            is_revoked=False,
            created_at=datetime.now(timezone.utc).isoformat(),
            allocated_epsilon=allocated_epsilon,
            used_epsilon=0.0
        )
        self.tokens[token] = info
        self.budget_manager.register_user(info.user_id, allocated_epsilon)

        self.audit_logger.log(
            user_id=info.user_id,
            user_role=role.value,
            action=ActionType.AUTHENTICATE,
            details={"event": "TOKEN_REGISTERED", "username": username},
            status="SUCCESS"
        )
        return info

    def get_token_info(self, token: str) -> Optional[TokenInfo]:
        return self.tokens.get(token)

    def revoke_token(self, token: str, reason: str, triggered_by: str = "SYSTEM") -> bool:
        """
        Revoke an active token immediately (NIS2 Art. 21(2)(b)).
        """
        info = self.tokens.get(token)
        if not info:
            return False

        info.is_revoked = True
        self.audit_logger.log(
            user_id=info.user_id,
            user_role=info.role.value,
            action=ActionType.REVOKE_TOKEN,
            details={"reason": reason, "triggered_by": triggered_by},
            status="ALERT"
        )
        return True

    def ingest_raw_data(self, token: str, new_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Healthcare Worker use case: Ingest/refresh raw clinical cohort data.
        """
        t_info = self.get_token_info(token)
        valid, msg = self.rules_engine.verify_token_authentication(t_info)
        if not valid:
            return False, msg

        valid, msg = self.rules_engine.verify_token_active(t_info)
        if not valid:
            return False, msg

        if t_info.role not in [UserRole.HEALTHCARE_WORKER, UserRole.ADMIN]:
            self.audit_logger.log(
                t_info.user_id, t_info.role.value, ActionType.INGEST_RAW_DATA,
                {"error": "Unauthorized role attempt to ingest raw data"}, "REJECTED"
            )
            return False, "Access Denied: Only Healthcare Workers or Admins may ingest raw data."

        self._raw_dataset = new_df.copy()
        raw_path = RAW_DATA_DIR / "patients.csv"
        self._raw_dataset.to_csv(raw_path, index=False)

        self.audit_logger.log(
            t_info.user_id, t_info.role.value, ActionType.INGEST_RAW_DATA,
            {"records_ingested": len(new_df), "columns": list(new_df.columns)}, "SUCCESS"
        )
        return True, f"Successfully ingested {len(new_df)} raw patient records."

    def execute_dp_query(
        self,
        token: str,
        request: DPQueryRequest
    ) -> Tuple[bool, Union[DPQueryResponse, str]]:
        """
        Researcher use case: Execute differentially private query with Laplace noise.
        """
        t_info = self.get_token_info(token)

        # Rule 1: Token Authentication
        valid, msg = self.rules_engine.verify_token_authentication(t_info)
        if not valid:
            return False, msg

        # Rule 2: Token Active
        valid, msg = self.rules_engine.verify_token_active(t_info)
        if not valid:
            return False, msg

        # Rule 4a: Rate Limiting
        rate_ok, rate_msg = self.rules_engine.check_rate_limit(t_info.user_id)
        if not rate_ok:
            # Trigger Incident & Revocation
            self.revoke_token(token, reason=rate_msg, triggered_by="RULES_ENGINE_RATE_LIMIT")
            self._trigger_security_incident(
                t_info, "Rate Limit Abuse Detected", rate_msg, IncidentSeverity.HIGH
            )
            return False, rate_msg

        # Rule 4b: Reconstruction Attack Pattern
        attack_detected, attack_msg = self.rules_engine.detect_reconstruction_attack(
            t_info.user_id, request.query_type, request.target_column, request.range_condition
        )
        if attack_detected:
            self.revoke_token(token, reason=attack_msg, triggered_by="RECONSTRUCTION_DETECTOR")
            self._trigger_security_incident(
                t_info, "Differential Privacy Reconstruction Attack", attack_msg, IncidentSeverity.CRITICAL
            )
            return False, attack_msg

        # Rule 3: Budget Verification
        admissible, budget_msg, rem_budget = self.budget_manager.check_and_consume_budget(
            t_info.user_id, request.epsilon, query_type=request.query_type
        )
        if not admissible:
            self.audit_logger.log(
                t_info.user_id, t_info.role.value, ActionType.EXECUTE_DP_QUERY,
                {"error": budget_msg, "requested_epsilon": request.epsilon}, "REJECTED"
            )
            return False, budget_msg

        # Filter cohort if requested (e.g. oncology)
        df_target = self._raw_dataset.copy()
        if request.filter_cohort == "oncology":
            df_target = df_target[df_target["is_oncology"] == True]

        if request.range_condition:
            col = request.range_condition.get("column")
            if col and col in df_target.columns:
                min_v = request.range_condition.get("min")
                max_v = request.range_condition.get("max")
                if min_v is not None:
                    df_target = df_target[df_target[col] >= min_v]
                if max_v is not None:
                    df_target = df_target[df_target[col] <= max_v]

        n_cohort = len(df_target)

        # Rule 5: Small-cell suppression
        valid_cell, cell_msg = self.rules_engine.verify_small_cell_threshold(n_cohort)
        if not valid_cell:
            self.audit_logger.log(
                t_info.user_id, t_info.role.value, ActionType.EXECUTE_DP_QUERY,
                {"error": cell_msg, "cohort_size": n_cohort}, "REJECTED"
            )
            return False, cell_msg

        # Compute query with Laplace Mechanism
        target_col = request.target_column
        if target_col not in df_target.columns:
            target_col = "blood_pressure_systolic"

        values = df_target[target_col].to_numpy()

        if request.query_type == "mean":
            noisy_res, bounds = self.dp_engine.private_mean(
                values=values,
                lower_bound=SYSTOLIC_BP_LOWER,
                upper_bound=SYSTOLIC_BP_UPPER,
                epsilon=request.epsilon,
                known_n=n_cohort
            )
            true_res = float(np.mean(values))
        elif request.query_type == "count":
            noisy_res, bounds = self.dp_engine.private_count(
                count=n_cohort,
                epsilon=request.epsilon
            )
            true_res = float(n_cohort)
        else:
            # Range count or generic count
            noisy_res, bounds = self.dp_engine.private_count(
                count=n_cohort,
                epsilon=request.epsilon
            )
            true_res = float(n_cohort)

        # Update token used epsilon
        t_info.used_epsilon += request.epsilon

        resp = DPQueryResponse(
            query_type=request.query_type,
            noisy_result=noisy_res,
            true_result=round(true_res, 4),
            epsilon_spent=request.epsilon,
            remaining_budget=round(rem_budget, 4),
            noise_scale=bounds["scale_b"],
            theoretical_mae=bounds["theoretical_mae"],
            theoretical_mse=bounds["theoretical_mse"]
        )

        self.audit_logger.log(
            user_id=t_info.user_id,
            user_role=t_info.role.value,
            action=ActionType.EXECUTE_DP_QUERY,
            details={
                "query_type": request.query_type,
                "epsilon_spent": request.epsilon,
                "cohort_size": n_cohort,
                "noisy_result": noisy_res,
                "noise_scale": bounds["scale_b"]
            },
            status="SUCCESS"
        )
        return True, resp

    def request_and_approve_kanon_dataset(
        self,
        token: str,
        k: int = 5,
        dataset_name: str = "health_cohort_anonymized",
        justification: str = "Scientific cardiovascular study"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Admin use case: Generate, evaluate risk/utility, approve, and export k-anonymized dataset.
        Pseudonymizes direct IDs with HMAC-SHA256 and applies Mondrian k-anonymity.
        """
        t_info = self.get_token_info(token)
        valid, msg = self.rules_engine.verify_token_authentication(t_info)
        if not valid:
            return False, {"error": msg}

        if t_info.role != UserRole.ADMIN:
            return False, {"error": "Only Privacy Officer / Admin can approve GDPR-compliant dataset releases."}

        # 1. Pseudonymize Direct Identifiers (HMAC-SHA256)
        pseudonymized_df = pseudonymize_dataframe(self._raw_dataset)

        # 2. Apply Mondrian k-Anonymity on Quasi-Identifiers
        anonymizer = MondrianAnonymizer(k=k, quasi_identifiers=QUASI_IDENTIFIERS)
        anon_df, summary = anonymizer.anonymize(pseudonymized_df)

        # 3. Save export to approved directory
        export_filename = f"{dataset_name}_k{k}.csv"
        export_path = ANONYMIZED_DATA_DIR / export_filename
        anon_df.to_csv(export_path, index=False)

        dataset_id = f"DS-{secrets.token_hex(4).upper()}"
        metadata = {
            "dataset_id": dataset_id,
            "filename": export_filename,
            "path": str(export_path),
            "k": k,
            "records": len(anon_df),
            "approved_by": t_info.username,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "justification": justification,
            "metrics": summary
        }
        self.approved_datasets[dataset_id] = metadata

        self.audit_logger.log(
            user_id=t_info.user_id,
            user_role=t_info.role.value,
            action=ActionType.APPROVE_DATASET,
            details=metadata,
            status="SUCCESS"
        )
        return True, metadata

    def download_anonymized_dataset(
        self,
        token: str,
        dataset_id: str
    ) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Researcher use case: Download an approved GDPR-compliant k-anonymized dataset.
        """
        t_info = self.get_token_info(token)
        valid, msg = self.rules_engine.verify_token_authentication(t_info)
        if not valid:
            return False, msg

        valid, msg = self.rules_engine.verify_token_active(t_info)
        if not valid:
            return False, msg

        if dataset_id not in self.approved_datasets:
            return False, f"Dataset ID '{dataset_id}' not found or not yet approved for release."

        ds_meta = self.approved_datasets[dataset_id]
        file_path = ds_meta["path"]
        df = pd.read_csv(file_path)

        self.audit_logger.log(
            user_id=t_info.user_id,
            user_role=t_info.role.value,
            action=ActionType.DOWNLOAD_DATASET,
            details={"dataset_id": dataset_id, "k": ds_meta["k"], "records": len(df)},
            status="SUCCESS"
        )
        return True, df

    def _trigger_security_incident(
        self,
        token_info: TokenInfo,
        title: str,
        reason: str,
        severity: IncidentSeverity
    ) -> Dict[str, Any]:
        """
        Internal incident automation: generates NIS2 Art. 23 and GDPR Art. 33 dossiers.
        """
        incident_id = f"INC-{secrets.token_hex(4).upper()}"

        nis2_report = self.incident_reporter.generate_nis2_early_warning(
            incident_id=incident_id,
            title=title,
            severity=severity,
            suspected_cause=reason,
            affected_assets=f"Health Privacy Gateway (User {token_info.username})",
            is_malicious=True,
            cross_border_impact=False
        )

        gdpr_report = self.incident_reporter.generate_gdpr_breach_notification(
            incident_id=incident_id,
            breach_nature=f"Attempted reconstruction attack / policy violation: {title}",
            estimated_affected_records=len(self._raw_dataset),
            likely_consequences="Attempted singling-out or inference mitigated by automated gateway revocation."
        )

        self.audit_logger.log(
            user_id=token_info.user_id,
            user_role=token_info.role.value,
            action=ActionType.SECURITY_INCIDENT,
            details={
                "incident_id": incident_id,
                "title": title,
                "severity": severity.value,
                "nis2_early_warning_generated": True,
                "gdpr_breach_notice_generated": True
            },
            status="ALERT"
        )
        return {"incident_id": incident_id, "nis2": nis2_report, "gdpr": gdpr_report}
