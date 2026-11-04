"""
Security and Table 8 Rules Verification Tests (FR-08, FR-09, FR-10, SEC-02, SEC-03, SEC-06).
"""

import pytest
import pandas as pd
from src.gateway.gateway import PrivacyGateway
from src.gateway.models import DPQueryRequest, UserRole
from src.gateway.audit_logger import AuditLogger
from src.gateway.incident_reporter import IncidentReporter


@pytest.fixture
def test_gateway(tmp_path):
    log_file = tmp_path / "test_audit.jsonl"
    inc_dir = tmp_path / "incidents"
    logger = AuditLogger(log_path=log_file)
    reporter = IncidentReporter(output_dir=inc_dir)
    gw = PrivacyGateway(
        default_epsilon=2.0,
        default_k=5,
        audit_logger=logger,
        incident_reporter=reporter
    )
    return gw


def test_rule_1_unauthenticated_request(test_gateway):
    req = DPQueryRequest(query_type="mean", epsilon=0.5)
    ok, resp = test_gateway.execute_dp_query("non-existent-token", req)
    assert ok is False
    assert "Rule 1 Violation" in str(resp)


def test_rule_2_revoked_token(test_gateway):
    token = "test-researcher-token"
    test_gateway.register_token(token, "Test Researcher", UserRole.RESEARCHER, allocated_epsilon=5.0)

    # Manually revoke
    test_gateway.revoke_token(token, reason="Administrative test revocation")

    req = DPQueryRequest(query_type="mean", epsilon=0.5)
    ok, resp = test_gateway.execute_dp_query(token, req)
    assert ok is False
    assert "Rule 2 Violation" in str(resp)


def test_rule_3_budget_exhaustion(test_gateway):
    token = "budget-test-token"
    test_gateway.register_token(token, "Budget Tester", UserRole.RESEARCHER, allocated_epsilon=1.0)

    # First query ok (eps = 0.8)
    req1 = DPQueryRequest(query_type="count", epsilon=0.8)
    ok1, resp1 = test_gateway.execute_dp_query(token, req1)
    assert ok1 is True

    # Second query exceeds remaining 0.2 (eps = 0.5)
    req2 = DPQueryRequest(query_type="count", epsilon=0.5)
    ok2, resp2 = test_gateway.execute_dp_query(token, req2)
    assert ok2 is False
    assert "budget exhausted" in str(resp2).lower()


def test_rule_4_reconstruction_attack_auto_revocation(test_gateway):
    token = "adversary-token"
    test_gateway.register_token(token, "Adversary Eve", UserRole.RESEARCHER, allocated_epsilon=5.0)

    # Send 3 identical queries in rapid succession
    req = DPQueryRequest(
        query_type="mean",
        target_column="blood_pressure_systolic",
        range_condition={"column": "age", "min": 40, "max": 41},
        epsilon=0.1
    )

    test_gateway.execute_dp_query(token, req)
    test_gateway.execute_dp_query(token, req)
    ok, resp = test_gateway.execute_dp_query(token, req)

    # The 3rd repeated query must trigger reconstruction abuse alert
    assert ok is False
    assert "reconstruction attempt detected" in str(resp).lower()

    # Token must now be revoked
    info = test_gateway.get_token_info(token)
    assert info.is_revoked is True


def test_rule_5_small_cell_suppression(test_gateway):
    token = "cell-test-token"
    test_gateway.register_token(token, "Cell Tester", UserRole.RESEARCHER, allocated_epsilon=5.0)

    # Target an empty or extremely narrow age bracket yielding < 5 patients
    req = DPQueryRequest(
        query_type="mean",
        target_column="blood_pressure_systolic",
        range_condition={"column": "age", "min": 87, "max": 88},  # very few patients in this extreme bracket
        epsilon=0.1
    )
    # If the filtered cohort has < 5 patients, Rule 5 triggers
    # Let's verify with an impossible condition (min > max or extreme age > 100)
    req_empty = DPQueryRequest(
        query_type="mean",
        target_column="blood_pressure_systolic",
        range_condition={"column": "age", "min": 95, "max": 99},
        epsilon=0.1
    )
    ok, resp = test_gateway.execute_dp_query(token, req_empty)
    assert ok is False
    assert "Rule 5 Violation" in str(resp)


def test_admin_kanon_approval_workflow(test_gateway):
    admin_token = "admin-token-secret-iscte-2026"
    researcher_token = "researcher-token-oncology-lab"

    # Researcher cannot approve datasets
    ok_res, msg_res = test_gateway.request_and_approve_kanon_dataset(researcher_token, k=5)
    assert ok_res is False
    assert "Only Privacy Officer / Admin" in msg_res["error"]

    # Admin approves dataset
    ok_adm, meta_adm = test_gateway.request_and_approve_kanon_dataset(
        admin_token, k=5, dataset_name="cardio_export", justification="Study approval"
    )
    assert ok_adm is True
    ds_id = meta_adm["dataset_id"]

    # Researcher can now download the approved dataset
    ok_dl, df_dl = test_gateway.download_anonymized_dataset(researcher_token, ds_id)
    assert ok_dl is True
    assert isinstance(df_dl, pd.DataFrame)
    assert "pseudonym_id" in df_dl.columns
