"""
Unit tests for Regulatory Incident Reporting under NIS2 Art. 23 & GDPR Art. 33 (SEC-04, SEC-05).
"""

import pytest
import json
from pathlib import Path
from src.gateway.incident_reporter import IncidentReporter
from src.gateway.models import IncidentSeverity


def test_nis2_early_warning_generation(tmp_path):
    reporter = IncidentReporter(output_dir=tmp_path)
    inc_id = "INC-TEST-001"

    report = reporter.generate_nis2_early_warning(
        incident_id=inc_id,
        title="Unauthorized Reconstruction Abuse",
        severity=IncidentSeverity.CRITICAL,
        suspected_cause="Adversary script attempted 50 overlapping count queries in 10s",
        affected_assets="Cardiovascular and Oncology Health Data",
        is_malicious=True,
        cross_border_impact=False
    )

    # Check structure
    assert report["regulatory_framework"] == "NIS2 Directive (EU) 2022/2555"
    assert "Article 23(4)(a)" in report["article"]
    assert report["incident_classification"]["severity_level"] == "CRITICAL"
    assert report["incident_classification"]["suspected_malicious_intent"] is True
    assert "CNCS" in report["competent_authority"]

    # Check deadlines
    tl = report["reporting_timeline"]
    assert "step_1_early_warning_due" in tl
    assert "step_2_incident_notification_due" in tl
    assert "step_3_final_report_due" in tl

    # Verify JSON file written to disk
    file_path = tmp_path / f"nis2_early_warning_{inc_id}.json"
    assert file_path.exists()
    disk_data = json.loads(file_path.read_text(encoding="utf-8"))
    assert disk_data["report_metadata"]["incident_id"] == inc_id


def test_gdpr_art33_breach_notification_generation(tmp_path):
    reporter = IncidentReporter(output_dir=tmp_path)
    inc_id = "INC-TEST-002"

    report = reporter.generate_gdpr_breach_notification(
        incident_id=inc_id,
        breach_nature="Attempted singling out attack on Oncology cohort",
        estimated_affected_records=64,
        data_categories=["Quasi-Identifiers (Age, Gender)", "Special Category Health Data (Oncology Diagnosis)"]
    )

    assert report["regulatory_framework"] == "GDPR (Regulation (EU) 2016/679)"
    assert "Article 33" in report["article"]
    assert "CNPD" in report["supervisory_authority"]
    assert report["nature_of_the_breach"]["special_category_health_data_involved"] is True
    assert report["nature_of_the_breach"]["approximate_number_of_data_subjects"] == 64

    file_path = tmp_path / f"gdpr_art33_breach_{inc_id}.json"
    assert file_path.exists()
