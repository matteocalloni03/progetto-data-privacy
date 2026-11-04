"""
Automated Regulatory Incident Reporting Module.
Implements NIS2 Directive Art. 23 (Early Warning < 24h, Incident Notification < 72h, Final Report)
and GDPR Art. 33 (Personal Data Breach Notification to supervisory authority CNPD).
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from src.config import INCIDENTS_DIR
from src.gateway.models import IncidentSeverity


class IncidentReporter:
    """
    Generates standardized, regulator-ready JSON incident draft dossiers
    for both NIS2 CSIRTs (e.g. CNCS in Portugal) and GDPR DPAs (CNPD in Portugal).
    """

    def __init__(self, output_dir: Path = INCIDENTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_nis2_early_warning(
        self,
        incident_id: str,
        title: str,
        severity: IncidentSeverity,
        suspected_cause: str,
        affected_assets: str = "Healthcare Privacy Gateway Core DB",
        is_malicious: bool = True,
        cross_border_impact: bool = False,
        mitigations_applied: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        NIS2 Art. 23(4)(a) Early Warning — Mandatory within 24 hours of becoming aware.
        """
        now = datetime.now(timezone.utc)
        deadline_24h = now + timedelta(hours=24)
        deadline_72h = now + timedelta(hours=72)
        deadline_1month = now + timedelta(days=30)

        report = {
            "regulatory_framework": "NIS2 Directive (EU) 2022/2555",
            "article": "Article 23(4)(a) - Early Warning",
            "competent_authority": "National Cybersecurity Centre (CNCS - Portugal)",
            "report_metadata": {
                "incident_id": incident_id,
                "timestamp_utc": now.isoformat(),
                "deadline_submission_utc": deadline_24h.isoformat(),
                "status": "DRAFT_EARLY_WARNING"
            },
            "incident_classification": {
                "title": title,
                "severity_level": severity.value,
                "sector": "High Criticality - Healthcare / Health Data Platform (Annex I)",
                "affected_assets": affected_assets,
                "suspected_malicious_intent": is_malicious,
                "cross_border_impact_potential": cross_border_impact
            },
            "initial_assessment": {
                "technical_summary": suspected_cause,
                "mitigations_immediate": mitigations_applied or [
                    "Revocation of offending user/token",
                    "Suspension of query interface for affected session",
                    "Hash-chain cryptographic log state preserved for forensics"
                ]
            },
            "reporting_timeline": {
                "step_1_early_warning_due": deadline_24h.isoformat(),
                "step_2_incident_notification_due": deadline_72h.isoformat(),
                "step_3_final_report_due": deadline_1month.isoformat()
            }
        }

        file_path = self.output_dir / f"nis2_early_warning_{incident_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def generate_gdpr_breach_notification(
        self,
        incident_id: str,
        breach_nature: str,
        estimated_affected_records: int,
        data_categories: Optional[list] = None,
        likely_consequences: str = "Potential re-identification or privacy degradation of health data",
        remedial_measures: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        GDPR Art. 33 Personal Data Breach Notification — Mandatory within 72 hours to DPA (CNPD).
        """
        now = datetime.now(timezone.utc)
        deadline_72h = now + timedelta(hours=72)

        report = {
            "regulatory_framework": "GDPR (Regulation (EU) 2016/679)",
            "article": "Article 33 - Notification of a personal data breach to the supervisory authority",
            "supervisory_authority": "Comissão Nacional de Proteção de Dados (CNPD - Portugal)",
            "notification_metadata": {
                "incident_id": incident_id,
                "timestamp_utc": now.isoformat(),
                "deadline_72h_utc": deadline_72h.isoformat(),
                "status": "DRAFT_DPA_NOTIFICATION"
            },
            "data_controller_information": {
                "organization": "Healthcare Data Management & Research Platform",
                "role": "Data Controller / Privacy Officer",
                "dpo_contact": "dpo@hospital-research.pt"
            },
            "nature_of_the_breach": {
                "breach_type": breach_nature,
                "special_category_health_data_involved": True,
                "gdpr_article_involved": "Article 9 (Processing of special categories of personal data)",
                "data_categories": data_categories or [
                    "Quasi-Identifiers (Age, Gender, Postal Area, BMI)",
                    "Sensitive Health Attributes (Diagnoses, Blood Pressure, Glucose)"
                ],
                "approximate_number_of_data_subjects": estimated_affected_records
            },
            "risk_and_consequences": {
                "severity_assessment": "High risk to rights and freedoms of natural persons",
                "likely_consequences": likely_consequences
            },
            "corrective_and_remedial_measures": {
                "applied_measures": remedial_measures or [
                    "Immediate token revocation and query endpoint isolation",
                    "Privacy Gateway rules updated to block suspicious pattern",
                    "Audit log hash-chain verification executed"
                ]
            }
        }

        file_path = self.output_dir / f"gdpr_art33_breach_{incident_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report
