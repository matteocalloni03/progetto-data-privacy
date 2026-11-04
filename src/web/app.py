"""
FastAPI REST Application and Gateway Web Interface.
Implements endpoints for the Admin, Healthcare Worker, and Researcher personas.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import FIGURES_DIR, INCIDENTS_DIR, DATA_DIR
from src.gateway.gateway import PrivacyGateway
from src.gateway.models import DPQueryRequest, DPQueryResponse, DatasetApprovalRequest
from src.experiments.run_all_benchmarks import run_all as run_benchmarks_exec

app = FastAPI(
    title="Healthcare Privacy Gateway API",
    description="GDPR & NIS2 Compliant Data Sharing System (k-Anonymity vs Differential Privacy)",
    version="1.0.0"
)

# Gateway singleton
gateway = PrivacyGateway()

# Pre-approve a default k=5 dataset for researcher immediate use
try:
    gateway.request_and_approve_kanon_dataset(
        token="admin-token-secret-iscte-2026",
        k=5,
        dataset_name="research_cardio_baseline",
        justification="Initial approved baseline research cohort"
    )
except Exception:
    pass

# Mount figures as static files
app.mount("/static/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


class WebDPRequest(BaseModel):
    token: str
    query_type: str = "mean"
    target_column: str = "blood_pressure_systolic"
    filter_cohort: Optional[str] = None
    range_condition: Optional[Dict[str, Any]] = None
    epsilon: float = 0.5


class WebApproveRequest(BaseModel):
    token: str
    k: int = 5
    dataset_name: str = "health_cohort"
    justification: str = "Research study"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_gateway_status():
    return {
        "status": "OPERATIONAL",
        "regulations": ["GDPR (EU) 2016/679", "NIS2 (EU) 2022/2555"],
        "default_epsilon": gateway.default_epsilon,
        "default_k": gateway.default_k,
        "registered_tokens": len(gateway.tokens),
        "approved_datasets": len(gateway.approved_datasets)
    }


@app.get("/api/budget")
async def get_budget(token: str):
    t_info = gateway.get_token_info(token)
    if not t_info:
        raise HTTPException(status_code=401, detail="Invalid token")
    return gateway.budget_manager.get_budget_status(t_info.user_id)


@app.post("/api/dp/query")
async def execute_query(req: WebDPRequest):
    dp_req = DPQueryRequest(
        query_type=req.query_type,
        target_column=req.target_column,
        filter_cohort=req.filter_cohort,
        range_condition=req.range_condition,
        epsilon=req.epsilon
    )
    success, result = gateway.execute_dp_query(req.token, dp_req)
    if not success:
        raise HTTPException(status_code=400, detail=str(result))
    return result


@app.post("/api/kanon/approve")
async def approve_dataset(req: WebApproveRequest):
    success, result = gateway.request_and_approve_kanon_dataset(
        token=req.token,
        k=req.k,
        dataset_name=req.dataset_name,
        justification=req.justification
    )
    if not success:
        raise HTTPException(status_code=403, detail=result.get("error", "Approval failed"))
    return result


@app.get("/api/kanon/datasets")
async def list_datasets():
    return list(gateway.approved_datasets.values())


@app.get("/api/kanon/download")
async def download_dataset(token: str, dataset_id: str):
    success, result = gateway.download_anonymized_dataset(token, dataset_id)
    if not success:
        raise HTTPException(status_code=403, detail=str(result))

    meta = gateway.approved_datasets.get(dataset_id, {})
    file_path = meta.get("path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=Path(file_path).name
    )


@app.get("/api/audit/logs")
async def get_audit_logs(limit: int = 30):
    return gateway.audit_logger.get_recent_entries(limit=limit)


@app.get("/api/audit/verify")
async def verify_audit_chain():
    is_valid, count, errors = gateway.audit_logger.verify_chain()
    return {
        "is_valid": is_valid,
        "records_checked": count,
        "errors": errors
    }


@app.get("/api/incidents")
async def list_incidents():
    incidents = []
    if INCIDENTS_DIR.exists():
        for f in INCIDENTS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                incidents.append({
                    "file": f.name,
                    "title": data.get("incident_classification", {}).get("title") or data.get("nature_of_the_breach", {}).get("breach_type"),
                    "severity": data.get("incident_classification", {}).get("severity_level") or "CRITICAL",
                    "article": data.get("article")
                })
            except Exception:
                pass
    return incidents


@app.get("/api/incidents/{filename}")
async def download_incident_json(filename: str):
    f_path = INCIDENTS_DIR / filename
    if not f_path.exists():
        raise HTTPException(status_code=404, detail="Incident file not found")
    return FileResponse(path=str(f_path), media_type="application/json", filename=filename)


@app.post("/api/benchmarks/run")
async def rerun_benchmarks():
    try:
        run_benchmarks_exec()
        return {"status": "SUCCESS", "message": "Benchmarks recalculated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
