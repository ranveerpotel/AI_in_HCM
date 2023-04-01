"""
Dashboard & REST API (Layer 1 — Experience).

FastAPI application exposing the AI-HCM platform capabilities as REST endpoints.
Implements the experience layer from §4.2.1:
  - /chat        — Employee Assistance Agent
  - /attrition   — Attrition risk predictions
  - /team        — Manager team health report
  - /career      — Career path recommendations
  - /learning    — Personalized learning path
  - /recruitment — Candidate matching
  - /governance  — Bias + drift monitoring dashboard
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..common.models import AgentType


def create_app(platform: Any) -> Any:
    """
    Factory that wires the AI-HCM platform into a FastAPI application.
    `platform` is the AIHCMPlatform orchestrator from main.py.
    """
    if not HAS_FASTAPI:
        raise ImportError("fastapi and pydantic are required for the REST API.")

    app = FastAPI(
        title="AI-HCM Platform API",
        description="Intelligent Workforce Management System — Ranveer Potel (IJAIBDCMS 2023)",
        version="1.0.0",
    )

    # ---------------------------------------------------------------- #
    # Request / Response schemas
    # ---------------------------------------------------------------- #

    class ChatRequest(BaseModel):
        session_id: str
        employee_id: str
        query: str

    class AttritionRequest(BaseModel):
        employee_id: str

    class TeamReportRequest(BaseModel):
        manager_id: str

    class CareerRequest(BaseModel):
        employee_id: str
        target_role_id: Optional[str] = None

    class LearningRequest(BaseModel):
        employee_id: str
        skill_gap: List[str] = []

    class CandidateRequest(BaseModel):
        name: str
        resume_text: str
        years_experience: float
        role_id: str
        demographic_group: Optional[str] = None

    # ---------------------------------------------------------------- #
    # Endpoints
    # ---------------------------------------------------------------- #

    @app.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    @app.post("/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        emp = platform.get_employee(req.employee_id)
        if not emp:
            raise HTTPException(404, f"Employee {req.employee_id} not found")
        return platform.employee_agent.handle_query(req.session_id, req.query, emp)

    @app.get("/attrition/{employee_id}")
    def attrition_risk(employee_id: str) -> Dict[str, Any]:
        emp = platform.get_employee(employee_id)
        if not emp:
            raise HTTPException(404, f"Employee {employee_id} not found")
        fv = platform.feature_store.get(employee_id)
        if not fv:
            raise HTTPException(422, "Feature vector not computed for employee")
        pred = platform.attrition_predictor.predict(employee_id, fv.to_array())
        explanation = platform.xai_explainer.explain_attrition_risk(pred)
        return {
            "employee_id": employee_id,
            "risk_score": pred.risk_score,
            "risk_level": pred.risk_level.value,
            "top_factors": pred.top_factors,
            "interventions": pred.recommended_interventions,
            "explanation": explanation,
        }

    @app.get("/team/{manager_id}")
    def team_report(manager_id: str) -> Dict[str, Any]:
        reports = platform.get_direct_reports(manager_id)
        if not reports:
            raise HTTPException(404, f"No team found for manager {manager_id}")
        report = platform.manager_agent.team_health_report(manager_id, reports)
        report["coaching_summary"] = platform.manager_agent.coaching_summary(manager_id, report)
        return report

    @app.get("/career/{employee_id}")
    def career_path(employee_id: str) -> Dict[str, Any]:
        emp = platform.get_employee(employee_id)
        if not emp:
            raise HTTPException(404, f"Employee {employee_id} not found")
        rec = platform.talent_agent.career_path_recommendation(emp, list(platform.roles.values()))
        return {
            "employee_id": employee_id,
            "recommended_roles": rec.recommended_roles,
            "succession_potential": rec.succession_potential,
            "development_priorities": rec.development_priorities,
        }

    @app.post("/learning")
    def learning_path(req: LearningRequest) -> Dict[str, Any]:
        emp = platform.get_employee(req.employee_id)
        if not emp:
            raise HTTPException(404, f"Employee {req.employee_id} not found")
        return platform.learning_agent.generate_learning_path(emp, req.skill_gap)

    @app.post("/recruitment/match")
    def match_candidate(req: CandidateRequest) -> Dict[str, Any]:
        from ..layer2_agents.recruitment_agent import Candidate
        cand = Candidate(
            id=str(uuid.uuid4()),
            name=req.name,
            resume_text=req.resume_text,
            years_experience=req.years_experience,
            demographic_group=req.demographic_group,
        )
        platform.recruitment_agent.process_resume(cand)
        role = platform.roles.get(req.role_id)
        required_skills = role.required_skills if role else []
        match = platform.recruitment_agent.match_candidate_to_role(cand, required_skills, req.role_id)
        return {
            "candidate_id": match.candidate_id,
            "role_id": match.role_id,
            "compatibility_score": match.compatibility_score,
            "skill_match": match.skill_match,
            "culture_fit_score": match.culture_fit_score,
            "explanation": match.explanation,
        }

    @app.get("/governance/bias/{model_id}")
    def bias_report(model_id: str) -> Dict[str, Any]:
        return {"model_id": model_id, "status": "No bias evaluation data yet — run evaluate() first"}

    @app.get("/governance/audit/{employee_id}")
    def audit_trail(employee_id: str) -> List[Dict[str, Any]]:
        records = platform.audit_logger.get_records_for_employee(employee_id)
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "agent": r.agent_type.value,
                "action": r.action_type,
                "human_reviewed": r.human_reviewed,
            }
            for r in records
        ]

    @app.get("/governance/summary")
    def governance_summary() -> Dict[str, Any]:
        return platform.audit_logger.summary()

    return app
