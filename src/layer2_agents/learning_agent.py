"""
Learning & Development Agent (Layer 2 — Intelligent Agents).

Implements §5.6:
  - Skill-demand-driven learning recommendations
  - At-risk learner identification
  - Personalized learning path generation
  - Progress monitoring and adaptive adjustment
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..common.models import AgentType, AuditRecord, Employee, LearningActivity
from ..layer3_ai_services.recommendation_engine import CollaborativeFilter
from ..layer6_governance.audit_logger import AuditLogger


CONTENT_CATALOG: List[Dict[str, Any]] = [
    {"id": "c001", "title": "Python for Data Science", "skill_ids": ["python", "data_analysis"], "hours": 8, "level": "intermediate"},
    {"id": "c002", "title": "Machine Learning Foundations", "skill_ids": ["machine learning"], "hours": 16, "level": "intermediate"},
    {"id": "c003", "title": "Leadership Essentials", "skill_ids": ["leadership", "communication"], "hours": 4, "level": "beginner"},
    {"id": "c004", "title": "Cloud Architecture on AWS", "skill_ids": ["aws", "kubernetes"], "hours": 12, "level": "advanced"},
    {"id": "c005", "title": "Agile Project Management", "skill_ids": ["agile", "project management"], "hours": 6, "level": "beginner"},
    {"id": "c006", "title": "Advanced SQL and Analytics", "skill_ids": ["sql", "data analysis"], "hours": 8, "level": "advanced"},
    {"id": "c007", "title": "NLP with Transformers", "skill_ids": ["nlp", "deep learning"], "hours": 20, "level": "advanced"},
    {"id": "c008", "title": "Team Communication", "skill_ids": ["communication", "teamwork"], "hours": 3, "level": "beginner"},
    {"id": "c009", "title": "DevOps Fundamentals", "skill_ids": ["docker", "ci/cd"], "hours": 10, "level": "intermediate"},
    {"id": "c010", "title": "Data Visualization with Tableau", "skill_ids": ["tableau", "data analysis"], "hours": 6, "level": "beginner"},
]


class LearningAgent:

    AGENT_VERSION = "la-v1.0"

    def __init__(
        self,
        collab_filter: Optional[CollaborativeFilter] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._cf = collab_filter
        self._audit = audit_logger

    # ---------------------------------------------------------------- #
    # Personalized Learning Path
    # ---------------------------------------------------------------- #

    def generate_learning_path(
        self,
        employee: Employee,
        skill_gap: List[str],
        org_demand_skills: Optional[List[str]] = None,
        max_hours: int = 40,
    ) -> Dict[str, Any]:
        priority_skills = set(skill_gap)
        if org_demand_skills:
            priority_skills.update(org_demand_skills)

        scored_content = []
        for content in CONTENT_CATALOG:
            overlap = len(set(content["skill_ids"]) & priority_skills)
            if overlap > 0:
                scored_content.append({**content, "priority_score": overlap})

        # Peer-based additions
        if self._cf:
            peer_content_ids = self._cf.recommend_learning_from_peers(
                employee.id, all_activities=[], top_k=3
            )
            for cid in peer_content_ids:
                match = next((c for c in CONTENT_CATALOG if c["id"] == cid), None)
                if match and not any(s["id"] == cid for s in scored_content):
                    scored_content.append({**match, "priority_score": 0.5})

        scored_content.sort(key=lambda x: x["priority_score"], reverse=True)

        # Fill path up to max_hours budget
        path: List[Dict] = []
        total_hours = 0
        for item in scored_content:
            if total_hours + item["hours"] <= max_hours:
                path.append(item)
                total_hours += item["hours"]

        return {
            "employee_id": employee.id,
            "learning_path": path,
            "total_hours": total_hours,
            "skills_addressed": list({s for item in path for s in item["skill_ids"]}),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ---------------------------------------------------------------- #
    # At-Risk Learner Detection
    # ---------------------------------------------------------------- #

    def identify_at_risk_learners(
        self,
        employees: List[Employee],
        learning_activities: List[LearningActivity],
        stall_threshold_days: int = 21,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow()
        at_risk = []
        for emp in employees:
            emp_activities = [la for la in learning_activities if la.employee_id == emp.id]
            in_progress = [la for la in emp_activities if 0 < la.progress < 1.0]
            for la in in_progress:
                last_date = la.completion_date or datetime.min
                days_stalled = (cutoff - last_date).days if last_date != datetime.min else stall_threshold_days
                if days_stalled >= stall_threshold_days:
                    at_risk.append({
                        "employee_id": emp.id,
                        "activity_id": la.id,
                        "content_title": la.content_title,
                        "progress": la.progress,
                        "days_stalled": days_stalled,
                        "recommendation": "Schedule a learning support check-in or adjust path complexity",
                    })
        return at_risk

    # ---------------------------------------------------------------- #
    # Progress Monitoring
    # ---------------------------------------------------------------- #

    def learning_progress_report(
        self, employee: Employee, learning_activities: List[LearningActivity]
    ) -> Dict[str, Any]:
        emp_activities = [la for la in learning_activities if la.employee_id == employee.id]
        completed = [la for la in emp_activities if la.progress >= 1.0]
        total_hours = sum(
            next((c["hours"] for c in CONTENT_CATALOG if c["id"] == la.content_id), 0)
            for la in completed
        )
        acquired_skills = list({s for la in completed for s in la.skill_ids})

        return {
            "employee_id": employee.id,
            "completed_count": len(completed),
            "completion_rate": round(len(completed) / len(emp_activities), 3) if emp_activities else 0.0,
            "total_learning_hours": total_hours,
            "acquired_skills": acquired_skills,
            "in_progress_count": sum(1 for la in emp_activities if 0 < la.progress < 1.0),
        }
