"""
Talent Intelligence Agent (Layer 2 — Intelligent Agents).

Implements §5.4:
  - Skill gap analysis
  - Career path prediction
  - Succession planning
  - Internal mobility optimization
  - Workforce mobility marketplace (§11.4)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..common.models import (
    AgentType, AuditRecord, CareerPathRecommendation, Employee, Role,
)
from ..layer3_ai_services.recommendation_engine import HybridRecommender
from ..layer4_data_intelligence.knowledge_graph import WorkforceKnowledgeGraph
from ..layer6_governance.audit_logger import AuditLogger


class TalentIntelligenceAgent:

    AGENT_VERSION = "tia-v1.0"

    def __init__(
        self,
        knowledge_graph: WorkforceKnowledgeGraph,
        recommender: HybridRecommender,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._kg = knowledge_graph
        self._rec = recommender
        self._audit = audit_logger

    # ---------------------------------------------------------------- #
    # Skill Gap Analysis
    # ---------------------------------------------------------------- #

    def skill_gap_analysis(
        self, employee: Employee, target_role_id: str
    ) -> Dict[str, Any]:
        gap_skill_ids = self._kg.get_skill_gap(employee.id, target_role_id)
        emp_skills = {sa.skill_id: sa.proficiency for sa in employee.skills}
        required = self._kg.get_role_required_skills(target_role_id)
        coverage = len(set(required) - set(gap_skill_ids)) / len(required) if required else 1.0

        return {
            "employee_id": employee.id,
            "target_role_id": target_role_id,
            "gap_skills": gap_skill_ids,
            "coverage_rate": round(coverage, 3),
            "current_skills": list(emp_skills.keys()),
            "readiness_level": (
                "ready" if coverage >= 0.9
                else "near-ready" if coverage >= 0.7
                else "development-needed"
            ),
        }

    # ---------------------------------------------------------------- #
    # Career Path Prediction
    # ---------------------------------------------------------------- #

    def career_path_recommendation(
        self,
        employee: Employee,
        all_roles: List[Role],
        top_k: int = 3,
    ) -> CareerPathRecommendation:
        current_role = None
        for role in all_roles:
            if role.id == employee.role_id:
                current_role = role
                break

        # Roles reachable from current via career ladder
        reachable_ids = set(self._kg.get_career_path_roles(employee.role_id, depth=3))
        candidate_roles = [r for r in all_roles if r.id in reachable_ids]

        if not candidate_roles:
            candidate_roles = all_roles  # fallback: consider all roles

        recommendations = self._rec.recommend_roles(employee, candidate_roles, top_k=top_k)

        # Enrich with skill gap
        for rec in recommendations:
            rec["skill_gap"] = self._kg.get_skill_gap(employee.id, rec["role_id"])

        succession_potential = min(
            employee.performance_score / 5.0 * 0.5 + employee.tenure_years / 10.0 * 0.3,
            1.0,
        )

        dev_priorities = []
        for rec in recommendations[:1]:
            dev_priorities.extend(rec.get("skill_gap", [])[:3])

        return CareerPathRecommendation(
            employee_id=employee.id,
            recommended_roles=recommendations,
            succession_potential=round(succession_potential, 3),
            development_priorities=dev_priorities,
        )

    # ---------------------------------------------------------------- #
    # Succession Planning
    # ---------------------------------------------------------------- #

    def succession_planning(
        self,
        critical_role_id: str,
        all_employees: List[Employee],
        all_roles: List[Role],
    ) -> List[Dict[str, Any]]:
        """Identify high-potential employees for a critical role."""
        candidates = []
        for emp in all_employees:
            gap = self.skill_gap_analysis(emp, critical_role_id)
            if gap["coverage_rate"] >= 0.6 and emp.performance_score >= 3.5:
                candidates.append({
                    "employee_id": emp.id,
                    "coverage_rate": gap["coverage_rate"],
                    "performance_score": emp.performance_score,
                    "tenure_years": emp.tenure_years,
                    "readiness_level": gap["readiness_level"],
                    "gap_skills": gap["gap_skills"],
                    "overall_fit": round(
                        gap["coverage_rate"] * 0.5 + emp.performance_score / 5.0 * 0.5, 3
                    ),
                })

        candidates.sort(key=lambda x: x["overall_fit"], reverse=True)

        if self._audit:
            self._audit.log(AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_type=AgentType.TALENT_INTELLIGENCE,
                action_type="succession_planning",
                employee_id="system",
                input_summary={"critical_role_id": critical_role_id, "candidates_evaluated": len(all_employees)},
                output_summary={"top_candidates": len(candidates[:3])},
                model_version=self.AGENT_VERSION,
            ))

        return candidates[:5]

    # ---------------------------------------------------------------- #
    # Internal Mobility Marketplace (§11.4)
    # ---------------------------------------------------------------- #

    def internal_mobility_matches(
        self,
        employee: Employee,
        open_roles: List[Role],
        open_projects: List[Any],  # List[Project]
        top_k: int = 5,
    ) -> Dict[str, Any]:
        role_recs = self._rec.recommend_roles(employee, open_roles, top_k=top_k)
        for rec in role_recs:
            rec["skill_gap"] = self._kg.get_skill_gap(employee.id, rec["role_id"])

        return {
            "employee_id": employee.id,
            "role_matches": role_recs,
            "match_generated_at": datetime.utcnow().isoformat(),
        }
