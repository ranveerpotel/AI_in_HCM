"""
Managerial Decision Agent (Layer 2 — Intelligent Agents).

Implements §5.3: synthesizes workforce data into actionable, contextualized
team insights for managers — reducing the cognitive bottleneck of multi-source
data navigation (paper §10.3).

Capabilities:
  - Team health summary
  - Compensation benchmarking
  - Attrition risk identification
  - Promotion readiness assessment
  - Headcount / succession planning support
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import anthropic

from ..common.config import config
from ..common.models import (
    AgentType, AuditRecord, AttritionPrediction, Employee,
    PerformancePrediction, RiskLevel,
)
from ..layer3_ai_services.predictive_models import AttritionPredictor, PerformancePredictor
from ..layer4_data_intelligence.knowledge_graph import WorkforceKnowledgeGraph
from ..layer4_data_intelligence.feature_store import FeatureStore
from ..layer6_governance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ManagerialDecisionAgent:
    """
    AI advisor that synthesises team intelligence for a manager.
    All recommendations include supporting rationale and confidence signals.
    """

    AGENT_VERSION = "mda-v1.0"

    def __init__(
        self,
        knowledge_graph: WorkforceKnowledgeGraph,
        feature_store: FeatureStore,
        attrition_predictor: AttritionPredictor,
        performance_predictor: PerformancePredictor,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._kg = knowledge_graph
        self._fs = feature_store
        self._attrition = attrition_predictor
        self._performance = performance_predictor
        self._audit = audit_logger
        if config.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            self._client = None

    # ---------------------------------------------------------------- #
    # Team Health Report
    # ---------------------------------------------------------------- #

    def team_health_report(
        self, manager_id: str, direct_reports: List[Employee]
    ) -> Dict[str, Any]:
        if not direct_reports:
            return {"error": "No direct reports found"}

        attrition_risks = []
        performance_summaries = []

        for emp in direct_reports:
            fv = self._fs.get(emp.id)
            if fv is None:
                continue
            features = fv.to_array()
            attrition_pred = self._attrition.predict(emp.id, features)
            perf_pred = self._performance.predict(emp.id, features)
            attrition_risks.append(attrition_pred)
            performance_summaries.append(perf_pred)

        skill_coverage = self._kg.get_team_skill_coverage(manager_id)
        high_risk = [p for p in attrition_risks if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        avg_performance = (
            sum(p.predicted_score for p in performance_summaries) / len(performance_summaries)
            if performance_summaries else 0.0
        )
        avg_engagement = (
            sum(e.engagement_score for e in direct_reports) / len(direct_reports)
        )

        report = {
            "manager_id": manager_id,
            "team_size": len(direct_reports),
            "average_performance": round(avg_performance, 2),
            "average_engagement": round(avg_engagement, 3),
            "high_attrition_risk_count": len(high_risk),
            "attrition_risk_details": [
                {
                    "employee_id": p.employee_id,
                    "risk_score": round(p.risk_score, 3),
                    "risk_level": p.risk_level.value,
                    "top_intervention": p.recommended_interventions[0] if p.recommended_interventions else "",
                }
                for p in high_risk
            ],
            "skill_coverage": skill_coverage,
            "performance_trajectories": {
                p.employee_id: p.growth_trajectory for p in performance_summaries
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

        if self._audit:
            self._audit.log(AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_type=AgentType.MANAGERIAL_DECISION,
                action_type="team_health_report",
                employee_id=manager_id,
                input_summary={"team_size": len(direct_reports)},
                output_summary={"high_risk_count": len(high_risk), "avg_perf": avg_performance},
                model_version=self.AGENT_VERSION,
            ))

        return report

    # ---------------------------------------------------------------- #
    # Compensation Benchmarking
    # ---------------------------------------------------------------- #

    def compensation_analysis(self, direct_reports: List[Employee]) -> Dict[str, Any]:
        below_market = [
            {"employee_id": e.id, "comp_ratio": round(e.market_comp_ratio, 3)}
            for e in direct_reports if e.market_comp_ratio < 0.90
        ]
        above_market = [
            {"employee_id": e.id, "comp_ratio": round(e.market_comp_ratio, 3)}
            for e in direct_reports if e.market_comp_ratio > 1.15
        ]
        return {
            "team_avg_comp_ratio": round(
                sum(e.market_comp_ratio for e in direct_reports) / len(direct_reports), 3
            ) if direct_reports else 0.0,
            "below_market_employees": below_market,
            "above_market_employees": above_market,
            "equity_action_needed": len(below_market) > 0,
        }

    # ---------------------------------------------------------------- #
    # Promotion Readiness
    # ---------------------------------------------------------------- #

    def promotion_readiness(self, direct_reports: List[Employee]) -> List[Dict[str, Any]]:
        ready = []
        for emp in direct_reports:
            fv = self._fs.get(emp.id)
            if fv is None:
                continue
            score = 0.0
            reasons = []
            if emp.performance_score >= 4.0:
                score += 0.35
                reasons.append("High performance rating")
            if emp.tenure_years >= 1.5:
                score += 0.20
                reasons.append("Sufficient tenure")
            if emp.promotion_velocity >= 0:
                score += 0.15
                reasons.append("On-track promotion velocity")
            if fv.skill_depth >= 0.7:
                score += 0.20
                reasons.append("Advanced skill proficiency")
            if fv.learning_completion_rate >= 0.8:
                score += 0.10
                reasons.append("Strong learning completion")
            if score >= 0.6:
                ready.append({
                    "employee_id": emp.id,
                    "readiness_score": round(score, 2),
                    "supporting_evidence": reasons,
                })
        ready.sort(key=lambda x: x["readiness_score"], reverse=True)
        return ready

    # ---------------------------------------------------------------- #
    # Natural-language coaching summary via Claude
    # ---------------------------------------------------------------- #

    def coaching_summary(self, manager_id: str, team_report: Dict[str, Any]) -> str:
        if not self._client:
            high = team_report.get("high_attrition_risk_count", 0)
            avg_e = team_report.get("average_engagement", 0)
            return (
                f"Team of {team_report.get('team_size')} — "
                f"{high} high attrition-risk members, "
                f"avg engagement {avg_e:.0%}. "
                "Focus 1:1s on career development for at-risk employees."
            )

        prompt = (
            f"You are a management coaching AI. A manager has the following team health data:\n"
            f"{str(team_report)}\n\n"
            "Produce a concise (3–4 bullet) coaching summary with the most important actions "
            "the manager should take this week to improve team retention and performance."
        )
        try:
            resp = self._client.messages.create(
                model=config.claude_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error("Coaching summary error: %s", e)
            return "Unable to generate coaching summary — see raw report data."
