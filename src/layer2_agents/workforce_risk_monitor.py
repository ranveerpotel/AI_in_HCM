"""
Workforce Risk Monitor Agent (Layer 2 — Intelligent Agents).

Implements §3.1 and §11.3:
  - Continuously analyses engagement signals for early attrition warning
  - Detects workload imbalance and engagement decline waves
  - Flags potential compliance or succession risk
  - Generates proactive intervention recommendations at organizational level

This agent operates continuously (like a background sensor) rather than
responding to individual employee queries — it monitors population-level
workforce health and surfaces emerging risks before they manifest.
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..common.models import AgentType, AuditRecord, Employee, RiskLevel, WorkforceSignal
from ..layer3_ai_services.predictive_models import AttritionPredictor
from ..layer3_ai_services.anomaly_detector import WorkforceAnomalyDetector
from ..layer4_data_intelligence.feature_store import FeatureStore
from ..layer4_data_intelligence.streaming_analytics import WorkforceStreamProcessor
from ..layer6_governance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class WorkforceRiskAlert:
    def __init__(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        scope: str,          # "individual" | "team" | "department" | "org"
        scope_id: str,
        description: str,
        affected_employees: List[str],
        recommended_actions: List[str],
        timestamp: datetime,
    ) -> None:
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.severity = severity
        self.scope = scope
        self.scope_id = scope_id
        self.description = description
        self.affected_employees = affected_employees
        self.recommended_actions = recommended_actions
        self.timestamp = timestamp
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "description": self.description,
            "affected_employees": self.affected_employees,
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
        }


class WorkforceRiskMonitor:
    """
    Continuously monitors workforce signals and surfaces population-level
    risks. Integrates attrition predictions, engagement anomalies, and
    workload imbalance detection into a unified alert stream.
    """

    AGENT_VERSION = "wrm-v1.0"

    def __init__(
        self,
        feature_store: FeatureStore,
        attrition_predictor: AttritionPredictor,
        anomaly_detector: WorkforceAnomalyDetector,
        stream_processor: WorkforceStreamProcessor,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._fs = feature_store
        self._attrition = attrition_predictor
        self._anomaly = anomaly_detector
        self._stream = stream_processor
        self._audit = audit_logger
        self._active_alerts: List[WorkforceRiskAlert] = []

    # ---------------------------------------------------------------- #
    # Full workforce scan (scheduled — runs on cadence, e.g. nightly)
    # ---------------------------------------------------------------- #

    def run_full_scan(self, employees: List[Employee]) -> List[WorkforceRiskAlert]:
        new_alerts: List[WorkforceRiskAlert] = []

        # 1. Attrition risk cluster detection
        attrition_scores: Dict[str, float] = {}
        for emp in employees:
            fv = self._fs.get(emp.id)
            if fv:
                pred = self._attrition.predict(emp.id, fv.to_array())
                attrition_scores[emp.id] = pred.risk_score

        cluster_alerts = self._anomaly.detect_attrition_cluster(employees, attrition_scores)
        for ca in cluster_alerts:
            alert = WorkforceRiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="attrition_cluster",
                severity=ca["severity"],
                scope="department",
                scope_id=ca["department"],
                description=(
                    f"Elevated attrition risk in {ca['department']}: "
                    f"mean risk {ca['mean_risk_score']:.0%} across {ca['employee_count']} employees, "
                    f"{ca['high_risk_count']} at critical level."
                ),
                affected_employees=[e.id for e in employees if e.department == ca["department"]],
                recommended_actions=[
                    "Initiate skip-level listening sessions with department leadership",
                    "Review compensation equity across department",
                    "Accelerate internal mobility opportunities for high-performers",
                ],
                timestamp=datetime.utcnow(),
            )
            new_alerts.append(alert)

        # 2. Engagement decline wave
        engagement_alerts = self._anomaly.detect_department_engagement_anomaly(employees)
        for ea in engagement_alerts:
            alert = WorkforceRiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="engagement_decline",
                severity=ea["severity"],
                scope="department",
                scope_id=ea["department"],
                description=(
                    f"Engagement decline in {ea['department']}: "
                    f"dept mean {ea['dept_mean']:.2f} vs org mean {ea['org_mean']:.2f} "
                    f"(z={ea['z_score']:.1f})."
                ),
                affected_employees=[e.id for e in employees if e.department == ea["department"]],
                recommended_actions=[
                    "Manager coaching: focused 1:1s on team conditions",
                    "HR business partner outreach for listening sessions",
                    "Review recent organizational changes impacting this team",
                ],
                timestamp=datetime.utcnow(),
            )
            new_alerts.append(alert)

        # 3. Individual critical-risk flags
        critical_employees = [
            emp for emp in employees
            if attrition_scores.get(emp.id, 0) >= 0.75
        ]
        for emp in critical_employees:
            fv = self._fs.get(emp.id)
            pred = self._attrition.predict(emp.id, fv.to_array()) if fv else None
            alert = WorkforceRiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="individual_critical_risk",
                severity="critical",
                scope="individual",
                scope_id=emp.id,
                description=(
                    f"Employee {emp.id} has critical attrition risk "
                    f"({attrition_scores[emp.id]:.0%}). Immediate manager attention needed."
                ),
                affected_employees=[emp.id],
                recommended_actions=pred.recommended_interventions if pred else [
                    "Manager 1:1 focused on career aspirations and role alignment"
                ],
                timestamp=datetime.utcnow(),
            )
            new_alerts.append(alert)

        # 4. Workload imbalance
        overloaded = self._anomaly.detect_workload_imbalance(employees)
        if overloaded:
            depts = list({o["department"] for o in overloaded})
            alert = WorkforceRiskAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="workload_imbalance",
                severity="high" if len(overloaded) > 3 else "medium",
                scope="org",
                scope_id="workforce",
                description=(
                    f"{len(overloaded)} employees at unsustainable workload levels "
                    f"across departments: {depts}."
                ),
                affected_employees=[o["employee_id"] for o in overloaded],
                recommended_actions=[
                    "Headcount capacity review with finance",
                    "Task rebalancing within affected teams",
                    "Contractor/contingent workforce assessment",
                ],
                timestamp=datetime.utcnow(),
            )
            new_alerts.append(alert)

        self._active_alerts.extend(new_alerts)

        if self._audit and new_alerts:
            self._audit.log(AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_type=AgentType.WORKFORCE_RISK,
                action_type="full_scan",
                employee_id="system",
                input_summary={"employees_scanned": len(employees)},
                output_summary={"alerts_generated": len(new_alerts)},
                model_version=self.AGENT_VERSION,
            ))

        return new_alerts

    # ---------------------------------------------------------------- #
    # Real-time signal ingestion
    # ---------------------------------------------------------------- #

    def ingest_signal(self, signal: WorkforceSignal) -> Optional[Dict]:
        return self._stream.ingest(signal)

    # ---------------------------------------------------------------- #
    # Alert management
    # ---------------------------------------------------------------- #

    def get_active_alerts(self, severity: Optional[str] = None) -> List[WorkforceRiskAlert]:
        alerts = [a for a in self._active_alerts if not a.resolved]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def resolve_alert(self, alert_id: str) -> bool:
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

    def alert_summary(self) -> Dict[str, Any]:
        active = self.get_active_alerts()
        return {
            "total_active": len(active),
            "critical": sum(1 for a in active if a.severity == "critical"),
            "high": sum(1 for a in active if a.severity == "high"),
            "medium": sum(1 for a in active if a.severity == "medium"),
            "by_type": {
                t: sum(1 for a in active if a.alert_type == t)
                for t in {"attrition_cluster", "engagement_decline", "individual_critical_risk", "workload_imbalance"}
            },
        }
