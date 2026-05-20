"""
Workforce Anomaly Detector (Layer 3 — AI Services).

Detects unusual patterns in workforce signals (§7.3):
  - Sudden attrition cluster in a department
  - Skill coverage gaps in critical teams
  - Engagement decline waves across organizational units
  - Workload distribution imbalance
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from scipy.stats import zscore  # type: ignore

from ..common.models import Employee, WorkforceSignal


class WorkforceAnomalyDetector:

    def __init__(self, z_threshold: float = 2.5) -> None:
        self._z_threshold = z_threshold

    def detect_department_engagement_anomaly(
        self, employees: List[Employee]
    ) -> List[Dict]:
        dept_scores: Dict[str, List[float]] = {}
        for emp in employees:
            dept_scores.setdefault(emp.department, []).append(emp.engagement_score)

        dept_means = {dept: np.mean(scores) for dept, scores in dept_scores.items()}
        overall_mean = np.mean(list(dept_means.values()))
        overall_std = np.std(list(dept_means.values())) + 1e-8

        alerts = []
        for dept, mean in dept_means.items():
            z = (mean - overall_mean) / overall_std
            if z < -self._z_threshold:
                alerts.append({
                    "type": "engagement_decline_cluster",
                    "department": dept,
                    "dept_mean": round(mean, 3),
                    "org_mean": round(overall_mean, 3),
                    "z_score": round(z, 2),
                    "detected_at": datetime.utcnow().isoformat(),
                    "severity": "high" if z < -3.0 else "medium",
                })
        return alerts

    def detect_workload_imbalance(
        self, employees: List[Employee]
    ) -> List[Dict]:
        overloaded = [
            {"employee_id": emp.id, "workload_index": emp.workload_index, "department": emp.department}
            for emp in employees
            if emp.workload_index > 0.85
        ]
        return overloaded

    def detect_skill_coverage_gap(
        self,
        team_employees: List[Employee],
        critical_skills: List[str],
        coverage_threshold: float = 0.5,
    ) -> List[Dict]:
        alerts = []
        for skill_id in critical_skills:
            covered = [
                emp for emp in team_employees
                if any(sa.skill_id == skill_id for sa in emp.skills)
            ]
            coverage = len(covered) / len(team_employees) if team_employees else 0.0
            if coverage < coverage_threshold:
                alerts.append({
                    "type": "skill_coverage_gap",
                    "skill_id": skill_id,
                    "coverage_rate": round(coverage, 3),
                    "threshold": coverage_threshold,
                    "detected_at": datetime.utcnow().isoformat(),
                    "severity": "critical" if coverage < 0.25 else "medium",
                })
        return alerts

    def detect_attrition_cluster(
        self,
        employees: List[Employee],
        attrition_predictions: Dict[str, float],
        cluster_threshold: float = 0.4,
    ) -> List[Dict]:
        dept_risks: Dict[str, List[float]] = {}
        for emp in employees:
            risk = attrition_predictions.get(emp.id, 0.0)
            dept_risks.setdefault(emp.department, []).append(risk)

        alerts = []
        for dept, risks in dept_risks.items():
            mean_risk = np.mean(risks)
            if mean_risk >= cluster_threshold:
                alerts.append({
                    "type": "attrition_cluster",
                    "department": dept,
                    "mean_risk_score": round(mean_risk, 3),
                    "employee_count": len(risks),
                    "high_risk_count": sum(1 for r in risks if r >= 0.65),
                    "detected_at": datetime.utcnow().isoformat(),
                    "severity": "critical" if mean_risk >= 0.6 else "high",
                })
        return alerts
