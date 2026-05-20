"""
Workforce Feature Store (Layer 4 — Data Intelligence).

Pre-computes and caches ML-ready feature vectors for employees so that all
agents and predictive models consume identical, consistently updated features.
Features align with §7.2 of the paper (tenure progression, performance trends,
collaboration patterns, skill trajectories, workload signals).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..common.models import Employee, LearningActivity, WorkforceSignal


class EmployeeFeatureVector:
    __slots__ = (
        "employee_id",
        "tenure_years",
        "tenure_zscore",
        "performance_score",
        "performance_trend",      # slope over last 4 quarters
        "engagement_score",
        "engagement_trend",
        "compensation",
        "market_comp_ratio",
        "promotion_velocity",
        "skill_breadth",          # number of distinct skill categories
        "skill_depth",            # avg proficiency
        "learning_hours_90d",
        "learning_completion_rate",
        "workload_index",
        "collaboration_density",  # edges in org network / team size
        "manager_tenure_gap",     # |employee_tenure - manager_tenure|
        "team_attrition_rate",    # fraction of team lost in last 12m
        "computed_at",
    )

    def __init__(self, **kwargs: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_array(self) -> np.ndarray:
        numeric_slots = [s for s in self.__slots__ if s not in ("employee_id", "computed_at")]
        return np.array([getattr(self, s) or 0.0 for s in numeric_slots], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class FeatureStore:
    def __init__(self) -> None:
        self._store: Dict[str, EmployeeFeatureVector] = {}
        self._history: Dict[str, List[EmployeeFeatureVector]] = {}  # rolling history per employee

    def compute_and_store(
        self,
        employee: Employee,
        learning_activities: List[LearningActivity],
        recent_signals: List[WorkforceSignal],
        team_attrition_rate: float = 0.0,
        collaboration_density: float = 0.5,
        manager_tenure_years: Optional[float] = None,
        performance_history: Optional[List[float]] = None,
        engagement_history: Optional[List[float]] = None,
    ) -> EmployeeFeatureVector:
        # Learning features
        cutoff = datetime.utcnow() - timedelta(days=90)
        recent_learning = [
            la for la in learning_activities
            if la.employee_id == employee.id
            and (la.completion_date or datetime.min) >= cutoff
        ]
        learning_hours_90d = len(recent_learning) * 4.0
        completed = [la for la in learning_activities if la.employee_id == employee.id and la.progress >= 1.0]
        total = [la for la in learning_activities if la.employee_id == employee.id]
        completion_rate = len(completed) / len(total) if total else 0.0

        # Skill features
        skill_cats = {sa.skill_id[:3] for sa in employee.skills}  # prefix as proxy for category
        skill_breadth = len(skill_cats)
        skill_depth = np.mean([sa.proficiency for sa in employee.skills]) if employee.skills else 0.0

        # Trend features (slope from history)
        perf_trend = self._slope(performance_history) if performance_history else 0.0
        eng_trend = self._slope(engagement_history) if engagement_history else 0.0

        # Workload from signals
        workload_signals = [s.value for s in recent_signals
                            if s.employee_id == employee.id and s.signal_type == "workload_spike"]
        workload_index = float(np.mean(workload_signals)) if workload_signals else employee.workload_index

        fv = EmployeeFeatureVector(
            employee_id=employee.id,
            tenure_years=employee.tenure_years,
            tenure_zscore=0.0,           # populated by normalize() below
            performance_score=employee.performance_score,
            performance_trend=perf_trend,
            engagement_score=employee.engagement_score,
            engagement_trend=eng_trend,
            compensation=employee.compensation,
            market_comp_ratio=employee.market_comp_ratio,
            promotion_velocity=employee.promotion_velocity,
            skill_breadth=float(skill_breadth),
            skill_depth=float(skill_depth),
            learning_hours_90d=learning_hours_90d,
            learning_completion_rate=completion_rate,
            workload_index=workload_index,
            collaboration_density=collaboration_density,
            manager_tenure_gap=abs(employee.tenure_years - (manager_tenure_years or employee.tenure_years)),
            team_attrition_rate=team_attrition_rate,
            computed_at=datetime.utcnow(),
        )
        self._store[employee.id] = fv
        self._history.setdefault(employee.id, []).append(fv)
        return fv

    def get(self, employee_id: str) -> Optional[EmployeeFeatureVector]:
        return self._store.get(employee_id)

    def get_all_arrays(self) -> tuple[List[str], np.ndarray]:
        ids = list(self._store.keys())
        arrays = np.stack([self._store[i].to_array() for i in ids]) if ids else np.empty((0, 17))
        return ids, arrays

    def normalize_tenure(self) -> None:
        """Z-score tenure across the stored population."""
        if not self._store:
            return
        tenures = np.array([fv.tenure_years for fv in self._store.values()])
        mu, sigma = tenures.mean(), tenures.std() + 1e-8
        for fv in self._store.values():
            fv.tenure_zscore = float((fv.tenure_years - mu) / sigma)

    @staticmethod
    def _slope(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values, 1)[0])
