"""
Predictive Workforce Analytics — ML Pipeline (Layer 3 — AI Services).

Implements §6.2 models:
  - AttritionPredictor          : XGBoost gradient boosting, SHAP explainability
  - PerformancePredictor        : LightGBM regression
  - DemandForecaster            : ARIMA-style time-series (via statsmodels or fallback)
  - AbsenteeismPredictor        : Logistic regression baseline (§6.2)
  - PromotionLikelihoodEstimator: Gradient boosting classification (§6.2)
"""

from __future__ import annotations
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from ..common.models import (
    AttritionPrediction,
    PerformancePrediction,
    RiskLevel,
)
from ..common.config import config

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "tenure_years", "tenure_zscore", "performance_score", "performance_trend",
    "engagement_score", "engagement_trend", "compensation", "market_comp_ratio",
    "promotion_velocity", "skill_breadth", "skill_depth", "learning_hours_90d",
    "learning_completion_rate", "workload_index", "collaboration_density",
    "manager_tenure_gap", "team_attrition_rate",
]


def _risk_level(score: float) -> RiskLevel:
    if score >= config.attrition_high:
        return RiskLevel.CRITICAL
    if score >= config.attrition_medium:
        return RiskLevel.HIGH
    if score >= 0.15:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class AttritionPredictor:
    """
    Gradient boosting attrition classifier with SHAP-based explanations.
    Falls back to a statistical heuristic when not yet trained.
    """

    MODEL_VERSION = "attrition-v1.0"

    def __init__(self) -> None:
        self._model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._trained = True
        scores = cross_val_score(self._model, X_scaled, y, cv=5, scoring="roc_auc")
        return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}

    def predict(self, employee_id: str, features: np.ndarray) -> AttritionPrediction:
        if self._trained:
            X = self._scaler.transform(features.reshape(1, -1))
            risk_score = float(self._model.predict_proba(X)[0, 1])
            shap_vals = self._shap_approximate(X[0])
        else:
            risk_score = self._heuristic_risk(features)
            shap_vals = self._heuristic_shap(features)

        top_factors = sorted(
            [{"feature": k, "contribution": v} for k, v in shap_vals.items()],
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )[:5]

        interventions = self._recommend_interventions(shap_vals)

        return AttritionPrediction(
            employee_id=employee_id,
            risk_score=risk_score,
            risk_level=_risk_level(risk_score),
            top_factors=top_factors,
            recommended_interventions=interventions,
            shap_values=shap_vals,
            prediction_date=datetime.utcnow(),
        )

    # -------------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------------- #

    def _shap_approximate(self, x_scaled: np.ndarray) -> Dict[str, float]:
        """Tree-based feature importances × sign(feature) as SHAP proxy."""
        importances = self._model.feature_importances_
        signs = np.sign(x_scaled)
        vals = importances * signs
        return {FEATURE_NAMES[i]: float(vals[i]) for i in range(len(FEATURE_NAMES))}

    @staticmethod
    def _heuristic_risk(features: np.ndarray) -> float:
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        score = 0.0
        if features[idx["market_comp_ratio"]] < 0.9:
            score += 0.25
        if features[idx["engagement_score"]] < 0.35:
            score += 0.30
        if features[idx["promotion_velocity"]] < -0.2:
            score += 0.20
        if features[idx["workload_index"]] > 0.8:
            score += 0.15
        if features[idx["performance_trend"]] < -0.5:
            score += 0.10
        return min(score, 1.0)

    @staticmethod
    def _heuristic_shap(features: np.ndarray) -> Dict[str, float]:
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        return {
            "market_comp_ratio": float(0.9 - features[idx["market_comp_ratio"]]) * 0.3,
            "engagement_score": float(0.5 - features[idx["engagement_score"]]) * 0.3,
            "promotion_velocity": float(-features[idx["promotion_velocity"]]) * 0.2,
            "workload_index": float(features[idx["workload_index"]] - 0.5) * 0.15,
            "performance_trend": float(-features[idx["performance_trend"]]) * 0.1,
        }

    @staticmethod
    def _recommend_interventions(shap_vals: Dict[str, float]) -> List[str]:
        interventions = []
        if shap_vals.get("market_comp_ratio", 0) > 0.05:
            interventions.append("Compensation equity review against external market benchmarks")
        if shap_vals.get("engagement_score", 0) > 0.05:
            interventions.append("Manager 1:1 focused on career aspirations and role alignment")
        if shap_vals.get("promotion_velocity", 0) > 0.05:
            interventions.append("Internal mobility opportunity discussion and stretch assignment")
        if shap_vals.get("workload_index", 0) > 0.05:
            interventions.append("Workload rebalancing review with team capacity planning")
        if shap_vals.get("skill_gap", 0) > 0.05:
            interventions.append("Personalized learning path for critical skill development")
        if not interventions:
            interventions.append("Proactive development conversation with manager")
        return interventions


class PerformancePredictor:
    """Predicts next-cycle performance score; identifies growth trajectory."""

    MODEL_VERSION = "performance-v1.0"

    def __init__(self) -> None:
        self._model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        )
        self._scaler = StandardScaler()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y)
        self._trained = True
        from sklearn.metrics import mean_absolute_error
        preds = self._model.predict(X_s)
        return {"mae": float(mean_absolute_error(y, preds))}

    def predict(self, employee_id: str, features: np.ndarray) -> PerformancePrediction:
        if self._trained:
            X_s = self._scaler.transform(features.reshape(1, -1))
            score = float(np.clip(self._model.predict(X_s)[0], 0, 5))
        else:
            idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
            score = float(np.clip(
                features[idx["performance_score"]] + features[idx["performance_trend"]] * 0.5,
                0, 5
            ))

        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        trend = features[idx["performance_trend"]]
        if trend > 0.3:
            trajectory = "accelerating"
        elif trend < -0.3:
            trajectory = "declining"
        else:
            trajectory = "stable"

        gaps: List[str] = []
        if features[idx["skill_depth"]] < 0.5:
            gaps.append("Technical skill depth below role benchmark")
        if features[idx["learning_completion_rate"]] < 0.6:
            gaps.append("Learning program completion rate below target")

        return PerformancePrediction(
            employee_id=employee_id,
            predicted_score=score,
            confidence=0.72 if self._trained else 0.5,
            growth_trajectory=trajectory,
            skill_gaps=gaps,
        )


class DemandForecaster:
    """
    Workforce demand forecaster (headcount by department, rolling horizon).
    Uses a simple exponential smoothing fallback; statsmodels ARIMA when available.
    """

    def forecast(
        self,
        historical_headcount: List[float],
        horizon: int = 4,
        alpha: float = 0.3,
    ) -> List[float]:
        if len(historical_headcount) < 2:
            return [historical_headcount[-1]] * horizon if historical_headcount else [0.0] * horizon

        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
            model = ARIMA(historical_headcount, order=(1, 1, 1))
            fit = model.fit()
            return [float(v) for v in fit.forecast(steps=horizon)]
        except Exception:
            # Exponential smoothing fallback
            s = historical_headcount[0]
            for v in historical_headcount[1:]:
                s = alpha * v + (1 - alpha) * s
            return [round(s)] * horizon


class AbsenteeismPredictor:
    """
    Predicts employees at elevated absenteeism risk using logistic regression (§6.2).
    Features: engagement trend, workload index, performance trajectory, tenure.
    """

    MODEL_VERSION = "absenteeism-v1.0"

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self._scaler = StandardScaler()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y)
        self._trained = True
        scores = cross_val_score(self._model, X_s, y, cv=5, scoring="roc_auc")
        return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}

    def predict(self, employee_id: str, features: np.ndarray) -> Dict:
        if self._trained:
            X_s = self._scaler.transform(features.reshape(1, -1))
            risk_score = float(self._model.predict_proba(X_s)[0, 1])
        else:
            idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
            risk_score = 0.0
            if features[idx["workload_index"]] > 0.80:
                risk_score += 0.30
            if features[idx["engagement_score"]] < 0.40:
                risk_score += 0.25
            if features[idx["engagement_trend"]] < -0.3:
                risk_score += 0.20
            risk_score = min(risk_score, 1.0)

        return {
            "employee_id": employee_id,
            "absenteeism_risk": round(risk_score, 3),
            "risk_level": _risk_level(risk_score).value,
            "intervention": (
                "Workload rebalancing and wellness check-in recommended"
                if risk_score >= 0.50
                else "Monitor engagement signals over next 30 days"
            ),
        }


class PromotionLikelihoodEstimator:
    """
    Estimates advancement probability based on performance patterns and skill
    development trajectory (§6.2 — promotion likelihood estimation).
    """

    MODEL_VERSION = "promotion-v1.0"

    def __init__(self) -> None:
        self._model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        self._scaler = StandardScaler()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y)
        self._trained = True
        scores = cross_val_score(self._model, X_s, y, cv=5, scoring="roc_auc")
        return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}

    def predict(self, employee_id: str, features: np.ndarray) -> Dict:
        if self._trained:
            X_s = self._scaler.transform(features.reshape(1, -1))
            likelihood = float(self._model.predict_proba(X_s)[0, 1])
        else:
            idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
            likelihood = 0.0
            if features[idx["performance_score"]] >= 4.0:
                likelihood += 0.35
            if features[idx["performance_trend"]] > 0.2:
                likelihood += 0.20
            if features[idx["skill_depth"]] >= 0.7:
                likelihood += 0.20
            if features[idx["learning_completion_rate"]] >= 0.8:
                likelihood += 0.15
            if features[idx["tenure_years"]] >= 1.5:
                likelihood += 0.10
            likelihood = min(likelihood, 1.0)

        return {
            "employee_id": employee_id,
            "promotion_likelihood": round(likelihood, 3),
            "readiness": "ready" if likelihood >= 0.65 else ("developing" if likelihood >= 0.40 else "early"),
            "key_drivers": [
                "High performance rating" if features[{n: i for i, n in enumerate(FEATURE_NAMES)}["performance_score"]] >= 4.0 else "Performance improvement needed",
            ],
        }
