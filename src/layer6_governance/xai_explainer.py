"""
Explainable AI Service (Layer 6 — Governance & Security).

Implements §9.5 explainability requirements:
  - Feature importance summaries (SHAP proxies)
  - Plain-language decision rationale for managers / employees
  - Counterfactual explanations ("what would need to change?")

Trust is the primary barrier to AI adoption in HR (paper §9.5).
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import anthropic

from ..common.config import config
from ..common.models import AttritionPrediction, PerformancePrediction


class XAIExplainer:

    def __init__(self) -> None:
        if config.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            self._client = None

    # ---------------------------------------------------------------- #
    # Plain-language attrition explanation
    # ---------------------------------------------------------------- #

    def explain_attrition_risk(
        self, prediction: AttritionPrediction, audience: str = "manager"
    ) -> str:
        factors_text = ", ".join(
            f"{f['feature']} ({f['contribution']:+.2f})" for f in prediction.top_factors[:3]
        )

        if not self._client:
            return self._rule_based_attrition_explanation(prediction, audience)

        system = (
            "You are an HR analytics explainer. Translate model predictions into clear, "
            "empathetic, plain-language explanations. Do not mention proprietary model internals. "
            "Audience: " + audience
        )
        prompt = (
            f"An employee has an attrition risk score of {prediction.risk_score:.0%} "
            f"({prediction.risk_level.value} risk). "
            f"The main contributing factors are: {factors_text}. "
            f"Recommended actions: {'; '.join(prediction.recommended_interventions[:2])}.\n\n"
            "Write a 2–3 sentence plain-language explanation suitable for a manager."
        )

        try:
            resp = self._client.messages.create(
                model=config.claude_model,
                max_tokens=200,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            return self._rule_based_attrition_explanation(prediction, audience)

    @staticmethod
    def _rule_based_attrition_explanation(
        prediction: AttritionPrediction, audience: str
    ) -> str:
        top = prediction.top_factors[0]["feature"] if prediction.top_factors else "engagement"
        return (
            f"This employee shows a {prediction.risk_score:.0%} likelihood of leaving "
            f"({prediction.risk_level.value} risk). The primary signal is {top.replace('_', ' ')}. "
            f"Recommended action: {prediction.recommended_interventions[0] if prediction.recommended_interventions else 'Schedule a development conversation'}."
        )

    # ---------------------------------------------------------------- #
    # Counterfactual: "what would reduce the risk?"
    # ---------------------------------------------------------------- #

    def counterfactual_attrition(
        self, prediction: AttritionPrediction, target_risk: float = 0.20
    ) -> List[Dict[str, Any]]:
        counterfactuals = []
        for factor in prediction.top_factors:
            feature = factor["feature"]
            contribution = factor["contribution"]
            if contribution > 0.01:
                counterfactuals.append({
                    "feature": feature,
                    "current_impact": round(contribution, 3),
                    "change_needed": f"Improve {feature.replace('_', ' ')} by addressing root cause",
                    "estimated_risk_reduction": round(contribution * 0.6, 3),
                })
        return counterfactuals

    # ---------------------------------------------------------------- #
    # Performance prediction explanation
    # ---------------------------------------------------------------- #

    def explain_performance_prediction(self, prediction: PerformancePrediction) -> str:
        return (
            f"Predicted performance score: {prediction.predicted_score:.1f}/5 "
            f"(trajectory: {prediction.growth_trajectory}). "
            + (f"Skill gaps to address: {', '.join(prediction.skill_gaps[:2])}." if prediction.skill_gaps else "No critical skill gaps identified.")
        )
