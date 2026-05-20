"""
Algorithmic Bias Monitor (Layer 6 — Governance & Security).

Implements §12.2 fairness metrics:
  - Demographic Parity  (equal selection rates across groups)
  - Equal Opportunity   (equal true positive rates)
  - Disparate Impact    (80% rule — DI ≥ 0.8)

Continuous monitoring ensures models maintain acceptable fairness thresholds
post-deployment.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common.config import config
from ..common.models import BiasReport

logger = logging.getLogger(__name__)

PROTECTED_GROUPS = {"gender", "race", "age_group", "disability"}


class BiasMonitor:
    def __init__(self, di_threshold: float = None) -> None:
        self._di_threshold = di_threshold or config.bias_threshold_di
        self._tracked_groups: Dict[str, int] = {}  # group → flag_count

    def evaluate(
        self,
        model_id: str,
        predictions: List[float],
        actuals: List[int],
        group_labels: List[str],
    ) -> BiasReport:
        groups = list(set(group_labels))
        demographic_parity: Dict[str, float] = {}
        equal_opportunity: Dict[str, float] = {}
        disparate_impact: Dict[str, float] = {}

        reference_rate = sum(1 for p in predictions if p >= 0.5) / len(predictions) if predictions else 0.0

        for group in groups:
            indices = [i for i, g in enumerate(group_labels) if g == group]
            group_preds = [predictions[i] for i in indices]
            group_acts = [actuals[i] for i in indices]

            # Demographic Parity: selection rate per group
            sel_rate = sum(1 for p in group_preds if p >= 0.5) / len(group_preds) if group_preds else 0.0
            demographic_parity[group] = round(sel_rate, 4)

            # Equal Opportunity: TPR per group
            pos_indices = [i for i, idx in enumerate(indices) if group_acts[i] == 1]
            if pos_indices:
                tpr = sum(1 for i in pos_indices if group_preds[i] >= 0.5) / len(pos_indices)
            else:
                tpr = 0.0
            equal_opportunity[group] = round(tpr, 4)

            # Disparate Impact: group_rate / reference_rate
            di = sel_rate / reference_rate if reference_rate > 0 else 1.0
            disparate_impact[group] = round(di, 4)

        flagged = any(di < self._di_threshold for di in disparate_impact.values())
        recommendations = []
        if flagged:
            low_di_groups = [g for g, di in disparate_impact.items() if di < self._di_threshold]
            recommendations.append(
                f"Disparate impact below threshold for groups: {low_di_groups}. "
                "Apply re-weighting or fairness constraint in training objective."
            )
            recommendations.append("Audit training data for historical representation gaps.")
            for g in low_di_groups:
                self._tracked_groups[g] = self._tracked_groups.get(g, 0) + 1

        return BiasReport(
            model_id=model_id,
            evaluation_date=datetime.utcnow(),
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            disparate_impact=disparate_impact,
            flagged=flagged,
            recommendations=recommendations,
        )

    def flag_for_review(self, demographic_group: str) -> bool:
        """Returns True when a demographic group is under-represented in pipeline."""
        return self._tracked_groups.get(demographic_group, 0) > 2

    def chronic_bias_groups(self) -> List[str]:
        return [g for g, cnt in self._tracked_groups.items() if cnt >= 3]
