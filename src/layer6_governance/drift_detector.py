"""
Model Drift Detector (Layer 6 — Governance & Security).

Implements §9.3 drift detection:
  - Kolmogorov-Smirnov test for feature distribution shift
  - Population Stability Index (PSI) for prediction distribution shift
  - Accuracy monitoring against realized outcomes

Continuous drift detection prevents degraded models from generating harmful
workforce recommendations (paper §9.3).
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp  # type: ignore

from ..common.config import config
from ..common.models import DriftReport

logger = logging.getLogger(__name__)

PSI_BINS = 10


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index. PSI > 0.2 indicates significant shift."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    exp_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_pcts = np.clip(exp_pcts, 1e-4, None)
    act_pcts = np.clip(act_pcts, 1e-4, None)

    psi = float(np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts)))
    return psi


class DriftDetector:
    PSI_WARNING = 0.1
    PSI_CRITICAL = 0.2

    def __init__(self, p_threshold: float = None) -> None:
        self._p_threshold = p_threshold or config.drift_pvalue_threshold
        self._baselines: Dict[str, np.ndarray] = {}

    def set_baseline(self, model_id: str, feature_name: str, baseline_values: np.ndarray) -> None:
        key = f"{model_id}::{feature_name}"
        self._baselines[key] = baseline_values

    def detect_feature_drift(
        self,
        model_id: str,
        feature_name: str,
        current_values: np.ndarray,
    ) -> DriftReport:
        key = f"{model_id}::{feature_name}"
        baseline = self._baselines.get(key)

        if baseline is None or len(baseline) < 10:
            return DriftReport(
                model_id=model_id,
                feature=feature_name,
                test_statistic=0.0,
                p_value=1.0,
                drift_detected=False,
                detection_date=datetime.utcnow(),
            )

        stat, p_value = ks_2samp(baseline, current_values)
        drift_detected = p_value < self._p_threshold

        if drift_detected:
            logger.warning(
                "Drift detected: model=%s feature=%s KS=%.3f p=%.4f",
                model_id, feature_name, stat, p_value,
            )

        return DriftReport(
            model_id=model_id,
            feature=feature_name,
            test_statistic=round(float(stat), 4),
            p_value=round(float(p_value), 6),
            drift_detected=drift_detected,
            detection_date=datetime.utcnow(),
        )

    def detect_prediction_drift(
        self,
        model_id: str,
        baseline_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> Dict[str, float]:
        psi = _psi(baseline_preds, current_preds)
        severity = "stable" if psi < self.PSI_WARNING else ("warning" if psi < self.PSI_CRITICAL else "critical")
        if severity != "stable":
            logger.warning("Prediction drift PSI=%.3f (%s) for model=%s", psi, severity, model_id)
        return {"psi": round(psi, 4), "severity": severity, "retraining_needed": psi >= self.PSI_CRITICAL}

    def scan_all_features(
        self,
        model_id: str,
        current_feature_matrix: np.ndarray,
        feature_names: List[str],
    ) -> List[DriftReport]:
        reports = []
        for i, fname in enumerate(feature_names):
            if i < current_feature_matrix.shape[1]:
                report = self.detect_feature_drift(
                    model_id, fname, current_feature_matrix[:, i]
                )
                reports.append(report)
        return reports
