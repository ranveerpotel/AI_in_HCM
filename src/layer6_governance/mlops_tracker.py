"""
MLOps Model Lifecycle Tracker (Layer 6 — Governance & Security).

Implements §9.2 MLOps lifecycle:
  Data Ingestion → Feature Engineering → Model Training →
  Validation & Explainability → Deployment → Monitoring →
  Retraining (triggered by DriftDetector)

Uses MLflow for experiment tracking, model versioning, and
performance monitoring. Falls back gracefully if MLflow is unavailable.
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelRecord:
    """Lightweight model registration when MLflow is unavailable."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        version: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
    ) -> None:
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.metrics = metrics
        self.params = params
        self.registered_at = datetime.utcnow()
        self.status = "staged"  # "staged" | "production" | "archived"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "metrics": self.metrics,
            "params": self.params,
            "registered_at": self.registered_at.isoformat(),
            "status": self.status,
        }


class MLOpsTracker:
    """
    Tracks model training runs, validates performance thresholds,
    and manages promotion from staging to production.
    Uses MLflow when available; falls back to in-memory registry.
    """

    AUC_THRESHOLD = 0.70       # minimum AUC for attrition model promotion
    MAE_THRESHOLD = 0.80       # maximum MAE for performance predictor

    def __init__(self, tracking_uri: Optional[str] = None) -> None:
        self._mlflow_available = False
        self._registry: Dict[str, ModelRecord] = {}
        self._run_history: list = []

        if tracking_uri:
            try:
                import mlflow  # type: ignore
                mlflow.set_tracking_uri(tracking_uri)
                self._mlflow = mlflow
                self._mlflow_available = True
                logger.info("MLflow tracking enabled: %s", tracking_uri)
            except ImportError:
                logger.info("MLflow not installed — using in-memory registry")

    # ---------------------------------------------------------------- #
    # Training run lifecycle
    # ---------------------------------------------------------------- #

    def start_run(self, model_id: str, params: Dict[str, Any]) -> str:
        run_id = f"{model_id}-{int(time.time())}"
        if self._mlflow_available:
            with self._mlflow.start_run(run_name=run_id) as run:
                self._mlflow.log_params(params)
                return run.info.run_id
        self._run_history.append({"run_id": run_id, "model_id": model_id, "params": params})
        return run_id

    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        if self._mlflow_available:
            with self._mlflow.start_run(run_id=run_id):
                self._mlflow.log_metrics(metrics)
        for run in self._run_history:
            if run["run_id"] == run_id:
                run.update({"metrics": metrics})

    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
    ) -> ModelRecord:
        record = ModelRecord(model_id, model_type, version, metrics, params)

        if self._mlflow_available:
            try:
                self._mlflow.register_model(f"runs:/{model_id}/model", model_type)
            except Exception as e:
                logger.debug("MLflow register_model skipped: %s", e)

        self._registry[model_id] = record
        logger.info("Model registered: %s v%s metrics=%s", model_id, version, metrics)
        return record

    # ---------------------------------------------------------------- #
    # Model promotion gate (§9.2 Validation & Explainability)
    # ---------------------------------------------------------------- #

    def validate_and_promote(self, model_id: str) -> Dict[str, Any]:
        record = self._registry.get(model_id)
        if not record:
            return {"status": "not_found", "model_id": model_id}

        passed = True
        reasons = []

        if "auc_mean" in record.metrics and record.metrics["auc_mean"] < self.AUC_THRESHOLD:
            passed = False
            reasons.append(f"AUC {record.metrics['auc_mean']:.3f} below threshold {self.AUC_THRESHOLD}")

        if "mae" in record.metrics and record.metrics["mae"] > self.MAE_THRESHOLD:
            passed = False
            reasons.append(f"MAE {record.metrics['mae']:.3f} above threshold {self.MAE_THRESHOLD}")

        if passed:
            record.status = "production"
            logger.info("Model %s promoted to production", model_id)
        else:
            record.status = "staged"
            logger.warning("Model %s failed validation: %s", model_id, reasons)

        return {
            "model_id": model_id,
            "promoted": passed,
            "status": record.status,
            "validation_failures": reasons,
            "metrics": record.metrics,
        }

    def get_production_models(self) -> Dict[str, ModelRecord]:
        return {mid: r for mid, r in self._registry.items() if r.status == "production"}

    def trigger_retraining(self, model_id: str, drift_report: Dict) -> Dict[str, Any]:
        logger.warning(
            "Retraining triggered for %s. PSI=%.3f drift_feature=%s",
            model_id,
            drift_report.get("psi", 0),
            drift_report.get("feature", "unknown"),
        )
        if model_id in self._registry:
            self._registry[model_id].status = "archived"
        return {
            "model_id": model_id,
            "action": "retrain_queued",
            "trigger": drift_report,
            "timestamp": datetime.utcnow().isoformat(),
        }
