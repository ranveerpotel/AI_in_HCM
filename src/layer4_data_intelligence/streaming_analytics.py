"""
Real-Time Workforce Streaming Analytics (Layer 4 — Data Intelligence).

Simulates an Apache Kafka / Flink style event-streaming pipeline (§7.3).
Signals arrive from engagement surveys, collaboration tools, and productivity
platforms.  The processor maintains a sliding-window aggregator and emits
alerts when anomalous patterns are detected.
"""

from __future__ import annotations
import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from ..common.models import WorkforceSignal

logger = logging.getLogger(__name__)


class SignalWindow:
    """Fixed-size rolling window of signal values for one (employee, type) pair."""

    def __init__(self, maxlen: int = 30) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        self._buf.append(value)

    @property
    def mean(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self._buf)) if len(self._buf) > 1 else 0.0

    def is_anomalous(self, new_value: float, z_threshold: float = 2.5) -> bool:
        if len(self._buf) < 5:
            return False
        return abs(new_value - self.mean) > z_threshold * (self.std + 1e-8)


AlertHandler = Callable[[Dict], None]


class WorkforceStreamProcessor:
    """
    Ingests WorkforceSignal events, maintains per-employee rolling windows,
    and fires alerts when:
      - engagement drops suddenly
      - workload spikes above sustainable threshold
      - collaboration density declines (early attrition signal)
    """

    ALERT_TYPES = {
        "engagement_drop": "Sudden engagement decline detected — potential early attrition signal",
        "workload_spike": "Workload index exceeds sustainable threshold",
        "collab_decline": "Collaboration density dropping — possible disengagement",
    }

    def __init__(self, alert_handler: Optional[AlertHandler] = None, window_size: int = 30) -> None:
        self._windows: Dict[str, Dict[str, SignalWindow]] = {}
        self._alert_handler = alert_handler or self._default_alert_handler
        self._window_size = window_size
        self._processed_count = 0

    def ingest(self, signal: WorkforceSignal) -> Optional[Dict]:
        emp_windows = self._windows.setdefault(signal.employee_id, {})
        window = emp_windows.setdefault(signal.signal_type, SignalWindow(self._window_size))

        alert = None
        if window.is_anomalous(signal.value):
            alert = self._build_alert(signal, window)
            self._alert_handler(alert)

        window.push(signal.value)
        self._processed_count += 1
        return alert

    def ingest_batch(self, signals: List[WorkforceSignal]) -> List[Dict]:
        alerts = []
        for sig in signals:
            result = self.ingest(sig)
            if result:
                alerts.append(result)
        return alerts

    async def ingest_stream(self, queue: asyncio.Queue) -> None:
        """Consume from an asyncio.Queue indefinitely."""
        while True:
            signal: WorkforceSignal = await queue.get()
            self.ingest(signal)
            queue.task_done()

    def get_employee_summary(self, employee_id: str) -> Dict[str, float]:
        emp_windows = self._windows.get(employee_id, {})
        return {sig_type: w.mean for sig_type, w in emp_windows.items()}

    def get_at_risk_employees(self, engagement_threshold: float = 0.4) -> List[str]:
        at_risk = []
        for eid, windows in self._windows.items():
            eng_window = windows.get("engagement_survey")
            if eng_window and eng_window.mean < engagement_threshold:
                at_risk.append(eid)
        return at_risk

    @staticmethod
    def _build_alert(signal: WorkforceSignal, window: SignalWindow) -> Dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "employee_id": signal.employee_id,
            "signal_type": signal.signal_type,
            "current_value": signal.value,
            "window_mean": window.mean,
            "window_std": window.std,
            "z_score": (signal.value - window.mean) / (window.std + 1e-8),
            "source": signal.source,
        }

    @staticmethod
    def _default_alert_handler(alert: Dict) -> None:
        logger.warning("WORKFORCE ALERT: %s", json.dumps(alert))
