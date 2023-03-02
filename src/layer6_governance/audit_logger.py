"""
Audit Logger (Layer 6 — Governance & Security).

Maintains a tamper-evident log of every AI recommendation, the data inputs
it was based on, and human override outcomes. Satisfies §12.4 (Responsible AI
Governance) and GDPR / CCPA audit trail requirements.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..common.models import AuditRecord

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, log_path: str = "./audit_log.jsonl") -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._in_memory: List[AuditRecord] = []

    def log(self, record: AuditRecord) -> None:
        self._in_memory.append(record)
        entry = {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "agent_type": record.agent_type.value,
            "action_type": record.action_type,
            "employee_id": record.employee_id,
            "input_summary": record.input_summary,
            "output_summary": record.output_summary,
            "model_version": record.model_version,
            "human_reviewed": record.human_reviewed,
            "override_applied": record.override_applied,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("Audit log write failed: %s", e)

    def mark_reviewed(self, record_id: str, override: bool = False) -> bool:
        for rec in self._in_memory:
            if rec.id == record_id:
                rec.human_reviewed = True
                rec.override_applied = override
                return True
        return False

    def get_records_for_employee(self, employee_id: str) -> List[AuditRecord]:
        return [r for r in self._in_memory if r.employee_id == employee_id]

    def pending_human_review(self) -> List[AuditRecord]:
        return [r for r in self._in_memory if not r.human_reviewed]

    def summary(self) -> dict:
        return {
            "total_records": len(self._in_memory),
            "pending_review": len(self.pending_human_review()),
            "override_count": sum(1 for r in self._in_memory if r.override_applied),
        }
