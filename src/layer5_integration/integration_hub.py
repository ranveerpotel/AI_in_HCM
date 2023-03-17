"""
Enterprise Integration Hub (Layer 5 — Integration).

Implements §4.2.5 and §15.4:
  - Event-driven integration with finance, identity, payroll, collaboration tools
  - API-first adapters for external system I/O
  - Event routing and downstream process triggering

In production these adapters communicate via REST/gRPC with actual enterprise
systems (Workday, SAP, Slack, etc.). Here they are simulated stubs that can be
swapped for real implementations without changing the agent layer.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

EventHandler = Callable[[Dict[str, Any]], None]


class IntegrationEvent:
    def __init__(self, event_type: str, payload: Dict[str, Any], source: str) -> None:
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.timestamp = datetime.utcnow().isoformat()


class IntegrationHub:
    """
    Central event bus that connects HCM agents with enterprise systems.
    Subscribes handlers to event types and dispatches inbound events.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_log: List[Dict] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: IntegrationEvent) -> None:
        self._event_log.append({"type": event.event_type, "source": event.source, "ts": event.timestamp})
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event.payload)
            except Exception as e:
                logger.error("Handler error for %s: %s", event.event_type, e)

    def event_count(self) -> int:
        return len(self._event_log)


# -------------------------------------------------------------------- #
# System adapters (stubs — replace with real API clients)
# -------------------------------------------------------------------- #

class PayrollAdapter:
    """Reads compensation data from payroll system."""

    def get_compensation(self, employee_id: str) -> Dict[str, float]:
        logger.info("PayrollAdapter: fetching comp for %s (stub)", employee_id)
        return {"base_salary": 95000.0, "bonus": 10000.0, "equity_value": 25000.0}

    def update_comp_band(self, employee_id: str, new_salary: float) -> bool:
        logger.info("PayrollAdapter: updating salary for %s to %.0f (stub)", employee_id, new_salary)
        return True


class IdentityManagementAdapter:
    """Interfaces with SSO / LDAP for role-based access controls."""

    def get_user_roles(self, employee_id: str) -> List[str]:
        return ["employee", "hcm_user"]

    def provision_access(self, employee_id: str, system: str, role: str) -> bool:
        logger.info("IdentityAdapter: provisioning %s access to %s (stub)", employee_id, system)
        return True


class CollaborationAdapter:
    """Reads signals from MS Teams / Slack (communication density proxy)."""

    def get_collaboration_score(self, employee_id: str, days: int = 30) -> float:
        logger.info("CollaborationAdapter: collaboration score for %s (stub)", employee_id)
        return 0.72

    def get_cross_team_connections(self, employee_id: str) -> int:
        return 14


class FinanceAdapter:
    """Provides headcount budget constraints for demand forecasting."""

    def get_headcount_budget(self, department: str, fiscal_year: int) -> Dict[str, Any]:
        return {"department": department, "approved_headcount": 25, "current_headcount": 22}

    def get_productivity_cost_data(self, department: str) -> Dict[str, float]:
        return {"cost_per_fte": 120000.0, "revenue_per_fte": 350000.0}
