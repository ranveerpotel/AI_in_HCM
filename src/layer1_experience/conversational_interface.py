"""
Conversational Interface (Layer 1 — Experience).

Implements §4.2.1 experience layer:
  - Multi-turn conversation manager with session state
  - Role-aware response calibration (employee vs. manager vs. HR)
  - Multimodal intent routing: routes queries to the correct agent
  - Context-aware adaptive UI response formatting

This module sits between the REST API (dashboard_api.py) and the
intelligent agents (Layer 2), orchestrating which agent handles each
intent and merging responses into a unified conversational reply.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..common.models import Employee
from ..layer2_agents.employee_assistance_agent import EmployeeAssistanceAgent
from ..layer2_agents.managerial_decision_agent import ManagerialDecisionAgent
from ..layer2_agents.talent_intelligence_agent import TalentIntelligenceAgent
from ..layer2_agents.learning_agent import LearningAgent
from ..layer3_ai_services.nlp_service import IntentClassifier, SentimentAnalyzer

logger = logging.getLogger(__name__)

MANAGER_INTENTS = {"team_management", "compensation_query", "performance_review"}
CAREER_INTENTS   = {"career_development"}
LEARNING_INTENTS = {"onboarding_help"}


@dataclass
class ConversationSession:
    session_id: str
    employee: Optional[Employee]
    role: str           # "employee" | "manager" | "hr"
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        self.last_active = datetime.utcnow()


class ConversationalInterface:
    """
    Unified conversational entry point that routes queries to the appropriate
    Layer 2 agent based on intent and user role (§4.2.1 context awareness).
    """

    def __init__(
        self,
        employee_agent: EmployeeAssistanceAgent,
        manager_agent: ManagerialDecisionAgent,
        talent_agent: TalentIntelligenceAgent,
        learning_agent: LearningAgent,
    ) -> None:
        self._employee_agent = employee_agent
        self._manager_agent = manager_agent
        self._talent_agent = talent_agent
        self._learning_agent = learning_agent
        self._intent_clf = IntentClassifier()
        self._sentiment = SentimentAnalyzer()
        self._sessions: Dict[str, ConversationSession] = {}

    # ---------------------------------------------------------------- #
    # Session management
    # ---------------------------------------------------------------- #

    def start_session(
        self,
        session_id: str,
        employee: Optional[Employee] = None,
        role: str = "employee",
    ) -> ConversationSession:
        session = ConversationSession(
            session_id=session_id, employee=employee, role=role
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)

    # ---------------------------------------------------------------- #
    # Message routing
    # ---------------------------------------------------------------- #

    def handle_message(
        self,
        session_id: str,
        message: str,
        employee: Optional[Employee] = None,
        role: str = "employee",
    ) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if not session:
            session = self.start_session(session_id, employee, role)

        intent, confidence = self._intent_clf.classify(message)
        sentiment = self._sentiment.analyze(message)

        response_text, agent_used = self._route(session, message, intent)

        session.add_turn(message, response_text)

        return {
            "session_id": session_id,
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "sentiment": sentiment["label"],
            "agent": agent_used,
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _route(
        self, session: ConversationSession, message: str, intent: str
    ) -> tuple[str, str]:
        employee = session.employee

        # Manager-role intents get routed to managerial agent when available
        if session.role == "manager" and intent in MANAGER_INTENTS:
            if intent == "team_management" and employee:
                try:
                    report = self._manager_agent.team_health_report(
                        employee.id, []  # real system would fetch direct reports
                    )
                    summary = self._manager_agent.coaching_summary(employee.id, report)
                    return summary, "managerial_decision_agent"
                except Exception:
                    pass

        # Career development routes to talent agent
        if intent in CAREER_INTENTS and employee:
            try:
                rec = self._talent_agent.career_path_recommendation(employee, [])
                if rec.recommended_roles:
                    top = rec.recommended_roles[0]
                    return (
                        f"Based on your profile, your strongest next-role match is "
                        f"{top['role_title']} (fit score: {top['combined_score']:.0%}). "
                        f"Development priorities: {', '.join(rec.development_priorities[:2]) or 'none identified'}.",
                        "talent_intelligence_agent",
                    )
            except Exception:
                pass

        # Default: employee assistance agent handles everything
        result = self._employee_agent.handle_query(session.session_id, message, employee)
        return result["response"], "employee_assistance_agent"

    # ---------------------------------------------------------------- #
    # Session analytics
    # ---------------------------------------------------------------- #

    def session_count(self) -> int:
        return len(self._sessions)

    def active_sessions(self, max_idle_minutes: int = 30) -> List[str]:
        cutoff = datetime.utcnow()
        return [
            sid for sid, s in self._sessions.items()
            if (cutoff - s.last_active).seconds < max_idle_minutes * 60
        ]
