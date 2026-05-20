"""
Employee Assistance Agent (Layer 2 — Intelligent Agents).

Implements the 6-stage functional flow from Figure 5 of the paper:
  Intent Recognition → Context Retrieval → Policy Interpretation →
  Transaction Orchestration → Response Delivery → Continuous Learning

Uses Claude API with RAG for grounded, policy-accurate answers.
Handles: leave inquiries, compensation queries, career guidance,
         benefits, performance, onboarding, compliance questions.
"""

from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import anthropic

from ..common.config import config
from ..common.models import AgentType, AuditRecord, Employee
from ..layer3_ai_services.nlp_service import IntentClassifier, PolicyExplainer, SentimentAnalyzer
from ..layer6_governance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE: Dict[str, str] = {
    "leave_inquiry": (
        "Leave Policy: Employees accrue 15 days PTO per year (20 days after 3 years). "
        "Parental leave: 12 weeks fully paid for primary caregiver, 6 weeks for secondary. "
        "Sick leave: 10 days per year, non-accruing. "
        "Submit leave requests via the HR portal at least 2 weeks in advance for planned leave."
    ),
    "compensation_query": (
        "Compensation Philosophy: We target the 65th percentile of market for base salary. "
        "Annual merit cycles run in Q1 with budget guidance from Finance. "
        "Bonus eligibility is based on role grade (Grade 5+) and individual performance rating. "
        "Equity refresh grants are issued annually to Grade 7+ employees."
    ),
    "career_development": (
        "Career Development: Career ladders define growth tracks for each function. "
        "Internal mobility is encouraged after 18 months in role. "
        "Development plans should be co-created with managers in Q1 each year. "
        "Tuition reimbursement up to $5,000/year is available for job-relevant programs."
    ),
    "benefits_query": (
        "Benefits: Health insurance (medical, dental, vision) effective day 1. "
        "401k with 4% company match vesting over 2 years. "
        "Life insurance 2× annual salary. EAP (Employee Assistance Program) available 24/7."
    ),
    "performance_review": (
        "Performance Management: Mid-year check-in in June, annual review in December. "
        "Ratings: Exceeds (5), Meets+ (4), Meets (3), Developing (2), Below (1). "
        "PIPs are initiated by HR for ratings below 2 for two consecutive cycles."
    ),
    "compliance_question": (
        "Code of Conduct: All employees must complete annual ethics training. "
        "Report concerns via the ethics hotline (anonymous). GDPR: Employee data processed "
        "under legitimate interest and contract necessity. Data subject requests handled within 30 days."
    ),
    "general_inquiry": (
        "HR Service Center is available Mon–Fri 8am–6pm local time. "
        "For urgent matters, contact your HRBP directly. "
        "Common forms and policies are available on the HR intranet portal."
    ),
}


class EmployeeAssistanceAgent:
    """
    Persistent conversational agent for employee HR self-service.
    Demonstrates the Data→Intelligence→Action→ContinuousLearning cycle.
    """

    AGENT_VERSION = "eaa-v1.0"

    def __init__(self, audit_logger: Optional[AuditLogger] = None) -> None:
        self._intent_clf = IntentClassifier()
        self._sentiment = SentimentAnalyzer()
        self._policy_explainer = PolicyExplainer()
        self._audit = audit_logger
        self._conversation_memory: Dict[str, List[Dict]] = {}  # session_id → history
        self._feedback_store: List[Dict] = []

        if config.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            self._client = None

    # ---------------------------------------------------------------- #
    # Stage 1 + 2: Intent Recognition + Context Retrieval
    # ---------------------------------------------------------------- #

    def _recognize_intent(self, query: str) -> tuple[str, float]:
        return self._intent_clf.classify(query)

    def _retrieve_context(self, intent: str, employee: Optional[Employee] = None) -> str:
        policy = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["general_inquiry"])
        employee_ctx = ""
        if employee:
            employee_ctx = (
                f"\n\nEmployee Profile: {employee.name}, "
                f"Tenure: {employee.tenure_years:.1f} years, "
                f"Department: {employee.department}, "
                f"Location: {employee.location}, "
                f"Role ID: {employee.role_id}"
            )
        return policy + employee_ctx

    # ---------------------------------------------------------------- #
    # Stage 3 + 4: Policy Interpretation + Transaction Orchestration
    # ---------------------------------------------------------------- #

    def _generate_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict],
    ) -> str:
        if not self._client:
            return self._fallback_response(query, context)

        system = (
            "You are an empathetic HR assistant for an enterprise workforce platform. "
            "Your answers must be grounded in the provided policy context. "
            "Be concise, accurate, and personalized. Do not fabricate policy details. "
            "If uncertain, direct the employee to their HRBP."
        )

        messages = conversation_history[-6:]  # sliding context window
        messages.append({"role": "user", "content": f"Policy Context:\n{context}\n\nEmployee Query: {query}"})

        try:
            resp = self._client.messages.create(
                model=config.claude_model,
                max_tokens=512,
                system=system,
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            logger.error("Claude API error in EAA: %s", e)
            return self._fallback_response(query, context)

    @staticmethod
    def _fallback_response(query: str, context: str) -> str:
        return f"Based on our policies:\n\n{context[:400]}\n\nFor specific questions, please contact your HRBP."

    # ---------------------------------------------------------------- #
    # Stage 5 + 6: Response Delivery + Continuous Learning
    # ---------------------------------------------------------------- #

    def handle_query(
        self,
        session_id: str,
        query: str,
        employee: Optional[Employee] = None,
    ) -> Dict[str, Any]:
        # Stage 1 – intent
        intent, confidence = self._recognize_intent(query)

        # Stage 2 – context
        context = self._retrieve_context(intent, employee)

        # Conversation memory
        history = self._conversation_memory.setdefault(session_id, [])

        # Stage 3 + 4 – generate
        response_text = self._generate_response(query, context, history)

        # Update memory
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        if len(history) > 20:
            history.pop(0)

        result = {
            "session_id": session_id,
            "intent": intent,
            "intent_confidence": confidence,
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Stage 6 – audit
        if self._audit and employee:
            self._audit.log(AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_type=AgentType.EMPLOYEE_ASSISTANCE,
                action_type=intent,
                employee_id=employee.id,
                input_summary={"query": query[:200], "intent": intent},
                output_summary={"response_length": len(response_text)},
                model_version=self.AGENT_VERSION,
            ))

        return result

    def record_feedback(self, session_id: str, rating: int, comment: str = "") -> None:
        """Stage 6: capture outcome feedback to retrain intent classifier over time."""
        self._feedback_store.append({
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def satisfaction_score(self) -> float:
        if not self._feedback_store:
            return 0.0
        ratings = [f["rating"] for f in self._feedback_store]
        return sum(r >= 4 for r in ratings) / len(ratings)
