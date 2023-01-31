"""
NLP Service (Layer 3 — AI Services).

Covers §6.3 use cases:
  - Intent recognition for employee queries
  - Sentiment analysis on free-text survey responses
  - Skill extraction from resumes / work summaries
  - Policy-to-context explanation using Claude API
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional, Tuple

import anthropic

from ..common.config import config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Rule-based intent classifier (production would use fine-tuned model)
# ------------------------------------------------------------------ #

INTENT_PATTERNS: Dict[str, List[str]] = {
    "leave_inquiry":       ["leave", "vacation", "pto", "parental", "sick day", "time off"],
    "compensation_query":  ["salary", "pay", "compensation", "bonus", "raise", "equity"],
    "career_development":  ["career", "promotion", "growth", "advance", "next role", "skills"],
    "benefits_query":      ["benefit", "health", "dental", "vision", "401k", "insurance"],
    "performance_review":  ["performance", "review", "feedback", "appraisal", "goal"],
    "onboarding_help":     ["onboarding", "new hire", "getting started", "first day"],
    "team_management":     ["team", "headcount", "hire", "backfill", "reorganiz"],
    "compliance_question": ["policy", "compliance", "regulation", "gdpr", "code of conduct"],
}


class IntentClassifier:
    def classify(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for intent, keywords in INTENT_PATTERNS.items():
            hit = sum(1 for kw in keywords if kw in text_lower)
            if hit:
                scores[intent] = hit
        if not scores:
            return "general_inquiry", 0.5
        top_intent = max(scores, key=lambda k: scores[k])
        confidence = min(scores[top_intent] / 3.0, 1.0)
        return top_intent, confidence


# ------------------------------------------------------------------ #
# Sentiment analysis (lexicon-based baseline)
# ------------------------------------------------------------------ #

POSITIVE_WORDS = {
    "great", "excellent", "love", "happy", "satisfied", "positive", "enjoy",
    "motivated", "engaged", "proud", "appreciate", "good", "amazing",
}
NEGATIVE_WORDS = {
    "bad", "poor", "unhappy", "frustrated", "stressed", "overwhelmed", "burnt",
    "tired", "difficult", "disappointed", "concern", "issue", "problem",
}


class SentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, float]:
        words = re.findall(r"\b\w+\b", text.lower())
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos + neg + 1e-8
        sentiment_score = (pos - neg) / total   # -1 to +1
        return {
            "sentiment_score": round(sentiment_score, 3),
            "positive_signals": pos,
            "negative_signals": neg,
            "label": "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral"),
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.analyze(t) for t in texts]


# ------------------------------------------------------------------ #
# Skill extractor from free text
# ------------------------------------------------------------------ #

SKILL_KEYWORDS: Dict[str, str] = {
    "python": "programming", "java": "programming", "sql": "data",
    "machine learning": "ai_ml", "deep learning": "ai_ml", "nlp": "ai_ml",
    "project management": "management", "agile": "management", "scrum": "management",
    "communication": "soft_skills", "leadership": "soft_skills", "teamwork": "soft_skills",
    "data analysis": "data", "tableau": "data", "power bi": "data",
    "aws": "cloud", "azure": "cloud", "gcp": "cloud",
    "kubernetes": "devops", "docker": "devops", "ci/cd": "devops",
}


class SkillExtractor:
    def extract(self, text: str) -> List[Dict[str, str]]:
        text_lower = text.lower()
        found = []
        seen = set()
        for skill_name, category in SKILL_KEYWORDS.items():
            if skill_name in text_lower and skill_name not in seen:
                found.append({"skill": skill_name, "category": category})
                seen.add(skill_name)
        return found


# ------------------------------------------------------------------ #
# Policy explainer using Claude RAG
# ------------------------------------------------------------------ #

class PolicyExplainer:
    """
    Uses Claude to interpret HR policies in the context of a specific employee.
    Implements RAG pattern from §14.3 of the paper.
    """

    def __init__(self) -> None:
        if config.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            self._client = None

    def explain(
        self,
        query: str,
        policy_text: str,
        employee_context: Optional[Dict] = None,
    ) -> str:
        if not self._client:
            return f"Policy reference: {policy_text[:200]}... (configure ANTHROPIC_API_KEY for AI explanation)"

        context_str = ""
        if employee_context:
            context_str = (
                f"\nEmployee context: tenure={employee_context.get('tenure_years', 'N/A')} years, "
                f"role={employee_context.get('role', 'N/A')}, "
                f"location={employee_context.get('location', 'N/A')}"
            )

        system_prompt = (
            "You are an HR policy assistant for an enterprise HCM system. "
            "Provide accurate, empathetic, and personalized policy explanations. "
            "Always ground your answer in the provided policy text. "
            "Never fabricate policy details."
        )

        user_message = (
            f"HR Policy:\n{policy_text}\n\n"
            f"Employee Query: {query}"
            f"{context_str}\n\n"
            "Provide a clear, personalized explanation relevant to this employee's situation."
        )

        try:
            response = self._client.messages.create(
                model=config.claude_model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Policy explainer error: %s", e)
            return f"Policy summary: {policy_text[:300]}..."
