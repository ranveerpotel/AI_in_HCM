"""
Recruitment Intelligence Agent (Layer 2 — Intelligent Agents).

Implements §5.5:
  - Semantic resume understanding (skill extraction + transferable skills)
  - Candidate-job matching (multi-dimensional compatibility)
  - Interview scheduling optimization
  - Diversity analytics and bias monitoring
  - Hiring outcome prediction
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..common.models import AgentType, AuditRecord, CandidateMatch
from ..layer3_ai_services.nlp_service import SkillExtractor
from ..layer6_governance.bias_monitor import BiasMonitor
from ..layer6_governance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class Candidate:
    def __init__(
        self,
        id: str,
        name: str,
        resume_text: str,
        years_experience: float,
        demographic_group: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.resume_text = resume_text
        self.years_experience = years_experience
        self.demographic_group = demographic_group
        self.extracted_skills: List[Dict] = []


class RecruitmentAgent:

    AGENT_VERSION = "rca-v1.0"

    def __init__(
        self,
        bias_monitor: Optional[BiasMonitor] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._skill_extractor = SkillExtractor()
        self._bias_monitor = bias_monitor
        self._audit = audit_logger

    # ---------------------------------------------------------------- #
    # Resume processing
    # ---------------------------------------------------------------- #

    def process_resume(self, candidate: Candidate) -> Candidate:
        candidate.extracted_skills = self._skill_extractor.extract(candidate.resume_text)
        return candidate

    # ---------------------------------------------------------------- #
    # Candidate-job matching
    # ---------------------------------------------------------------- #

    def match_candidate_to_role(
        self,
        candidate: Candidate,
        role_required_skills: List[str],
        role_id: str,
        culture_keywords: Optional[List[str]] = None,
    ) -> CandidateMatch:
        candidate_skill_names = {s["skill"] for s in candidate.extracted_skills}
        required_set = set(role_required_skills)

        skill_match = len(candidate_skill_names & required_set) / len(required_set) if required_set else 0.0

        culture_fit = 0.7
        if culture_keywords:
            culture_hits = sum(1 for kw in culture_keywords if kw.lower() in candidate.resume_text.lower())
            culture_fit = min(culture_hits / len(culture_keywords), 1.0)

        exp_factor = min(candidate.years_experience / 5.0, 1.0)
        compatibility = 0.50 * skill_match + 0.30 * culture_fit + 0.20 * exp_factor

        explanation = (
            f"Skill match: {skill_match:.0%} of required skills identified. "
            f"Culture alignment: {culture_fit:.0%}. "
            f"Experience factor: {exp_factor:.0%}."
        )

        diversity_flag = False
        if self._bias_monitor and candidate.demographic_group:
            diversity_flag = self._bias_monitor.flag_for_review(candidate.demographic_group)

        return CandidateMatch(
            candidate_id=candidate.id,
            role_id=role_id,
            compatibility_score=round(compatibility, 3),
            skill_match=round(skill_match, 3),
            culture_fit_score=round(culture_fit, 3),
            diversity_flag=diversity_flag,
            explanation=explanation,
        )

    def rank_candidates(
        self,
        candidates: List[Candidate],
        role_required_skills: List[str],
        role_id: str,
    ) -> List[CandidateMatch]:
        matches = [
            self.match_candidate_to_role(c, role_required_skills, role_id)
            for c in candidates
        ]
        matches.sort(key=lambda m: m.compatibility_score, reverse=True)

        if self._audit:
            self._audit.log(AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_type=AgentType.RECRUITMENT,
                action_type="candidate_ranking",
                employee_id="system",
                input_summary={"role_id": role_id, "candidate_count": len(candidates)},
                output_summary={"top_score": matches[0].compatibility_score if matches else 0},
                model_version=self.AGENT_VERSION,
            ))

        return matches

    # ---------------------------------------------------------------- #
    # Diversity analytics
    # ---------------------------------------------------------------- #

    def pipeline_diversity_report(
        self, matches: List[CandidateMatch], candidates: List[Candidate]
    ) -> Dict[str, Any]:
        candidate_map = {c.id: c for c in candidates}
        groups: Dict[str, List[float]] = {}
        for m in matches:
            cand = candidate_map.get(m.candidate_id)
            if cand and cand.demographic_group:
                groups.setdefault(cand.demographic_group, []).append(m.compatibility_score)

        report: Dict[str, Any] = {
            "total_candidates": len(matches),
            "representation": {g: len(v) for g, v in groups.items()},
            "avg_score_by_group": {g: round(sum(v)/len(v), 3) for g, v in groups.items()},
            "flagged_for_review": sum(1 for m in matches if m.diversity_flag),
        }
        return report
