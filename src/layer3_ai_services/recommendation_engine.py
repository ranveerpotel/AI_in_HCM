"""
Hybrid Recommendation Engine (Layer 3 — AI Services).

Implements §6.4:  Collaborative Filtering  +  Content-Based  +  Organizational
Network Analysis (ONA).  Produces:
  - Internal job / project recommendations
  - Learning content recommendations
  - Mentor matching
  - Internal mobility marketplace matches
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity

from ..common.models import Employee, Role, Project, LearningActivity, Skill


class ContentBasedEngine:
    """Match employees to roles/projects based on skill feature vectors."""

    def __init__(self, skill_index: Dict[str, int]) -> None:
        self._skill_index = skill_index
        self._dim = len(skill_index)

    def employee_vector(self, employee: Employee) -> np.ndarray:
        vec = np.zeros(self._dim)
        for sa in employee.skills:
            idx = self._skill_index.get(sa.skill_id)
            if idx is not None:
                vec[idx] = sa.proficiency
        return vec

    def role_vector(self, role: Role) -> np.ndarray:
        vec = np.zeros(self._dim)
        for sid in role.required_skills:
            idx = self._skill_index.get(sid)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def match_employee_to_roles(
        self, employee: Employee, roles: List[Role], top_k: int = 5
    ) -> List[Tuple[Role, float]]:
        emp_vec = self.employee_vector(employee).reshape(1, -1)
        results = []
        for role in roles:
            r_vec = self.role_vector(role).reshape(1, -1)
            score = float(cosine_similarity(emp_vec, r_vec)[0, 0])
            results.append((role, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def match_employee_to_projects(
        self, employee: Employee, projects: List[Project], skill_index: Dict[str, int], top_k: int = 5
    ) -> List[Tuple[Project, float]]:
        emp_vec = self.employee_vector(employee)
        results = []
        for proj in projects:
            proj_vec = np.zeros(self._dim)
            for sid in proj.required_skills:
                idx = skill_index.get(sid)
                if idx is not None:
                    proj_vec[idx] = 1.0
            score = float(cosine_similarity(emp_vec.reshape(1, -1), proj_vec.reshape(1, -1))[0, 0])
            results.append((proj, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class CollaborativeFilter:
    """
    Employee-employee collaborative filtering based on career outcome similarity.
    Uses cosine similarity on outcome-enriched skill vectors.
    """

    def __init__(self) -> None:
        self._matrix: Optional[np.ndarray] = None
        self._employee_ids: List[str] = []

    def fit(self, employee_ids: List[str], feature_matrix: np.ndarray) -> None:
        self._employee_ids = employee_ids
        self._matrix = feature_matrix

    def similar_peers(self, employee_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._matrix is None or employee_id not in self._employee_ids:
            return []
        idx = self._employee_ids.index(employee_id)
        sims = cosine_similarity(self._matrix[idx].reshape(1, -1), self._matrix)[0]
        sims[idx] = -1.0  # exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self._employee_ids[i], float(sims[i])) for i in top_indices]

    def recommend_learning_from_peers(
        self,
        employee_id: str,
        all_activities: List[LearningActivity],
        top_k: int = 5,
    ) -> List[str]:
        """Return content_ids completed by similar successful peers but not yet by target."""
        peers = [pid for pid, _ in self.similar_peers(employee_id, top_k=10)]
        target_content = {la.content_id for la in all_activities if la.employee_id == employee_id}
        peer_completions: Dict[str, int] = {}
        for la in all_activities:
            if la.employee_id in peers and la.progress >= 1.0 and la.content_id not in target_content:
                peer_completions[la.content_id] = peer_completions.get(la.content_id, 0) + 1
        ranked = sorted(peer_completions.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]


class MentorMatcher:
    """Match employees to mentors based on skill complementarity and career alignment."""

    def match(
        self,
        mentee: Employee,
        mentee_target_role_id: str,
        candidate_mentors: List[Employee],
        top_k: int = 3,
    ) -> List[Tuple[Employee, float, str]]:
        results = []
        mentee_skills = {sa.skill_id for sa in mentee.skills}
        for mentor in candidate_mentors:
            if mentor.id == mentee.id:
                continue
            mentor_skills = {sa.skill_id for sa in mentor.skills}
            complementarity = len(mentor_skills - mentee_skills) / (len(mentor_skills) + 1)
            dept_match = 1.0 if mentor.department == mentee.department else 0.5
            score = 0.6 * complementarity + 0.4 * dept_match
            explanation = (
                f"Mentor has {len(mentor_skills - mentee_skills)} skills you can develop; "
                f"{'same' if dept_match == 1.0 else 'cross'}-department pairing"
            )
            results.append((mentor, score, explanation))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class HybridRecommender:
    """
    Combines content-based, collaborative, and ONA signals (§6.4).
    Weights: CB=0.40, CF=0.35, ONA=0.25.
    """

    def __init__(
        self,
        content_engine: ContentBasedEngine,
        collab_filter: CollaborativeFilter,
        cb_weight: float = 0.40,
        cf_weight: float = 0.35,
        ona_weight: float = 0.25,
    ) -> None:
        self._cb = content_engine
        self._cf = collab_filter
        self._cb_w = cb_weight
        self._cf_w = cf_weight
        self._ona_w = ona_weight

    def recommend_roles(
        self,
        employee: Employee,
        all_roles: List[Role],
        ona_scores: Optional[Dict[str, float]] = None,  # role_id → network centrality
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        cb_matches = {role.id: score for role, score in self._cb.match_employee_to_roles(employee, all_roles, top_k=20)}
        ona_scores = ona_scores or {}

        peer_role_ids: List[str] = []
        for pid, _ in self._cf.similar_peers(employee.id, top_k=10):
            pass  # CF for roles would need role-completion history; use CB for now

        combined: Dict[str, float] = {}
        for role in all_roles:
            cb = cb_matches.get(role.id, 0.0)
            ona = ona_scores.get(role.id, 0.5)
            combined[role.id] = self._cb_w * cb + self._ona_w * ona

        role_map = {r.id: r for r in all_roles}
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "role_id": rid,
                "role_title": role_map[rid].title if rid in role_map else rid,
                "combined_score": score,
                "skill_gap": [],  # populated by talent agent
            }
            for rid, score in ranked
        ]
