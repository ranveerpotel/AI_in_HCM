"""
Workforce Knowledge Graph (Layer 4 — Data Intelligence).

Encodes rich relationships between employees, skills, roles, projects, and
learning activities as a directed property graph backed by NetworkX.

Relationship types (from paper §7.1):
  HAS_SKILL        Employee → Skill   (proficiency score)
  HOLDS_ROLE       Employee → Role
  REPORTS_TO       Employee → Employee
  WORKED_ON        Employee → Project  (contribution_score)
  LEARNING_PROGRESS Employee → LearningActivity
  SIMILAR_TO       Skill   → Skill    (semantic similarity)
  REQUIRES_SKILL   Role    → Skill    (required_proficiency)
  NEEDS_SKILL      Project → Skill
"""

from __future__ import annotations
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..common.models import Employee, Skill, Role, Project, LearningActivity


class WorkforceKnowledgeGraph:
    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # Node registration
    # ------------------------------------------------------------------ #

    def add_employee(self, emp: Employee) -> None:
        self._g.add_node(
            f"emp:{emp.id}",
            type="Employee",
            name=emp.name,
            department=emp.department,
            tenure_years=emp.tenure_years,
            performance_score=emp.performance_score,
            engagement_score=emp.engagement_score,
            compensation=emp.compensation,
            market_comp_ratio=emp.market_comp_ratio,
            promotion_velocity=emp.promotion_velocity,
            workload_index=emp.workload_index,
        )
        for sa in emp.skills:
            self._g.add_edge(
                f"emp:{emp.id}",
                f"skill:{sa.skill_id}",
                relation="HAS_SKILL",
                proficiency=sa.proficiency,
                source=sa.source,
            )
        if emp.manager_id:
            self._g.add_edge(
                f"emp:{emp.id}",
                f"emp:{emp.manager_id}",
                relation="REPORTS_TO",
            )
        self._g.add_edge(
            f"emp:{emp.id}",
            f"role:{emp.role_id}",
            relation="HOLDS_ROLE",
        )

    def add_skill(self, skill: Skill) -> None:
        self._g.add_node(
            f"skill:{skill.id}",
            type="Skill",
            name=skill.name,
            category=skill.category,
        )

    def add_role(self, role: Role) -> None:
        self._g.add_node(
            f"role:{role.id}",
            type="Role",
            title=role.title,
            department=role.department,
            grade_level=role.grade_level,
        )
        for sid in role.required_skills:
            self._g.add_edge(
                f"role:{role.id}",
                f"skill:{sid}",
                relation="REQUIRES_SKILL",
            )
        for next_role_id in role.career_paths:
            self._g.add_edge(
                f"role:{role.id}",
                f"role:{next_role_id}",
                relation="LEADS_TO",
            )

    def add_project(self, project: Project) -> None:
        self._g.add_node(
            f"proj:{project.id}",
            type="Project",
            name=project.name,
            department=project.department,
            status=project.status,
            outcome_score=project.outcome_score,
        )
        for sid in project.required_skills:
            self._g.add_edge(
                f"proj:{project.id}",
                f"skill:{sid}",
                relation="NEEDS_SKILL",
            )
        for eid in project.team_members:
            self._g.add_edge(
                f"emp:{eid}",
                f"proj:{project.id}",
                relation="WORKED_ON",
            )

    def add_learning_activity(self, la: LearningActivity) -> None:
        self._g.add_node(
            f"learn:{la.id}",
            type="LearningActivity",
            title=la.content_title,
            progress=la.progress,
            outcome_score=la.outcome_score,
        )
        self._g.add_edge(
            f"emp:{la.employee_id}",
            f"learn:{la.id}",
            relation="LEARNING_PROGRESS",
            progress=la.progress,
        )
        for sid in la.skill_ids:
            self._g.add_edge(
                f"learn:{la.id}",
                f"skill:{sid}",
                relation="DEVELOPS_SKILL",
            )

    def add_skill_similarity(self, skill_a: str, skill_b: str, similarity: float) -> None:
        self._g.add_edge(
            f"skill:{skill_a}",
            f"skill:{skill_b}",
            relation="SIMILAR_TO",
            weight=similarity,
        )

    # ------------------------------------------------------------------ #
    # Traversal queries
    # ------------------------------------------------------------------ #

    def get_employee_skills(self, employee_id: str) -> List[Dict[str, Any]]:
        node = f"emp:{employee_id}"
        skills = []
        for _, target, data in self._g.out_edges(node, data=True):
            if data.get("relation") == "HAS_SKILL":
                skill_data = self._g.nodes.get(target, {})
                skills.append({
                    "skill_id": target.replace("skill:", ""),
                    "name": skill_data.get("name"),
                    "category": skill_data.get("category"),
                    "proficiency": data.get("proficiency"),
                })
        return skills

    def get_role_required_skills(self, role_id: str) -> List[str]:
        node = f"role:{role_id}"
        return [
            t.replace("skill:", "")
            for _, t, d in self._g.out_edges(node, data=True)
            if d.get("relation") == "REQUIRES_SKILL"
        ]

    def get_skill_gap(self, employee_id: str, role_id: str) -> List[str]:
        emp_skill_ids = {s["skill_id"] for s in self.get_employee_skills(employee_id)}
        required = set(self.get_role_required_skills(role_id))
        return list(required - emp_skill_ids)

    def get_direct_reports(self, manager_id: str) -> List[str]:
        node = f"emp:{manager_id}"
        return [
            s.replace("emp:", "")
            for s, _, d in self._g.in_edges(node, data=True)
            if d.get("relation") == "REPORTS_TO"
        ]

    def get_similar_employees(self, employee_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return employees with the most skill overlap (collaborative-filter proxy)."""
        target_skills = {s["skill_id"] for s in self.get_employee_skills(employee_id)}
        scores: List[tuple] = []
        for n, attrs in self._g.nodes(data=True):
            if attrs.get("type") == "Employee" and n != f"emp:{employee_id}":
                peer_skills = {s["skill_id"] for s in self.get_employee_skills(n.replace("emp:", ""))}
                union = target_skills | peer_skills
                jaccard = len(target_skills & peer_skills) / len(union) if union else 0.0
                scores.append({"employee_id": n.replace("emp:", ""), "similarity": jaccard})
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]

    def get_career_path_roles(self, role_id: str, depth: int = 3) -> List[str]:
        node = f"role:{role_id}"
        reachable = []
        for target in nx.dfs_preorder_nodes(self._g, node, depth_limit=depth):
            if target != node and self._g.nodes[target].get("type") == "Role":
                reachable.append(target.replace("role:", ""))
        return reachable

    def get_team_skill_coverage(self, manager_id: str) -> Dict[str, float]:
        """Average proficiency per skill across direct reports."""
        reports = self.get_direct_reports(manager_id)
        skill_totals: Dict[str, List[float]] = {}
        for eid in reports:
            for s in self.get_employee_skills(eid):
                skill_totals.setdefault(s["name"], []).append(s["proficiency"])
        return {k: sum(v) / len(v) for k, v in skill_totals.items()}

    def employee_count(self) -> int:
        return sum(1 for _, d in self._g.nodes(data=True) if d.get("type") == "Employee")

    def export_adjacency(self) -> Dict[str, Any]:
        return nx.node_link_data(self._g)
