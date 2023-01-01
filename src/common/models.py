from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EmploymentStatus(Enum):
    ACTIVE = "active"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentType(Enum):
    EMPLOYEE_ASSISTANCE = "employee_assistance"
    MANAGERIAL_DECISION = "managerial_decision"
    TALENT_INTELLIGENCE = "talent_intelligence"
    RECRUITMENT = "recruitment"
    LEARNING = "learning"
    WORKFORCE_RISK = "workforce_risk"


@dataclass
class Skill:
    id: str
    name: str
    category: str
    description: str = ""


@dataclass
class SkillAssessment:
    skill_id: str
    skill_name: str
    proficiency: float  # 0.0–1.0
    assessed_date: datetime
    source: str  # "self", "manager", "system"


@dataclass
class Employee:
    id: str
    name: str
    role_id: str
    department: str
    manager_id: Optional[str]
    hire_date: datetime
    location: str
    employment_status: EmploymentStatus
    skills: List[SkillAssessment] = field(default_factory=list)
    compensation: float = 0.0
    market_comp_ratio: float = 1.0   # compensation / market midpoint
    performance_score: float = 0.0   # 0–5
    engagement_score: float = 0.0    # 0–1
    promotion_velocity: float = 0.0  # promotions per year vs cohort norm
    workload_index: float = 0.0      # 0–1, 1 = overloaded

    @property
    def tenure_years(self) -> float:
        return (datetime.utcnow() - self.hire_date).days / 365.25


@dataclass
class Role:
    id: str
    title: str
    department: str
    grade_level: int
    required_skills: List[str] = field(default_factory=list)
    career_paths: List[str] = field(default_factory=list)


@dataclass
class Project:
    id: str
    name: str
    department: str
    required_skills: List[str]
    team_members: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    status: str
    outcome_score: Optional[float] = None


@dataclass
class LearningActivity:
    id: str
    employee_id: str
    content_id: str
    content_title: str
    skill_ids: List[str]
    completion_date: Optional[datetime]
    progress: float  # 0.0–1.0
    outcome_score: Optional[float] = None


@dataclass
class AttritionPrediction:
    employee_id: str
    risk_score: float             # 0.0–1.0
    risk_level: RiskLevel
    top_factors: List[Dict[str, float]]
    recommended_interventions: List[str]
    shap_values: Dict[str, float]
    prediction_date: datetime


@dataclass
class PerformancePrediction:
    employee_id: str
    predicted_score: float
    confidence: float
    growth_trajectory: str        # "accelerating" | "stable" | "declining"
    skill_gaps: List[str]


@dataclass
class CareerPathRecommendation:
    employee_id: str
    recommended_roles: List[Dict[str, Any]]  # {role_id, match_score, gap_skills}
    succession_potential: float
    development_priorities: List[str]


@dataclass
class CandidateMatch:
    candidate_id: str
    role_id: str
    compatibility_score: float
    skill_match: float
    culture_fit_score: float
    diversity_flag: bool
    explanation: str


@dataclass
class AuditRecord:
    id: str
    timestamp: datetime
    agent_type: AgentType
    action_type: str
    employee_id: str
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    model_version: str
    human_reviewed: bool = False
    override_applied: bool = False


@dataclass
class BiasReport:
    model_id: str
    evaluation_date: datetime
    demographic_parity: Dict[str, float]    # group → selection rate
    equal_opportunity: Dict[str, float]     # group → TPR
    disparate_impact: Dict[str, float]      # group → DI ratio
    flagged: bool
    recommendations: List[str]


@dataclass
class DriftReport:
    model_id: str
    feature: str
    test_statistic: float
    p_value: float
    drift_detected: bool
    detection_date: datetime


@dataclass
class WorkforceSignal:
    """Real-time signal ingested from collaboration / productivity tools."""
    employee_id: str
    signal_type: str   # "engagement_survey", "collaboration_change", "workload_spike"
    value: float
    timestamp: datetime
    source: str
