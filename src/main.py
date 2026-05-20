"""
AI-HCM Platform — Main Orchestrator.

Wires together all 6 architectural layers from Ranveer Potel's paper
"Artificial Intelligence in Human Capital Management: A Comprehensive
Framework for Intelligent Workforce Systems" (IJAIBDCMS, 2023).

Core operating cycle (Figure 3):
  Data → Intelligence → Action → Continuous Learning
"""

from __future__ import annotations
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .common.config import config
from .common.models import (
    Employee, EmploymentStatus, LearningActivity, Role, Project,
    Skill, SkillAssessment, WorkforceSignal,
)

# Layer 4 — Data Intelligence
from .layer4_data_intelligence.knowledge_graph import WorkforceKnowledgeGraph
from .layer4_data_intelligence.feature_store import FeatureStore
from .layer4_data_intelligence.streaming_analytics import WorkforceStreamProcessor

# Layer 3 — AI Services
from .layer3_ai_services.predictive_models import (
    AttritionPredictor, PerformancePredictor, DemandForecaster,
    AbsenteeismPredictor, PromotionLikelihoodEstimator,
)
from .layer3_ai_services.recommendation_engine import (
    ContentBasedEngine, CollaborativeFilter, MentorMatcher, HybridRecommender
)
from .layer3_ai_services.nlp_service import IntentClassifier, SentimentAnalyzer, SkillExtractor
from .layer3_ai_services.anomaly_detector import WorkforceAnomalyDetector

# Layer 2 — Intelligent Agents
from .layer2_agents.employee_assistance_agent import EmployeeAssistanceAgent
from .layer2_agents.managerial_decision_agent import ManagerialDecisionAgent
from .layer2_agents.talent_intelligence_agent import TalentIntelligenceAgent
from .layer2_agents.recruitment_agent import RecruitmentAgent
from .layer2_agents.learning_agent import LearningAgent
from .layer2_agents.workforce_risk_monitor import WorkforceRiskMonitor

# Layer 1 — Experience
from .layer1_experience.conversational_interface import ConversationalInterface

# Layer 5 — Integration
from .layer5_integration.integration_hub import IntegrationHub, PayrollAdapter, CollaborationAdapter

# Layer 6 — Governance
from .layer6_governance.audit_logger import AuditLogger
from .layer6_governance.bias_monitor import BiasMonitor
from .layer6_governance.xai_explainer import XAIExplainer
from .layer6_governance.drift_detector import DriftDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AIHCMPlatform:
    """
    Top-level orchestrator for the AI-Native HCM platform.
    Instantiates and wires all layers; exposes a unified interface
    for use by the REST API and the demo smoke test.
    """

    def __init__(self) -> None:
        logger.info("Initializing AI-HCM Platform...")

        # --- Layer 6: Governance (initialized first — others depend on it) ---
        self.audit_logger = AuditLogger()
        self.bias_monitor = BiasMonitor()
        self.xai_explainer = XAIExplainer()
        self.drift_detector = DriftDetector()

        # --- Layer 4: Data Intelligence ---
        self.knowledge_graph = WorkforceKnowledgeGraph()
        self.feature_store = FeatureStore()
        self.stream_processor = WorkforceStreamProcessor()

        # --- Layer 3: AI Services ---
        self.skill_index: Dict[str, int] = {}
        self.content_engine = ContentBasedEngine(self.skill_index)
        self.collab_filter = CollaborativeFilter()
        self.recommender = HybridRecommender(self.content_engine, self.collab_filter)
        self.attrition_predictor = AttritionPredictor()
        self.performance_predictor = PerformancePredictor()
        self.demand_forecaster = DemandForecaster()
        self.absenteeism_predictor = AbsenteeismPredictor()
        self.promotion_estimator = PromotionLikelihoodEstimator()
        self.anomaly_detector = WorkforceAnomalyDetector()

        # --- Layer 2: Intelligent Agents ---
        self.employee_agent = EmployeeAssistanceAgent(self.audit_logger)
        self.manager_agent = ManagerialDecisionAgent(
            self.knowledge_graph, self.feature_store,
            self.attrition_predictor, self.performance_predictor,
            self.audit_logger,
        )
        self.talent_agent = TalentIntelligenceAgent(
            self.knowledge_graph, self.recommender, self.audit_logger
        )
        self.recruitment_agent = RecruitmentAgent(self.bias_monitor, self.audit_logger)
        self.learning_agent = LearningAgent(self.collab_filter, self.audit_logger)
        self.risk_monitor = WorkforceRiskMonitor(
            self.feature_store, self.attrition_predictor,
            self.anomaly_detector, self.stream_processor,
            self.audit_logger,
        )

        # --- Layer 1: Experience ---
        self.conversational_interface = ConversationalInterface(
            self.employee_agent, self.manager_agent,
            self.talent_agent, self.learning_agent,
        )

        # --- Layer 5: Integration ---
        self.integration_hub = IntegrationHub()
        self.payroll_adapter = PayrollAdapter()
        self.collab_adapter = CollaborationAdapter()

        # --- In-memory registries ---
        self.employees: Dict[str, Employee] = {}
        self.roles: Dict[str, Role] = {}
        self.projects: Dict[str, Project] = {}
        self.skills: Dict[str, Skill] = {}
        self.learning_activities: List[LearningActivity] = []

        logger.info("AI-HCM Platform initialized.")

    # ---------------------------------------------------------------- #
    # Data loading helpers
    # ---------------------------------------------------------------- #

    def register_skill(self, skill: Skill) -> None:
        self.skills[skill.id] = skill
        if skill.id not in self.skill_index:
            self.skill_index[skill.id] = len(self.skill_index)
        self.knowledge_graph.add_skill(skill)

    def register_role(self, role: Role) -> None:
        self.roles[role.id] = role
        self.knowledge_graph.add_role(role)

    def register_employee(self, emp: Employee) -> None:
        self.employees[emp.id] = emp
        self.knowledge_graph.add_employee(emp)

    def register_project(self, proj: Project) -> None:
        self.projects[proj.id] = proj
        self.knowledge_graph.add_project(proj)

    def add_learning_activity(self, la: LearningActivity) -> None:
        self.learning_activities.append(la)
        self.knowledge_graph.add_learning_activity(la)

    def compute_features(self) -> None:
        for emp in self.employees.values():
            mgr_tenure = (
                self.employees[emp.manager_id].tenure_years
                if emp.manager_id and emp.manager_id in self.employees else None
            )
            self.feature_store.compute_and_store(
                emp,
                self.learning_activities,
                [],  # signals — populated by streaming processor in production
                manager_tenure_years=mgr_tenure,
            )
        self.feature_store.normalize_tenure()

        # Fit collaborative filter
        ids, matrix = self.feature_store.get_all_arrays()
        if ids:
            self.collab_filter.fit(ids, matrix)

    # ---------------------------------------------------------------- #
    # Lookup helpers (used by API layer)
    # ---------------------------------------------------------------- #

    def get_employee(self, employee_id: str) -> Optional[Employee]:
        return self.employees.get(employee_id)

    def get_direct_reports(self, manager_id: str) -> List[Employee]:
        return [
            emp for emp in self.employees.values()
            if emp.manager_id == manager_id
        ]

    # ---------------------------------------------------------------- #
    # Factory: seed synthetic data for demo / testing
    # ---------------------------------------------------------------- #

    @classmethod
    def create_demo(cls) -> "AIHCMPlatform":
        platform = cls()

        skills_data = [
            Skill("python", "Python", "programming"),
            Skill("ml", "Machine Learning", "ai_ml"),
            Skill("sql", "SQL", "data"),
            Skill("leadership", "Leadership", "soft_skills"),
            Skill("communication", "Communication", "soft_skills"),
            Skill("aws", "AWS", "cloud"),
            Skill("agile", "Agile", "management"),
            Skill("nlp", "NLP", "ai_ml"),
            Skill("data_analysis", "Data Analysis", "data"),
            Skill("docker", "Docker", "devops"),
        ]
        for s in skills_data:
            platform.register_skill(s)

        roles_data = [
            Role("swe_l3", "Software Engineer L3", "Engineering", 3,
                 ["python", "sql", "docker"], ["swe_l4", "ml_eng_l4"]),
            Role("swe_l4", "Software Engineer L4", "Engineering", 4,
                 ["python", "sql", "docker", "aws"], ["swe_l5", "eng_mgr"]),
            Role("ml_eng_l4", "ML Engineer L4", "Data Science", 4,
                 ["python", "ml", "nlp", "aws"], ["ml_eng_l5"]),
            Role("eng_mgr", "Engineering Manager", "Engineering", 5,
                 ["leadership", "communication", "agile"], []),
            Role("data_analyst", "Data Analyst", "Analytics", 3,
                 ["sql", "data_analysis", "communication"], ["sr_data_analyst"]),
            Role("sr_data_analyst", "Senior Data Analyst", "Analytics", 4,
                 ["sql", "data_analysis", "python", "leadership"], []),
        ]
        for r in roles_data:
            platform.register_role(r)

        rng = random.Random(42)
        managers = []

        def _make_emp(eid, name, role_id, dept, mgr_id, tenure, perf, eng, comp, comp_ratio, promo, workload, skill_ids):
            hire = datetime.utcnow() - timedelta(days=int(tenure * 365))
            skills = [
                SkillAssessment(sid, sid, rng.uniform(0.5, 1.0), hire, "system")
                for sid in skill_ids
            ]
            return Employee(
                id=eid, name=name, role_id=role_id, department=dept,
                manager_id=mgr_id, hire_date=hire, location="US",
                employment_status=EmploymentStatus.ACTIVE,
                skills=skills, compensation=comp, market_comp_ratio=comp_ratio,
                performance_score=perf, engagement_score=eng,
                promotion_velocity=promo, workload_index=workload,
            )

        emp_mgr = _make_emp("mgr1", "Alice Chen", "eng_mgr", "Engineering", None,
                             8.0, 4.5, 0.85, 160000, 1.05, 0.1, 0.5,
                             ["leadership", "communication", "agile", "python"])
        platform.register_employee(emp_mgr)

        employees_data = [
            ("e001", "Bob Smith",    "swe_l3", "Engineering",  "mgr1", 1.2, 3.5, 0.70, 95000,  0.88, -0.1, 0.60, ["python", "sql", "docker"]),
            ("e002", "Carol Diaz",   "swe_l4", "Engineering",  "mgr1", 3.5, 4.2, 0.80, 125000, 1.02,  0.1, 0.55, ["python", "sql", "docker", "aws"]),
            ("e003", "David Park",   "ml_eng_l4", "Data Science", "mgr1", 2.1, 4.8, 0.90, 140000, 0.95, 0.2, 0.45, ["python", "ml", "nlp", "aws"]),
            ("e004", "Eva Johnson",  "swe_l3", "Engineering",  "mgr1", 0.8, 2.8, 0.45, 88000,  0.82, -0.3, 0.85, ["python", "docker"]),
            ("e005", "Frank Liu",    "data_analyst", "Analytics", "mgr1", 4.2, 3.9, 0.75, 92000, 1.01,  0.0, 0.60, ["sql", "data_analysis"]),
        ]
        for row in employees_data:
            platform.register_employee(_make_emp(*row))

        # Learning activities
        for eid in ["e001", "e002", "e003"]:
            la = LearningActivity(
                id=str(uuid.uuid4()), employee_id=eid,
                content_id="c002", content_title="Machine Learning Foundations",
                skill_ids=["ml"], completion_date=datetime.utcnow() - timedelta(days=30),
                progress=1.0, outcome_score=4.2,
            )
            platform.add_learning_activity(la)

        platform.compute_features()
        logger.info("Demo platform ready: %d employees, %d roles", len(platform.employees), len(platform.roles))
        return platform


def main() -> None:
    platform = AIHCMPlatform.create_demo()

    logger.info("=== Demo: Employee Assistance Agent ===")
    result = platform.employee_agent.handle_query(
        session_id="demo-session-1",
        query="How many days of parental leave am I entitled to?",
        employee=platform.get_employee("e001"),
    )
    logger.info("Chat response: %s", result["response"][:200])

    logger.info("=== Demo: Attrition Risk ===")
    for eid in ["e004", "e003"]:
        fv = platform.feature_store.get(eid)
        if fv:
            pred = platform.attrition_predictor.predict(eid, fv.to_array())
            xpl = platform.xai_explainer.explain_attrition_risk(pred)
            logger.info("Employee %s | Risk: %.0f%% (%s) | %s", eid, pred.risk_score * 100, pred.risk_level.value, xpl[:100])

    logger.info("=== Demo: Team Health Report ===")
    report = platform.manager_agent.team_health_report("mgr1", platform.get_direct_reports("mgr1"))
    logger.info("Team size: %d | Avg perf: %.1f | High-risk: %d",
                report["team_size"], report["average_performance"], report["high_attrition_risk_count"])

    logger.info("=== Demo: Career Path ===")
    emp = platform.get_employee("e001")
    career = platform.talent_agent.career_path_recommendation(emp, list(platform.roles.values()))
    logger.info("Top role for %s: %s (score %.2f)",
                emp.name,
                career.recommended_roles[0]["role_title"] if career.recommended_roles else "N/A",
                career.recommended_roles[0]["combined_score"] if career.recommended_roles else 0)

    logger.info("=== Demo: Anomaly Detection ===")
    alerts = platform.anomaly_detector.detect_department_engagement_anomaly(
        list(platform.employees.values())
    )
    logger.info("Engagement anomalies detected: %d", len(alerts))

    logger.info("=== Demo: Governance Summary ===")
    logger.info(platform.audit_logger.summary())

    try:
        from .layer1_experience.dashboard_api import create_app
        import uvicorn
        app = create_app(platform)
        logger.info("Starting REST API on http://%s:%d", config.api_host, config.api_port)
        uvicorn.run(app, host=config.api_host, port=config.api_port, log_level="info")
    except ImportError:
        logger.info("uvicorn / fastapi not installed — skipping REST API startup.")


if __name__ == "__main__":
    main()
