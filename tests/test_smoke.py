"""
Smoke test suite for the AI-HCM Platform.
Validates all 6 architectural layers without requiring API keys or external services.
"""

import pytest
import numpy as np
from datetime import datetime


# ------------------------------------------------------------------ #
# Layer 4 — Data Intelligence
# ------------------------------------------------------------------ #

def test_knowledge_graph_basic():
    from src.common.models import Employee, EmploymentStatus, Skill, SkillAssessment, Role
    from src.layer4_data_intelligence.knowledge_graph import WorkforceKnowledgeGraph

    kg = WorkforceKnowledgeGraph()
    kg.add_skill(Skill("python", "Python", "programming"))
    kg.add_skill(Skill("ml", "Machine Learning", "ai_ml"))

    role = Role("swe", "Software Engineer", "Eng", 3, ["python", "ml"], [])
    kg.add_role(role)

    emp = Employee(
        id="emp1", name="Alice", role_id="swe", department="Eng",
        manager_id=None, hire_date=datetime(2022, 1, 1), location="US",
        employment_status=EmploymentStatus.ACTIVE,
        skills=[SkillAssessment("python", "Python", 0.8, datetime.now(), "system")],
    )
    kg.add_employee(emp)

    skills = kg.get_employee_skills("emp1")
    assert len(skills) == 1
    assert skills[0]["skill_id"] == "python"

    gap = kg.get_skill_gap("emp1", "swe")
    assert "ml" in gap
    assert "python" not in gap


def test_feature_store():
    from src.common.models import Employee, EmploymentStatus
    from src.layer4_data_intelligence.feature_store import FeatureStore
    from datetime import timedelta

    fs = FeatureStore()
    emp = Employee(
        id="e1", name="Bob", role_id="r1", department="Eng",
        manager_id=None, hire_date=datetime.utcnow() - timedelta(days=365),
        location="US", employment_status=EmploymentStatus.ACTIVE,
        performance_score=3.8, engagement_score=0.75,
        compensation=100000, market_comp_ratio=0.95,
    )
    fv = fs.compute_and_store(emp, [], [])
    assert fv.tenure_years > 0.9
    arr = fv.to_array()
    assert arr.shape == (17,)


def test_streaming_processor_alert():
    from src.common.models import WorkforceSignal
    from src.layer4_data_intelligence.streaming_analytics import WorkforceStreamProcessor

    processor = WorkforceStreamProcessor()
    # Seed with 10 normal signals
    for i in range(10):
        sig = WorkforceSignal("emp1", "engagement_survey", 0.75, datetime.now(), "survey")
        processor.ingest(sig)

    # Inject anomaly
    anomaly = WorkforceSignal("emp1", "engagement_survey", 0.05, datetime.now(), "survey")
    alert = processor.ingest(anomaly)
    assert alert is not None
    assert alert["employee_id"] == "emp1"


# ------------------------------------------------------------------ #
# Layer 3 — AI Services
# ------------------------------------------------------------------ #

def test_attrition_predictor_heuristic():
    from src.layer3_ai_services.predictive_models import AttritionPredictor, FEATURE_NAMES
    from src.common.models import RiskLevel

    predictor = AttritionPredictor()
    # Build low-risk feature vector
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    features = np.zeros(len(FEATURE_NAMES))
    features[idx["market_comp_ratio"]] = 1.05
    features[idx["engagement_score"]] = 0.85
    features[idx["promotion_velocity"]] = 0.1
    features[idx["workload_index"]] = 0.4

    pred = predictor.predict("emp1", features)
    assert 0.0 <= pred.risk_score <= 1.0
    assert pred.risk_level == RiskLevel.LOW

    # High-risk employee
    features[idx["market_comp_ratio"]] = 0.75
    features[idx["engagement_score"]] = 0.20
    features[idx["workload_index"]] = 0.95
    pred_high = predictor.predict("emp2", features)
    assert pred_high.risk_score > pred.risk_score


def test_recommendation_engine():
    from src.common.models import Employee, EmploymentStatus, SkillAssessment, Role
    from src.layer3_ai_services.recommendation_engine import ContentBasedEngine

    skill_index = {"python": 0, "ml": 1, "sql": 2, "leadership": 3}
    engine = ContentBasedEngine(skill_index)

    emp = Employee(
        id="e1", name="Alice", role_id="r1", department="Eng",
        manager_id=None, hire_date=datetime.utcnow(),
        location="US", employment_status=EmploymentStatus.ACTIVE,
        skills=[
            SkillAssessment("python", "Python", 0.9, datetime.now(), "system"),
            SkillAssessment("ml", "ML", 0.8, datetime.now(), "system"),
        ],
    )
    roles = [
        Role("ml_eng", "ML Engineer", "DS", 4, ["python", "ml"], []),
        Role("data_analyst", "Analyst", "Analytics", 3, ["sql"], []),
    ]
    matches = engine.match_employee_to_roles(emp, roles)
    assert matches[0][0].id == "ml_eng"
    assert matches[0][1] > matches[1][1]


def test_nlp_intent_classifier():
    from src.layer3_ai_services.nlp_service import IntentClassifier

    clf = IntentClassifier()
    intent, conf = clf.classify("How many days of parental leave am I entitled to?")
    assert intent == "leave_inquiry"
    assert conf > 0

    intent2, _ = clf.classify("What is my bonus for this year?")
    assert intent2 == "compensation_query"


def test_sentiment_analyzer():
    from src.layer3_ai_services.nlp_service import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    pos = analyzer.analyze("I love my team, great culture and amazing work!")
    assert pos["label"] == "positive"

    neg = analyzer.analyze("I am frustrated and stressed with the poor management")
    assert neg["label"] == "negative"


# ------------------------------------------------------------------ #
# Layer 6 — Governance
# ------------------------------------------------------------------ #

def test_bias_monitor():
    from src.layer6_governance.bias_monitor import BiasMonitor

    monitor = BiasMonitor(di_threshold=0.8)
    predictions = [0.8, 0.3, 0.9, 0.2, 0.75, 0.1, 0.85, 0.6]
    actuals     = [1,   0,   1,   0,   1,    0,   1,   1]
    groups      = ["A", "B", "A", "B", "A",  "B", "A", "B"]

    report = monitor.evaluate("attrition_v1", predictions, actuals, groups)
    assert "A" in report.demographic_parity
    assert "B" in report.demographic_parity


def test_drift_detector():
    from src.layer6_governance.drift_detector import DriftDetector

    detector = DriftDetector(p_threshold=0.05)
    baseline = np.random.normal(0.7, 0.1, 200)
    detector.set_baseline("model1", "engagement_score", baseline)

    # Same distribution — no drift
    current_same = np.random.normal(0.7, 0.1, 100)
    report = detector.detect_feature_drift("model1", "engagement_score", current_same)
    assert not report.drift_detected

    # Shifted distribution — drift expected
    current_shifted = np.random.normal(0.3, 0.1, 100)
    report_shifted = detector.detect_feature_drift("model1", "engagement_score", current_shifted)
    assert report_shifted.drift_detected


# ------------------------------------------------------------------ #
# Full platform smoke test
# ------------------------------------------------------------------ #

def test_platform_demo():
    from src.main import AIHCMPlatform

    platform = AIHCMPlatform.create_demo()

    assert len(platform.employees) == 6
    assert len(platform.roles) == 6

    # Employee agent
    result = platform.employee_agent.handle_query(
        "test-session", "What are my leave benefits?",
        platform.get_employee("e001"),
    )
    assert "response" in result
    assert len(result["response"]) > 10

    # Team report
    reports = platform.get_direct_reports("mgr1")
    assert len(reports) == 5
    report = platform.manager_agent.team_health_report("mgr1", reports)
    assert "team_size" in report
    assert report["team_size"] == 5

    # Audit trail
    summary = platform.audit_logger.summary()
    assert summary["total_records"] >= 1


# ------------------------------------------------------------------ #
# New components: AbsenteeismPredictor, PromotionLikelihoodEstimator
# ------------------------------------------------------------------ #

def test_absenteeism_predictor():
    from src.layer3_ai_services.predictive_models import AbsenteeismPredictor, FEATURE_NAMES

    predictor = AbsenteeismPredictor()
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    features = np.zeros(len(FEATURE_NAMES))

    # Low-risk employee
    features[idx["workload_index"]] = 0.3
    features[idx["engagement_score"]] = 0.80
    result = predictor.predict("emp1", features)
    assert 0.0 <= result["absenteeism_risk"] <= 1.0
    assert result["risk_level"] in ("low", "medium", "high", "critical")

    # High-risk employee
    features[idx["workload_index"]] = 0.95
    features[idx["engagement_score"]] = 0.20
    features[idx["engagement_trend"]] = -0.5
    result_high = predictor.predict("emp2", features)
    assert result_high["absenteeism_risk"] > result["absenteeism_risk"]


def test_promotion_likelihood_estimator():
    from src.layer3_ai_services.predictive_models import PromotionLikelihoodEstimator, FEATURE_NAMES

    estimator = PromotionLikelihoodEstimator()
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    features = np.zeros(len(FEATURE_NAMES))

    # Strong candidate
    features[idx["performance_score"]] = 4.5
    features[idx["performance_trend"]] = 0.4
    features[idx["skill_depth"]] = 0.8
    features[idx["learning_completion_rate"]] = 0.9
    features[idx["tenure_years"]] = 2.0
    result = estimator.predict("emp1", features)
    assert result["promotion_likelihood"] >= 0.5
    assert result["readiness"] in ("ready", "developing", "early")


# ------------------------------------------------------------------ #
# WorkforceRiskMonitor
# ------------------------------------------------------------------ #

def test_workforce_risk_monitor():
    from src.main import AIHCMPlatform

    platform = AIHCMPlatform.create_demo()
    alerts = platform.risk_monitor.run_full_scan(list(platform.employees.values()))
    # Should produce at least some alerts on the synthetic demo data
    assert isinstance(alerts, list)
    summary = platform.risk_monitor.alert_summary()
    assert "total_active" in summary
    assert "critical" in summary


# ------------------------------------------------------------------ #
# ConversationalInterface
# ------------------------------------------------------------------ #

def test_conversational_interface():
    from src.main import AIHCMPlatform

    platform = AIHCMPlatform.create_demo()
    ci = platform.conversational_interface
    emp = platform.get_employee("e001")

    result = ci.handle_message(
        session_id="conv-test-1",
        message="What are my leave entitlements?",
        employee=emp,
        role="employee",
    )
    assert "response" in result
    assert "intent" in result
    assert len(result["response"]) > 10
    assert ci.session_count() >= 1


# ------------------------------------------------------------------ #
# MLOps Tracker
# ------------------------------------------------------------------ #

def test_mlops_tracker():
    from src.layer6_governance.mlops_tracker import MLOpsTracker

    tracker = MLOpsTracker()  # no MLflow URI — in-memory mode

    record = tracker.register_model(
        model_id="attrition_v1",
        model_type="attrition",
        version="1.0",
        metrics={"auc_mean": 0.82, "auc_std": 0.03},
        params={"n_estimators": 200, "learning_rate": 0.05},
    )
    assert record.status == "staged"

    result = tracker.validate_and_promote("attrition_v1")
    assert result["promoted"] is True
    assert result["status"] == "production"

    # Failing model
    tracker.register_model(
        model_id="attrition_v2",
        model_type="attrition",
        version="2.0",
        metrics={"auc_mean": 0.60},
        params={},
    )
    result2 = tracker.validate_and_promote("attrition_v2")
    assert result2["promoted"] is False
