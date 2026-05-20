# Technical Design Document
## AI-Native Human Capital Management (AI-HCM) Platform

**Paper Reference:** Ranveer Potel, "Artificial Intelligence in Human Capital Management: A Comprehensive Framework for Intelligent Workforce Systems," *International Journal of AI, Big Data, Computational and Management Studies* (IJAIBDCMS), Vol. 4, Issue 4, pp. 147–174, 2023. DOI: 10.63282/3050-9416.IJAIBDCMS-V4I4P116

---

## 1. Overview

This platform is a full Python implementation of the AI-Native HCM reference architecture proposed in the paper. It operationalizes the **Data → Intelligence → Action → Continuous Learning** operating cycle (Figure 3) across a 6-layer architecture, five intelligent agent types, and a complete MLOps governance framework.

### 1.1 System Goals

| Goal | Paper Reference | Implementation |
|---|---|---|
| Predict employee attrition before resignation | §6.2, §11.3 | `AttritionPredictor` + SHAP |
| Enable conversational HR self-service | §5.2, §14.2 | `EmployeeAssistanceAgent` + Claude API |
| Surface actionable team intelligence for managers | §5.3, §10.3 | `ManagerialDecisionAgent` |
| Identify skill gaps and career paths | §5.4, §7.1 | `TalentIntelligenceAgent` + Knowledge Graph |
| Semantic candidate matching with bias monitoring | §5.5, §12.2 | `RecruitmentAgent` + `BiasMonitor` |
| Personalized learning path generation | §5.6, §6.4 | `LearningAgent` + Hybrid Recommender |
| Detect model drift and ensure fairness | §9.3, §9.5 | `DriftDetector` + `XAIExplainer` |

---

## 2. Architecture Decision: Tech Stack

### 2.1 Stack Selection

**Chosen: Python-native ML Stack**

| Concern | Chosen Approach | Rationale |
|---|---|---|
| ML modeling | scikit-learn + XGBoost + LightGBM | Paper §6.2 explicitly names GBMs as enterprise-dominant; interpretable, tabular-data-native |
| Conversational AI | Anthropic Claude API (claude-sonnet-4-6) | Highest quality RAG + reasoning; enterprise-ready; matches paper's LLM requirement §14.3 |
| Knowledge graph | NetworkX (dev) / Neo4j (prod) | Paper §7.1 names Neo4j/Amazon Neptune; NetworkX enables testing without external deps |
| Vector store | ChromaDB | Open-source, embeds natively; enables RAG without cloud lock-in |
| REST API | FastAPI | Async, auto-OpenAPI docs, Pydantic validation |
| MLOps | MLflow | Open-source; paper §9.2 recommends MLflow/Kubeflow class platforms |
| Streaming | asyncio + Kafka-python | Paper §7.3 recommends Kafka/Flink; asyncio enables dev without external broker |
| Explainability | SHAP | Paper §6.2 identifies SHAP as mandatory for HR acceptance |
| Time-series | statsmodels ARIMA | Paper §6.2 names ARIMA/Prophet for temporal demand forecasting |
| Containerization | Docker + docker-compose | Full stack including Kafka, Neo4j, MLflow, PostgreSQL |

### 2.2 Alternative Considered: TypeScript / Node.js Stack

| Criterion | Python Stack | Node.js Stack | Winner |
|---|---|---|---|
| ML/AI native support | Excellent (sklearn, XGBoost, SHAP) | Poor (no mature tabular ML) | Python |
| Conversational AI | Anthropic SDK (official) | Anthropic SDK (official) | Tie |
| Knowledge graph | NetworkX + Neo4j drivers | Neo4j JS driver only | Python |
| REST API maturity | FastAPI (excellent) | Express/NestJS (excellent) | Tie |
| Data pipeline | pandas, numpy, statsmodels | Limited | Python |
| Enterprise HR ecosystem | Dominant (Workday, SAP integrations in Python) | Emerging | Python |
| Type safety | Pydantic v2 + dataclasses | TypeScript native | Tie |

**Verdict: Python wins decisively** for an AI-first HCM platform due to the ML ecosystem dominance, SHAP/statsmodels availability, and the paper's explicit recommendation of GBM models and ARIMA forecasting.

---

## 3. Six-Layer Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — EXPERIENCE                                                    │
│  FastAPI REST endpoints · Employee chat (Claude RAG) · Manager dashboard │
│  src/layer1_experience/dashboard_api.py                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — INTELLIGENT AGENTS                                            │
│  EmployeeAssistanceAgent  ManagerialDecisionAgent  TalentIntelligence    │
│  RecruitmentAgent         LearningAgent                                  │
│  src/layer2_agents/                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — AI SERVICES                                                   │
│  AttritionPredictor · PerformancePredictor · DemandForecaster            │
│  HybridRecommender (CB + CF + ONA) · NLP Service · AnomalyDetector      │
│  src/layer3_ai_services/                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — DATA INTELLIGENCE                                             │
│  WorkforceKnowledgeGraph · FeatureStore · WorkforceStreamProcessor       │
│  src/layer4_data_intelligence/                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 5 — INTEGRATION                                                   │
│  IntegrationHub (event bus) · PayrollAdapter · CollaborationAdapter      │
│  FinanceAdapter · IdentityManagementAdapter                              │
│  src/layer5_integration/                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 6 — GOVERNANCE & SECURITY                                         │
│  AuditLogger · BiasMonitor (DI/EOpp/DP) · XAIExplainer · DriftDetector  │
│  src/layer6_governance/                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Workforce Knowledge Graph (§7.1)

**File:** `src/layer4_data_intelligence/knowledge_graph.py`

Implements the workforce knowledge graph using NetworkX `DiGraph`. Nodes represent entity types; edges represent typed relationships.

**Entity Types:**

| Node Type | Key Attributes |
|---|---|
| Employee | tenure, performance_score, engagement_score, market_comp_ratio |
| Skill | category, name |
| Role | grade_level, required_skills, career_paths |
| Project | status, outcome_score, team_members |
| LearningActivity | progress, outcome_score, skill_ids |

**Relationship Types:**

| Relation | Source → Target | Properties |
|---|---|---|
| `HAS_SKILL` | Employee → Skill | proficiency (0–1), source |
| `HOLDS_ROLE` | Employee → Role | — |
| `REPORTS_TO` | Employee → Employee | — |
| `WORKED_ON` | Employee → Project | — |
| `LEARNING_PROGRESS` | Employee → LearningActivity | progress |
| `REQUIRES_SKILL` | Role → Skill | — |
| `LEADS_TO` | Role → Role | — |
| `NEEDS_SKILL` | Project → Skill | — |
| `SIMILAR_TO` | Skill → Skill | similarity weight |

**Key Queries:**
- `get_skill_gap(employee_id, role_id)` — drives career recommendations
- `get_team_skill_coverage(manager_id)` — drives team health reporting
- `get_career_path_roles(role_id, depth)` — depth-first traversal for career ladder

### 4.2 Feature Store (§7.2)

**File:** `src/layer4_data_intelligence/feature_store.py`

Pre-computes 17-dimensional `EmployeeFeatureVector` per employee. Features are shared across all ML models to ensure consistent representations.

**Feature Vector (17 dimensions):**

```
tenure_years, tenure_zscore, performance_score, performance_trend,
engagement_score, engagement_trend, compensation, market_comp_ratio,
promotion_velocity, skill_breadth, skill_depth, learning_hours_90d,
learning_completion_rate, workload_index, collaboration_density,
manager_tenure_gap, team_attrition_rate
```

`performance_trend` and `engagement_trend` are computed as linear regression slopes over historical quarter-by-quarter values — capturing **career velocity**, not just point-in-time snapshots (§7.2).

### 4.3 Predictive ML Models (§6.2)

**File:** `src/layer3_ai_services/predictive_models.py`

**AttritionPredictor:**
- Model: `GradientBoostingClassifier` (XGBoost class, n_estimators=200, lr=0.05)
- Explainability: SHAP proxy via `feature_importances_ × sign(feature_value)`
- Heuristic fallback (untrained): weighted rule engine using comp_ratio, engagement, workload
- Outputs: risk_score (0–1), RiskLevel enum, top_factors, interventions, SHAP values

**PerformancePredictor:**
- Model: `GradientBoostingRegressor` (n_estimators=150)
- Outputs: predicted_score (0–5), confidence, growth_trajectory, skill_gaps

**DemandForecaster:**
- Primary: `statsmodels ARIMA(1,1,1)` for time-series headcount
- Fallback: Exponential smoothing (α=0.3)

**Training Protocol (§9.2 MLOps):**
```
Raw Workforce Data → Feature Engineering → Model Training →
Validation & Explainability → Prediction Output → Business Action
→ [Continuous retraining feedback loop]
```

### 4.4 Hybrid Recommendation Engine (§6.4)

**File:** `src/layer3_ai_services/recommendation_engine.py`

Combines three methods with tunable weights:

| Component | Weight | Method |
|---|---|---|
| Content-Based (CB) | 0.40 | Cosine similarity on skill feature vectors |
| Collaborative Filter (CF) | 0.35 | Employee-employee similarity from feature matrix |
| Organizational Network Analysis (ONA) | 0.25 | Role centrality scores |

**MentorMatcher:** Pairs employees with mentors using skill complementarity (60%) + department alignment (40%).

### 4.5 Employee Assistance Agent (§5.2, Figure 5)

**File:** `src/layer2_agents/employee_assistance_agent.py`

**6-Stage Functional Flow:**

```
1. Intent Recognition   → IntentClassifier (rule-based NLP, 8 intent classes)
2. Context Retrieval    → KNOWLEDGE_BASE lookup + employee profile
3. Policy Interpretation → Claude API with RAG-grounded system prompt
4. Transaction Orchestration → HR system stubs (extensible via Layer 5)
5. Response Delivery    → Personalized, employee-context-aware answer
6. Continuous Learning  → Feedback store + satisfaction scoring
```

**Conversation Memory:** Sliding window of 10 turns per session.

**Supported Intents:**
`leave_inquiry`, `compensation_query`, `career_development`, `benefits_query`, `performance_review`, `onboarding_help`, `team_management`, `compliance_question`

### 4.6 Governance Framework (§9.3–§9.5, §12)

**Bias Monitor** (`BiasMonitor`):
- Demographic Parity: selection rate per demographic group
- Equal Opportunity: TPR per group
- Disparate Impact: group rate / reference rate (threshold: 0.8, §12.2)
- Triggers `BiasReport` with remediation recommendations

**Drift Detector** (`DriftDetector`):
- Kolmogorov-Smirnov test per feature (p < 0.05 → drift)
- Population Stability Index for prediction distribution (PSI > 0.2 → retrain)
- Baseline stored at training time; compared against production window

**XAI Explainer** (`XAIExplainer`):
- Feature importance summaries (SHAP proxy)
- Plain-language rationale via Claude API
- Counterfactual explanations ("what would reduce this score?")

**Audit Logger** (`AuditLogger`):
- JSONL append-only log of every AI recommendation
- Records: input_summary, output_summary, model_version, human_reviewed, override_applied
- Supports GDPR/CCPA data subject request tracing

---

## 5. Data Flow Diagram

```
External Signals
(Surveys, Collaboration, Payroll)
         │
         ▼
┌─────────────────────┐
│ Layer 5: Integration│  ← PayrollAdapter, CollaborationAdapter
│       Hub           │
└────────┬────────────┘
         │ Events
         ▼
┌─────────────────────┐
│ Layer 4: Streaming  │  ← WorkforceStreamProcessor (sliding window Z-score)
│    Processor        │
└────────┬────────────┘
         │ Signals
         ▼
┌─────────────────────┐
│ Layer 4: Feature    │  ← 17-dim EmployeeFeatureVector (tenure, perf, etc.)
│     Store           │
└────────┬────────────┘
         │ Feature vectors
         ▼
┌─────────────────────┐
│  Layer 3: ML Models │  ← AttritionPredictor, PerformancePredictor
│  + Recommenders     │  ← HybridRecommender, NLP Service
└────────┬────────────┘
         │ Predictions / recommendations
         ▼
┌─────────────────────┐
│  Layer 2: Agents    │  ← EAA, MDA, TIA, Recruitment, Learning
│                     │
└────────┬────────────┘
         │ Actions / responses
         ▼
┌─────────────────────┐
│  Layer 1: API/Chat  │  ← FastAPI endpoints, Claude conversational interface
└────────┬────────────┘
         │ All events logged
         ▼
┌─────────────────────┐
│  Layer 6: Governance│  ← AuditLogger, BiasMonitor, DriftDetector, XAI
└─────────────────────┘
         │ Feedback
         └──────────────► Continuous retraining loop (MLflow)
```

---

## 6. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/chat` | POST | Employee Assistance Agent (RAG + Claude) |
| `/attrition/{employee_id}` | GET | Attrition risk prediction + XAI explanation |
| `/team/{manager_id}` | GET | Team health report + coaching summary |
| `/career/{employee_id}` | GET | Career path recommendations |
| `/learning` | POST | Personalized learning path |
| `/recruitment/match` | POST | Candidate-role compatibility scoring |
| `/governance/bias/{model_id}` | GET | Bias evaluation report |
| `/governance/audit/{employee_id}` | GET | Audit trail for employee |
| `/governance/summary` | GET | Platform-wide governance summary |

---

## 7. MLOps Lifecycle (§9.2)

```
Data Ingestion → Feature Engineering → Model Training →
Validation & Explainability → Deployment → Monitoring →
[Retraining triggered by DriftDetector] → Data Ingestion
```

**Drift Triggers:**
- KS test p-value < 0.05 for any feature
- PSI > 0.20 for prediction distribution
- Accuracy drop > 10% against realized outcomes

**Human-in-the-Loop (§9.4):**
- All HIGH/CRITICAL risk predictions queued for HRBP review
- `AuditLogger.mark_reviewed()` records human override
- Override rate tracked as model quality signal

---

## 8. Deployment Architecture (§15)

### Development
```bash
pip install -r requirements.txt
python -m src.main            # Demo mode
pytest tests/                 # Smoke tests
```

### Production (Docker)
```bash
ANTHROPIC_API_KEY=sk-... docker-compose up
# API:    http://localhost:8000
# Neo4j:  http://localhost:7474
# MLflow: http://localhost:5001
```

### Production Services

| Service | Role | Port |
|---|---|---|
| FastAPI | REST API / Experience Layer | 8000 |
| PostgreSQL 16 | Transactional employee records | 5432 |
| Neo4j 5 | Workforce Knowledge Graph (production) | 7687/7474 |
| MLflow | Model tracking + experiment registry | 5001 |
| Kafka + Zookeeper | Real-time signal streaming | 9092 |

### Hybrid Deployment (§15.2)
Sensitive workforce data (compensation, health, disciplinary) remains on-premise. The cloud layer handles NLP inference and recommendation services. This matches the paper's hybrid deployment pattern for regulated industries.

---

## 9. Ethical & Governance Compliance (§12)

| Requirement | Implementation |
|---|---|
| Algorithmic transparency | XAIExplainer: SHAP + plain-language rationale |
| Bias monitoring | BiasMonitor: DI ≥ 0.8, demographic parity, equal opportunity |
| Privacy (GDPR/CCPA) | Role-based access; data minimization in AuditRecord |
| Audit trail | AuditLogger: JSONL append-only, human review tracking |
| Model drift | DriftDetector: KS test per feature + PSI for predictions |
| Human override | AuditLogger.mark_reviewed() + HITL queue |
| Fairness-aware training | BiasMonitor.evaluate() runs before deployment |

---

## 10. File Structure

```
AI_IN_HCM/
├── src/
│   ├── common/
│   │   ├── config.py              # Environment-driven configuration
│   │   └── models.py              # Shared dataclasses (Employee, Skill, etc.)
│   ├── layer1_experience/
│   │   ├── conversational_interface.py  # (reserved for future UI)
│   │   └── dashboard_api.py       # FastAPI REST endpoints
│   ├── layer2_agents/
│   │   ├── employee_assistance_agent.py  # 6-stage RAG HR assistant
│   │   ├── managerial_decision_agent.py  # Team intelligence + coaching
│   │   ├── talent_intelligence_agent.py  # Career path + succession
│   │   ├── recruitment_agent.py          # Semantic candidate matching
│   │   └── learning_agent.py             # Personalized L&D paths
│   ├── layer3_ai_services/
│   │   ├── predictive_models.py    # AttritionPredictor, PerformancePredictor, DemandForecaster
│   │   ├── recommendation_engine.py # Hybrid CB+CF+ONA recommender
│   │   ├── nlp_service.py          # Intent, sentiment, skill extraction
│   │   └── anomaly_detector.py     # Engagement/workload/attrition anomalies
│   ├── layer4_data_intelligence/
│   │   ├── knowledge_graph.py      # Workforce Knowledge Graph (NetworkX)
│   │   ├── feature_store.py        # 17-dim feature vectors, normalization
│   │   └── streaming_analytics.py  # Real-time sliding-window signal processor
│   ├── layer5_integration/
│   │   └── integration_hub.py      # Event bus + payroll/collab/finance adapters
│   ├── layer6_governance/
│   │   ├── audit_logger.py         # JSONL audit trail
│   │   ├── bias_monitor.py         # DI / equal opportunity / demographic parity
│   │   ├── xai_explainer.py        # SHAP + counterfactual explanations
│   │   └── drift_detector.py       # KS test + PSI drift detection
│   └── main.py                     # Platform orchestrator + demo
├── tests/
│   └── test_smoke.py               # Full platform smoke test (no API keys needed)
├── docs/
│   └── TECHNICAL_DESIGN.md         # This document
├── requirements.txt
├── docker-compose.yml
└── .gitignore
```
