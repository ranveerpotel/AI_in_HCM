# AI in Human Capital Management
### A Comprehensive Framework for Intelligent Workforce Systems

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20passing-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/DOI-10.63282%2F3050--9416.IJAIBDCMS--V4I4P116-orange.svg)](https://doi.org/10.63282/3050-9416.IJAIBDCMS-V4I4P116)

Full Python implementation of the AI-Native HCM reference architecture from:

> **Ranveer Potel**, "Artificial Intelligence in Human Capital Management: A Comprehensive Framework for Intelligent Workforce Systems," *International Journal of AI, Big Data, Computational and Management Studies (IJAIBDCMS)*, Vol. 4, Issue 4, pp. 147–174, 2023.
> DOI: [10.63282/3050-9416.IJAIBDCMS-V4I4P116](https://doi.org/10.63282/3050-9416.IJAIBDCMS-V4I4P116) · ISSN: 3050-9416

---

## Overview

Human Capital Management systems have historically evolved from administrative record-keeping platforms into enterprise decision systems. This platform operationalises the paper's proposed AI-Native HCM architecture — positioning intelligence as the central operating layer rather than an analytics add-on.

**Core operating cycle (Figure 3 of the paper):**

```
Data  →  Intelligence  →  Action  →  Continuous Learning
```

Every workforce transaction generates data that enriches organisational knowledge; ML models transform this into actionable insights; agents execute decisions on behalf of users; outcomes feed back into model training to improve future recommendations.

---

## Six-Layer Reference Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — EXPERIENCE                                               │
│  REST API (FastAPI)  ·  Conversational Interface  ·  Dashboards     │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — INTELLIGENT AGENTS                                       │
│  Employee Assistance  ·  Managerial Decision  ·  Talent Intelligence│
│  Recruitment Intelligence  ·  Learning & Development  ·  Risk Monitor│
├────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — AI SERVICES                                              │
│  Attrition Predictor (GBM+SHAP)  ·  Performance Predictor          │
│  Absenteeism Predictor  ·  Promotion Estimator  ·  Demand Forecaster│
│  Hybrid Recommender (CB+CF+ONA)  ·  NLP Service  ·  Anomaly Detector│
├────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — DATA INTELLIGENCE                                        │
│  Workforce Knowledge Graph  ·  Feature Store  ·  Streaming Analytics│
├────────────────────────────────────────────────────────────────────┤
│  LAYER 5 — INTEGRATION                                              │
│  Event Hub  ·  Payroll  ·  Identity Mgmt  ·  Collaboration  ·  Finance│
├────────────────────────────────────────────────────────────────────┤
│  LAYER 6 — GOVERNANCE & SECURITY                                    │
│  Audit Logger  ·  Bias Monitor  ·  XAI Explainer                   │
│  Drift Detector  ·  MLOps Tracker                                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

### Predictive Workforce Analytics (§6.2)
| Model | Algorithm | Paper Section |
|---|---|---|
| Attrition Predictor | Gradient Boosting + SHAP | §6.2 |
| Performance Predictor | Gradient Boosting Regressor | §6.2 |
| Absenteeism Predictor | Logistic Regression | §6.2 |
| Promotion Likelihood | Gradient Boosting Classifier | §6.2 |
| Demand Forecaster | ARIMA(1,1,1) + Exponential Smoothing | §6.2 |

### Intelligent Agents (§5.2–5.6)
| Agent | Capability | Paper Section |
|---|---|---|
| Employee Assistance Agent | 6-stage RAG HR self-service (Figure 5) | §5.2 |
| Managerial Decision Agent | Team health, compensation, coaching | §5.3 |
| Talent Intelligence Agent | Skill gap, career paths, succession | §5.4 |
| Recruitment Agent | Semantic matching, diversity analytics | §5.5 |
| Learning & Development Agent | Personalised paths, at-risk learners | §5.6 |
| Workforce Risk Monitor | Population-level alert scanning | §3.1 |

### Responsible AI Governance (§9, §12)
- **Bias Monitor**: Demographic Parity, Equal Opportunity, Disparate Impact (≥0.8 threshold)
- **Drift Detector**: Kolmogorov-Smirnov test + Population Stability Index
- **XAI Explainer**: SHAP-proxy feature importance + Claude plain-language rationale + counterfactuals
- **Audit Logger**: JSONL append-only trail, human review tracking, GDPR/CCPA compliant
- **MLOps Tracker**: Model registration → validation gate → production promotion → retraining trigger

---

## Project Structure

```
AI_in_HCM/
├── src/
│   ├── common/                         # Shared models & config
│   ├── layer1_experience/              # FastAPI REST API, conversational interface
│   ├── layer2_agents/                  # 6 intelligent agents
│   ├── layer3_ai_services/             # ML models, NLP, recommenders, anomaly detection
│   ├── layer4_data_intelligence/       # Knowledge graph, feature store, streaming
│   ├── layer5_integration/             # Enterprise system adapters & event hub
│   ├── layer6_governance/              # Audit, bias, XAI, drift, MLOps
│   └── main.py                         # Platform orchestrator
├── tests/
│   └── test_smoke.py                   # 15 smoke tests (all layers)
├── docs/
│   └── TECHNICAL_DESIGN.md             # Full architecture & design specification
├── Dockerfile
├── docker-compose.yml                  # API + PostgreSQL + Neo4j + MLflow + Kafka
└── requirements.txt
```

---

## Getting Started

### Local (Development)

```bash
git clone https://github.com/ranveerpotel/AI_in_HCM.git
cd AI_in_HCM

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run smoke tests (no API key required)
pytest tests/test_smoke.py -v

# Run full demo
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main
```

### Docker (Full Stack)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker-compose up

# API:     http://localhost:8000/docs
# Neo4j:   http://localhost:7474
# MLflow:  http://localhost:5001
```

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/chat` | POST | Employee Assistance Agent (RAG + Claude) |
| `/attrition/{employee_id}` | GET | Risk score + SHAP explanation |
| `/team/{manager_id}` | GET | Team health report + coaching summary |
| `/career/{employee_id}` | GET | Career path recommendations |
| `/learning` | POST | Personalised learning path |
| `/recruitment/match` | POST | Candidate-role compatibility |
| `/governance/audit/{employee_id}` | GET | Per-employee audit trail |
| `/governance/summary` | GET | Platform governance dashboard |

Full OpenAPI docs available at `http://localhost:8000/docs` when running.

---

## AI Capability Maturity (§2.3, Table 1)

This implementation targets **Level 3–4** of the paper's five-level maturity model:

| Level | Capability | Status |
|---|---|---|
| 1 | Automation | ✅ Implemented |
| 2 | Assisted Analytics | ✅ Implemented |
| 3 | Predictive Intelligence | ✅ Implemented |
| 4 | Decision Augmentation | ✅ Implemented |
| 5 | Autonomous Workforce Systems | Partially (risk monitor) |

---

## Tech Stack Decision

Python was chosen over a TypeScript/Node.js stack for this implementation:

| Concern | Rationale |
|---|---|
| ML modeling | Paper §6.2 names GBM (XGBoost/LightGBM) and ARIMA — Python-native |
| Explainability | SHAP library mandatory per §9.5 — Python-native |
| Knowledge graph | NetworkX (dev) → Neo4j (prod) |
| Conversational AI | Anthropic Claude API (claude-sonnet-4-6) |
| REST API | FastAPI — async, Pydantic validation, auto OpenAPI docs |
| MLOps | MLflow — paper §9.2 recommendation |

---

## Paper Reference

```bibtex
@article{potel2023aihcm,
  author    = {Ranveer Potel},
  title     = {Artificial Intelligence in Human Capital Management:
               A Comprehensive Framework for Intelligent Workforce Systems},
  journal   = {International Journal of AI, Big Data, Computational
               and Management Studies},
  volume    = {4},
  number    = {4},
  pages     = {147--174},
  year      = {2023},
  issn      = {3050-9416},
  doi       = {10.63282/3050-9416.IJAIBDCMS-V4I4P116},
  publisher = {Noble Scholar Research Group}
}
```

**Direct link:** [https://doi.org/10.63282/3050-9416.IJAIBDCMS-V4I4P116](https://doi.org/10.63282/3050-9416.IJAIBDCMS-V4I4P116)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Independent Research, USA · Ranveer Potel*
