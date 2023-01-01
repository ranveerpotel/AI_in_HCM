import os
from dataclasses import dataclass


@dataclass
class Config:
    # Claude / Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Storage
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./ai_hcm.db")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # Vector store (ChromaDB)
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Governance
    bias_threshold_di: float = float(os.getenv("BIAS_THRESHOLD_DI", "0.8"))
    drift_pvalue_threshold: float = float(os.getenv("DRIFT_PVALUE", "0.05"))
    hitl_risk_threshold: float = float(os.getenv("HITL_RISK_THRESHOLD", "0.75"))

    # Attrition thresholds
    attrition_high: float = 0.65
    attrition_medium: float = 0.35


config = Config()
