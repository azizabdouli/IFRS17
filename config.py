# Configuration du système ML IFRS17 Optimisé v2.0

import os
from pathlib import Path

# ==== 📁 Chemins du projet ====
PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT / "Data"
MODELS_FOLDER = PROJECT_ROOT / "models"

# ==== 🌐 API Configuration Optimisée ====
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8001"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
API_WORKERS = int(os.getenv("API_WORKERS", "4"))  # Multi-workers pour performance

# ==== 🎨 Streamlit Configuration Optimisée ====
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8504"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "127.0.0.1")
STREAMLIT_CACHE_TTL = int(os.getenv("STREAMLIT_CACHE_TTL", "600"))  # 10 minutes

# ==== 🤖 ML Configuration Optimisée ====
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "xgboost")
DEFAULT_N_CLUSTERS = int(os.getenv("DEFAULT_N_CLUSTERS", "5"))
DEFAULT_CONTAMINATION = float(os.getenv("DEFAULT_CONTAMINATION", "0.1"))

# Configuration Cache ML
ML_CACHE_CONFIG = {
    "model_cache_ttl": int(os.getenv("MODEL_CACHE_TTL", "3600")),  # 1 heure
    "data_cache_ttl": int(os.getenv("DATA_CACHE_TTL", "1800")),   # 30 minutes
    "max_cache_size": int(os.getenv("MAX_CACHE_SIZE", "128")),
    "enable_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
}

# ==== 📊 Data Configuration Optimisée ====
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(200 * 1024 * 1024)))  # 200MB
SUPPORTED_FORMATS = [".xlsx", ".csv", ".xls"]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50000"))  # Taille des chunks pour gros fichiers
ENABLE_ASYNC_PROCESSING = os.getenv("ENABLE_ASYNC", "true").lower() == "true"

# ==== ⚡ Performance Configuration ====
PERFORMANCE_CONFIG = {
    "max_workers": int(os.getenv("MAX_WORKERS", "4")),
    "memory_limit_mb": int(os.getenv("MEMORY_LIMIT_MB", "2048")),  # 2GB
    "enable_gc_optimization": os.getenv("ENABLE_GC_OPT", "true").lower() == "true",
    "lazy_loading": os.getenv("LAZY_LOADING", "true").lower() == "true"
}

# ==== 📝 Logging Configuration Optimisée ====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "ifrs17.log"
ENABLE_PERFORMANCE_LOGGING = os.getenv("PERF_LOGGING", "true").lower() == "true"

# ==== 🗄️ Database Configuration ====
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/ifrs17.db")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))

# ==== 🔒 Security Configuration ====
SECRET_KEY = os.getenv("SECRET_KEY", "ifrs17-optimized-secret-key-change-in-production")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ==== 🧪 Development Configuration ====
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
VERSION = "2.0.0-optimized"

# ==== 📈 Monitoring Configuration ====
MONITORING_CONFIG = {
    "enabled": os.getenv("MONITORING_ENABLED", "true").lower() == "true",
    "metrics_endpoint": "/metrics",
    "health_endpoint": "/health",
    "cache_stats_endpoint": "/cache/stats"
}

# ==== 🎯 Model Performance Targets ====
MODEL_TARGETS = {
    "profitability": {"r2_score": 0.95, "cache_enabled": True},
    "risk_classification": {"accuracy": 0.85, "cache_enabled": True}, 
    "claims_prediction": {"r2_score": 0.75, "cache_enabled": True},
    "lrc_prediction": {"r2_score": 0.90, "cache_enabled": True}
}

def get_config_summary():
    """Résumé de la configuration pour debugging"""
    return {
        "version": VERSION,
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "api_port": API_PORT,
        "streamlit_port": STREAMLIT_PORT,
        "cache_enabled": ML_CACHE_CONFIG["model_cache_ttl"] > 0,
        "async_enabled": ENABLE_ASYNC_PROCESSING,
        "performance_mode": "optimized"
    }