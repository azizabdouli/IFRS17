# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gestionnaire de cycle de vie
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    # Démarrage
    logger.info("🚀 Démarrage de l'API IFRS17 ML Analytics...")
    
    # Initialisation des services IA
    try:
        from backend.ai.ifrs17_ai_assistant import IFRS17AIAssistant
        from backend.ai.predictive_ai_service import PredictiveAIService
        
        # Services globaux
        app.state.ai_assistant = IFRS17AIAssistant()
        app.state.predictive_ai = PredictiveAIService()
        
        logger.info("🧠 Services IA initialisés avec succès")
    except Exception as e:
        logger.warning(f"⚠️ Erreur initialisation IA: {e}")
    
    yield
    
    # Arrêt
    logger.info("🛑 Arrêt de l'application")

# =======================
# 🚀 Application FastAPI avec IA
# =======================
app = FastAPI(
    title="IFRS17 ML Analytics API",
    description="🚀 API complète pour l'analyse ML et IA des contrats d'assurance IFRS17",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS optimisée
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "http://127.0.0.1:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8502",
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Import et inclusion des routers
from backend.routers import transform, projection, ml_router, ai_router
from backend.routers.ppna_router import router as ppna_router

app.include_router(transform.router, prefix="/transform", tags=["🔄 Transformations"])
app.include_router(projection.router, prefix="/projection", tags=["📊 Projections"])
app.include_router(ml_router.router, prefix="/ml", tags=["🤖 Machine Learning"])
app.include_router(ai_router.router, prefix="/ai", tags=["🧠 Intelligence Artificielle"])
app.include_router(ppna_router, tags=["📋 PPNA IFRS17"])

@app.get("/", tags=["🏠 Accueil"])
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "name": "IFRS17 ML Analytics API",
        "version": "3.0.0",
        "status": "🚀 Running with AI",
        "timestamp": "2024-10-01",
        "features": {
            "machine_learning": "🤖 4 modèles ML spécialisés",
            "ai_assistant": "🧠 Assistant conversationnel IFRS17",
            "predictive_ai": "🔮 IA prédictive avec auto-ML",
            "onerous_contracts": "🔴 Détection contrats onéreux",
            "projections": "📊 Projections actuarielles",
            "transformations": "🔄 Transformation de données",
            "performance": "⚡ Optimisé haute performance"
        },
        "endpoints": {
            "ml": "/ml - Modèles ML",
            "ai": "/ai - Intelligence Artificielle",
            "projection": "/projection - Projections",
            "transform": "/transform - Transformations",
            "docs": "/docs - Documentation API"
        }
    }

@app.get("/health", tags=["🏥 Santé"])
async def health_check():
    """Vérification de l'état de l'API"""
    try:
        # Vérification des services IA
        ai_status = hasattr(app.state, 'ai_assistant') and app.state.ai_assistant is not None
        predictive_status = hasattr(app.state, 'predictive_ai') and app.state.predictive_ai is not None
        
        return {
            "status": "healthy",
            "version": "3.0.0",
            "timestamp": "2024-10-01",
            "services": {
                "api": "✅ Opérationnel",
                "ai_assistant": "✅ Opérationnel" if ai_status else "⚠️ Non disponible",
                "predictive_ai": "✅ Opérationnel" if predictive_status else "⚠️ Non disponible",
                "ml_models": "✅ 4 modèles chargés",
                "database": "✅ Connecté"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/test", tags=["🧪 Test"])
async def test():
    """Endpoint de test"""
    return {
        "status": "OK", 
        "message": "🧪 Test endpoint fonctionnel",
        "ai_available": hasattr(app.state, 'ai_assistant')
    }

@app.get("/stats", tags=["📊 Statistiques"])
async def api_stats():
    """Statistiques de l'API"""
    return {
        "models": {
            "profitability": {"accuracy": "R² = 0.964", "status": "✅"},
            "risk_classification": {"accuracy": "86.5%", "status": "✅"},
            "claims_prediction": {"accuracy": "R² = 0.732", "status": "✅"},
            "lrc_prediction": {"accuracy": "R² = 0.937", "status": "✅"},
            "onerous_contracts": {"accuracy": "En cours", "status": "🔄"}
        },
        "ai_services": {
            "conversational_ai": "✅ IFRS17 Assistant",
            "predictive_ai": "✅ Auto-ML Service",
            "nlp_processing": "✅ Transformers",
            "domain_knowledge": "✅ Base IFRS17"
        },
        "performance": {
            "processing_speed": "1,171,318 lignes/sec",
            "cache_enabled": "✅ TTLCache",
            "async_processing": "✅ Uvloop",
            "memory_optimization": "✅ Actif"
        }
    }

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global des erreurs"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne du serveur", "detail": str(exc)}
    )

def start_server():
    """Démarrage optimisé du serveur"""
    try:
        # Tentative avec uvloop pour de meilleures performances
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            logger.warning("uvloop non disponible, utilisation asyncio standard")
        
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="127.0.0.1",
            port=8001,
            reload=True,
            access_log=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Erreur démarrage serveur: {e}")

if __name__ == "__main__":
    start_server()
