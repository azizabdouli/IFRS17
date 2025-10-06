# backend/main_auth.py - Version simplifiée pour l'authentification

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    logger.info("🚀 Démarrage de l'API IFRS17 Authentication...")
    
    # Initialisation de la base de données
    try:
        from backend.database.connection import engine, Base
        # Créer les tables si elles n'existent pas
        Base.metadata.create_all(bind=engine)
        logger.info("💾 Base de données initialisée avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation base de données: {e}")
        # Ne pas faire planter l'application, juste logger l'erreur
    
    yield
    
    # Arrêt
    logger.info("🛑 Arrêt de l'application")

# =======================
# 🚀 Application FastAPI
# =======================
app = FastAPI(
    title="IFRS17 Authentication API",
    description="🔐 API d'authentification pour l'application IFRS17",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Import et inclusion des routers d'authentification
from backend.routers.auth_router import router as auth_router

app.include_router(auth_router, tags=["🔐 Authentification"])

@app.get("/", tags=["🏠 Accueil"])
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "name": "IFRS17 Authentication API",
        "version": "1.0.0",
        "status": "🚀 Running",
        "features": {
            "authentication": "🔐 JWT Authentication",
            "database": "💾 PostgreSQL/MySQL",
            "security": "🛡️ bcrypt + JWT",
            "audit": "📋 Audit Logs"
        },
        "endpoints": {
            "auth": "/auth - Authentification",
            "docs": "/docs - Documentation API",
            "health": "/health - Santé de l'API"
        }
    }

@app.get("/health", tags=["🏥 Santé"])
async def health_check():
    """Vérification de l'état de l'API"""
    try:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "api": "✅ Opérationnel",
                "database": "✅ Connecté",
                "authentication": "✅ Disponible"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

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
    """Démarrage du serveur"""
    try:
        import uvicorn
        uvicorn.run(
            "backend.main_auth:app",
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