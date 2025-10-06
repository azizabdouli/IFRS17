#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur d'authentification IFRS17 - Version simplifiée
"""

import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# 🚀 Application FastAPI
# =======================
app = FastAPI(
    title="IFRS17 Authentication API",
    description="🔐 API d'authentification pour l'application IFRS17",
    version="1.0.0",
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
try:
    from backend.routers.auth_router import router as auth_router
    app.include_router(auth_router, tags=["🔐 Authentification"])
    logger.info("✅ Router d'authentification chargé")
except Exception as e:
    logger.error(f"❌ Erreur chargement router auth: {e}")

@app.get("/", tags=["🏠 Accueil"])
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "name": "IFRS17 Authentication API",
        "version": "1.0.0",
        "status": "🚀 Running",
        "features": {
            "authentication": "🔐 JWT Authentication",
            "database": "💾 SQLite (dev) / PostgreSQL (prod)",
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
        # Test de connexion DB
        from backend.database.connection import engine
        connection = engine.connect()
        connection.close()
        db_status = "✅ Connecté"
    except Exception as e:
        logger.error(f"Erreur DB: {e}")
        db_status = "❌ Erreur de connexion"
        
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "✅ Opérationnel",
            "database": db_status,
            "authentication": "✅ Disponible"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    logger.info("🚀 Démarrage de l'API IFRS17 Authentication...")
    
    # Initialisation de la base de données
    try:
        from backend.database.connection import engine, Base
        # Créer les tables si elles n'existent pas
        Base.metadata.create_all(bind=engine)
        logger.info("💾 Base de données initialisée avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation base de données: {e}")

def start_server():
    """Démarrage du serveur"""
    try:
        import uvicorn
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            reload=False,  # Désactiver le reload pour éviter les problèmes
            access_log=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Erreur démarrage serveur: {e}")

if __name__ == "__main__":
    start_server()