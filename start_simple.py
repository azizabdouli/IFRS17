#!/usr/bin/env python3
"""
Script de démarrage simple pour l'API IFRS17
"""
import os
import sys

# Ajouter le répertoire racine au path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🔧 Configuration du serveur...")

try:
    # Import et configuration
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    print("✅ FastAPI importé avec succès")
    
    # Créer l'application FastAPI
    app = FastAPI(
        title="API IFRS17 - BNA",
        description="API pour l'analyse IFRS17 de la BNA",
        version="1.0.0"
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4200"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    print("✅ Configuration CORS appliquée")
    
    @app.get("/")
    async def root():
        return {"message": "API IFRS17 - BNA"}
    
    @app.get("/health")
    async def health():
        return {"status": "OK", "service": "IFRS17 API"}
    
    # Démarrer le serveur
    print("🚀 Démarrage du serveur sur http://localhost:8001")
    
    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            reload=False,
            log_level="info"
        )
        
except Exception as e:
    print(f"❌ Erreur lors du démarrage: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)