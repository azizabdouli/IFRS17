#!/usr/bin/env python3
"""
Script de démarrage pour l'API IFRS17 Backend - Version Directe
"""

import sys
import os

# Ajouter le répertoire courant au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 Démarrage de l'API IFRS17...")

# Import direct de l'application
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    
    # Configuration pour éviter les problèmes de multiprocessing
    uvicorn.run(
        app,  # Utiliser l'objet app directement
        host="0.0.0.0",
        port=8001,
        reload=False,  # Désactiver le reload pour éviter les problèmes
        access_log=True,
        log_level="info"
    )