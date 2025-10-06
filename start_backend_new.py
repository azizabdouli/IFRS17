#!/usr/bin/env python3
"""
Script de démarrage pour l'API IFRS17 Backend
"""

import sys
import os

# Ajouter le répertoire courant au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 Démarrage de l'API IFRS17...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[current_dir]
    )