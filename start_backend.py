#!/usr/bin/env python3#!/usr/bin/env python3

"""# -*- coding: utf-8 -*-

Script de démarrage pour l'API IFRS17 Backend"""

"""Script de démarrage pour l'API IFRS17 avec authentification

"""

import sys

import osimport sys

import os

# Ajouter le répertoire parent au path Python

current_dir = os.path.dirname(os.path.abspath(__file__))# Ajouter le répertoire racine au path

parent_dir = os.path.dirname(current_dir)sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, parent_dir)

try:

# Maintenant importer l'application    from backend.main import start_server

from backend.main import app    print("🚀 Démarrage de l'API IFRS17...")

    start_server()

if __name__ == "__main__":except Exception as e:

    import uvicorn    print(f"❌ Erreur lors du démarrage: {e}")

    uvicorn.run(    sys.exit(1)
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[parent_dir]
    )