#!/usr/bin/env python3
"""
Script de démarrage du système ML IFRS17
Lance automatiquement l'API et l'interface utilisateur
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading

def check_dependencies():
    """Vérification des dépendances"""
    print("🔍 Vérification des dépendances...")
    
    try:
        import uvicorn
        import streamlit
        import fastapi
        import pandas
        import numpy
        import sklearn
        import xgboost
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("💡 Installez les dépendances avec: pip install -r requirements.txt")
        return False

def start_api():
    """Démarrer l'API FastAPI"""
    print("🚀 Démarrage de l'API FastAPI...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8001", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API arrêtée")
    except Exception as e:
        print(f"❌ Erreur API: {e}")

def start_streamlit():
    """Démarrer l'interface Streamlit"""
    print("🖥️ Démarrage de l'interface Streamlit...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "frontend/ml_interface.py", 
            "--server.port", "8504",
            "--server.headless", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Interface arrêtée")
    except Exception as e:
        print(f"❌ Erreur Streamlit: {e}")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🎯 SYSTÈME ML IFRS17 - DÉMARRAGE")
    print("=" * 60)
    
    # Vérification des dépendances
    if not check_dependencies():
        sys.exit(1)
    
    print("\n📋 Services à démarrer:")
    print("1. API FastAPI (port 8001)")
    print("2. Interface ML Streamlit (port 8504)")
    
    choice = input("\n🔧 Que voulez-vous démarrer? (1=API, 2=Interface, 3=Les deux): ")
    
    if choice == "1":
        start_api()
    elif choice == "2":
        start_streamlit()
    elif choice == "3":
        print("\n🚀 Démarrage des deux services...")
        
        # Démarrer l'API en arrière-plan
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        
        # Attendre un peu avant de démarrer Streamlit
        time.sleep(3)
        
        # Ouvrir les URLs dans le navigateur
        print("\n🌐 Ouverture des interfaces...")
        webbrowser.open("http://127.0.0.1:8001/docs")  # API docs
        webbrowser.open("http://127.0.0.1:8504")       # Streamlit
        
        # Démarrer Streamlit (bloquant)
        start_streamlit()
    else:
        print("❌ Choix invalide")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du système")
        print("✅ Merci d'avoir utilisé le système ML IFRS17!")
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        sys.exit(1)