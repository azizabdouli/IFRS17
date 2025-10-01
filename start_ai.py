#!/usr/bin/env python3
# start_ai.py

"""
🚀 IFRS17 ML Analytics Platform - Launcher IA
Lanceur optimisé pour l'application avec IA intégrée
"""

import subprocess
import sys
import time
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IFRS17Launcher:
    """Lanceur intelligent de l'application IFRS17"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.backend_dir = self.base_dir / "backend"
        self.frontend_dir = self.base_dir / "frontend"
        
    def check_dependencies(self):
        """Vérification des dépendances"""
        logger.info("🔍 Vérification des dépendances...")
        
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "pandas", 
            "numpy", "scikit-learn", "xgboost", "transformers", "torch"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} - MANQUANT")
        
        if missing_packages:
            logger.error(f"📦 Packages manquants: {missing_packages}")
            logger.info("💡 Exécutez: pip install -r requirements.txt")
            return False
        
        logger.info("✅ Toutes les dépendances sont installées")
        return True
    
    def start_backend(self):
        """Démarrage du serveur backend"""
        logger.info("🚀 Démarrage du serveur backend...")
        
        backend_cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "127.0.0.1",
            "--port", "8001", 
            "--reload"
        ]
        
        try:
            # Démarrage en arrière-plan
            process = subprocess.Popen(
                backend_cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("⏳ Attente démarrage backend...")
            time.sleep(5)
            
            # Vérification que le processus est toujours en vie
            if process.poll() is None:
                logger.info("✅ Backend démarré avec succès sur http://127.0.0.1:8001")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Erreur démarrage backend: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur démarrage backend: {str(e)}")
            return None
    
    def start_frontend(self):
        """Démarrage de l'interface frontend"""
        logger.info("🎨 Démarrage de l'interface frontend...")
        
        frontend_cmd = [
            sys.executable, "-m", "streamlit", "run",
            "main_app.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false"
        ]
        
        try:
            process = subprocess.Popen(
                frontend_cmd,
                cwd=self.frontend_dir
            )
            
            logger.info("⏳ Attente démarrage frontend...")
            time.sleep(3)
            
            if process.poll() is None:
                logger.info("✅ Frontend démarré avec succès sur http://127.0.0.1:8501")
                return process
            else:
                logger.error("❌ Erreur démarrage frontend")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur démarrage frontend: {str(e)}")
            return None
    
    def check_api_health(self):
        """Vérification de la santé de l'API"""
        try:
            import requests
            
            logger.info("🏥 Vérification de la santé de l'API...")
            response = requests.get("http://127.0.0.1:8001/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API en bonne santé - Version: {data.get('version', 'N/A')}")
                
                # Affichage des services
                services = data.get('services', {})
                for service, status in services.items():
                    logger.info(f"   - {service}: {status}")
                
                return True
            else:
                logger.warning(f"⚠️ API répond avec code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Impossible de vérifier l'API: {str(e)}")
            return False
    
    def launch(self):
        """Lancement complet de l'application"""
        logger.info("🏢 Démarrage IFRS17 ML Analytics Platform v3.0.0")
        logger.info("=" * 60)
        
        # 1. Vérification des dépendances
        if not self.check_dependencies():
            logger.error("💥 Arrêt - Dépendances manquantes")
            return False
        
        # 2. Démarrage du backend
        backend_process = self.start_backend()
        if not backend_process:
            logger.error("💥 Arrêt - Impossible de démarrer le backend")
            return False
        
        # 3. Vérification de la santé de l'API
        time.sleep(2)
        self.check_api_health()
        
        # 4. Démarrage du frontend
        frontend_process = self.start_frontend()
        if not frontend_process:
            logger.error("💥 Arrêt - Impossible de démarrer le frontend")
            backend_process.terminate()
            return False
        
        # 5. Instructions utilisateur
        logger.info("=" * 60)
        logger.info("🎉 APPLICATION DÉMARRÉE AVEC SUCCÈS !")
        logger.info("")
        logger.info("📱 Accès à l'application:")
        logger.info("   🌐 Frontend: http://127.0.0.1:8501")
        logger.info("   🔧 API Backend: http://127.0.0.1:8001")
        logger.info("   📖 Documentation API: http://127.0.0.1:8001/docs")
        logger.info("")
        logger.info("🆕 Nouvelles fonctionnalités v3.0.0:")
        logger.info("   🧠 Assistant IA conversationnel IFRS17")
        logger.info("   🔮 IA prédictive avec auto-ML")
        logger.info("   🔴 Détection contrats onéreux")
        logger.info("   ⚡ Performance optimisée (1.17M lignes/sec)")
        logger.info("")
        logger.info("🛑 Pour arrêter: Ctrl+C")
        logger.info("=" * 60)
        
        # 6. Attente et gestion des processus
        try:
            # Attendre l'interruption utilisateur
            while True:
                time.sleep(1)
                
                # Vérifier que les processus sont toujours en vie
                if backend_process.poll() is not None:
                    logger.error("❌ Backend s'est arrêté de façon inattendue")
                    break
                    
                if frontend_process.poll() is not None:
                    logger.error("❌ Frontend s'est arrêté de façon inattendue")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Arrêt demandé par l'utilisateur")
        
        finally:
            # Nettoyage
            logger.info("🧹 Arrêt des services...")
            try:
                frontend_process.terminate()
                backend_process.terminate()
                
                # Attendre l'arrêt complet
                frontend_process.wait(timeout=5)
                backend_process.wait(timeout=5)
                
                logger.info("✅ Services arrêtés proprement")
            except:
                logger.warning("⚠️ Forçage de l'arrêt des services")
                frontend_process.kill()
                backend_process.kill()
        
        return True

def main():
    """Point d'entrée principal"""
    launcher = IFRS17Launcher()
    success = launcher.launch()
    
    if success:
        logger.info("👋 Au revoir !")
    else:
        logger.error("💥 Échec du démarrage de l'application")
        sys.exit(1)

if __name__ == "__main__":
    main()