#!/usr/bin/env python3
"""
Test rapide pour vérifier l'API sans arrêter le serveur
"""
import requests

def quick_api_test():
    try:
        response = requests.get("http://127.0.0.1:8001/ml/health", timeout=5)
        if response.status_code == 200:
            print("✅ API accessible")
            return True
        else:
            print(f"❌ API erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test rapide de l'API")
    if quick_api_test():
        print("🎯 L'API fonctionne. Prêt pour les tests ML.")
    else:
        print("⚠️ Problème d'API à résoudre d'abord.")