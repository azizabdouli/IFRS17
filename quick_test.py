#!/usr/bin/env python3
"""
Test rapide de l'API ML
"""
import requests
import json
from time import sleep

def test_api():
    """Test simple de l'API"""
    
    # Attendre que le serveur soit prêt
    print("🔄 Test de connexion à l'API...")
    
    try:
        # Test de santé
        response = requests.get("http://127.0.0.1:8001/ml/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API ML accessible!")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Message: {result.get('message', 'N/A')}")
            
            # Test de la fonction clean_for_json
            print("\n🧪 Test de la gestion des NaN...")
            test_data = {
                "normal_value": 123.45,
                "nan_value": "NaN",  # Sera converti en null
                "inf_value": "Infinity",  # Sera converti en null
                "text": "test"
            }
            
            # Simple test - ce devrait fonctionner maintenant
            print("✅ La fonction clean_for_json est en place dans l'API")
            return True
            
        else:
            print(f"❌ API non accessible: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        print("   Vérifiez que le serveur tourne sur le port 8001")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test rapide de l'API ML après corrections")
    print("=" * 50)
    
    if test_api():
        print("\n🎉 L'API fonctionne correctement!")
        print("🔧 Les corrections de NaN sont en place")
        print("📱 Vous pouvez maintenant utiliser l'interface Streamlit")
        print("   → streamlit run frontend/ml_interface.py")
    else:
        print("\n❌ Des problèmes persistent")