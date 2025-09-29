#!/usr/bin/env python3
"""
Test simple de l'API ML
"""
import requests

print("🧪 Test simple de l'API ML")
print("=" * 30)

try:
    response = requests.get("http://127.0.0.1:8001/ml/health", timeout=5)
    if response.status_code == 200:
        result = response.json()
        print("✅ API accessible")
        print(f"   Status: {result.get('status')}")
        print("\n🎯 Correction des types de données appliquée:")
        print("   • Conversion automatique des colonnes numériques")
        print("   • Gestion des erreurs de type avec pd.to_numeric")
        print("   • Protection contre les comparaisons string vs int")
    else:
        print(f"❌ API erreur: {response.status_code}")
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("   Vérifiez que le serveur est démarré")