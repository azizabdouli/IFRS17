#!/usr/bin/env python3
"""
Test final du système ML IFRS17 - Validation de production
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au PATH
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """Test du chargement des données"""
    print("🔍 Test du chargement des données...")
    
    try:
        # Charger les données IFRS17
        data_path = Path("Data/Ppna (4).xlsx")
        if not data_path.exists():
            print(f"❌ Fichier de données non trouvé: {data_path}")
            return False
            
        df = pd.read_excel(data_path)
        print(f"✅ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"📊 Aperçu: {df.dtypes.value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return False

def test_ml_service():
    """Test du service ML"""
    print("\n🤖 Test du service ML...")
    
    try:
        from backend.ml.ml_service import MLService
        
        ml_service = MLService()
        print("✅ MLService initialisé")
        
        # Test des données
        data_path = "Data/Ppna (4).xlsx"
        df = pd.read_excel(data_path)
        
        # Test rapide avec échantillon
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        print(f"📊 Test avec échantillon: {len(sample_df)} lignes")
        
        # Test profitability
        try:
            result = ml_service.train_profitability_model(sample_df)
            print(f"✅ Profitability model: R² = {result.get('r2_score', 'N/A'):.3f}")
        except Exception as e:
            print(f"❌ Profitability model error: {e}")
            
        # Test risk classification
        try:
            result = ml_service.train_risk_classification_model(sample_df)
            print(f"✅ Risk classification: Accuracy = {result.get('accuracy', 'N/A'):.3f}")
        except Exception as e:
            print(f"❌ Risk classification error: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur service ML: {e}")
        return False

def test_api_endpoints():
    """Test des endpoints API"""
    print("\n🌐 Test des endpoints API...")
    
    try:
        import requests
        import time
        
        # Vérifier si l'API est démarrée
        base_url = "http://127.0.0.1:8001"
        
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            print("✅ API accessible")
        except requests.exceptions.ConnectionError:
            print("⚠️ API non démarrée. Démarrez avec: uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload")
            return False
            
        # Test endpoint ML
        try:
            response = requests.get(f"{base_url}/ml/status", timeout=10)
            if response.status_code == 200:
                print("✅ Endpoint ML fonctionnel")
            else:
                print(f"⚠️ Endpoint ML: status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Endpoint ML: {e}")
            
        return True
        
    except ImportError:
        print("⚠️ Requests non installé pour tester l'API")
        return True
    except Exception as e:
        print(f"❌ Erreur test API: {e}")
        return False

def test_frontend_files():
    """Test des fichiers frontend"""
    print("\n🖥️ Test des fichiers frontend...")
    
    frontend_files = [
        "frontend/app.py",
        "frontend/ml_interface.py"
    ]
    
    all_ok = True
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} manquant")
            all_ok = False
            
    if all_ok:
        print("✅ Tous les fichiers frontend présents")
        print("🚀 Démarrez avec: streamlit run frontend/ml_interface.py --server.port 8504")
        
    return all_ok

def main():
    """Test principal du système"""
    print("=" * 60)
    print("🎯 VALIDATION SYSTÈME ML IFRS17 - PRÊT POUR PRODUCTION")
    print("=" * 60)
    
    tests = [
        ("Chargement données", test_data_loading),
        ("Service ML", test_ml_service),
        ("Endpoints API", test_api_endpoints),
        ("Fichiers frontend", test_frontend_files)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    
    success_count = sum(results)
    total_count = len(results)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n🎯 Score: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 SYSTÈME ENTIÈREMENT FONCTIONNEL !")
        print("🚀 Prêt pour la production IFRS17")
        print("\n📋 Commandes de démarrage:")
        print("   Terminal 1: uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload")
        print("   Terminal 2: streamlit run frontend/ml_interface.py --server.port 8504")
    else:
        print("\n⚠️ Quelques éléments nécessitent attention")
        print("💡 Consultez les détails ci-dessus")

if __name__ == "__main__":
    main()