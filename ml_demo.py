#!/usr/bin/env python3
"""
Test de diagnostic pour identifier le problème None dans le preprocessing
"""
import requests
import pandas as pd
import numpy as np
import io

def test_preprocessing_debug():
    """Test diagnostic du preprocessing ML"""
    
    print("🔍 Test de diagnostic du preprocessing ML")
    print("=" * 50)
    
    # Création de données de test très simples
    np.random.seed(42)
    data = {
        'NUMQUITT': [1, 2, 3, 4, 5],
        'DUREE': [12, 24, 36, 12, 24],
        'MNTPRNET': [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        'MNTPPNA': [100.0, 200.0, 150.0, 300.0, 250.0],
        'CODFAM': [1, 2, 1, 3, 2],
        'CODPROD': [111, 112, 111, 113, 112],
        'target_test': [0, 1, 0, 1, 1]  # Variable cible simple
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données de test créées: {df.shape}")
    print("Aperçu des données:")
    print(df.head())
    
    # Conversion en CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    try:
        # 1. Test d'upload
        print("\n📤 1. Test d'upload...")
        files = {'file': ('debug_test.csv', csv_content, 'text/csv')}
        response = requests.post("http://127.0.0.1:8001/ml/upload-data", 
                               files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Upload échoué: {response.status_code}")
            print(f"Erreur: {response.text}")
            return False
        
        result = response.json()
        print("✅ Upload réussi")
        print(f"   Lignes: {result['data_info']['n_rows']}")
        print(f"   Colonnes: {result['data_info']['n_columns']}")
        
        # 2. Test d'insights pour vérifier le preprocessing
        print("\n🔍 2. Test des insights (preprocessing simple)...")
        insights_response = requests.get("http://127.0.0.1:8001/ml/insights", timeout=10)
        
        if insights_response.status_code != 200:
            print(f"❌ Insights échoué: {insights_response.status_code}")
            print(f"Erreur: {insights_response.text}")
            return False
            
        print("✅ Insights réussi (preprocessing fonctionne)")
        
        # 3. Test d'entraînement simple
        print("\n🤖 3. Test d'entraînement...")
        train_response = requests.post(
            "http://127.0.0.1:8001/ml/train/claims-prediction",
            params={
                "target_column": "target_test",
                "model_type": "random_forest"
            },
            timeout=30
        )
        
        if train_response.status_code == 200:
            result = train_response.json()
            print("✅ Entraînement réussi !")
            print(f"   Score: {result.get('score', 'N/A')}")
            return True
        else:
            print(f"❌ Entraînement échoué: {train_response.status_code}")
            print(f"Erreur: {train_response.text}")
            
            # Essayons d'analyser l'erreur
            if "None" in train_response.text or "NoneType" in train_response.text:
                print("\n🔍 Analyse: Problème de données None détecté")
                print("   Possible causes:")
                print("   - Preprocessing retourne None")
                print("   - Colonne target introuvable")
                print("   - DataFrame vide après nettoyage")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

if __name__ == "__main__":
    if test_preprocessing_debug():
        print("\n🎉 Diagnostic réussi - Le système fonctionne")
    else:
        print("\n🔧 Diagnostic révèle des problèmes à corriger")
