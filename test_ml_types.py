#!/usr/bin/env python3
"""
Test de l'entraînement ML après correction du type de données
"""
import requests
import pandas as pd
import numpy as np
import io

def test_ml_training():
    """Test d'entraînement ML avec des données mixtes"""
    
    print("🧪 Test d'entraînement ML après correction des types")
    print("=" * 50)
    
    # Création de données de test avec types mixtes (similaire aux vraies données)
    np.random.seed(42)
    data = {
        'NUMQUITT': range(1, 21),
        'CODFAM': np.random.choice(['1', '2', '3'], 20),  # String qui devrait être numérique
        'CODPROD': np.random.choice([111, 112, 113], 20),
        'DUREE': np.random.choice(['12', '24', '36'], 20),  # String qui devrait être numérique
        'MNTPRNET': np.random.exponential(1000, 20),
        'MNTPPNA': np.random.exponential(100, 20),
        'MNTACCESS': np.random.exponential(50, 20),
        'MNTPRASSI': np.random.exponential(800, 20),
        'NBPPNATOT': np.random.randint(1000, 5000, 20),
        'NBPPNAJ': np.random.randint(100, 500, 20),
        'DEBEFFQUI': 20230101,
        'FINEFFQUI': 20241231,
        'DATEEMISS': 20230101,
        'TYPEEMMIS': np.random.choice(['A', 'B'], 20),
        'STATQUIT': np.random.choice([0, 1], 20),
        'STATU': np.random.choice([0, 1, 2], 20),
        # Ajoutons une colonne target pour l'entraînement
        'sinistres': np.random.choice([0, 1], 20)  # Variable cible
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données créées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Types de données:")
    print(f"  DUREE: {df['DUREE'].dtype} (devrait être converti en numérique)")
    print(f"  CODFAM: {df['CODFAM'].dtype} (devrait être converti en numérique)")
    
    # Conversion en CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    try:
        # 1. Upload des données
        print("\n📤 1. Upload des données...")
        files = {'file': ('test_mixed_types.csv', csv_content, 'text/csv')}
        response = requests.post("http://127.0.0.1:8001/ml/upload-data", 
                               files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Upload échoué: {response.status_code}")
            print(f"Erreur: {response.text}")
            return False
        
        print("✅ Upload réussi")
        
        # 2. Test d'entraînement
        print("\n🤖 2. Test d'entraînement du modèle...")
        train_response = requests.post(
            "http://127.0.0.1:8001/ml/train/claims-prediction",
            params={
                "target_column": "sinistres",
                "model_type": "random_forest"
            },
            timeout=30
        )
        
        if train_response.status_code == 200:
            result = train_response.json()
            print("✅ Entraînement réussi !")
            print(f"  Score: {result.get('score', 'N/A')}")
            print(f"  Modèle: {result.get('model_type', 'N/A')}")
            return True
        else:
            print(f"❌ Entraînement échoué: {train_response.status_code}")
            print(f"Erreur: {train_response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        print("   Vérifiez que le serveur est démarré sur le port 8001")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

if __name__ == "__main__":
    if test_ml_training():
        print("\n🎉 Succès ! Le preprocessing gère maintenant les types mixtes")
        print("🔧 Les conversions automatiques de types fonctionnent")
    else:
        print("\n❌ Il reste des problèmes à résoudre")