#!/usr/bin/env python3
"""
Test spécifique pour le problème de create_risk_labels avec des types mixtes
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.models.insurance_models import RiskClassificationModel

def test_create_risk_labels_type_fix():
    print("=" * 60)
    print("🧪 TEST: Fix create_risk_labels avec types mixtes")
    print("=" * 60)
    
    # Créer des données avec des types mixtes (le problème exact)
    data = {
        'MNTPPNA': [1000, 2000, '3000', '4000', 5000],  # Mixte string/numeric
        'MNTPRNET': ['500', 1000, 1500, '2000', 2500],  # Mixte string/numeric  
        'DUREE': ['12', 24, '36', 48, '60'],            # Mixte string/numeric
        'CODFAM': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'LIBELLE_SOUS_PRODUIT': ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données test créées: {df.shape}")
    print("Types de colonnes:")
    for col in ['MNTPPNA', 'MNTPRNET', 'DUREE']:
        print(f"  {col}: {df[col].dtype} - Exemple: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
    
    # Test de la méthode create_risk_labels
    print("\n" + "=" * 40)
    print("🎯 TEST create_risk_labels")
    print("=" * 40)
    
    try:
        model = RiskClassificationModel()
        risk_labels = model.create_risk_labels(df)
        
        print(f"✅ create_risk_labels réussi!")
        print(f"📊 Labels créés: {len(risk_labels)}")
        print(f"🏷️ Catégories: {risk_labels.value_counts().to_dict()}")
        print(f"📝 Exemples: {risk_labels.head().tolist()}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur create_risk_labels: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_risk_classification():
    """Test complet du modèle de classification des risques"""
    print("\n" + "=" * 60)
    print("🤖 TEST: Entraînement complet modèle classification")
    print("=" * 60)
    
    # Données plus complètes pour l'entraînement
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'MNTPPNA': np.random.choice(['1000', '2000', 3000, '4000', 5000], n_samples),
        'MNTPRNET': np.random.choice([500, '1000', 1500, '2000', 2500], n_samples),
        'DUREE': np.random.choice(['12', 24, '36', 48, '60'], n_samples),
        'CODFAM': np.random.choice(['AUTO', 'VIE', 'HABITATION', 'SANTE'], n_samples),
        'LIBELLE_SOUS_PRODUIT': np.random.choice(['SP1', 'SP2', 'SP3', 'SP4'], n_samples),
        'NBPPNATOT': np.random.randint(1, 50, n_samples),
        'FRACT': np.random.choice(['0.25', '0.5', 0.75, '1.0'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    try:
        from backend.ml.data_preprocessing import DataPreprocessor
        
        # Créer les labels de risque
        model = RiskClassificationModel()
        risk_labels = model.create_risk_labels(df)
        
        # Ajouter les labels au dataframe comme target
        df['RISK_LABEL'] = risk_labels
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data_for_training(df, target_column='RISK_LABEL')
        
        # Entraînement
        results = model.train(X, y)
        
        print(f"✅ Entraînement complet réussi!")
        print(f"📊 Métriques: {results.get('validation_metrics', {})}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False
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
    success1 = test_create_risk_labels_type_fix()
    success2 = test_full_risk_classification()
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ")
    print("=" * 60)
    print(f"✅ Test create_risk_labels: {'SUCCÈS' if success1 else 'ÉCHEC'}")
    print(f"✅ Test entraînement complet: {'SUCCÈS' if success2 else 'ÉCHEC'}")
    
    if success1 and success2:
        print("\n🎉 FIX RÉUSSI!")
        print("✅ Le problème de types mixtes dans create_risk_labels est résolu")
    else:
        print("\n❌ Le fix nécessite encore des ajustements")