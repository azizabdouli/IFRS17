#!/usr/bin/env python3
"""
Test direct du preprocessing pour diagnostiquer l'erreur NoneType
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from backend.ml.data_preprocessing import DataPreprocessor
from backend.ml.models.insurance_models import ClaimsPredictionModel

def test_preprocessing_direct():
    """Test direct du preprocessing sans API"""
    
    print("🔍 Test Direct du Preprocessing ML")
    print("=" * 50)
    
    # Création de données de test simples
    data = {
        'NUMQUITT': [1, 2, 3, 4, 5],
        'DUREE': [12, 24, 36, 12, 24],
        'MNTPRNET': [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        'MNTPPNA': [100.0, 200.0, 150.0, 300.0, 250.0],
        'CODFAM': [1, 2, 1, 3, 2],
        'CODPROD': [111, 112, 111, 113, 112],
        'MNTPRASSI': [800.0, 1600.0, 1200.0, 2400.0, 2000.0],
        'NBPPNATOT': [1000, 2000, 1500, 3000, 2500],
        'target_test': [0, 1, 0, 1, 1]  # Variable cible
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données créées: {df.shape}")
    print("Colonnes:", df.columns.tolist())
    
    try:
        # Test du preprocessing
        print("\n🔧 Test du preprocessing...")
        preprocessor = DataPreprocessor()
        
        X, y = preprocessor.prepare_data_for_training(df, 'target_test')
        
        print(f"✅ Preprocessing réussi!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {len(y) if y is not None else 'None'}")
        print(f"   Features: {X.columns.tolist()[:5]}...")
        
        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            print("\n🤖 Test d'entraînement de modèle...")
            model = ClaimsPredictionModel()
            results = model.train(X, y)
            print(f"✅ Entraînement réussi! Score: {results.get('score', 'N/A')}")
            return True
        else:
            print("❌ Données invalides après preprocessing")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_preprocessing_direct():
        print("\n🎉 SUCCESS! Le problème NoneType est résolu")
        print("✅ Le système ML fonctionne maintenant correctement")
    else:
        print("\n🔧 Il reste des problèmes à corriger")