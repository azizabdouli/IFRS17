#!/usr/bin/env python3
"""
Test spécifique pour le problème XGBoost avec FRACT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from backend.ml.data_preprocessing import DataPreprocessor
from backend.ml.models.insurance_models import ClaimsPredictionModel

def test_xgboost_dtype_fix():
    """Test de la correction pour XGBoost avec colonne FRACT"""
    
    print("🔧 Test de Correction XGBoost - Colonne FRACT")
    print("=" * 55)
    
    # Création de données simulant le problème réel
    data = {
        'NUMQUITT': [1, 2, 3, 4, 5],
        'DUREE': [12, 24, 36, 12, 24],
        'MNTPRNET': [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        'MNTPPNA': [100.0, 200.0, 150.0, 300.0, 250.0],
        'CODFAM': [1, 2, 1, 3, 2],
        'CODPROD': [111, 112, 111, 113, 112],
        'MNTPRASSI': [800.0, 1600.0, 1200.0, 2400.0, 2000.0],
        'NBPPNATOT': [1000, 2000, 1500, 3000, 2500],
        'FRACT': ['0.5', '1.0', '0.75', '0.25', '1.25'],  # PROBLÈME : type object
        'target_test': [0, 1, 0, 1, 1]
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données créées: {df.shape}")
    print(f"⚠️  Type FRACT AVANT: {df['FRACT'].dtype}")
    print(f"Valeurs FRACT: {df['FRACT'].tolist()}")
    
    try:
        # Test du preprocessing
        print("\n🔧 Test du preprocessing...")
        preprocessor = DataPreprocessor()
        
        X, y = preprocessor.prepare_data_for_training(df, 'target_test')
        
        print(f"✅ Preprocessing réussi!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {len(y) if y is not None else 'None'}")
        
        # Vérification des types après preprocessing
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            print(f"❌ Colonnes object restantes: {object_cols}")
            for col in object_cols:
                print(f"   {col}: {X[col].dtype} - Valeurs: {X[col].head().tolist()}")
        else:
            print("✅ Aucune colonne object - Compatible XGBoost")
        
        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            print("\n🤖 Test d'entraînement XGBoost...")
            model = ClaimsPredictionModel()
            results = model.train(X, y)
            print(f"✅ Entraînement XGBoost réussi! Score: {results.get('score', 'N/A')}")
            return True
        else:
            print("❌ Données invalides après preprocessing")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if "object" in str(e):
            print("🔍 Erreur liée aux types object - Correction nécessaire")
        return False

if __name__ == "__main__":
    if test_xgboost_dtype_fix():
        print("\n🎉 SUCCESS! Problème XGBoost/FRACT résolu")
        print("✅ Le système peut maintenant traiter vos données IFRS17")
    else:
        print("\n🔧 Correction supplémentaire nécessaire")