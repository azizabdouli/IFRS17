#!/usr/bin/env python3
"""
Test spécifique pour le problème d'imputation FRACT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from backend.ml.data_preprocessing import DataPreprocessor

def test_imputation_fix():
    """Test de la correction pour l'imputation avec colonnes entièrement NaN"""
    
    print("🔧 Test Correction Imputation - Colonnes NaN")
    print("=" * 50)
    
    # Données simulant le problème réel : FRACT entièrement NaN
    data = {
        'NUMQUITT': [1, 2, 3, 4, 5],
        'DUREE': [12, 24, 36, 12, 24],
        'MNTPRNET': [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        'MNTPPNA': [100.0, 200.0, 150.0, 300.0, 250.0],
        'CODFAM': [1, 2, 1, 3, 2],
        'CODPROD': [111, 112, 111, 113, 112],
        'MNTPRASSI': [800.0, 1600.0, 1200.0, 2400.0, 2000.0],
        'NBPPNATOT': [1000, 2000, 1500, 3000, 2500],
        'FRACT': [np.nan, np.nan, np.nan, np.nan, np.nan],  # PROBLÈME : tout NaN
        'target_test': [0, 1, 0, 1, 1]
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données créées: {df.shape}")
    print(f"⚠️  FRACT valeurs NaN: {df['FRACT'].isna().sum()}/{len(df)}")
    print(f"⚠️  FRACT valeurs valides: {df['FRACT'].notna().sum()}")
    
    try:
        # Test du preprocessing complet
        print("\n🔧 Test du preprocessing complet...")
        preprocessor = DataPreprocessor()
        
        X, y = preprocessor.prepare_data_for_training(df, 'target_test')
        
        print(f"✅ Preprocessing réussi!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {len(y) if y is not None else 'None'}")
        
        # Vérification que FRACT a été traité
        if 'FRACT' in X.columns:
            fract_values = X['FRACT'].unique()
            print(f"✅ FRACT traité - Valeurs uniques: {fract_values}")
            if np.isnan(fract_values).any():
                print("❌ FRACT contient encore des NaN")
                return False
            else:
                print("✅ FRACT ne contient plus de NaN")
        
        return True
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if "Columns must be same length" in str(e):
            print("🔍 Erreur de dimension pandas - Correction nécessaire")
        elif "median" in str(e):
            print("🔍 Erreur d'imputation médiane - Correction nécessaire")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_imputation_fix():
        print("\n🎉 SUCCESS! Problème d'imputation résolu")
        print("✅ Le système peut traiter les colonnes entièrement NaN")
    else:
        print("\n🔧 Correction supplémentaire nécessaire")