"""
Test direct du nettoyage target dans la classe BaseMLModel
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.models.insurance_models import ProfitabilityModel

def test_target_cleaning_direct():
    print("=" * 60)
    print("🧪 TEST DIRECT: Nettoyage du target dans BaseMLModel")
    print("=" * 60)
    
    # Créer des données simples mais propres pour les features
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 0.5, n_samples),
        'feature3': np.random.randint(0, 5, n_samples),
        'feature4': np.random.uniform(-1, 1, n_samples)
    })
    
    # Target avec des valeurs problématiques
    y_good = np.random.normal(100, 20, n_samples-10)  # 90 bonnes valeurs
    y_bad = [np.nan, np.inf, -np.inf, 999999999, -999999999,  # 5 valeurs problématiques
             np.nan, np.inf, -np.inf, 1e20, -1e20]              # 5 autres valeurs problématiques
    
    y = pd.Series(list(y_good) + y_bad)
    
    print(f"📊 Données créées: X={X.shape}, y={len(y)}")
    print(f"Target - NaN: {y.isna().sum()}, Inf: {np.isinf(y).sum()}")
    print(f"Target min: {y[~np.isnan(y) & ~np.isinf(y)].min()}")
    print(f"Target max: {y[~np.isnan(y) & ~np.isinf(y)].max()}")
    
    # Test du modèle avec nettoyage du target
    print("\n" + "=" * 40)
    print("🤖 ENTRAINEMENT AVEC NETTOYAGE TARGET")
    print("=" * 40)
    
    try:
        model = ProfitabilityModel()
        results = model.train(X, y)
        
        print(f"✅ Entraînement réussi!")
        print(f"📊 Score validation: {results.get('validation_metrics', {})}")
        print(f"🎯 Features utilisées: {len(model.feature_names)}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_target_xgboost_specific():
    """Test spécifique pour l'erreur XGBoost qu'on a vue dans les logs"""
    print("\n" + "=" * 60)
    print("🧪 TEST SPÉCIFIQUE: Simulation erreur XGBoost")
    print("=" * 60)
    
    # Simuler les données qui ont causé l'erreur XGBoost
    np.random.seed(123)
    n_samples = 1000  # Plus proche de la vraie taille de données
    
    X = pd.DataFrame({
        'FRACT': np.random.uniform(0, 1, n_samples),
        'contrat_long_terme': np.random.randint(0, 2, n_samples),
        'prime_elevee': np.random.randint(0, 2, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(10, 3, n_samples)
    })
    
    # Target mixte avec quelques valeurs problématiques dans un dataset plus large
    y_good = np.random.normal(50, 15, n_samples-50)
    y_problems = [np.nan] * 20 + [np.inf] * 10 + [-np.inf] * 10 + [1e15] * 10
    
    y = pd.Series(list(y_good) + y_problems)
    
    print(f"📊 Données: X={X.shape}, y={len(y)}")
    print(f"Valeurs problématiques: NaN={y.isna().sum()}, Inf={np.isinf(y).sum()}")
    
    try:
        model = ProfitabilityModel()
        results = model.train(X, y)
        
        print(f"✅ Entraînement réussi avec nettoyage!")
        print(f"📊 Score: {results.get('validation_metrics', {})}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if "Label contains NaN" in str(e) or "infinity" in str(e):
            print("🎯 C'est exactement l'erreur XGBoost qu'on cherche à résoudre!")
        return False

if __name__ == "__main__":
    success1 = test_target_cleaning_direct()
    success2 = test_target_xgboost_specific()
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ")
    print("=" * 60)
    print(f"✅ Test direct nettoyage: {'SUCCÈS' if success1 else 'ÉCHEC'}")
    print(f"✅ Test spécifique XGBoost: {'SUCCÈS' if success2 else 'ÉCHEC'}")
    
    if success1 and success2:
        print("\n🎉 NETTOYAGE TARGET FONCTIONNE!")
        print("✅ XGBoost peut maintenant traiter les données problématiques")
    else:
        print("\n🔧 Le nettoyage du target nécessite encore des ajustements")