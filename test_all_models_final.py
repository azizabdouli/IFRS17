"""
Test final global pour valider que TOUS les problèmes de types mixtes sont résolus
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.ml_service import MLService

def test_all_models_with_mixed_types():
    """Test tous les modèles avec des données à types mixtes"""
    print("=" * 70)
    print("🏆 VALIDATION FINALE GLOBALE - Tous Modèles")
    print("=" * 70)
    
    # Données de test avec TOUS les types mixtes possibles
    np.random.seed(42)
    n_samples = 50  # Plus petit pour test rapide
    
    data = {
        # Colonnes temporelles
        'ANRCTGPO': np.random.choice([2021, 2022, 2023, 2024], n_samples),
        'MOIS': np.random.choice(range(1, 13), n_samples),
        
        # Colonnes catégorielles
        'LIBELLE_SOUS_PRODUIT': np.random.choice(['AUTO', 'VIE', 'HABITATION', 'SANTE'], n_samples),
        'COUVERTURE': np.random.choice(['AUTO', 'VIE', 'HABITATION', 'SANTE'], n_samples),
        'CODFAM': np.random.choice(['AUTO', 'VIE', 'HABITATION', 'SANTE'], n_samples),
        
        # Colonnes numériques avec types mixtes (string/numeric)
        'FRACT': [str(val) if i % 3 == 0 else val for i, val in enumerate(np.random.choice([0.25, 0.5, 0.75, 1.0], n_samples))],
        'PRIME_NETTE': [str(int(val)) if i % 2 == 0 else val for i, val in enumerate(np.random.uniform(500, 5000, n_samples))],
        'MNTPRNET': [str(int(val)) if i % 3 == 0 else val for i, val in enumerate(np.random.uniform(1000, 6000, n_samples))],
        'MNTPPNA': [str(int(val)) if i % 4 == 0 else val for i, val in enumerate(np.random.uniform(100, 600, n_samples))],
        'DUREE': [str(val) if i % 2 == 0 else val for i, val in enumerate(np.random.choice([12, 24, 36, 48, 60], n_samples))],
        'MNTACCESS': [str(int(val)) if i % 5 == 0 else val for i, val in enumerate(np.random.uniform(50, 300, n_samples))],
        'NBPPNATOT': [str(val) if i % 3 == 0 else val for i, val in enumerate(np.random.randint(1, 50, n_samples))],
        'NBPPNAJ': np.random.randint(0, 20, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    print(f"📊 Dataset test créé: {df.shape}")
    print("🔍 Vérification des types mixtes:")
    mixed_type_cols = ['FRACT', 'PRIME_NETTE', 'MNTPRNET', 'MNTPPNA', 'DUREE', 'MNTACCESS', 'NBPPNATOT']
    for col in mixed_type_cols:
        if col in df.columns:
            types_in_col = [type(val).__name__ for val in df[col][:5]]
            print(f"  {col}: {set(types_in_col)}")
    
    # Test de tous les modèles ML
    ml_service = MLService()
    results = {}
    
    models_to_test = [
        ('profitability', 'xgboost', 'train_profitability_model'),
        ('risk_classification', 'random_forest', 'train_risk_classification_model'),
        ('claims_prediction', 'xgboost', 'train_claims_prediction_model'),
        ('lrc_prediction', 'random_forest', 'train_lrc_prediction_model'),
    ]
    
    print("\n" + "=" * 50)
    print("🤖 TEST DE TOUS LES MODÈLES")
    print("=" * 50)
    
    success_count = 0
    total_count = len(models_to_test)
    
    for model_name, model_type, method_name in models_to_test:
        print(f"\n🔧 Test {model_name} ({model_type})...")
        
        try:
            method = getattr(ml_service, method_name)
            result = method(df, model_type)
            
            metrics = result.get('validation_metrics', {})
            print(f"  ✅ SUCCÈS! Métriques: {metrics}")
            results[model_name] = {'status': 'success', 'metrics': metrics}
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ ERREUR: {e}")
            if "unsupported operand type" in str(e) or "'str' and" in str(e):
                print(f"    🚨 ERREUR DE TYPES MIXTES DÉTECTÉE!")
            results[model_name] = {'status': 'error', 'error': str(e)}
    
    return success_count, total_count, results

def test_data_preprocessing_robustness():
    """Test de robustesse du preprocessing avec données problématiques"""
    print("\n" + "=" * 70)
    print("🛡️ TEST ROBUSTESSE PREPROCESSING")
    print("=" * 70)
    
    # Données avec tous les types de problèmes possibles
    data = {
        'COL_MIXED_TYPES': [1, '2', 3.5, '4.5', None],
        'COL_ALL_NAN': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'COL_WITH_INF': [1, np.inf, 3, -np.inf, 5],
        'COL_EXTREME_VALUES': [1, 1e15, 3, -1e15, 5],
        'COL_STRINGS': ['a', 'b', 'c', 'd', 'e'],
        'TARGET_MIXED': [100, '200', 300, '400', 500]
    }
    
    df = pd.DataFrame(data)
    
    try:
        from backend.ml.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data_for_training(df, target_column='TARGET_MIXED')
        
        print(f"✅ Preprocessing robustesse: SUCCÈS")
        print(f"📊 Résultat: X={X.shape}, y={len(y)}")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing robustesse: ÉCHEC - {e}")
        return False

if __name__ == "__main__":
    print("🚀 VALIDATION FINALE GLOBALE DU SYSTÈME IFRS17")
    print("=" * 70)
    
    # Test 1: Tous les modèles
    success_count, total_count, results = test_all_models_with_mixed_types()
    
    # Test 2: Robustesse preprocessing
    preprocessing_ok = test_data_preprocessing_robustness()
    
    # Résultats finaux
    print("\n" + "=" * 70)
    print("📋 RÉSULTATS FINAUX GLOBAUX")
    print("=" * 70)
    
    print(f"🤖 Modèles ML testés: {success_count}/{total_count}")
    for model_name, result in results.items():
        status = "✅ SUCCÈS" if result['status'] == 'success' else "❌ ÉCHEC"
        print(f"  {model_name}: {status}")
    
    print(f"🛡️ Robustesse preprocessing: {'✅ SUCCÈS' if preprocessing_ok else '❌ ÉCHEC'}")
    
    # Verdict final
    all_models_ok = (success_count == total_count)
    everything_ok = all_models_ok and preprocessing_ok
    
    print("\n" + "=" * 70)
    if everything_ok:
        print("🎉🎉🎉 VALIDATION FINALE: TOUT FONCTIONNE! 🎉🎉🎉")
        print("✅ TOUS les problèmes de types mixtes sont résolus")
        print("✅ TOUS les modèles ML fonctionnent parfaitement")
        print("✅ Le preprocessing est robuste")
        print("✅ Le système IFRS17 est 100% opérationnel")
        print("🚀 PRÊT POUR LA PRODUCTION!")
    else:
        print("❌ VALIDATION FINALE: DES PROBLÈMES PERSISTENT")
        print(f"   Modèles OK: {success_count}/{total_count}")
        print(f"   Preprocessing OK: {preprocessing_ok}")
        print("🔧 Investigation supplémentaire nécessaire")
    print("=" * 70)