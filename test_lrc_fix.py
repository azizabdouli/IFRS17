"""
Test spécifique pour le fix de l'erreur LRC avec types mixtes
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.models.insurance_models import LRCPredictionModel
from backend.ml.ml_service import MLService

def test_lrc_create_target_fix():
    print("=" * 60)
    print("🧪 TEST: Fix create_lrc_target avec types mixtes")
    print("=" * 60)
    
    # Créer des données avec des types mixtes (le problème exact)
    data = {
        'MNTPPNA': [1000, '2000', 3000, '4000', 5000],      # Mixte string/numeric
        'MNTPRNET': ['1500', 2500, '3500', 4500, '5500'],   # Mixte string/numeric  
        'DUREE': ['12', 24, '36', 48, '60'],                # Mixte string/numeric (origine erreur)
        'MNTACCESS': [100, '200', 300, '400', 500],         # Mixte string/numeric
        'CODFAM': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'LIBELLE_SOUS_PRODUIT': ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données test créées: {df.shape}")
    print("Types de colonnes:")
    for col in ['MNTPPNA', 'MNTPRNET', 'DUREE', 'MNTACCESS']:
        print(f"  {col}: {df[col].dtype} - Exemple: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
    
    # Test de la méthode create_lrc_target
    print("\n" + "=" * 40)
    print("🎯 TEST create_lrc_target")
    print("=" * 40)
    
    try:
        model = LRCPredictionModel()
        lrc_target = model.create_lrc_target(df)
        
        print(f"✅ create_lrc_target réussi!")
        print(f"📊 Target LRC créé: {len(lrc_target)} valeurs")
        print(f"📈 Statistiques: min={lrc_target.min():.2f}, max={lrc_target.max():.2f}, mean={lrc_target.mean():.2f}")
        print(f"📝 Exemples: {lrc_target.head().tolist()}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur create_lrc_target: {e}")
        if "unsupported operand type(s) for /: 'str' and 'int'" in str(e):
            print("🚨 C'est exactement l'erreur originale ! Le fix n'a pas fonctionné.")
        import traceback
        traceback.print_exc()
        return False

def test_full_lrc_training():
    """Test complet du modèle LRC"""
    print("\n" + "=" * 60)
    print("🤖 TEST: Entraînement complet modèle LRC")
    print("=" * 60)
    
    # Données plus complètes pour l'entraînement
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'MNTPPNA': [str(val) if i % 3 == 0 else val for i, val in enumerate(np.random.uniform(500, 3000, n_samples))],
        'MNTPRNET': [str(int(val)) if i % 2 == 0 else val for i, val in enumerate(np.random.uniform(1000, 5000, n_samples))],
        'DUREE': [str(val) if i % 4 == 0 else val for i, val in enumerate(np.random.choice([12, 24, 36, 48, 60], n_samples))],
        'MNTACCESS': [str(int(val)) if i % 5 == 0 else val for i, val in enumerate(np.random.uniform(50, 500, n_samples))],
        'CODFAM': np.random.choice(['AUTO', 'VIE', 'HABITATION', 'SANTE'], n_samples),
        'LIBELLE_SOUS_PRODUIT': np.random.choice(['SP1', 'SP2', 'SP3', 'SP4'], n_samples),
        'NBPPNATOT': np.random.randint(1, 50, n_samples),
        'FRACT': np.random.choice(['0.25', '0.5', 0.75, '1.0'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    try:
        # Test avec MLService exactement comme l'API
        ml_service = MLService()
        results = ml_service.train_lrc_prediction_model(df, 'random_forest')
        
        print(f"✅ Entraînement LRC complet réussi!")
        print(f"📊 Métriques: {results.get('validation_metrics', {})}")
        print(f"🎯 R²: {results.get('validation_metrics', {}).get('r2', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement LRC: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lrc_api_simulation():
    """Simulation exacte de l'appel API qui échouait"""
    print("\n" + "=" * 60)
    print("🌐 TEST: Simulation appel API LRC")
    print("=" * 60)
    
    # Données exactes comme dans l'erreur API
    data = {
        'ANRCTGPO': [2021, 2022, 2023, 2024, 2025],
        'MOIS': [1, 2, 3, 4, 5],
        'LIBELLE_SOUS_PRODUIT': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'FRACT': ['0.25', '0.5', 0.75, '1.0', 0.25],
        'PRIME_NETTE': [1000, '2000', 3000, '4000', 5000],
        'COUVERTURE': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'MNTPRNET': ['1500', 2500, '3500', 4500, '5500'],  # Types mixtes
        'MNTPPNA': [100, '200', 300, '400', 500],          # Types mixtes
        'DUREE': ['12', 24, '36', 48, '60'],               # Types mixtes (origine erreur)
        'MNTACCESS': [50, '100', 150, '200', 250],         # Types mixtes
        'NBPPNATOT': [10, '20', 30, '40', 50],
    }
    
    df = pd.DataFrame(data)
    
    try:
        ml_service = MLService()
        results = ml_service.train_lrc_prediction_model(df, 'xgboost')
        
        print(f"✅ API LRC simulation réussie!")
        print(f"📊 Métriques: {results.get('validation_metrics', {})}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur simulation API: {e}")
        return False

if __name__ == "__main__":
    success1 = test_lrc_create_target_fix()
    success2 = test_full_lrc_training()
    success3 = test_lrc_api_simulation()
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ")
    print("=" * 60)
    print(f"✅ Test create_lrc_target: {'SUCCÈS' if success1 else 'ÉCHEC'}")
    print(f"✅ Test entraînement complet: {'SUCCÈS' if success2 else 'ÉCHEC'}")
    print(f"✅ Test simulation API: {'SUCCÈS' if success3 else 'ÉCHEC'}")
    
    if success1 and success2 and success3:
        print("\n🎉 FIX LRC RÉUSSI!")
        print("✅ Le problème de types mixtes dans create_lrc_target est résolu")
        print("✅ L'API LRC fonctionne maintenant sans erreur")
    else:
        print("\n❌ Le fix LRC nécessite encore des ajustements")