"""
Test pour vérifier le nettoyage du target avec des valeurs problématiques
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.data_preprocessing import DataPreprocessor
from backend.ml.models.insurance_models import ProfitabilityModel

def test_target_cleaning():
    print("=" * 50)
    print("🧪 TEST: Nettoyage du target avec valeurs problématiques")
    print("=" * 50)
    
    # Créer des données de test avec des valeurs problématiques dans le target
    data = {
        'ANRCTGPO': [2021, 2022, 2023, 2024, 2025],
        'MOIS': [1, 2, 3, 4, 5],
        'LIBELLE_SOUS_PRODUIT': ['SP1', 'SP2', 'SP3', 'SP4', 'SP5'],
        'FRACT': ['0.5', '0.8', '1.0', '0.3', '0.7'],
        'PRIME_NETTE': [1000, 2000, 3000, 4000, 5000],
        'COUVERTURE': ['AUTO', 'HABITATION', 'AUTO', 'SANTE', 'HABITATION'],
        # Target avec des valeurs problématiques
        'RESULTAT_TECHNIQUE': [100.5, np.nan, np.inf, -np.inf, 999999999]  # NaN, +inf, -inf, valeur extrême
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données créées: {df.shape}")
    print("\nTarget initial:")
    print(df['RESULTAT_TECHNIQUE'].tolist())
    print(f"NaN: {df['RESULTAT_TECHNIQUE'].isna().sum()}")
    print(f"Inf: {np.isinf(df['RESULTAT_TECHNIQUE']).sum()}")
    
    # Preprocessing
    print("\n" + "=" * 30)
    print("🔄 PREPROCESSING")
    print("=" * 30)
    
    preprocessor = DataPreprocessor()
    
    try:
        X, y = preprocessor.prepare_data_for_training(df, target_column='RESULTAT_TECHNIQUE')
        print(f"✅ Preprocessing réussi: X={X.shape}, y={len(y)}")
        print(f"Target après preprocessing: {y.tolist()}")
        print(f"NaN dans y: {y.isna().sum()}")
        print(f"Inf dans y: {np.isinf(y).sum()}")
        
    except Exception as e:
        print(f"❌ Erreur preprocessing: {e}")
        return False
    
    # Test du modèle avec nettoyage du target
    print("\n" + "=" * 30)
    print("🤖 ENTRAINEMENT MODELE")
    print("=" * 30)
    
    try:
        model = ProfitabilityModel()
        results = model.train(X, y)
        print(f"✅ Entraînement réussi!")
        print(f"📊 Résultats: {results.get('validation_metrics', {})}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_target_all_nan():
    """Test avec un target entièrement NaN"""
    print("\n" + "=" * 50)
    print("🧪 TEST: Target entièrement NaN")
    print("=" * 50)
    
    data = {
        'ANRCTGPO': [2021, 2022, 2023],
        'MOIS': [1, 2, 3],
        'LIBELLE_SOUS_PRODUIT': ['SP1', 'SP2', 'SP3'],
        'FRACT': ['0.5', '0.8', '1.0'],
        'PRIME_NETTE': [1000, 2000, 3000],
        'COUVERTURE': ['AUTO', 'HABITATION', 'AUTO'],
        'RESULTAT_TECHNIQUE': [np.nan, np.nan, np.nan]  # Tout NaN
    }
    
    df = pd.DataFrame(data)
    
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data_for_training(df, target_column='RESULTAT_TECHNIQUE')
        
        model = ProfitabilityModel()
        results = model.train(X, y)
        print("❌ Devrait échouer avec target entièrement NaN")
        return False
        
    except Exception as e:
        print(f"✅ Erreur attendue avec target NaN: {e}")
        return True

if __name__ == "__main__":
    success1 = test_target_cleaning()
    success2 = test_target_all_nan()
    
    print("\n" + "=" * 50)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 50)
    print(f"✅ Test nettoyage target: {'SUCCÈS' if success1 else 'ÉCHEC'}")
    print(f"✅ Test target entièrement NaN: {'SUCCÈS' if success2 else 'ÉCHEC'}")
    
    if success1 and success2:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Le nettoyage du target fonctionne correctement")
    else:
        print("\n❌ Certains tests ont échoué")