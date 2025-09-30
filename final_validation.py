"""
Test final pour valider la résolution complète de l'erreur TypeError
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.ml_service import MLService

def test_original_error_scenario():
    """
    Simulation exacte du scénario qui provoquait l'erreur dans l'API
    """
    print("=" * 70)
    print("🎯 VALIDATION FINALE: Erreur TypeError résolue")
    print("=" * 70)
    
    # Données exactes avec types mixtes qui causaient l'erreur
    data = {
        'ANRCTGPO': [2021, 2022, 2023, 2024, 2025],
        'MOIS': [1, 2, 3, 4, 5],
        'LIBELLE_SOUS_PRODUIT': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'FRACT': ['0.25', '0.5', 0.75, '1.0', 0.25],  # Types mixtes !
        'PRIME_NETTE': [1000, '2000', 3000, '4000', 5000],  # Types mixtes !
        'COUVERTURE': ['AUTO', 'VIE', 'HABITATION', 'SANTE', 'AUTO'],
        'MNTPRNET': ['1500', 2500, '3500', 4500, '5500'],  # Types mixtes !
        'MNTPPNA': [100, '200', 300, '400', 500],  # Types mixtes !
        'DUREE': ['12', 24, '36', 48, '60'],  # Types mixtes ! (origine erreur)
        'NBPPNATOT': [10, '20', 30, '40', 50],  # Types mixtes !
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Données avec types mixtes: {df.shape}")
    
    try:
        ml_service = MLService()
        results = ml_service.train_risk_classification_model(df, 'xgboost')
        
        print(f"✅ SUCCÈS TOTAL!")
        print(f"📊 Métriques: {results.get('validation_metrics', {})}")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return False

if __name__ == "__main__":
    if test_original_error_scenario():
        print("\n🎉 PROBLÈME RÉSOLU! Le système fonctionne parfaitement!")
    else:
        print("\n❌ Le problème persiste")