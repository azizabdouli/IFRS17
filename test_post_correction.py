#!/usr/bin/env python3
"""
Test rapide des modèles ML après correction des index
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Ajouter le répertoire parent au PATH
sys.path.append(str(Path(__file__).parent))

def test_models_after_fix():
    """Test rapide des modèles après correction"""
    print("🔧 Test des modèles ML après correction des index...")
    
    try:
        from backend.ml.ml_service import MLService
        
        ml_service = MLService()
        print("✅ MLService initialisé")
        
        # Charger échantillon de données
        data_path = "Data/Ppna (4).xlsx"
        df = pd.read_excel(data_path)
        sample_df = df.sample(n=500, random_state=42)  # Plus petit échantillon
        print(f"📊 Test avec échantillon: {len(sample_df)} lignes")
        
        results = {}
        
        # Test 1: Profitability
        print("\n💰 Test modèle de rentabilité...")
        try:
            result = ml_service.train_profitability_model(sample_df)
            r2 = result.get('r2_score', 0)
            results['profitability'] = f"✅ R² = {r2:.3f}"
            print(f"✅ Profitability: R² = {r2:.3f}")
        except Exception as e:
            results['profitability'] = f"❌ {str(e)[:100]}..."
            print(f"❌ Profitability error: {e}")
        
        # Test 2: Risk Classification
        print("\n⚠️ Test classification des risques...")
        try:
            result = ml_service.train_risk_classification_model(sample_df)
            acc = result.get('accuracy', 0)
            results['risk_classification'] = f"✅ Accuracy = {acc:.3f}"
            print(f"✅ Risk classification: Accuracy = {acc:.3f}")
        except Exception as e:
            results['risk_classification'] = f"❌ {str(e)[:100]}..."
            print(f"❌ Risk classification error: {e}")
        
        # Test 3: Claims Prediction
        print("\n🔍 Test prédiction des sinistres...")
        try:
            result = ml_service.train_claims_prediction_model(sample_df)
            r2 = result.get('r2_score', 0)
            results['claims_prediction'] = f"✅ R² = {r2:.3f}"
            print(f"✅ Claims prediction: R² = {r2:.3f}")
        except Exception as e:
            results['claims_prediction'] = f"❌ {str(e)[:100]}..."
            print(f"❌ Claims prediction error: {e}")
        
        # Test 4: LRC Prediction
        print("\n📈 Test prédiction LRC...")
        try:
            result = ml_service.train_lrc_prediction_model(sample_df)
            r2 = result.get('r2_score', 0)
            results['lrc_prediction'] = f"✅ R² = {r2:.3f}"
            print(f"✅ LRC prediction: R² = {r2:.3f}")
        except Exception as e:
            results['lrc_prediction'] = f"❌ {str(e)[:100]}..."
            print(f"❌ LRC prediction error: {e}")
        
        # Résumé
        print("\n" + "="*60)
        print("📊 RÉSULTATS DU TEST POST-CORRECTION")
        print("="*60)
        
        success_count = 0
        for model_name, result in results.items():
            print(f"{result}")
            if "✅" in result:
                success_count += 1
        
        print(f"\n🎯 Modèles fonctionnels: {success_count}/{len(results)}")
        
        if success_count == len(results):
            print("\n🎉 TOUTES LES CORRECTIONS RÉUSSIES !")
            print("🚀 Système ML IFRS17 totalement opérationnel")
        else:
            print(f"\n⚠️ {len(results) - success_count} modèle(s) nécessitent encore attention")
        
        return success_count == len(results)
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        return False

if __name__ == "__main__":
    test_models_after_fix()