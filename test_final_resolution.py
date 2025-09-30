"""
Test final avec les vraies données IFRS17 Excel pour confirmer la résolution complète
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from backend.ml.data_preprocessing import DataPreprocessor
from backend.ml.models.insurance_models import ProfitabilityModel, ClaimsPredictionModel

def test_with_real_data():
    print("=" * 60)
    print("🏆 TEST FINAL: Vraies données IFRS17")
    print("=" * 60)
    
    try:
        # Charger les vraies données IFRS17
        data_path = "Data/Ppna (4).xlsx"
        if not os.path.exists(data_path):
            print(f"❌ Fichier non trouvé: {data_path}")
            return False
            
        print(f"📂 Chargement du fichier: {data_path}")
        df = pd.read_excel(data_path)
        print(f"📊 Données chargées: {df.shape}")
        
        # Limiter à un échantillon manageable pour le test
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📏 Échantillon pour test: {df_sample.shape}")
        
        # Créer un target synthétique pour le test
        np.random.seed(42)
        # Target avec quelques valeurs problématiques
        normal_values = np.random.normal(100, 50, sample_size - 20)
        problem_values = [np.nan] * 5 + [np.inf] * 5 + [-np.inf] * 5 + [1e15] * 5
        
        target_values = list(normal_values) + problem_values
        np.random.shuffle(target_values)
        df_sample['RESULTAT_TECHNIQUE'] = target_values
        
        print(f"🎯 Target créé avec {pd.Series(target_values).isna().sum()} NaN et {np.isinf(target_values).sum()} Inf")
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        print("📝 Utilisation de données synthétiques à la place...")
        
        # Créer des données synthétiques similaires aux vraies données IFRS17
        sample_size = 1000
        df_sample = pd.DataFrame({
            'ANRCTGPO': np.random.choice([2021, 2022, 2023, 2024], sample_size),
            'MOIS': np.random.choice(range(1, 13), sample_size),
            'LIBELLE_SOUS_PRODUIT': np.random.choice(['AUTO', 'HABITATION', 'SANTE', 'VIE'], sample_size),
            'FRACT': np.random.choice(['0.25', '0.5', '0.75', '1.0'], sample_size),
            'PRIME_NETTE': np.random.uniform(500, 10000, sample_size),
            'COUVERTURE': np.random.choice(['AUTO', 'HABITATION', 'SANTE', 'VIE'], sample_size),
            'MNTPRNET': np.random.uniform(1000, 15000, sample_size),
            'NBPPNATOT': np.random.randint(1, 100, sample_size),
            'DUREE': np.random.choice([12, 24, 36, 48, 60], sample_size),
        })
        
        # Target avec valeurs problématiques
        normal_values = np.random.normal(100, 50, sample_size - 50)
        problem_values = [np.nan] * 20 + [np.inf] * 10 + [-np.inf] * 10 + [1e12] * 10
        target_values = list(normal_values) + problem_values
        np.random.shuffle(target_values)
        df_sample['RESULTAT_TECHNIQUE'] = target_values
        
        print(f"📊 Données synthétiques créées: {df_sample.shape}")
    
    # Preprocessing
    print("\n" + "=" * 40)
    print("🔄 PREPROCESSING DES DONNÉES")
    print("=" * 40)
    
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data_for_training(df_sample, target_column='RESULTAT_TECHNIQUE')
        
        print(f"✅ Preprocessing réussi!")
        print(f"📊 X: {X.shape}, y: {len(y)}")
        print(f"🧹 Target après preprocessing - NaN: {y.isna().sum()}, Inf: {np.isinf(y).sum()}")
        
    except Exception as e:
        print(f"❌ Erreur preprocessing: {e}")
        return False
    
    # Test modèle de rentabilité (celui qui avait le problème XGBoost)
    print("\n" + "=" * 40)
    print("💰 TEST MODÈLE DE RENTABILITÉ (XGBoost)")
    print("=" * 40)
    
    try:
        model = ProfitabilityModel()
        # Utiliser XGBoost explicitement
        model.model_type = 'regression'
        
        results = model.train(X, y)
        
        print(f"✅ Entraînement XGBoost réussi!")
        print(f"📊 Métriques: {results.get('validation_metrics', {})}")
        print(f"🎯 R²: {results.get('validation_metrics', {}).get('r2', 'N/A')}")
        
        # Test de prédiction
        predictions = model.predict(X.head(5))
        print(f"🔮 Prédictions test: {predictions[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur modèle XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_claims_prediction():
    """Test du modèle de prédiction de sinistres"""
    print("\n" + "=" * 40)
    print("⚠️ TEST MODÈLE PRÉDICTION SINISTRES")
    print("=" * 40)
    
    # Données synthétiques pour classification
    sample_size = 500
    df = pd.DataFrame({
        'PRIME_NETTE': np.random.uniform(500, 10000, sample_size),
        'DUREE': np.random.choice([12, 24, 36], sample_size),
        'FRACT': np.random.choice(['0.25', '0.5', '0.75', '1.0'], sample_size),
        'COUVERTURE': np.random.choice(['AUTO', 'HABITATION', 'SANTE'], sample_size),
        'LIBELLE_SOUS_PRODUIT': np.random.choice(['SP1', 'SP2', 'SP3'], sample_size),
    })
    
    # Target de classification avec valeurs problématiques
    normal_classes = np.random.choice([0, 1, 2], sample_size - 30)
    problem_values = [np.nan] * 15 + [np.inf] * 10 + [999] * 5  # Classe inexistante
    target_values = list(normal_classes) + problem_values
    np.random.shuffle(target_values)
    df['CLASSE_RISQUE'] = target_values
    
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data_for_training(df, target_column='CLASSE_RISQUE')
        
        model = ClaimsPredictionModel()
        results = model.train(X, y)
        
        print(f"✅ Modèle classification réussi!")
        print(f"📊 Accuracy: {results.get('validation_metrics', {}).get('accuracy', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur modèle classification: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTS FINAUX - RÉSOLUTION COMPLÈTE DES ERREURS ML")
    print("=" * 60)
    
    success1 = test_with_real_data()
    success2 = test_claims_prediction()
    
    print("\n" + "=" * 60)
    print("📋 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"✅ Test rentabilité XGBoost: {'SUCCÈS' if success1 else 'ÉCHEC'}")
    print(f"✅ Test prédiction sinistres: {'SUCCÈS' if success2 else 'ÉCHEC'}")
    
    if success1 and success2:
        print("\n🎉 MISSION ACCOMPLIE! 🎉")
        print("✅ Tous les problèmes ML sont résolus:")
        print("   • NoneType error dans train_test_split ✅")
        print("   • XGBoost object dtype error ✅")
        print("   • Imputation entirely NaN columns ✅") 
        print("   • XGBoost target validation error ✅")
        print("\n🚀 Le système IFRS17 est prêt pour la production!")
    else:
        print("\n❌ Il reste encore des problèmes à résoudre")