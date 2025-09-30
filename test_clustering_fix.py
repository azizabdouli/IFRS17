#!/usr/bin/env python3
"""
Test spécifique du clustering après correction des types
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au PATH
sys.path.append(str(Path(__file__).parent))

def test_clustering():
    """Test du clustering après correction"""
    print("🎯 Test du clustering après correction des types...")
    
    try:
        from backend.ml.ml_service import MLService
        
        ml_service = MLService()
        print("✅ MLService initialisé")
        
        # Charger échantillon de données
        data_path = "Data/Ppna (4).xlsx"
        df = pd.read_excel(data_path)
        sample_df = df.sample(n=1000, random_state=42)  # Échantillon pour test rapide
        print(f"📊 Test avec échantillon: {len(sample_df)} lignes")
        
        # Test clustering
        print("\n🎯 Test clustering K-means...")
        try:
            results = ml_service.perform_contract_clustering(sample_df, n_clusters=5, clustering_type='kmeans')
            
            print(f"✅ Clustering réussi:")
            print(f"   📊 Nombre de clusters: {results.get('n_clusters', 'N/A')}")
            print(f"   📈 Distribution: {results.get('cluster_distribution', {})}")
            print(f"   🔍 Caractéristiques: {len(results.get('cluster_characteristics', {}))} clusters analysés")
            
            # Test des caractéristiques
            characteristics = results.get('cluster_characteristics', {})
            if characteristics:
                for cluster_id, stats in list(characteristics.items())[:3]:  # Afficher 3 premiers
                    print(f"   Cluster {cluster_id}: {stats.get('size', 0)} contrats, Prime moy: {stats.get('avg_prime', 0):.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Clustering error: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        return False

def test_json_serialization():
    """Test de la sérialisation JSON"""
    print("\n🧪 Test de sérialisation JSON...")
    
    try:
        import json
        from backend.routers.ml_router import clean_for_json
        
        # Test avec différents types problématiques
        test_data = {
            'int_numpy': np.int64(42),
            'float_numpy': np.float64(3.14),
            'array_numpy': np.array([1, 2, 3]),
            'nan_value': np.nan,
            'inf_value': np.inf,
            'mixed_dict': {
                'normal_str': 'test',
                'numpy_int': np.int32(123)
            }
        }
        
        # Nettoyage
        cleaned_data = clean_for_json(test_data)
        
        # Test sérialisation
        json_str = json.dumps(cleaned_data)
        print(f"✅ Sérialisation JSON réussie: {len(json_str)} caractères")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur sérialisation: {e}")
        return False

def main():
    """Test principal"""
    print("=" * 60)
    print("🔧 TEST CLUSTERING APRÈS CORRECTION DES TYPES")
    print("=" * 60)
    
    tests = [
        ("Clustering", test_clustering),
        ("Sérialisation JSON", test_json_serialization)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)
    
    success_count = sum(results)
    total_count = len(results)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n🎯 Score: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 TOUTES LES CORRECTIONS RÉUSSIES !")
        print("🚀 Clustering prêt pour la production")
    else:
        print(f"\n⚠️ {total_count - success_count} test(s) en échec")

if __name__ == "__main__":
    main()