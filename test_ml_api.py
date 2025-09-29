# test_ml_api.py

import requests
import json
import pandas as pd
import numpy as np
import io

# Configuration de l'API
API_URL = "http://localhost:8001"

def test_health_endpoint():
    """Test de l'endpoint de santé"""
    try:
        response = requests.get(f"{API_URL}/ml/health")
        if response.status_code == 200:
            print("✅ Health check réussi")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check échoué: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def create_test_data():
    """Création de données de test"""
    np.random.seed(42)
    data = {
        'NUMQUITT': range(1, 51),
        'CODFAM': np.random.choice([1, 2, 3], 50),
        'CODPROD': np.random.choice([111, 112, 113], 50),
        'DUREE': np.random.choice([12, 24, 36], 50),
        'MNTPRNET': np.random.exponential(1000, 50),
        'MNTPPNA': np.random.exponential(100, 50),
        'MNTACCESS': np.random.exponential(50, 50),
        'MNTPRASSI': np.random.exponential(800, 50),
        'NBPPNATOT': np.random.randint(1000, 5000, 50),
        'NBPPNAJ': np.random.randint(100, 500, 50),
        'DEBEFFQUI': 20230101,
        'FINEFFQUI': 20241231,
        'DATEEMISS': 20230101,
        'TYPEEMMIS': np.random.choice(['A', 'B'], 50),
        'STATQUIT': np.random.choice([0, 1], 50),
        'STATU': np.random.choice([0, 1, 2], 50)
    }
    return pd.DataFrame(data)

def test_upload_data():
    """Test d'upload de données"""
    print("\n📤 Test d'upload de données...")
    
    df = create_test_data()
    
    # Conversion en CSV pour upload
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    files = {
        'file': ('test_data.csv', csv_content, 'text/csv')
    }
    
    try:
        response = requests.post(f"{API_URL}/ml/upload-data", files=files)
        if response.status_code == 200:
            print("✅ Upload réussi")
            result = response.json()
            print(f"Données uploadées: {result['data_info']['n_rows']} lignes, {result['data_info']['n_columns']} colonnes")
            return True
        else:
            print(f"❌ Upload échoué: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur d'upload: {e}")
        return False

def test_clustering():
    """Test du clustering"""
    print("\n🎯 Test du clustering...")
    
    try:
        response = requests.post(f"{API_URL}/ml/clustering", params={
            "n_clusters": 3,
            "clustering_type": "kmeans"
        })
        
        if response.status_code == 200:
            print("✅ Clustering réussi")
            result = response.json()
            print(f"Clusters créés: {result['results']['n_clusters']}")
            print(f"Distribution: {result['results']['cluster_distribution']}")
            return True
        else:
            print(f"❌ Clustering échoué: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur de clustering: {e}")
        return False

def test_anomaly_detection():
    """Test de la détection d'anomalies"""
    print("\n🔍 Test de détection d'anomalies...")
    
    try:
        response = requests.post(f"{API_URL}/ml/anomaly-detection", params={
            "method": "isolation_forest",
            "contamination": 0.1
        })
        
        if response.status_code == 200:
            print("✅ Détection d'anomalies réussie")
            result = response.json()
            print(f"Anomalies détectées: {result['results']['n_anomalies']}")
            print(f"Taux d'anomalies: {result['results']['anomaly_rate']}")
            return True
        else:
            print(f"❌ Détection d'anomalies échouée: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur de détection d'anomalies: {e}")
        return False

def test_insights():
    """Test des insights ML"""
    print("\n🧠 Test des insights ML...")
    
    try:
        response = requests.get(f"{API_URL}/ml/insights")
        
        if response.status_code == 200:
            print("✅ Insights générés avec succès")
            result = response.json()
            print(f"Contrats analysés: {result['data_overview']['n_contracts']}")
            print(f"Algorithme recommandé: {result['model_recommendations']['preferred_algorithm']}")
            return True
        else:
            print(f"❌ Génération d'insights échouée: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur de génération d'insights: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🤖 Test de l'API Machine Learning IFRS17")
    print("=" * 50)
    
    # Test de connexion
    if not test_health_endpoint():
        print("❌ Impossible de se connecter à l'API. Vérifiez que le serveur est démarré.")
        return
    
    # Test d'upload
    if not test_upload_data():
        print("❌ Impossible d'uploader les données. Tests interrompus.")
        return
    
    # Tests des fonctionnalités ML
    test_clustering()
    test_anomaly_detection()
    test_insights()
    
    print("\n🎉 Tests terminés!")
    print("\n📋 Pour tester manuellement:")
    print(f"- API Documentation: {API_URL}/docs")
    print(f"- Interface Streamlit: http://localhost:8502")

if __name__ == "__main__":
    main()