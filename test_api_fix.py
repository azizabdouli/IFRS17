#!/usr/bin/env python3
"""
Test de l'API ML après correction des NaN
"""
import requests
import pandas as pd
import numpy as np
import io

def test_api_nan_handling():
    """Test l'API avec des données contenant des NaN/inf"""
    print('🧪 Test de l\'API ML après correction des NaN')
    
    # Création de données de test avec NaN intentionnels
    np.random.seed(42)
    data = {
        'NUMQUITT': range(1, 21),
        'CODFAM': np.random.choice([1, 2, 3], 20),
        'CODPROD': np.random.choice([111, 112, 113], 20),
        'DUREE': np.random.choice([12, 24, 36], 20),
        'MNTPRNET': np.random.exponential(1000, 20),
        'MNTPPNA': np.random.exponential(100, 20),
        'MNTACCESS': np.random.exponential(50, 20),
        'MNTPRASSI': np.random.exponential(800, 20),
        'NBPPNATOT': np.random.randint(1000, 5000, 20),
        'NBPPNAJ': np.random.randint(100, 500, 20),
        'DEBEFFQUI': 20230101,
        'FINEFFQUI': 20241231,
        'DATEEMISS': 20230101,
        'TYPEEMMIS': np.random.choice(['A', 'B'], 20),
        'STATQUIT': np.random.choice([0, 1], 20),
        'STATU': np.random.choice([0, 1, 2], 20)
    }
    
    df = pd.DataFrame(data)
    # Ajoutons quelques NaN pour tester
    df.loc[0, 'MNTPRNET'] = np.nan
    df.loc[1, 'MNTPPNA'] = np.inf
    df.loc[2, 'DUREE'] = -np.inf
    
    print(f'Données créées: {df.shape[0]} lignes avec NaN/inf')
    
    # Test d'upload
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    files = {'file': ('test_nan.csv', csv_content, 'text/csv')}
    
    try:
        print('Tentative d\'upload...')
        response = requests.post('http://127.0.0.1:8001/ml/upload-data', 
                               files=files, timeout=10)
        
        print(f'Status upload: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('✅ Upload réussi avec NaN gérés!')
            data_info = result.get('data_info', {})
            print(f'Lignes: {data_info.get("n_rows", "N/A")}')
            print(f'Colonnes: {data_info.get("n_cols", "N/A")}')
            print(f'Valeurs manquantes: {data_info.get("missing_values", "N/A")}')
            
            # Test des insights
            print('\nTest des insights...')
            insights_response = requests.get('http://127.0.0.1:8001/ml/insights')
            if insights_response.status_code == 200:
                print('✅ Insights OK')
            else:
                print(f'❌ Insights erreur: {insights_response.status_code}')
                
        else:
            print(f'❌ Erreur upload: {response.text}')
            
    except requests.exceptions.ConnectionError:
        print('❌ Impossible de se connecter à l\'API. Vérifiez que le serveur tourne sur le port 8001')
    except Exception as e:
        print(f'❌ Erreur inattendue: {e}')

if __name__ == "__main__":
    test_api_nan_handling()