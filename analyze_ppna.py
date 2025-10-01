import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.ppna_service import PPNAService

def test_ppna_analysis():
    """Test de l'analyse des données PPNA"""
    print("🔍 Test d'analyse des données PPNA")
    
    # Initialiser le service
    service = PPNAService()
    
    # Charger les données
    print("\n📁 Chargement des données...")
    result = service.load_ppna_data()
    print(f"Status: {result.get('status')}")
    print(f"Feuilles: {result.get('sheets')}")
    
    if result.get('status') == 'success':
        print(f"Total feuilles: {result.get('total_sheets')}")
        
        # Analyser la première feuille
        sheet_names = result.get('sheets', [])
        if sheet_names:
            first_sheet = sheet_names[0]
            df = service.ppna_data[first_sheet]
            
            print(f"\n📊 Analyse de la feuille '{first_sheet}':")
            print(f"Dimensions: {df.shape}")
            print(f"Colonnes: {list(df.columns)}")
            
            # Vérifier les colonnes importantes
            print(f"\n🔑 Colonnes importantes détectées:")
            print(f"MNTPRNET (primes): {'✅' if 'MNTPRNET' in df.columns else '❌'}")
            print(f"MNTPPNA (provisions): {'✅' if 'MNTPPNA' in df.columns else '❌'}")
            print(f"CODPROD (produit): {'✅' if 'CODPROD' in df.columns else '❌'}")
            
            # Calculs LRC
            print(f"\n💰 Calcul LRC...")
            lrc_result = service.calculate_lrc_paa()
            print(f"Résultat LRC: {lrc_result.get('status', 'N/A')}")
            
            if 'metriques' in lrc_result:
                metrics = lrc_result['metriques']
                print(f"Lignes traitées: {metrics.get('lignes_traitees', 0)}")
                print(f"PPNA Total: {metrics.get('ppna_total', 0):,.2f}")
                print(f"Primes Total: {metrics.get('primes_totales', 0):,.2f}")
                print(f"LRC Total: {metrics.get('lrc_total', 0):,.2f}")
            
            # Métriques dashboard
            print(f"\n📋 Métriques Dashboard...")
            dashboard_metrics = service.get_dashboard_metrics()
            print(f"LRC Dashboard: {dashboard_metrics.get('lrc_total', 0):,.2f}")
            
    else:
        print(f"❌ Erreur: {result.get('message')}")

if __name__ == "__main__":
    test_ppna_analysis()