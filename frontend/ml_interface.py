# frontend/ml_interface.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
from typing import Dict, Any
import json

# Configuration de la page
st.set_page_config(
    page_title="IFRS 17 - Machine Learning",
    page_icon="🤖",
    layout="wide"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:8001/ml"

def upload_data_to_api(uploaded_file):
    """Upload des données vers l'API ML"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    try:
        response = requests.post(f"{API_BASE_URL}/upload-data", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de l'upload: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

def train_model(model_type: str, algorithm: str = "xgboost"):
    """Entraînement d'un modèle via l'API"""
    try:
        response = requests.post(f"{API_BASE_URL}/train/{model_type}", 
                               params={"model_type": algorithm})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de l'entraînement: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

def get_models_summary():
    """Récupération du résumé des modèles"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/summary")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

def perform_clustering(n_clusters: int = 5, method: str = "kmeans"):
    """Clustering via l'API"""
    try:
        response = requests.post(f"{API_BASE_URL}/clustering", 
                               params={"n_clusters": n_clusters, "clustering_type": method})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors du clustering: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

def detect_anomalies(method: str = "isolation_forest", contamination: float = 0.1):
    """Détection d'anomalies via l'API"""
    try:
        response = requests.post(f"{API_BASE_URL}/anomaly-detection", 
                               params={"method": method, "contamination": contamination})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la détection d'anomalies: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

def get_ml_insights():
    """Récupération des insights ML"""
    try:
        response = requests.get(f"{API_BASE_URL}/insights")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

# Interface principale
def main():
    st.title("🤖 IFRS 17 - Machine Learning Dashboard")
    st.markdown("---")
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Sélectionnez une section",
        ["🏠 Accueil", "📊 Upload & Insights", "🎯 Modèles Prédictifs", 
         "🔍 Clustering", "⚠️ Détection d'Anomalies", "📈 Résultats"]
    )
    
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "📊 Upload & Insights":
        show_upload_insights_page()
    elif page == "🎯 Modèles Prédictifs":
        show_predictive_models_page()
    elif page == "🔍 Clustering":
        show_clustering_page()
    elif page == "⚠️ Détection d'Anomalies":
        show_anomaly_detection_page()
    elif page == "📈 Résultats":
        show_results_page()

def show_home_page():
    """Page d'accueil"""
    st.header("🏠 Bienvenue dans le module Machine Learning")
    
    st.markdown("""
    ### 🎯 Fonctionnalités disponibles
    
    Cette interface vous permet d'utiliser des algorithmes de machine learning avancés 
    pour analyser vos données IFRS 17:
    
    **🔮 Modèles Prédictifs:**
    - 📊 Prédiction des sinistres
    - 💰 Prédiction de la rentabilité
    - ⚠️ Classification des risques
    - 📈 Prédiction du LRC (Liability for Remaining Coverage)
    
    **🔍 Analyse Exploratoire:**
    - 🎯 Clustering de contrats (segmentation du portefeuille)
    - 🚨 Détection d'anomalies
    - 📊 Analyse des patterns cachés
    
    **🚀 Algorithmes Supportés:**
    - XGBoost (recommandé pour la performance)
    - LightGBM
    - Random Forest
    - Modèles linéaires (Ridge, Lasso)
    - SVM
    """)
    
    # Vérification de l'état de l'API
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ Service ML opérationnel")
            st.json(health_data)
        else:
            st.error("❌ Service ML non disponible")
    except:
        st.error("❌ Impossible de se connecter au service ML")

def show_upload_insights_page():
    """Page d'upload et insights"""
    st.header("📊 Upload des Données & Insights")
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier de données",
        type=['csv', 'xlsx'],
        help="Formats supportés: CSV, Excel"
    )
    
    if uploaded_file is not None:
        st.success(f"Fichier uploadé: {uploaded_file.name}")
        
        # Upload vers l'API
        if st.button("📤 Envoyer vers le service ML"):
            with st.spinner("Upload en cours..."):
                result = upload_data_to_api(uploaded_file)
                
                if result:
                    st.success("✅ Données uploadées avec succès!")
                    
                    # Affichage des informations sur les données
                    st.subheader("📋 Informations sur les données")
                    data_info = result["data_info"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nombre de lignes", data_info["n_rows"])
                        st.metric("Nombre de colonnes", data_info["n_columns"])
                    
                    with col2:
                        st.write("**Colonnes disponibles:**")
                        st.write(data_info["columns"])
                    
                    # Échantillon des données
                    st.subheader("🔍 Aperçu des données")
                    sample_df = pd.DataFrame(data_info["sample_data"])
                    st.dataframe(sample_df, width="stretch")
                    
                    # Génération des insights
                    if st.button("🧠 Générer les insights ML"):
                        with st.spinner("Analyse en cours..."):
                            insights = get_ml_insights()
                            
                            if insights:
                                display_insights(insights)

def display_insights(insights: Dict[str, Any]):
    """Affichage des insights ML"""
    st.subheader("🧠 Insights Machine Learning")
    
    # Vue d'ensemble des données
    if "data_overview" in insights:
        overview = insights["data_overview"]
        st.write("### 📊 Vue d'ensemble")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Contrats", overview["n_contracts"])
        with col2:
            st.metric("Features", overview["n_features"])
        with col3:
            if overview["date_range"]["min"]:
                st.write(f"**Période:** {overview['date_range']['min']} - {overview['date_range']['max']}")
    
    # Métriques business
    if "business_metrics" in insights:
        metrics = insights["business_metrics"]
        st.write("### 💰 Métriques Business")
        
        if "total_premium" in metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prime Totale", f"{metrics['total_premium']:,.0f}")
            with col2:
                st.metric("Prime Moyenne", f"{metrics['avg_premium']:,.0f}")
            with col3:
                if "total_ppna" in metrics:
                    st.metric("PPNA Total", f"{metrics['total_ppna']:,.0f}")
    
    # Recommandations
    if "model_recommendations" in insights:
        reco = insights["model_recommendations"]
        st.write("### 🎯 Recommandations")
        st.info(f"**Algorithme recommandé:** {reco['preferred_algorithm']}")
        st.write(f"**Raison:** {reco['reason']}")

def show_predictive_models_page():
    """Page des modèles prédictifs"""
    st.header("🎯 Modèles Prédictifs")
    
    # Sélection du type de modèle
    model_type = st.selectbox(
        "Choisissez le type de modèle",
        ["claims-prediction", "profitability", "risk-classification", "lrc-prediction"],
        format_func=lambda x: {
            "claims-prediction": "📊 Prédiction des Sinistres",
            "profitability": "💰 Prédiction de Rentabilité", 
            "risk-classification": "⚠️ Classification des Risques",
            "lrc-prediction": "📈 Prédiction LRC (IFRS 17)"
        }[x]
    )
    
    # Sélection de l'algorithme
    algorithm = st.selectbox(
        "Algorithme",
        ["xgboost", "lightgbm", "random_forest", "linear"],
        help="XGBoost est généralement recommandé pour les meilleures performances"
    )
    
    # Description du modèle sélectionné
    descriptions = {
        "claims-prediction": "Prédit le ratio sinistres/primes basé sur les caractéristiques du contrat",
        "profitability": "Estime la rentabilité future d'un contrat d'assurance", 
        "risk-classification": "Classe les contrats en catégories de risque (Faible/Moyen/Élevé)",
        "lrc-prediction": "Prédit le montant LRC selon la norme IFRS 17"
    }
    
    st.info(f"**Description:** {descriptions[model_type]}")
    
    # Entraînement du modèle
    if st.button(f"🚀 Entraîner le modèle {model_type}"):
        with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
            result = train_model(model_type, algorithm)
            
            if result:
                st.success(f"✅ Entraînement du modèle {model_type} démarré!")
                st.json(result)

def show_clustering_page():
    """Page de clustering"""
    st.header("🔍 Clustering de Contrats")
    
    st.markdown("""
    Le clustering permet de segmenter automatiquement votre portefeuille de contrats
    en groupes homogènes basés sur leurs caractéristiques.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Nombre de clusters", 2, 10, 5)
        method = st.selectbox("Méthode", ["kmeans", "dbscan"])
    
    with col2:
        st.write("**Applications:**")
        st.write("- Segmentation du portefeuille")
        st.write("- Tarification par segment")
        st.write("- Allocation des ressources")
        st.write("- Stratégies marketing ciblées")
    
    if st.button("🎯 Effectuer le clustering"):
        with st.spinner("Clustering en cours..."):
            result = perform_clustering(n_clusters, method)
            
            if result:
                st.success("✅ Clustering terminé!")
                
                # Affichage des résultats
                results = result["results"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clusters identifiés", results["n_clusters"])
                
                with col2:
                    # Distribution des clusters
                    distribution = results["cluster_distribution"]
                    fig = px.pie(
                        values=list(distribution.values()),
                        names=[f"Cluster {k}" for k in distribution.keys()],
                        title="Distribution des Clusters"
                    )
                    st.plotly_chart(fig, width="stretch")
                
                # Caractéristiques des clusters
                st.subheader("📊 Caractéristiques des Clusters")
                
                if "cluster_characteristics" in results:
                    chars = results["cluster_characteristics"]
                    
                    for cluster_id, char in chars.items():
                        with st.expander(f"Cluster {cluster_id} ({char['size']} contrats)"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prime moyenne", f"{char['avg_prime']:,.0f}")
                            with col2:
                                st.metric("Durée moyenne", f"{char['avg_duration']:.1f}")
                            with col3:
                                st.metric("PPNA moyenne", f"{char['avg_ppna']:,.0f}")
                            
                            st.write(f"**Produit principal:** {char['main_product']}")

def show_anomaly_detection_page():
    """Page de détection d'anomalies"""
    st.header("⚠️ Détection d'Anomalies")
    
    st.markdown("""
    La détection d'anomalies identifie les contrats atypiques qui pourraient nécessiter
    une attention particulière.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Méthode de détection",
            ["isolation_forest", "local_outlier_factor", "one_class_svm"],
            format_func=lambda x: {
                "isolation_forest": "Isolation Forest (recommandé)",
                "local_outlier_factor": "Local Outlier Factor",
                "one_class_svm": "One-Class SVM"
            }[x]
        )
        
        contamination = st.slider(
            "Taux d'anomalies attendu (%)",
            1, 20, 10
        ) / 100
    
    with col2:
        st.write("**Cas d'usage:**")
        st.write("- Détection de fraude")
        st.write("- Contrôle qualité des données")
        st.write("- Identification de contrats à risque")
        st.write("- Audit automatisé")
    
    if st.button("🔍 Détecter les anomalies"):
        with st.spinner("Détection en cours..."):
            result = detect_anomalies(method, contamination)
            
            if result:
                st.success("✅ Détection terminée!")
                
                results = result["results"]
                
                # Métriques principales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies détectées", results["n_anomalies"])
                with col2:
                    st.metric("Taux d'anomalies", results["anomaly_rate"])
                with col3:
                    st.metric("Méthode utilisée", method)
                
                # Contrats anormaux
                if results["anomalous_contracts"]:
                    st.subheader("🚨 Contrats Anormaux (échantillon)")
                    anomalous_df = pd.DataFrame(results["anomalous_contracts"])
                    st.dataframe(anomalous_df, width="stretch")

def show_results_page():
    """Page des résultats"""
    st.header("📈 Résultats des Modèles")
    
    # Récupération du résumé des modèles
    summary = get_models_summary()
    
    if summary and summary["trained_models"]:
        st.success(f"✅ {len(summary['trained_models'])} modèle(s) entraîné(s)")
        
        # Liste des modèles
        st.subheader("🎯 Modèles Disponibles")
        for model in summary["trained_models"]:
            st.write(f"- {model}")
        
        # Performance des modèles
        if "model_performance" in summary and summary["model_performance"]:
            st.subheader("📊 Performance des Modèles")
            
            for model_name, metrics in summary["model_performance"].items():
                with st.expander(f"📈 {model_name}"):
                    
                    # Affichage des métriques selon le type
                    if "r2" in metrics:  # Modèle de régression
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R²", f"{metrics['r2']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.2f}")
                        with col3:
                            st.metric("MAE", f"{metrics['mae']:.2f}")
                        with col4:
                            st.metric("MSE", f"{metrics['mse']:.2f}")
                    
                    elif "accuracy" in metrics:  # Modèle de classification
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        # Bouton de sauvegarde
        if st.button("💾 Sauvegarder tous les modèles"):
            try:
                response = requests.post(f"{API_BASE_URL}/models/save")
                if response.status_code == 200:
                    st.success("✅ Modèles sauvegardés avec succès!")
                else:
                    st.error("❌ Erreur lors de la sauvegarde")
            except:
                st.error("❌ Impossible de se connecter au service")
    
    else:
        st.info("ℹ️ Aucun modèle entraîné. Utilisez la section 'Modèles Prédictifs' pour commencer.")

if __name__ == "__main__":
    main()