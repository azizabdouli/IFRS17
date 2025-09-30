# ml/ml_service.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from datetime import datetime

from .data_preprocessing import DataPreprocessor
from .models.insurance_models import (
    ClaimsPredictionModel, ProfitabilityModel, RiskClassificationModel,
    ContractClusteringModel, AnomalyDetectionModel, LRCPredictionModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLService:
    """
    Service principal pour toutes les fonctionnalités ML du projet IFRS17
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.model_results = {}
        
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Chargement et préprocessing des données
        """
        logger.info(f"📁 Chargement des données depuis {data_path}")
        
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .xlsx ou .csv")
        
        logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    def train_claims_prediction_model(self, df: pd.DataFrame, target_column: str = None, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Entraînement du modèle de prédiction des sinistres
        """
        logger.info("🎯 Entraînement du modèle de prédiction des sinistres")
        
        # Création d'une cible synthétique si pas fournie
        if target_column is None or target_column not in df.columns:
            # Créer une variable cible basée sur le ratio PPNA/Prime (proxy pour les sinistres)
            if 'MNTPPNA' in df.columns and 'MNTPRNET' in df.columns:
                # Conversion sécurisée des types mixtes
                mntppna_numeric = pd.to_numeric(df['MNTPPNA'], errors='coerce').fillna(0)
                mntprnet_numeric = pd.to_numeric(df['MNTPRNET'], errors='coerce').fillna(1)  # éviter division par 0
                df['claims_ratio'] = mntppna_numeric / (mntprnet_numeric + 1e-8)
                target_column = 'claims_ratio'
            else:
                raise ValueError("Impossible de créer une variable cible pour les sinistres")
        
        # Preprocessing
        X, y = self.preprocessor.prepare_data_for_training(df, target_column)
        
        # Modèle
        model = ClaimsPredictionModel(model_type)
        results = model.train(X, y)
        
        # Sauvegarde
        model_key = f'claims_prediction_{model_type}'
        self.models[model_key] = model
        self.model_results[model_key] = results
        
        logger.info("✅ Modèle de prédiction des sinistres entraîné avec succès")
        return results
    
    def train_profitability_model(self, df: pd.DataFrame, target_column: str = None, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Entraînement du modèle de prédiction de rentabilité
        """
        logger.info("💰 Entraînement du modèle de rentabilité")
        
        # Création d'une cible de rentabilité
        if target_column is None or target_column not in df.columns:
            if 'MNTPRNET' in df.columns and 'MNTPPNA' in df.columns:
                # Profit estimé = Prime - PPNA - Coûts estimés
                # Conversion sécurisée des types mixtes
                mntprnet_numeric = pd.to_numeric(df['MNTPRNET'], errors='coerce').fillna(0)
                mntppna_numeric = pd.to_numeric(df['MNTPPNA'], errors='coerce').fillna(0)
                estimated_costs = mntprnet_numeric * 0.15  # 15% de coûts estimés
                df['profitability'] = mntprnet_numeric - mntppna_numeric - estimated_costs
                target_column = 'profitability'
            else:
                raise ValueError("Impossible de créer une variable cible pour la rentabilité")
        
        # Preprocessing - ne pas inclure la target dans le preprocessing
        df_copy = df.copy()
        target_data = df_copy[target_column]
        df_copy = df_copy.drop(columns=[target_column])
        
        X, _ = self.preprocessor.prepare_data_for_training(df_copy)
        y = target_data
        
        # Modèle
        model = ProfitabilityModel(model_type)
        results = model.train(X, y)
        
        # Sauvegarde
        model_key = f'profitability_{model_type}'
        self.models[model_key] = model
        self.model_results[model_key] = results
        
        logger.info("✅ Modèle de rentabilité entraîné avec succès")
        return results
    
    def train_risk_classification_model(self, df: pd.DataFrame, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Entraînement du modèle de classification des risques
        """
        logger.info("⚠️ Entraînement du modèle de classification des risques")
        
        # Création des labels de risque
        model = RiskClassificationModel(model_type)
        risk_labels = model.create_risk_labels(df)
        
        # Vérification que risk_labels n'est pas None
        if risk_labels is None:
            raise ValueError("Impossible de créer les labels de risque")
        
        # Ajout des labels au dataframe
        df_copy = df.copy()
        df_copy['risk_level'] = risk_labels
        
        # Preprocessing - ne pas inclure la target dans le preprocessing
        target_data = df_copy['risk_level']
        df_copy = df_copy.drop(columns=['risk_level'])
        
        X, _ = self.preprocessor.prepare_data_for_training(df_copy)
        y = target_data
        
        # Filtrage des valeurs non nulles
        valid_indices = y.dropna().index
        X_filtered = X.loc[valid_indices]
        y_filtered = y.loc[valid_indices]
        
        # Encodage des labels de risque
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_filtered.astype(str))
        
        # Entraînement
        results = model.train(X_filtered, pd.Series(y_encoded, index=X_filtered.index))
        
        # Sauvegarde
        model_key = f'risk_classification_{model_type}'
        self.models[model_key] = model
        self.model_results[model_key] = results
        self.model_results[model_key]['label_encoder'] = le
        
        logger.info("✅ Modèle de classification des risques entraîné avec succès")
        return results
        self.models[model_key] = model
        self.model_results[model_key] = results
        self.model_results[model_key]['label_encoder'] = le
        
        logger.info("✅ Modèle de classification des risques entraîné avec succès")
        return results
    
    def perform_contract_clustering(self, df: pd.DataFrame, n_clusters: int = 5, clustering_type: str = 'kmeans') -> Dict[str, Any]:
        """
        Clustering des contrats
        """
        logger.info(f"🎯 Clustering des contrats en {n_clusters} groupes")
        
        # Preprocessing pour clustering
        X, _ = self.preprocessor.prepare_data_for_training(df)
        
        # Modèle de clustering
        model = ContractClusteringModel(clustering_type)
        model.build_model(n_clusters=n_clusters)
        
        # Clustering
        cluster_labels = model.fit_predict(X)
        
        # Analyse des clusters
        cluster_characteristics = model.get_cluster_characteristics(df)
        
        # Conversion des clés en string pour la sérialisation JSON
        cluster_characteristics_clean = {
            str(k): v for k, v in cluster_characteristics.items()
        }
        
        # Sauvegarde
        model_key = f'clustering_{clustering_type}'
        self.models[model_key] = model
        
        results = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_characteristics': cluster_characteristics_clean,
            'n_clusters': int(len(np.unique(cluster_labels))),
            'cluster_distribution': {str(k): int(v) for k, v in pd.Series(cluster_labels).value_counts().to_dict().items()}
        }
        
        self.model_results[model_key] = results
        
        logger.info("✅ Clustering terminé avec succès")
        return results
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'isolation_forest', contamination: float = 0.1) -> Dict[str, Any]:
        """
        Détection d'anomalies dans les contrats
        """
        logger.info(f"🔍 Détection d'anomalies avec {method}")
        
        # Preprocessing
        X, _ = self.preprocessor.prepare_data_for_training(df)
        
        # Modèle de détection d'anomalies
        model = AnomalyDetectionModel(method)
        model.build_model(contamination=contamination)
        
        # Détection
        anomaly_labels = model.fit_predict(X)
        anomaly_scores = model.get_anomaly_scores(X)
        
        # Analyse des anomalies
        n_anomalies = np.sum(anomaly_labels == 0)
        anomaly_rate = n_anomalies / len(anomaly_labels)
        
        # Identifier les contrats anormaux
        anomaly_indices = np.where(anomaly_labels == 0)[0]
        anomalous_contracts = df.iloc[anomaly_indices]
        
        # Sauvegarde
        model_key = f'anomaly_detection_{method}'
        self.models[model_key] = model
        
        results = {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'anomalous_contracts': anomalous_contracts.to_dict('records')
        }
        
        self.model_results[model_key] = results
        
        logger.info(f"✅ Détection terminée: {n_anomalies} anomalies détectées ({anomaly_rate:.2%})")
        return results
    
    def train_lrc_prediction_model(self, df: pd.DataFrame, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Entraînement du modèle de prédiction LRC (IFRS 17)
        """
        logger.info("📊 Entraînement du modèle de prédiction LRC")
        
        # Création de la variable cible LRC
        model = LRCPredictionModel(model_type)
        lrc_target = model.create_lrc_target(df)
        df['lrc_estimate'] = lrc_target
        
        # Preprocessing
        X, y = self.preprocessor.prepare_data_for_training(df, 'lrc_estimate')
        
        # Entraînement
        results = model.train(X, y)
        
        # Sauvegarde
        model_key = f'lrc_prediction_{model_type}'
        self.models[model_key] = model
        self.model_results[model_key] = results
        
        logger.info("✅ Modèle LRC entraîné avec succès")
        return results
    
    def predict_with_model(self, model_name: str, df: pd.DataFrame) -> np.ndarray:
        """
        Prédiction avec un modèle entraîné
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé. Modèles disponibles: {list(self.models.keys())}")
        
        # Preprocessing des nouvelles données
        X, _ = self.preprocessor.prepare_data_for_training(df)
        
        # Prédiction
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Résumé de tous les modèles entraînés
        """
        summary = {
            'trained_models': list(self.models.keys()),
            'model_performance': {}
        }
        
        for model_name, results in self.model_results.items():
            if 'validation_metrics' in results:
                summary['model_performance'][model_name] = results['validation_metrics']
        
        return summary
    
    def save_all_models(self, save_dir: str = "models"):
        """
        Sauvegarde de tous les modèles entraînés
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            if hasattr(model, 'save_model'):
                filepath = os.path.join(save_dir, f"{model_name}_{timestamp}.joblib")
                model.save_model(filepath)
        
        logger.info(f"💾 Tous les modèles sauvegardés dans {save_dir}")
    
    def generate_ml_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Génération d'insights ML globaux
        """
        insights = {
            'data_overview': {
                'n_contracts': len(df),
                'n_features': len(df.columns),
                'date_range': {
                    'min': df['DEBEFFQUI'].min() if 'DEBEFFQUI' in df.columns else None,
                    'max': df['DEBEFFQUI'].max() if 'DEBEFFQUI' in df.columns else None
                }
            },
            'business_metrics': {},
            'model_recommendations': {}
        }
        
        # Métriques business
        if 'MNTPRNET' in df.columns:
            insights['business_metrics']['total_premium'] = df['MNTPRNET'].sum()
            insights['business_metrics']['avg_premium'] = df['MNTPRNET'].mean()
            insights['business_metrics']['premium_std'] = df['MNTPRNET'].std()
        
        if 'MNTPPNA' in df.columns:
            insights['business_metrics']['total_ppna'] = df['MNTPPNA'].sum()
            insights['business_metrics']['avg_ppna'] = df['MNTPPNA'].mean()
        
        # Recommandations de modèles
        if len(df) > 10000:
            insights['model_recommendations']['preferred_algorithm'] = 'xgboost'
            insights['model_recommendations']['reason'] = 'Dataset volumineux - XGBoost recommandé pour la performance'
        else:
            insights['model_recommendations']['preferred_algorithm'] = 'random_forest'
            insights['model_recommendations']['reason'] = 'Dataset modéré - Random Forest recommandé pour l\'interprétabilité'
        
        return insights