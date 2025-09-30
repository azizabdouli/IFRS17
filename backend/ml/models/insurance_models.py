# ml/models/insurance_models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple, Optional
import logging

from .base_model import BaseMLModel

logger = logging.getLogger(__name__)

class ClaimsPredictionModel(BaseMLModel):
    """
    Modèle de prédiction des sinistres/réclamations
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(f"claims_prediction_{model_type}")
        self.model_type = 'regression'
        self.model_variant = model_type
        
    def build_model(self, **kwargs):
        """Construction du modèle de prédiction des sinistres"""
        if self.model_variant == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif self.model_variant == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif self.model_variant == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif self.model_variant == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_variant}")

class ProfitabilityModel(BaseMLModel):
    """
    Modèle de prédiction de la rentabilité des contrats
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(f"profitability_{model_type}")
        self.model_type = 'regression'
        self.model_variant = model_type
        
    def build_model(self, **kwargs):
        """Construction du modèle de rentabilité"""
        if self.model_variant == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42
            )
        elif self.model_variant == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.05),
                random_state=42
            )
        elif self.model_variant == 'linear':
            self.model = Ridge(
                alpha=kwargs.get('alpha', 1.0),
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )

class RiskClassificationModel(BaseMLModel):
    """
    Modèle de classification du risque des contrats
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(f"risk_classification_{model_type}")
        self.model_type = 'classification'
        self.model_variant = model_type
        
    def build_model(self, **kwargs):
        """Construction du modèle de classification des risques"""
        if self.model_variant == 'xgboost':
            # Configuration adaptée pour XGBoost avec classes multiples
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                objective='multi:softprob',  # Classification multiclasse
                num_class=3,  # 3 classes de risque
                random_state=42
            )
        elif self.model_variant == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif self.model_variant == 'logistic':
            self.model = LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=1000,
                random_state=42
            )
        else:
            # Par défaut, utiliser RandomForest (plus robuste)
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    
    def create_risk_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Création des labels de risque basés sur les données historiques
        """
        # Calcul d'un score de risque composite
        risk_score = pd.Series(0.0, index=df.index)
        
        if 'MNTPPNA' in df.columns and 'MNTPRNET' in df.columns:
            # Ratio PPNA/Prime (plus élevé = plus risqué)
            # Conversion sécurisée en numérique
            mntppna_numeric = pd.to_numeric(df['MNTPPNA'], errors='coerce').fillna(0)
            mntprnet_numeric = pd.to_numeric(df['MNTPRNET'], errors='coerce').fillna(0)
            ppna_ratio = mntppna_numeric / (mntprnet_numeric + 1e-8)
            if ppna_ratio.max() > 0:
                risk_score += ppna_ratio / ppna_ratio.max()
        
        if 'DUREE' in df.columns:
            # Contrats plus longs sont potentiellement plus risqués
            # Conversion sécurisée en numérique
            duree_numeric = pd.to_numeric(df['DUREE'], errors='coerce').fillna(0)
            if duree_numeric.max() > 0:
                duration_score = duree_numeric / duree_numeric.max()
                risk_score += duration_score
        
        if 'MNTPRNET' in df.columns:
            # Primes très faibles ou très élevées peuvent être risquées
            # Conversion sécurisée en numérique
            mntprnet_numeric = pd.to_numeric(df['MNTPRNET'], errors='coerce').fillna(0)
            if mntprnet_numeric.std() > 0:  # Éviter division par zéro
                prime_score = np.abs(mntprnet_numeric - mntprnet_numeric.median()) / (mntprnet_numeric.std() + 1e-8)
                if prime_score.max() > 0:
                    risk_score += prime_score / prime_score.max()
        
        # Classification en 3 catégories de risque
        try:
            risk_labels = pd.cut(risk_score, bins=3, labels=['Faible', 'Moyen', 'Élevé'])
        except ValueError:
            # Fallback si les données ne permettent pas la classification
            risk_labels = pd.Series(['Moyen'] * len(df), index=df.index)
        
        # Assurer la continuité des index pour éviter les erreurs de masquage
        return risk_labels.reset_index(drop=True)

class ContractClusteringModel:
    """
    Modèle de clustering des contrats d'assurance
    """
    
    def __init__(self, clustering_type: str = 'kmeans'):
        self.clustering_type = clustering_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cluster_labels = None
        
    def build_model(self, n_clusters: int = 5, **kwargs):
        """Construction du modèle de clustering"""
        if self.clustering_type == 'kmeans':
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                **kwargs
            )
        elif self.clustering_type == 'dbscan':
            self.model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
        else:
            raise ValueError(f"Type de clustering non supporté: {self.clustering_type}")
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Entraînement et prédiction du clustering
        """
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Clustering
        if self.clustering_type == 'kmeans':
            self.cluster_labels = self.model.fit_predict(X_scaled)
        elif self.clustering_type == 'dbscan':
            self.cluster_labels = self.model.fit_predict(X_scaled)
        
        self.is_fitted = True
        return self.cluster_labels
    
    def get_cluster_characteristics(self, X: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Analyse des caractéristiques de chaque cluster
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'analyser les clusters")
        
        cluster_stats = {}
        df_with_clusters = X.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        for cluster_id in np.unique(self.cluster_labels):
            if cluster_id == -1:  # Outliers dans DBSCAN
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Calculer les moyennes de façon sécurisée
            avg_prime = 0.0
            if 'MNTPRNET' in cluster_data.columns:
                try:
                    avg_prime = float(cluster_data['MNTPRNET'].mean())
                except (TypeError, ValueError):
                    avg_prime = 0.0
            
            avg_duration = 0.0
            if 'DUREE' in cluster_data.columns:
                try:
                    # Convertir DUREE en numérique si possible, sinon compter les modes
                    duree_numeric = pd.to_numeric(cluster_data['DUREE'], errors='coerce')
                    if not duree_numeric.isna().all():
                        avg_duration = float(duree_numeric.mean())
                    else:
                        # Pour les colonnes catégorielles, compter le mode le plus fréquent
                        avg_duration = float(len(cluster_data['DUREE'].mode()))
                except (TypeError, ValueError):
                    avg_duration = 0.0
            
            avg_ppna = 0.0
            if 'MNTPPNA' in cluster_data.columns:
                try:
                    avg_ppna = float(cluster_data['MNTPPNA'].mean())
                except (TypeError, ValueError):
                    avg_ppna = 0.0
            
            main_product = 'N/A'
            if 'CODPROD' in cluster_data.columns:
                try:
                    mode_values = cluster_data['CODPROD'].mode()
                    if len(mode_values) > 0:
                        main_product = str(mode_values.iloc[0])
                except (TypeError, ValueError, IndexError):
                    main_product = 'N/A'
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'avg_prime': avg_prime,
                'avg_duration': avg_duration,
                'avg_ppna': avg_ppna,
                'main_product': main_product
            }
        
        return cluster_stats

class AnomalyDetectionModel:
    """
    Modèle de détection d'anomalies dans les contrats
    """
    
    def __init__(self, method: str = 'isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def build_model(self, **kwargs):
        """Construction du modèle de détection d'anomalies"""
        if self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=42
            )
        elif self.method == 'local_outlier_factor':
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.1)
            )
        elif self.method == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            self.model = OneClassSVM(
                nu=kwargs.get('nu', 0.1),
                kernel='rbf'
            )
        else:
            raise ValueError(f"Méthode non supportée: {self.method}")
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Détection des anomalies
        """
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Détection
        if self.method == 'local_outlier_factor':
            predictions = self.model.fit_predict(X_scaled)
        else:
            self.model.fit(X_scaled)
            predictions = self.model.predict(X_scaled)
        
        self.is_fitted = True
        # Convertir en 0/1 (0 = anomalie, 1 = normal)
        return (predictions == 1).astype(int)
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Scores d'anomalie
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_scaled)
        elif hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X_scaled)
        else:
            return None

class LRCPredictionModel(BaseMLModel):
    """
    Modèle de prédiction du Liability for Remaining Coverage (LRC)
    Spécifique à IFRS 17
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(f"lrc_prediction_{model_type}")
        self.model_type = 'regression'
        self.model_variant = model_type
        
    def build_model(self, **kwargs):
        """Construction du modèle LRC"""
        if self.model_variant == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                reg_alpha=kwargs.get('reg_alpha', 0.1),
                reg_lambda=kwargs.get('reg_lambda', 1),
                random_state=42
            )
        elif self.model_variant == 'ensemble':
            from sklearn.ensemble import VotingRegressor
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            ridge = Ridge(alpha=1.0)
            
            self.model = VotingRegressor([
                ('rf', rf),
                ('xgb', xgb_model),
                ('ridge', ridge)
            ])
        else:
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 150),
                max_depth=kwargs.get('max_depth', 12),
                random_state=42
            )
    
    def create_lrc_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Création d'une variable cible LRC synthétique basée sur les données disponibles
        """
        # Estimation du LRC basée sur PPNA et autres facteurs
        lrc_estimate = pd.Series(0.0, index=df.index)
        
        if 'MNTPPNA' in df.columns:
            # Conversion sécurisée des types mixtes
            mntppna_numeric = pd.to_numeric(df['MNTPPNA'], errors='coerce').fillna(0)
            lrc_estimate += mntppna_numeric
        
        if 'MNTPRNET' in df.columns and 'DUREE' in df.columns:
            # Estimation basée sur la prime et la durée restante
            # Conversion sécurisée des types mixtes
            mntprnet_numeric = pd.to_numeric(df['MNTPRNET'], errors='coerce').fillna(0)
            duree_numeric = pd.to_numeric(df['DUREE'], errors='coerce').fillna(12)  # Default 12 mois
            remaining_premium = mntprnet_numeric * (duree_numeric / 12)  # Conversion en années
            lrc_estimate += remaining_premium * 0.8  # Facteur d'ajustement
        
        if 'MNTACCESS' in df.columns:
            # Conversion sécurisée des types mixtes
            mntaccess_numeric = pd.to_numeric(df['MNTACCESS'], errors='coerce').fillna(0)
            lrc_estimate += mntaccess_numeric
        
        return lrc_estimate