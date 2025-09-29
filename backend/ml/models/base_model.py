# ml/models/base_model.py

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseMLModel(ABC):
    """
    Classe de base pour tous les modèles ML du projet IFRS17
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.model_type = None  # 'regression' ou 'classification'
        
    @abstractmethod
    def build_model(self, **kwargs):
        """Construction du modèle"""
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Entraînement du modèle avec validation
        """
        logger.info(f"🚀 Début de l'entraînement du modèle {self.model_name}")
        
        # Vérifications de sécurité des données
        if X is None:
            raise ValueError("❌ Les données d'entrée X sont None")
        if y is None:
            raise ValueError("❌ Les données cible y sont None")
        if len(X) == 0:
            raise ValueError("❌ Le DataFrame X est vide")
        if len(y) == 0:
            raise ValueError("❌ La série y est vide")
        if len(X) != len(y):
            raise ValueError(f"❌ Tailles incompatibles: X={len(X)}, y={len(y)}")
        
        logger.info(f"✅ Validation des données: X={X.shape}, y={len(y)}")
        
        # Sauvegarde des noms de features
        self.feature_names = X.columns.tolist()
        
        # Split train/validation - gestion robuste de la stratification
        try:
            if self.model_type == 'classification' and len(y.unique()) > 1:
                # Vérifier si la stratification est possible
                min_class_count = y.value_counts().min()
                if min_class_count >= 2:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=validation_split, random_state=42, stratify=y
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=validation_split, random_state=42
                    )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
        except Exception:
            # Fallback sans stratification
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        
        # Construction du modèle si pas encore fait
        if self.model is None:
            self.build_model(**kwargs)
        
        # Entraînement
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Validation
        train_score = self.evaluate(X_train, y_train)
        val_score = self.evaluate(X_val, y_val)
        
        # Cross-validation avec gestion d'erreurs
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(X)//2))
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean = val_score.get('r2', val_score.get('accuracy', 0))
            cv_std = 0
        
        results = {
            'train_metrics': train_score,
            'validation_metrics': val_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'feature_importance': self.get_feature_importance()
        }
        
        logger.info(f"✅ Entraînement terminé pour {self.model_name}")
        logger.info(f"📊 Score validation: {val_score}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction
        """
        if not self.is_trained:
            raise ValueError(f"Le modèle {self.model_name} n'est pas encore entraîné")
        
        # Vérifier que les features correspondent
        if self.feature_names and X.columns.tolist() != self.feature_names:
            logger.warning("⚠️ Les features ne correspondent pas exactement au modèle entraîné")
            # Réorganiser les colonnes pour correspondre
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction de probabilités (pour classification)
        """
        if not self.is_trained:
            raise ValueError(f"Le modèle {self.model_name} n'est pas encore entraîné")
        
        if self.model_type != 'classification':
            raise ValueError("predict_proba n'est disponible que pour les modèles de classification")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Ce modèle ne supporte pas predict_proba")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Évaluation du modèle
        """
        predictions = self.predict(X)
        
        if self.model_type == 'regression':
            return {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions)
            }
        elif self.model_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted'),
                'recall': recall_score(y, predictions, average='weighted'),
                'f1': f1_score(y, predictions, average='weighted')
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Importance des features
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # Pour les modèles linéaires
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return None
    
    def save_model(self, filepath: str):
        """
        Sauvegarde du modèle
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modèle sauvegardé : {filepath}")
    
    def load_model(self, filepath: str):
        """
        Chargement du modèle
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"📂 Modèle chargé : {filepath}")