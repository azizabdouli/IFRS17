# ml/optimized_ml_service.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from datetime import datetime
from functools import lru_cache
import asyncio
import concurrent.futures
from cachetools import TTLCache
import hashlib

from .data_preprocessing import OptimizedDataPreprocessor
from .ml_service import OptimizedMLService as BaseMLService  # Service de base
from .models.insurance_models import (
    ContractClusteringModel,
    ClaimsPredictionModel,
    ProfitabilityModel,
    RiskClassificationModel,
    AnomalyDetectionModel,
    LRCPredictionModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLService(BaseMLService):
    """
    Service ML optimisé avec cache, lazy loading et traitement asynchrone
    Hérite du service original pour compatibilité
    """
    
    def __init__(self, max_workers: int = 4, cache_ttl: int = 3600):
        # Initialiser le service parent
        super().__init__()
        
        # Remplacer le preprocessor par la version optimisée
        self.preprocessor = OptimizedDataPreprocessor()
        
        # Optimisations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.model_cache = TTLCache(maxsize=64, ttl=cache_ttl)
        self.data_cache = TTLCache(maxsize=32, ttl=cache_ttl)
        
        logger.info("🚀 Service ML optimisé initialisé")
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Génère un hash pour les données"""
        sample_data = df.head(100) if len(df) > 100 else df
        return hashlib.md5(pd.util.hash_pandas_object(sample_data).values).hexdigest()
    
    @lru_cache(maxsize=16)
    def _get_file_info(self, data_path: str) -> dict:
        """Cache des infos de fichier"""
        return {
            'size': os.path.getsize(data_path),
            'modified': os.path.getmtime(data_path),
            'extension': os.path.splitext(data_path)[1]
        }
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Chargement optimisé des données avec cache
        """
        file_info = self._get_file_info(data_path)
        cache_key = f"{data_path}_{file_info['modified']}"
        
        # Vérifier le cache
        if cache_key in self.data_cache:
            logger.info("📈 Données récupérées du cache")
            return self.data_cache[cache_key]
        
        logger.info(f"📁 Chargement optimisé ({file_info['size']/1024/1024:.1f}MB)")
        
        # Chargement optimisé selon la taille
        if file_info['size'] > 50 * 1024 * 1024:  # > 50MB
            df = self._load_large_file(data_path, file_info['extension'])
        else:
            df = self._load_small_file(data_path, file_info['extension'])
        
        logger.info(f"✅ Données chargées: {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
        
        # Cache le résultat
        self.data_cache[cache_key] = df
        return df
    
    def _load_small_file(self, data_path: str, extension: str) -> pd.DataFrame:
        """Chargement des petits fichiers"""
        if extension == '.xlsx':
            # Optimisations pour Excel
            return pd.read_excel(data_path, engine='openpyxl')
        elif extension == '.csv':
            # Optimisations pour CSV
            return pd.read_csv(data_path, low_memory=False)
        else:
            raise ValueError("Format non supporté. Utilisez .xlsx ou .csv")
    
    def _load_large_file(self, data_path: str, extension: str) -> pd.DataFrame:
        """Chargement optimisé des gros fichiers"""
        logger.info("📊 Fichier volumineux - chargement par chunks")
        
        if extension == '.csv':
            # Chargement par chunks pour CSV
            chunks = []
            chunk_size = 10000
            
            for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                
            return pd.concat(chunks, ignore_index=True)
        else:
            # Pour Excel, chargement normal (chunks non supportés)
            return self._load_small_file(data_path, extension)
    
    def train_claims_prediction_model(self, df: pd.DataFrame, target_column: str = None, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Version optimisée avec cache
        """
        # Générer clé de cache
        data_hash = self._get_data_hash(df)
        cache_key = f"claims_{model_type}_{data_hash}"
        
        # Vérifier le cache
        if cache_key in self.model_cache:
            logger.info("📈 Modèle de sinistres récupéré du cache")
            return self.model_cache[cache_key]
        
        # Utiliser la méthode parent si pas en cache
        logger.info("🎯 Entraînement optimisé du modèle de sinistres")
        result = super().train_claims_prediction_model(df, target_column, model_type)
        
        # Mettre en cache
        self.model_cache[cache_key] = result
        return result
    
    def train_profitability_model(self, df: pd.DataFrame, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Version optimisée avec cache
        """
        data_hash = self._get_data_hash(df)
        cache_key = f"profitability_{model_type}_{data_hash}"
        
        if cache_key in self.model_cache:
            logger.info("📈 Modèle de rentabilité récupéré du cache")
            return self.model_cache[cache_key]
        
        logger.info("💰 Entraînement optimisé du modèle de rentabilité")
        result = super().train_profitability_model(df, model_type)
        
        self.model_cache[cache_key] = result
        return result
    
    def train_risk_classification_model(self, df: pd.DataFrame, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Version optimisée avec cache
        """
        data_hash = self._get_data_hash(df)
        cache_key = f"risk_{model_type}_{data_hash}"
        
        if cache_key in self.model_cache:
            logger.info("📈 Modèle de classification récupéré du cache")
            return self.model_cache[cache_key]
        
        logger.info("⚠️ Entraînement optimisé du modèle de classification")
        result = super().train_risk_classification_model(df, model_type)
        
        self.model_cache[cache_key] = result
        return result
    
    def perform_contract_clustering(self, df: pd.DataFrame, n_clusters: int = 5, clustering_type: str = 'kmeans') -> Dict[str, Any]:
        """
        Version optimisée avec cache
        """
        data_hash = self._get_data_hash(df)
        cache_key = f"clustering_{clustering_type}_{n_clusters}_{data_hash}"
        
        if cache_key in self.model_cache:
            logger.info("📈 Clustering récupéré du cache")
            return self.model_cache[cache_key]
        
        logger.info("🎯 Clustering optimisé")
        result = super().perform_contract_clustering(df, n_clusters, clustering_type)
        
        self.model_cache[cache_key] = result
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Statistiques du cache pour monitoring
        """
        return {
            'model_cache_size': len(self.model_cache),
            'model_cache_maxsize': self.model_cache.maxsize,
            'data_cache_size': len(self.data_cache),
            'data_cache_maxsize': self.data_cache.maxsize,
            'cache_hit_info': {
                'model_keys': list(self.model_cache.keys()),
                'data_keys': list(self.data_cache.keys())
            }
        }
    
    def clear_cache(self):
        """
        Vider les caches
        """
        self.model_cache.clear()
        self.data_cache.clear()
        logger.info("🧹 Caches vidés")
    
    def __del__(self):
        """Nettoyage à la destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)