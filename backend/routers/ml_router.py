# backend/routers/ml_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import io
import logging
from datetime import datetime
import json
from cachetools import TTLCache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from backend.ml.optimized_ml_service import EnhancedMLService

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache pour les résultats de requêtes
query_cache = TTLCache(maxsize=100, ttl=600)  # 10 minutes
executor = ThreadPoolExecutor(max_workers=4)

# Instance globale du service ML optimisé
ml_service = EnhancedMLService()

def clean_for_json(obj):
    """
    Nettoie les données pour la sérialisation JSON
    Remplace NaN, inf par None et convertit les types numpy
    """
    if isinstance(obj, dict):
        # S'assurer que les clés sont des strings
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj) or obj in [np.inf, -np.inf]:
        return None
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif obj is None:
        return None
    else:
        return obj

@router.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload des données pour l'entraînement des modèles ML
    """
    try:
        # Vérification du format de fichier
        if not file.filename.endswith(('.xlsx', '.csv')):
            raise HTTPException(status_code=400, detail="Format de fichier non supporté. Utilisez .xlsx ou .csv")
        
        # Lecture du fichier
        contents = await file.read()
        
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Sauvegarde temporaire des données
        ml_service.current_dataset = df
        
        # Nettoyage des données pour JSON
        sample_data = df.head().fillna("null").to_dict('records')
        
        response_data = {
            "message": "Données uploadées avec succès",
            "data_info": {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "sample_data": sample_data
            }
        }
        
        return clean_for_json(response_data)
    
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")

@router.post("/train/claims-prediction")
async def train_claims_prediction(
    background_tasks: BackgroundTasks,
    model_type: str = "xgboost",
    target_column: Optional[str] = None
):
    """
    Entraînement du modèle de prédiction des sinistres
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées. Utilisez /upload-data d'abord.")
        
        df = ml_service.current_dataset
        
        # Entraînement en arrière-plan
        def train_model():
            results = ml_service.train_claims_prediction_model(df, target_column, model_type)
            logger.info(f"Modèle de prédiction des sinistres entraîné: {results}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Entraînement du modèle de prédiction des sinistres démarré",
            "model_type": model_type,
            "status": "training_started"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/train/profitability")
async def train_profitability_model(
    background_tasks: BackgroundTasks,
    model_type: str = "xgboost",
    target_column: Optional[str] = None
):
    """
    Entraînement du modèle de prédiction de rentabilité
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        
        def train_model():
            results = ml_service.train_profitability_model(df, target_column, model_type)
            logger.info(f"Modèle de rentabilité entraîné: {results}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Entraînement du modèle de rentabilité démarré",
            "model_type": model_type,
            "status": "training_started"
        }
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/train/risk-classification")
async def train_risk_classification(
    background_tasks: BackgroundTasks,
    model_type: str = "xgboost"
):
    """
    Entraînement du modèle de classification des risques
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        
        def train_model():
            results = ml_service.train_risk_classification_model(df, model_type)
            logger.info(f"Modèle de classification des risques entraîné: {results}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Entraînement du modèle de classification des risques démarré",
            "model_type": model_type,
            "status": "training_started"
        }
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/clustering")
async def perform_clustering(
    n_clusters: int = 5,
    clustering_type: str = "kmeans"
):
    """
    Clustering des contrats d'assurance
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        results = ml_service.perform_contract_clustering(df, n_clusters, clustering_type)
        
        response_data = {
            "message": "Clustering terminé avec succès",
            "results": {
                "n_clusters": results['n_clusters'],
                "cluster_distribution": results['cluster_distribution'],
                "cluster_characteristics": results['cluster_characteristics']
            }
        }
        
        return clean_for_json(response_data)
    
    except Exception as e:
        logger.error(f"Erreur lors du clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/anomaly-detection")
async def detect_anomalies(
    method: str = "isolation_forest",
    contamination: float = 0.1
):
    """
    Détection d'anomalies dans les contrats
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        results = ml_service.detect_anomalies(df, method, contamination)
        
        response_data = {
            "message": "Détection d'anomalies terminée",
            "results": {
                "n_anomalies": results['n_anomalies'],
                "anomaly_rate": f"{results['anomaly_rate']:.2%}",
                "method": method,
                "anomalous_contracts": results['anomalous_contracts'][:10]  # Limiter à 10 pour l'API
            }
        }
        
        return clean_for_json(response_data)
    
    except Exception as e:
        logger.error(f"Erreur lors de la détection d'anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/train/lrc-prediction")
async def train_lrc_prediction(
    background_tasks: BackgroundTasks,
    model_type: str = "xgboost"
):
    """
    Entraînement du modèle de prédiction LRC (IFRS 17)
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        
        def train_model():
            results = ml_service.train_lrc_prediction_model(df, model_type)
            logger.info(f"Modèle LRC entraîné: {results}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Entraînement du modèle LRC démarré",
            "model_type": model_type,
            "status": "training_started"
        }
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/predict/{model_name}")
async def make_prediction(
    model_name: str,
    file: UploadFile = File(...)
):
    """
    Prédiction avec un modèle entraîné
    """
    try:
        # Lecture des nouvelles données
        contents = await file.read()
        
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Prédiction
        predictions = ml_service.predict_with_model(model_name, df)
        
        return {
            "message": "Prédictions générées avec succès",
            "model_used": model_name,
            "n_predictions": len(predictions),
            "predictions": predictions.tolist()
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/models/summary")
async def get_models_summary():
    """
    Résumé de tous les modèles entraînés
    """
    try:
        summary = ml_service.get_model_summary()
        return clean_for_json(summary)
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du résumé: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/insights")
async def get_ml_insights():
    """
    Insights ML sur les données actuelles
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        insights = ml_service.generate_ml_insights(df)
        
        return clean_for_json(insights)
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/models/save")
async def save_models(save_directory: str = "models"):
    """
    Sauvegarde de tous les modèles entraînés
    """
    try:
        ml_service.save_all_models(save_directory)
        return {
            "message": "Modèles sauvegardés avec succès",
            "save_directory": save_directory,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Vérification de l'état du service ML
    """
    # Cache des statistiques de santé
    cache_key = "health_stats"
    if cache_key in query_cache:
        cached_stats = query_cache[cache_key]
    else:
        cached_stats = {
            "models_loaded": len(ml_service.models),
            "models_available": list(ml_service.models.keys()),
            "cache_stats": ml_service.get_cache_stats() if hasattr(ml_service, 'get_cache_stats') else {}
        }
        query_cache[cache_key] = cached_stats
    
    return {
        "status": "healthy",
        "service": "IFRS17 ML Service Optimized",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        **cached_stats
    }

@router.get("/data/paginated")
async def get_data_paginated(
    page: int = Query(1, ge=1, description="Numéro de page"),
    size: int = Query(50, ge=1, le=1000, description="Taille de la page"),
    columns: Optional[str] = Query(None, description="Colonnes spécifiques (séparées par virgules)")
):
    """
    Récupération paginée des données avec cache
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        # Clé de cache basée sur les paramètres
        cache_key = f"data_page_{page}_{size}_{columns or 'all'}"
        
        if cache_key in query_cache:
            logger.info(f"Cache hit pour {cache_key}")
            return query_cache[cache_key]
        
        df = ml_service.current_dataset
        
        # Sélection des colonnes
        if columns:
            selected_cols = [col.strip() for col in columns.split(',')]
            available_cols = [col for col in selected_cols if col in df.columns]
            if available_cols:
                df = df[available_cols]
        
        # Pagination
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        
        paginated_data = df.iloc[start_idx:end_idx]
        total_rows = len(df)
        total_pages = (total_rows + size - 1) // size
        
        result = {
            "data": clean_for_json(paginated_data.to_dict('records')),
            "pagination": {
                "page": page,
                "size": size,
                "total_rows": total_rows,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Mise en cache
        query_cache[cache_key] = result
        logger.info(f"Données mises en cache pour {cache_key}")
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération paginée: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/data/summary")
async def get_data_summary_cached():
    """
    Résumé statistique des données avec cache
    """
    try:
        cache_key = "data_summary"
        
        if cache_key in query_cache:
            logger.info("Cache hit pour le résumé des données")
            return query_cache[cache_key]
        
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        # Génération du résumé en arrière-plan si possible
        def generate_summary():
            df = ml_service.current_dataset
            summary = {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "numerical_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                "categorical_summary": {
                    col: {
                        "unique_count": df[col].nunique(),
                        "top_values": df[col].value_counts().head(5).to_dict()
                    }
                    for col in df.select_dtypes(include=['object']).columns
                }
            }
            return clean_for_json(summary)
        
        # Exécution asynchrone
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, generate_summary)
        
        # Mise en cache
        query_cache[cache_key] = result
        logger.info("Résumé des données mis en cache")
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Statistiques du cache
    """
    ml_cache_stats = ml_service.get_cache_stats() if hasattr(ml_service, 'get_cache_stats') else {}
    
    return {
        "query_cache": {
            "size": len(query_cache),
            "maxsize": query_cache.maxsize,
            "ttl": query_cache.ttl,
            "keys": list(query_cache.keys())
        },
        "ml_service_cache": ml_cache_stats,
        "performance": {
            "executor_threads": executor._max_workers,
            "cache_hit_rate": "Données non disponibles pour le moment"
        }
    }

@router.post("/cache/clear")
async def clear_cache():
    """
    Nettoyage du cache
    """
    try:
        # Nettoyage du cache de requêtes
        query_cache.clear()
        
        # Nettoyage du cache ML si disponible
        if hasattr(ml_service, 'clear_cache'):
            ml_service.clear_cache()
        
        return {
            "message": "Cache nettoyé avec succès",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage du cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/train/onerous-contracts")
async def train_onerous_contracts(
    background_tasks: BackgroundTasks,
    model_type: str = "xgboost"
):
    """
    Entraînement du modèle de détection des contrats onéreux
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        
        def train_model():
            results = ml_service.train_onerous_contracts_model(df, model_type)
            logger.info(f"Modèle contrats onéreux entraîné: {results}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Entraînement du modèle de contrats onéreux démarré",
            "model_type": model_type,
            "status": "training_started"
        }
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/predict/onerous-contracts")
async def predict_onerous_contracts(model_type: str = "xgboost"):
    """
    Prédiction des contrats onéreux
    """
    try:
        if not hasattr(ml_service, 'current_dataset'):
            raise HTTPException(status_code=400, detail="Aucune données uploadées.")
        
        df = ml_service.current_dataset
        predictions = ml_service.predict_onerous_contracts(df, model_type)
        
        return clean_for_json({
            "message": "Prédictions contrats onéreux générées",
            "model_used": model_type,
            "results": predictions
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/onerous-analysis")
async def get_onerous_analysis():
    """
    Analyse détaillée des contrats onéreux
    """
    try:
        # Chercher les résultats d'analyse onéreuse
        onerous_results = None
        for key, results in ml_service.model_results.items():
            if 'onerous_contracts' in key:
                onerous_results = results
                break
        
        if not onerous_results:
            raise HTTPException(status_code=404, detail="Aucune analyse de contrats onéreux trouvée. Entraînez d'abord le modèle.")
        
        return clean_for_json({
            "analysis": onerous_results.get('onerous_analysis', {}),
            "insights": onerous_results.get('insights', {}),
            "performance": onerous_results.get('performance_metrics', {}),
            "recommendations": onerous_results.get('onerous_analysis', {}).get('recommendations', [])
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse onéreuse: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")