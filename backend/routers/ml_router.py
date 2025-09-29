# backend/routers/ml_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import io
import logging
from datetime import datetime
import json

from backend.ml.ml_service import MLService

router = APIRouter()
logger = logging.getLogger(__name__)

# Instance globale du service ML
ml_service = MLService()

def clean_for_json(obj):
    """
    Nettoie les données pour la sérialisation JSON
    Remplace NaN, inf par None
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj) or obj in [np.inf, -np.inf]:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
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
    return {
        "status": "healthy",
        "service": "IFRS17 ML Service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ml_service.models),
        "models_available": list(ml_service.models.keys())
    }