# backend/routers/ai_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import io
import logging
from datetime import datetime
import asyncio

from backend.ai.ifrs17_ai_assistant import IFRS17AIAssistant
from backend.ai.predictive_ai_service import PredictiveAIService

router = APIRouter()
logger = logging.getLogger(__name__)

# Instances globales des services IA
ai_assistant = IFRS17AIAssistant()
predictive_ai = PredictiveAIService()

def clean_for_json(obj):
    """Nettoie les données pour la sérialisation JSON"""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
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
    else:
        return obj

@router.post("/chat")
async def ai_chat(query: str, context: Optional[Dict[str, Any]] = None):
    """
    Chat avec l'assistant IA IFRS17
    """
    try:
        logger.info(f"🤖 Requête IA: {query[:100]}...")
        
        # Traiter la requête avec l'assistant IA
        response = await ai_assistant.process_query(query, context)
        
        return clean_for_json({
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": len(ai_assistant.conversation_history)
        })
    
    except Exception as e:
        logger.error(f"Erreur chat IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur IA: {str(e)}")

@router.get("/chat/history")
async def get_conversation_history():
    """
    Récupère l'historique de conversation
    """
    try:
        history = ai_assistant.get_conversation_history()
        return clean_for_json({"history": history})
    
    except Exception as e:
        logger.error(f"Erreur récupération historique: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.delete("/chat/history")
async def clear_conversation():
    """
    Vide l'historique de conversation
    """
    try:
        ai_assistant.clear_history()
        return {"message": "Historique de conversation vidé", "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Erreur vidage historique: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/analyze-dataset")
async def ai_analyze_dataset(file: UploadFile = File(...)):
    """
    Analyse automatique IA d'un dataset
    """
    try:
        logger.info(f"🔍 Analyse IA du dataset: {file.filename}")
        
        # Lire le fichier
        content = await file.read()
        
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez .xlsx ou .csv")
        
        # Analyse IA automatique
        analysis = await predictive_ai.auto_analyze_dataset(df)
        
        return clean_for_json({
            "filename": file.filename,
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": len(content) / (1024 * 1024)
            },
            "ai_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erreur analyse IA dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/smart-model-selection")
async def smart_model_selection(
    task_type: str,
    dataset_info: Dict[str, Any]
):
    """
    Sélection intelligente de modèle avec Auto-ML
    """
    try:
        logger.info(f"🧠 Sélection intelligente de modèle pour: {task_type}")
        
        # Créer un DataFrame fictif basé sur les infos
        df = pd.DataFrame({
            'col1': np.random.randn(dataset_info.get('rows', 1000)),
            'col2': np.random.randn(dataset_info.get('rows', 1000))
        })
        
        # Ajouter plus de colonnes si spécifié
        for i in range(3, dataset_info.get('columns', 5)):
            df[f'col{i}'] = np.random.randn(len(df))
        
        recommendation = await predictive_ai.smart_model_selection(df, task_type)
        
        return clean_for_json({
            "task_type": task_type,
            "recommendation": recommendation,
            "explanation": predictive_ai.get_model_explanation(recommendation["recommended_algorithm"]),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erreur sélection modèle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/generate-insights")
async def generate_ai_insights(
    results: Dict[str, Any],
    dataset_size: int = 1000
):
    """
    Génère des insights IA à partir des résultats ML
    """
    try:
        logger.info("💡 Génération d'insights IA...")
        
        # Créer un DataFrame fictif pour l'analyse
        df = pd.DataFrame({'dummy': range(dataset_size)})
        
        insights = await predictive_ai.generate_ai_insights(df, results)
        
        return clean_for_json({
            "insights": insights,
            "analysis_date": datetime.now().isoformat(),
            "confidence": "high"
        })
    
    except Exception as e:
        logger.error(f"Erreur génération insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/quick-help/{topic}")
async def get_quick_help(topic: str):
    """
    Aide rapide contextuelle
    """
    try:
        help_text = ai_assistant.get_quick_help(topic)
        return {"topic": topic, "help": help_text}
    
    except Exception as e:
        logger.error(f"Erreur aide rapide: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.get("/ai-status")
async def get_ai_status():
    """
    Statut des services IA
    """
    try:
        status = {
            "ai_assistant": {
                "status": "active",
                "conversations": len(ai_assistant.conversation_history),
                "llm_available": hasattr(ai_assistant, 'llm_pipeline') and ai_assistant.llm_pipeline is not None
            },
            "predictive_ai": {
                "status": "active",
                "models_loaded": len(predictive_ai.models),
                "cache_size": len(predictive_ai.predictions_cache)
            },
            "capabilities": [
                "Conversation intelligente IFRS17",
                "Analyse automatique de datasets",
                "Sélection de modèles Auto-ML",
                "Génération d'insights business",
                "Recommandations personnalisées"
            ]
        }
        
        return clean_for_json(status)
    
    except Exception as e:
        logger.error(f"Erreur statut IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/ai-recommendations")
async def get_ai_recommendations(
    data_summary: Dict[str, Any],
    user_goals: List[str] = ["analysis", "prediction"]
):
    """
    Recommandations IA personnalisées
    """
    try:
        logger.info("🎯 Génération de recommandations IA personnalisées...")
        
        recommendations = []
        
        # Recommandations basées sur la taille des données
        data_size = data_summary.get("rows", 0)
        if data_size > 50000:
            recommendations.append({
                "type": "performance",
                "title": "🚀 Optimisation Big Data",
                "description": "Dataset volumineux détecté - utiliser le chunking et le cache",
                "priority": "high",
                "action": "enable_chunking"
            })
        
        # Recommandations basées sur les objectifs
        for goal in user_goals:
            if goal == "analysis":
                recommendations.append({
                    "type": "analysis",
                    "title": "📊 Analyse Exploratoire",
                    "description": "Commencer par l'analyse automatique IA du dataset",
                    "priority": "medium",
                    "action": "auto_analyze"
                })
            elif goal == "prediction":
                recommendations.append({
                    "type": "ml",
                    "title": "🔮 Modèles Prédictifs",
                    "description": "Utiliser l'Auto-ML pour sélection optimale",
                    "priority": "high",
                    "action": "auto_ml"
                })
        
        # Recommandations IFRS17 spécifiques
        if any(col in str(data_summary.get("columns", "")).lower() for col in ["prime", "lrc", "contrat"]):
            recommendations.append({
                "type": "domain",
                "title": "🏢 Analyse IFRS17 Spécialisée",
                "description": "Données IFRS17 détectées - analyses spécialisées disponibles",
                "priority": "high",
                "action": "ifrs17_analysis"
            })
        
        return clean_for_json({
            "recommendations": recommendations,
            "personalization": {
                "user_goals": user_goals,
                "data_profile": data_summary
            },
            "next_steps": [
                "Choisir une recommandation prioritaire",
                "Consulter l'assistant IA pour détails",
                "Lancer l'analyse automatique"
            ]
        })
    
    except Exception as e:
        logger.error(f"Erreur recommandations IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@router.post("/save-ai-session")
async def save_ai_session(session_name: str):
    """
    Sauvegarde la session IA
    """
    try:
        filepath = f"ai_sessions/{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        predictive_ai.save_ai_state(filepath)
        
        return {
            "message": "Session IA sauvegardée avec succès",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erreur sauvegarde session IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")