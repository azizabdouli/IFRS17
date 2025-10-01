# backend/routers/ppna_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import pandas as pd
import tempfile
import os
from pathlib import Path

from backend.services.ppna_service import PPNAService

router = APIRouter(prefix="/ppna", tags=["PPNA IFRS17"])

# Instance globale du service PPNA
ppna_service = PPNAService()

@router.get("/load-data", response_model=Dict[str, Any])
async def load_ppna_data():
    """Charge les données PPNA depuis le fichier Excel"""
    try:
        result = ppna_service.load_ppna_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement PPNA: {str(e)}")

@router.get("/calculate-lrc", response_model=Dict[str, Any])
async def calculate_lrc_paa(sheet_name: Optional[str] = None):
    """Calcule la LRC selon l'approche PAA IFRS17"""
    try:
        result = ppna_service.calculate_lrc_paa(sheet_name)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur calcul LRC: {str(e)}")

@router.get("/dashboard-metrics", response_model=Dict[str, Any])
async def get_dashboard_metrics():
    """Obtient les métriques IFRS17 pour le dashboard"""
    try:
        metrics = ppna_service.get_dashboard_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "approche": "PAA (Premium Allocation Approach)",
            "source": "PPNA Excel Data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur métriques: {str(e)}")

@router.post("/upload-file")
async def upload_ppna_file(file: UploadFile = File(...)):
    """Upload et traitement d'un nouveau fichier PPNA Excel"""
    tmp_file_path = None
    try:
        # Vérifier le type de fichier
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Seuls les fichiers Excel (.xlsx, .xls) sont acceptés")
        
        # Vérifier la taille du fichier (max 50MB)
        content = await file.read()
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 50MB)")
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()  # S'assurer que les données sont écrites
            tmp_file_path = tmp_file.name
        
        # Traiter le fichier
        result = ppna_service.upload_and_process_file(tmp_file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(content),
            "processing_result": result
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")
    finally:
        # Nettoyer le fichier temporaire avec gestion d'erreur
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                # Attendre un peu pour que pandas ferme le fichier
                import time
                time.sleep(0.5)
                os.unlink(tmp_file_path)
            except PermissionError:
                # Si le fichier est encore verrouillé, l'ignorer
                # Il sera nettoyé par le système plus tard
                pass

@router.get("/sheets", response_model=Dict[str, Any])
async def get_available_sheets():
    """Obtient la liste des feuilles Excel disponibles"""
    try:
        if not ppna_service.ppna_data:
            ppna_service.load_ppna_data()
        
        return {
            "status": "success",
            "sheets": list(ppna_service.ppna_data.keys()) if ppna_service.ppna_data else [],
            "total_sheets": len(ppna_service.ppna_data) if ppna_service.ppna_data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur feuilles: {str(e)}")

@router.get("/sheet-data/{sheet_name}", response_model=Dict[str, Any])
async def get_sheet_data(sheet_name: str, limit: int = 100):
    """Obtient les données d'une feuille spécifique"""
    try:
        if not ppna_service.ppna_data:
            ppna_service.load_ppna_data()
        
        if sheet_name not in ppna_service.ppna_data:
            raise HTTPException(status_code=404, detail=f"Feuille '{sheet_name}' non trouvée")
        
        df = ppna_service.ppna_data[sheet_name]
        
        return {
            "status": "success",
            "sheet_name": sheet_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "data": df.head(limit).to_dict('records'),
            "dtypes": df.dtypes.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur données feuille: {str(e)}")

@router.get("/analysis/segments", response_model=Dict[str, Any])
async def analyze_by_segments(sheet_name: Optional[str] = None):
    """Analyse les données par segments"""
    try:
        lrc_data = ppna_service.calculate_lrc_paa(sheet_name)
        
        segments_analysis = lrc_data.get("analyse_segments", [])
        
        return {
            "status": "success",
            "segments": segments_analysis,
            "total_segments": len(segments_analysis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse segments: {str(e)}")

@router.get("/analysis/onerous-contracts", response_model=Dict[str, Any])
async def analyze_onerous_contracts(sheet_name: Optional[str] = None):
    """Analyse des contrats onéreux"""
    try:
        lrc_data = ppna_service.calculate_lrc_paa(sheet_name)
        
        onerous_analysis = lrc_data.get("contrats_onereux", {})
        
        return {
            "status": "success",
            "onerous_contracts": onerous_analysis,
            "recommendations": ppna_service._get_onerous_recommendations(onerous_analysis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse contrats onéreux: {str(e)}")

# Ajouter la méthode manquante au service
def _get_onerous_recommendations(self, onerous_data: Dict) -> List[str]:
    """Génère des recommandations pour les contrats onéreux"""
    recommendations = []
    
    if onerous_data.get("detected", False):
        nombre = onerous_data.get("nombre_contrats_onereux", 0)
        ratio = onerous_data.get("ratio_moyen_onereux", 0)
        
        recommendations.append(f"⚠️ {nombre} contrat(s) onéreux détecté(s)")
        
        if ratio > 90:
            recommendations.append("🔴 Ratio très élevé - Révision urgente des tarifs")
        elif ratio > 80:
            recommendations.append("🟡 Ratio élevé - Surveillance renforcée")
            
        recommendations.append("📊 Constituer une loss component pour ces contrats")
        recommendations.append("🔍 Analyser les causes : sinistralité, frais de gestion")
        
    else:
        recommendations.append("✅ Aucun contrat onéreux détecté")
        
    return recommendations

# Ajouter la méthode au service
PPNAService._get_onerous_recommendations = _get_onerous_recommendations