# backend/routers/ppna_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import pandas as pd
import tempfile
import os
from pathlib import Path

from backend.services.ppna_service import PPNAService
from backend.ml.ml_instance import ml_service

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
        
        # Partager les données avec le service ML
        try:
            # Charger le fichier pour le ML
            df = pd.read_excel(tmp_file_path) if tmp_file_path.endswith('.xlsx') else pd.read_csv(tmp_file_path)
            ml_service.current_dataset = df
            print(f"💾 Données partagées avec ML service: {len(df):,} lignes, {len(df.columns)} colonnes")
        except Exception as e:
            print(f"⚠️ Erreur lors du partage avec ML service: {str(e)}")
        
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
# Ajouter la méthode au service
PPNAService._get_onerous_recommendations = _get_onerous_recommendations

@router.post("/projection/calculate")
async def calculate_monthly_projection(
    start_year: int = Query(..., description="Année de début"),
    end_year: int = Query(..., description="Année de fin"),
    products: Optional[List[str]] = Query(None, description="Liste des produits (optionnel)")
):
    """
    Calcule la projection mensuelle IFRS 17 (revenue & DAC amortissement)
    """
    try:
        # Vérifier que les données PPNA sont chargées
        if not ppna_service.ppna_data:
            raise HTTPException(status_code=400, detail="Aucune donnée PPNA chargée. Veuillez uploader un fichier Excel.")
        
        # Extraire le DataFrame de la première feuille
        sheet_name = list(ppna_service.ppna_data.keys())[0]
        df = ppna_service.ppna_data[sheet_name].copy()
        
        # Filtrer par produits si spécifié
        if products and len(products) > 0:
            df = df[df['CODPROD'].isin(products)]
        
        # Générer les dates mensuelles
        import pandas as pd
        from datetime import datetime
        
        dates = pd.date_range(
            start=f'{start_year}-01-01',
            end=f'{end_year}-12-31',
            freq='MS'
        )
        
        projections = []
        for date in dates:
            month_str = date.strftime('%Y-%m')
            
            # Calculs réels basés sur les données
            monthly_revenue = df['MNTPRNET'].sum() / 12  # Simplification - peut être amélioré
            monthly_dac_amort = df.get('DAC_AMORT', df['MNTPRNET'] * 0.05).sum() / 12
            
            projections.append({
                'mois': month_str,
                'revenue_mois': round(monthly_revenue, 2),
                'dac_amort_mois': round(monthly_dac_amort, 2),
                'n_contracts': len(df)
            })
        
        return {
            "status": "success",
            "projections": projections,
            "period": f"{start_year}-{end_year}",
            "n_months": len(projections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur projection: {str(e)}")

@router.get("/export/excel")
async def export_ppna_excel():
    """
    Exporte les données PPNA en Excel
    """
    try:
        # Vérifier que les données PPNA sont chargées
        if not ppna_service.ppna_data:
            raise HTTPException(status_code=400, detail="Aucune donnée PPNA chargée. Veuillez uploader un fichier Excel.")
        
        import io
        from fastapi.responses import StreamingResponse
        
        # Créer le fichier Excel en mémoire avec toutes les feuilles
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Exporter toutes les feuilles
            for sheet_name, df in ppna_service.ppna_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename=PPNA_Export.xlsx'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur export Excel: {str(e)}")

@router.get("/export/pdf")
async def export_ppna_pdf():
    """
    Exporte un rapport PPNA en PDF
    """
    try:
        # Vérifier que les données PPNA sont chargées
        if not ppna_service.ppna_data:
            raise HTTPException(status_code=400, detail="Aucune donnée PPNA chargée. Veuillez uploader un fichier Excel.")
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        import io
        from datetime import datetime
        
        # Créer le PDF en mémoire
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Titre
        title = Paragraph(f"<b>Rapport PPNA IFRS 17</b><br/>{datetime.now().strftime('%d/%m/%Y')}", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Extraire le DataFrame de la première feuille
        sheet_name = list(ppna_service.ppna_data.keys())[0]
        df = ppna_service.ppna_data[sheet_name]
        stats_data = [
            ['Métrique', 'Valeur'],
            ['Nombre de contrats', f"{len(df):,}"],
            ['Prime totale (TND)', f"{df['MNTPRNET'].sum():,.2f}"],
            ['PPNA totale (TND)', f"{df['MNTPPNA'].sum():,.2f}"],
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(stats_table)
        doc.build(elements)
        
        buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=PPNA_Rapport.pdf'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur export PDF: {str(e)}")
