from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.schemas import DashboardResponse, UserResponse
from backend.services.dashboard_service import DashboardService
from backend.auth.user_service import UserService
from backend.routers.auth_router import get_current_user
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du router
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/", response_model=DashboardResponse)
async def get_unified_dashboard(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer le dashboard unifié pour l'analyste IFRS17
    
    Fournit une vue d'ensemble personnalisée avec:
    - KPIs personnalisés basés sur l'expertise
    - Alertes contextuelles intelligentes
    - Actions recommandées
    - Progression de l'utilisateur
    """
    try:
        # Récupérer l'utilisateur complet depuis la base
        user = UserService.get_user_by_id(db, current_user.id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Générer le dashboard personnalisé
        dashboard = await DashboardService.get_unified_dashboard(db, user)
        
        logger.info(f"Dashboard unifié généré pour {user.email}")
        return dashboard
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la génération du dashboard"
        )

@router.post("/award-points/{points}")
async def award_points_to_user(
    points: int,
    action: str = None,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Attribuer des points à l'utilisateur pour une action spécifique
    
    Actions supportées:
    - onerous_detection: Détection de contrats onéreux
    - ml_analysis: Utilisation d'analyse ML
    - monthly_report: Génération de rapport mensuel
    - perfect_week: Semaine parfaite
    - compliance_expert: Expertise en conformité
    """
    try:
        success = UserService.award_points(db, current_user.id, points, action)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Impossible d'attribuer les points"
            )
        
        # Récupérer l'utilisateur mis à jour
        user = UserService.get_user_by_id(db, current_user.id)
        progress = UserService.get_user_progress(user)
        
        logger.info(f"Points attribués: {points} à {current_user.email} pour {action}")
        
        return {
            "message": f"Points attribués avec succès: +{points}",
            "total_points": user.points,
            "level": user.level,
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'attribution des points: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'attribution des points"
        )

@router.get("/user-progress")
async def get_user_progress(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Récupérer la progression détaillée de l'utilisateur"""
    try:
        user = UserService.get_user_by_id(db, current_user.id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        progress = UserService.get_user_progress(user)
        
        return {
            "user_id": user.id,
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la progression: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération de la progression"
        )

@router.get("/recommended-actions")
async def get_recommended_actions(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Récupérer les actions recommandées pour l'utilisateur"""
    try:
        user = UserService.get_user_by_id(db, current_user.id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        recommended_actions = await DashboardService.get_recommended_actions(db, user)
        
        return {
            "user_level": user.level,
            "recommended_actions": recommended_actions
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des actions recommandées: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des actions recommandées"
        )

@router.get("/alerts")
async def get_contextual_alerts(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Récupérer les alertes contextuelles pour l'utilisateur"""
    try:
        user = UserService.get_user_by_id(db, current_user.id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        alerts = await DashboardService.get_contextual_alerts(db, user)
        
        return {
            "alerts_count": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des alertes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des alertes"
        )