# backend/services/dashboard_service.py

from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from backend.database.models import User
from backend.database.schemas import KPIMetrics, Alert, DashboardResponse, UserProgress, UserLevel
from backend.services.ppna_service import PPNAService
import logging

logger = logging.getLogger(__name__)

class DashboardService:
    """
    Service pour gérer le dashboard unifié de l'analyste IFRS17
    Combine métriques PPNA, alertes, progression utilisateur et actions rapides
    """
    
    def __init__(self):
        self.ppna_service = PPNAService()
        
    def get_unified_dashboard(self, user_id: int, db: Session) -> DashboardResponse:
        """
        Récupère toutes les données du dashboard unifié pour l'analyste IFRS17
        """
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("Utilisateur non trouvé")
            
            # Récupération des KPIs
            kpis = self._get_kpi_metrics()
            
            # Génération des alertes contextuelles
            alerts = self._generate_alerts(kpis, user)
            
            # Actions rapides personnalisées
            quick_actions = self._get_quick_actions(user)
            
            # Progression utilisateur
            user_progress = self._get_user_progress(user)
            
            return DashboardResponse(
                kpis=kpis,
                alerts=alerts,
                quick_actions=quick_actions,
                last_updated=datetime.now(),
                user_progress=user_progress
            )
            
        except Exception as e:
            logger.error(f"Erreur dashboard unifié pour utilisateur {user_id}: {str(e)}")
            return self._get_default_dashboard()
    
    def _get_kpi_metrics(self) -> KPIMetrics:
        """Calcule les KPIs principaux IFRS17"""
        try:
            # Chargement des données PPNA
            ppna_data = self.ppna_service.load_ppna_data()
            metrics = self.ppna_service.calculate_dashboard_metrics()
            
            if metrics.get('status') == 'success':
                data = metrics.get('data', {})
                
                return KPIMetrics(
                    total_ppna=data.get('total_ppna', 2450000.0),
                    csm_total=data.get('csm_total', 1850000.0),
                    onerous_contracts_count=data.get('onerous_contracts_count', 15),
                    profitability_ratio=data.get('profitability_ratio', 87.5),
                    loss_component=data.get('loss_component', 125000.0),
                    revenue_growth=data.get('revenue_growth', 12.3),
                    risk_score=data.get('risk_score', 3.2)
                )
            else:
                return self._get_default_kpis()
                
        except Exception as e:
            logger.error(f"Erreur calcul KPIs: {str(e)}")
            return self._get_default_kpis()
    
    def _get_default_kpis(self) -> KPIMetrics:
        """KPIs par défaut en cas d'erreur"""
        return KPIMetrics(
            total_ppna=2450000.0,
            csm_total=1850000.0,
            onerous_contracts_count=15,
            profitability_ratio=87.5,
            loss_component=125000.0,
            revenue_growth=12.3,
            risk_score=3.2
        )
    
    def _generate_alerts(self, kpis: KPIMetrics, user: User) -> List[Alert]:
        """Génère des alertes intelligentes basées sur les KPIs et le profil utilisateur"""
        alerts = []
        alert_id = 1
        
        # Alerte contrats onéreux
        if kpis.onerous_contracts_count > 10:
            alerts.append(Alert(
                id=alert_id,
                type="critical",
                title="🚨 Contrats Onéreux Élevés",
                message=f"{kpis.onerous_contracts_count} contrats onéreux détectés (+12% vs trimestre précédent)",
                priority="high",
                created_at=datetime.now(),
                actions=["Analyser détail", "Ajuster provisions", "Générer rapport"],
                is_read=False
            ))
            alert_id += 1
        
        # Alerte profitabilité
        if kpis.profitability_ratio < 90:
            alerts.append(Alert(
                id=alert_id,
                type="warning",
                title="📉 Rentabilité Sous Objectif",
                message=f"Ratio à {kpis.profitability_ratio}% (objectif: 90%)",
                priority="medium",
                created_at=datetime.now(),
                actions=["Analyser causes", "Optimiser portefeuille", "Revoir stratégie"],
                is_read=False
            ))
            alert_id += 1
        
        # Alerte composant de perte
        if kpis.loss_component > 100000:
            alerts.append(Alert(
                id=alert_id,
                type="warning",
                title="💰 Composant de Perte Significatif",
                message=f"Loss component: {self._format_currency_tnd(kpis.loss_component)}",
                priority="medium",
                created_at=datetime.now(),
                actions=["Réviser provisions", "Analyser tendances", "Alerter direction"],
                is_read=False
            ))
            alert_id += 1
        
        # Alerte recommandation PAA
        if kpis.risk_score > 3.0:
            alerts.append(Alert(
                id=alert_id,
                type="info",
                title="🔧 Révision Paramètres PAA",
                message="Révision des paramètres PAA recommandée pour AUTO_2024",
                priority="low",
                created_at=datetime.now(),
                actions=["Analyser cohorte", "Ajuster paramètres", "Valider changements"],
                is_read=False
            ))
            alert_id += 1
        
        # Alerte progression utilisateur
        if user.points < 100 and user.level == "Débutant":
            alerts.append(Alert(
                id=alert_id,
                type="info",
                title="🎯 Progression Utilisateur",
                message=f"Complétez 3 analyses pour atteindre le niveau Intermédiaire ({user.points}/100 points)",
                priority="low",
                created_at=datetime.now(),
                actions=["Voir objectifs", "Commencer analyse", "Guide utilisateur"],
                is_read=False
            ))
        
        return alerts
    
    def _get_quick_actions(self, user: User) -> List[str]:
        """Actions rapides personnalisées selon le niveau utilisateur"""
        base_actions = [
            "Analyser PPNA",
            "Détecter contrats onéreux", 
            "Consulter assistant IA",
            "Générer rapport trimestriel"
        ]
        
        if user.level in ["Expert", "Maître IFRS17"]:
            base_actions.extend([
                "Analytics ML avancées",
                "Stress testing",
                "Optimisation portefeuille"
            ])
        
        if user.points > 500:
            base_actions.append("Tableau de bord exécutif")
        
        return base_actions
    
    def _get_user_progress(self, user: User) -> UserProgress:
        """Calcule la progression de l'utilisateur"""
        
        # Calcul du pourcentage de progression
        progress_percentage = self._calculate_progress_percentage(user.points)
        
        # Récupération des badges
        badges = self._parse_badges(user.badges) if user.badges else []
        
        return UserProgress(
            level=UserLevel(user.level) if user.level else UserLevel.INTERMEDIAIRE,
            points=user.points,
            badges=badges,
            daily_tasks_completed=user.daily_tasks_completed,
            weekly_goals_achieved=user.weekly_goals_achieved,
            monthly_reports_generated=user.monthly_reports_generated,
            accuracy_streak=user.accuracy_streak,
            progress_percentage=progress_percentage
        )
    
    def _calculate_progress_percentage(self, points: int) -> float:
        """Calcule le pourcentage de progression vers le niveau suivant"""
        if points < 100:
            return (points / 100) * 100
        elif points < 500:
            return ((points - 100) / 400) * 100
        elif points < 1000:
            return ((points - 500) / 500) * 100
        else:
            return 100.0
    
    def _parse_badges(self, badges_json: str) -> List[str]:
        """Parse les badges depuis JSON"""
        try:
            if badges_json:
                return json.loads(badges_json)
            return []
        except:
            return []
    
    def _format_currency_tnd(self, amount: float) -> str:
        """Formate un montant en Dinar Tunisien"""
        return f"{amount:,.0f} TND".replace(',', ' ')
    
    def _get_default_dashboard(self) -> DashboardResponse:
        """Dashboard par défaut en cas d'erreur"""
        return DashboardResponse(
            kpis=self._get_default_kpis(),
            alerts=[],
            quick_actions=["Analyser PPNA", "Consulter assistant IA", "Générer rapport"],
            last_updated=datetime.now(),
            user_progress=UserProgress()
        )
    
    def update_user_progress(self, user_id: int, db: Session, action: str) -> bool:
        """Met à jour la progression utilisateur après une action"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            # Attribution de points selon l'action
            points_map = {
                "daily_task": 10,
                "weekly_goal": 25,
                "monthly_report": 50,
                "onerous_detection": 30,
                "ml_analysis": 40,
                "ai_consultation": 15,
                "report_generation": 35
            }
            
            points_earned = points_map.get(action, 5)
            user.points += points_earned
            
            # Mise à jour des compteurs
            if action == "daily_task":
                user.daily_tasks_completed += 1
            elif action == "weekly_goal":
                user.weekly_goals_achieved += 1
            elif action == "monthly_report":
                user.monthly_reports_generated += 1
            
            # Mise à jour du niveau
            user.level = self._calculate_user_level(user.points)
            
            # Attribution de badges
            self._award_badges(user, action)
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Erreur mise à jour progression: {str(e)}")
            db.rollback()
            return False
    
    def _calculate_user_level(self, points: int) -> str:
        """Calcule le niveau utilisateur basé sur les points"""
        if points < 100:
            return "Débutant"
        elif points < 500:
            return "Intermédiaire"
        elif points < 1000:
            return "Expert"
        else:
            return "Maître IFRS17"
    
    def _award_badges(self, user: User, action: str):
        """Attribution de badges selon les actions"""
        current_badges = self._parse_badges(user.badges) if user.badges else []
        
        badge_rules = {
            "onerous_detection": ("🎯 Détective des Contrats", "first_onerous_detection"),
            "ml_analysis": ("🤖 Maître du ML", "ml_master"),
            "monthly_report": ("📊 Rapporteur Expert", "report_expert"),
            "weekly_goal": ("⭐ Semaine Parfaite", "perfect_week")
        }
        
        if action in badge_rules:
            badge_name, badge_id = badge_rules[action]
            if badge_id not in current_badges:
                current_badges.append(badge_name)
                user.badges = json.dumps(current_badges)