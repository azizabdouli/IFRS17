from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from backend.database.models import User
from backend.database.schemas import UserCreate, UserUpdate, UserResponse, UserProgress, UserLevel
from backend.auth.security import get_password_hash, verify_password
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import secrets
import json

class UserService:
    """Service unifié pour la gestion des utilisateurs Analyste IFRS17"""
    
    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """Créer un nouvel analyste IFRS17"""
        # Vérifier si l'email existe déjà
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ValueError("Un utilisateur avec cet email existe déjà")
        
        # Hasher le mot de passe
        hashed_password = get_password_hash(user_data.password)
        
        # Créer l'utilisateur avec rôle unifié
        db_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role="analyste_ifrs17",  # Rôle unifié
            company=user_data.company or "BNA",
            phone=user_data.phone,
            department=user_data.department or "Assurance",
            employee_id=user_data.employee_id,
            level="Intermédiaire",  # Niveau par défaut
            points=0,
            badges=json.dumps([]),  # Liste vide de badges
            is_active=True,
            is_verified=True  # Auto-vérification pour analystes IFRS17
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return db_user
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """Authentifier un analyste IFRS17"""
        user = db.query(User).filter(
            and_(
                User.email == email,
                User.is_active == True
            )
        ).first()
        
        if not user:
            return None
        
        # Vérifier si le compte est verrouillé
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return None
        
        # Vérifier le mot de passe
        if not verify_password(password, user.hashed_password):
            # Incrémenter les tentatives échouées
            user.failed_login_attempts += 1
            
            # Verrouiller le compte après 5 tentatives
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            
            db.commit()
            return None
        
        # Réinitialiser les tentatives échouées et mettre à jour la dernière connexion
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        user.login_count += 1
        
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Récupérer un utilisateur par email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Récupérer un utilisateur par ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def update_user(db: Session, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Mettre à jour un utilisateur"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        
        # Mettre à jour les champs fournis
        for field, value in user_data.dict(exclude_unset=True).items():
            setattr(user, field, value)
        
        user.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def change_password(db: Session, user_id: int, current_password: str, new_password: str) -> bool:
        """Changer le mot de passe d'un utilisateur"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        # Vérifier le mot de passe actuel
        if not verify_password(current_password, user.hashed_password):
            return False
        
        # Mettre à jour le mot de passe
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        return True
    
    @staticmethod
    def get_all_users(db: Session) -> List[User]:
        """Récupérer tous les utilisateurs actifs"""
        return db.query(User).filter(User.is_active == True).all()
    
    @staticmethod
    def deactivate_user(db: Session, user_id: int) -> bool:
        """Désactiver un utilisateur"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        return True

    @staticmethod
    def award_points(db: Session, user_id: int, points: int, action: str = None) -> bool:
        """Attribue des points à l'utilisateur et met à jour son niveau"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            old_level = user.level
            user.points += points
            
            # Mise à jour du niveau
            new_level = UserService._calculate_user_level(user.points)
            user.level = new_level
            
            # Attribution de badges si changement de niveau
            if old_level != new_level:
                UserService._award_level_badge(user, new_level)
            
            # Attribution de badge spécifique à l'action
            if action:
                UserService._award_action_badge(user, action)
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            return False

    @staticmethod
    def _calculate_user_level(points: int) -> str:
        """Calcule le niveau utilisateur basé sur les points"""
        if points < 100:
            return "Débutant"
        elif points < 500:
            return "Intermédiaire"
        elif points < 1000:
            return "Expert"
        else:
            return "Maître IFRS17"

    @staticmethod
    def _award_level_badge(user: User, level: str):
        """Attribue un badge de niveau"""
        badges = []
        if user.badges:
            try:
                badges = json.loads(user.badges)
            except:
                badges = []
        
        level_badges = {
            "Intermédiaire": "🥉 Analyste Confirmé",
            "Expert": "🥈 Expert IFRS17",
            "Maître IFRS17": "🥇 Maître IFRS17"
        }
        
        if level in level_badges:
            badge = level_badges[level]
            if badge not in badges:
                badges.append(badge)
                user.badges = json.dumps(badges)

    @staticmethod
    def _award_action_badge(user: User, action: str):
        """Attribue un badge d'action"""
        badges = []
        if user.badges:
            try:
                badges = json.loads(user.badges)
            except:
                badges = []
        
        action_badges = {
            "onerous_detection": "🎯 Détective des Contrats",
            "ml_analysis": "🤖 Maître du ML",
            "monthly_report": "📊 Rapporteur Expert",
            "perfect_week": "⭐ Semaine Parfaite",
            "compliance_expert": "🏆 Champion de la Conformité"
        }
        
        if action in action_badges:
            badge = action_badges[action]
            if badge not in badges:
                badges.append(badge)
                user.badges = json.dumps(badges)

    @staticmethod
    def get_user_progress(user: User) -> UserProgress:
        """Calcule la progression détaillée de l'utilisateur"""
        
        # Parsing des badges
        badges = []
        if user.badges:
            try:
                badges = json.loads(user.badges)
            except:
                badges = []
        
        # Calcul du pourcentage de progression
        progress_percentage = UserService._calculate_progress_percentage(user.points)
        
        # Détermination du niveau
        level = UserLevel(user.level) if user.level in [l.value for l in UserLevel] else UserLevel.INTERMEDIAIRE
        
        return UserProgress(
            level=level,
            points=user.points,
            badges=badges,
            daily_tasks_completed=user.daily_tasks_completed,
            weekly_goals_achieved=user.weekly_goals_achieved,
            monthly_reports_generated=user.monthly_reports_generated,
            accuracy_streak=user.accuracy_streak,
            progress_percentage=progress_percentage
        )

    @staticmethod
    def _calculate_progress_percentage(points: int) -> float:
        """Calcule le pourcentage de progression vers le niveau suivant"""
        if points < 100:
            return (points / 100) * 100
        elif points < 500:
            return ((points - 100) / 400) * 100
        elif points < 1000:
            return ((points - 500) / 500) * 100
        else:
            return 100.0

    @staticmethod
    def to_user_response(user: User) -> UserResponse:
        """Convertit un User en UserResponse avec progression"""
        progress = UserService.get_user_progress(user)
        
        return UserResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=f"{user.first_name} {user.last_name}",
            role=user.role,
            company=user.company,
            department=user.department,
            level=UserLevel(user.level) if user.level in [l.value for l in UserLevel] else UserLevel.INTERMEDIAIRE,
            points=user.points,
            progress=progress,
            created_at=user.created_at,
            last_login=user.last_login,
            login_count=user.login_count,
            phone=user.phone,
            employee_id=user.employee_id,
            is_active=user.is_active,
            is_verified=user.is_verified
        )

    @staticmethod
    def create_default_users(db: Session):
        """Créer les utilisateurs par défaut pour l'application"""
        default_users = [
            {
                "email": "analyste@bna.tn",
                "password": "password123",
                "first_name": "Analyste",
                "last_name": "IFRS17",
                "company": "BNA",
                "department": "Assurance",
                "employee_id": "A001"
            },
            {
                "email": "expert@bna.tn", 
                "password": "password123",
                "first_name": "Expert",
                "last_name": "Senior",
                "company": "BNA",
                "department": "Assurance",
                "employee_id": "E001"
            }
        ]
        
        for user_data in default_users:
            existing = db.query(User).filter(User.email == user_data["email"]).first()
            if not existing:
                try:
                    user_create = UserCreate(**user_data)
                    UserService.create_user(db, user_create)
                    print(f"✅ Utilisateur créé: {user_data['email']}")
                except Exception as e:
                    print(f"❌ Erreur création utilisateur {user_data['email']}: {e}")