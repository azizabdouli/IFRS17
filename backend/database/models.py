from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.sql import func
from backend.database.connection import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(50), default="analyste_ifrs17", nullable=False)  # Rôle unifié
    company = Column(String(200), default="BNA", nullable=False)
    
    # Profil IFRS17 
    department = Column(String(100), default="Assurance")
    employee_id = Column(String(50), nullable=True)
    level = Column(String(50), default="Intermédiaire")  # Débutant, Intermédiaire, Expert, Maître
    points = Column(Integer, default=0)  # Système de gamification
    badges = Column(Text, nullable=True)  # JSON des badges obtenus
    
    # Métriques utilisateur
    daily_tasks_completed = Column(Integer, default=0)
    weekly_goals_achieved = Column(Integer, default=0)
    monthly_reports_generated = Column(Integer, default=0)
    accuracy_streak = Column(Integer, default=0)
    
    # Status et sécurité
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Métadonnées
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0)
    
    # Contact
    phone = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role}', level='{self.level}')>"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def progress_percentage(self):
        """Calcule le pourcentage de progression basé sur les points"""
        if self.points < 100:
            return (self.points / 100) * 100
        elif self.points < 500:
            return ((self.points - 100) / 400) * 100
        elif self.points < 1000:
            return ((self.points - 500) / 500) * 100
        else:
            return 100

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    session_token = Column(String(255), unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    success = Column(Boolean, default=True)