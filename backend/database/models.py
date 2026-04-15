from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, Date, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
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

# =========================
# 🏢 ERP Assurance - Modèles
# =========================

class Portfolio(Base):
    __tablename__ = "erp_portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    currency = Column(String(10), default="TND", nullable=False)
    manager = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    policies = relationship("Policy", back_populates="portfolio", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Portfolio(name='{self.name}', currency='{self.currency}')>"


class Client(Base):
    __tablename__ = "erp_clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    client_type = Column(String(50), default="particulier", nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(30), nullable=True)
    address = Column(Text, nullable=True)
    status = Column(String(30), default="actif", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    policies = relationship("Policy", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(name='{self.name}', type='{self.client_type}')>"


class Policy(Base):
    __tablename__ = "erp_policies"

    id = Column(Integer, primary_key=True, index=True)
    policy_number = Column(String(100), unique=True, nullable=False)
    client_id = Column(Integer, ForeignKey("erp_clients.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("erp_portfolios.id"), nullable=True)
    effective_date = Column(Date, nullable=False)
    expiry_date = Column(Date, nullable=True)
    premium_amount = Column(Float, default=0.0, nullable=False)
    currency = Column(String(10), default="TND", nullable=False)
    status = Column(String(30), default="active", nullable=False)
    ifrs17_group = Column(String(100), nullable=True)
    cohort_year = Column(Integer, nullable=True)
    measurement_model = Column(String(20), default="PAA", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    client = relationship("Client", back_populates="policies")
    portfolio = relationship("Portfolio", back_populates="policies")
    coverages = relationship("Coverage", back_populates="policy", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="policy", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="policy", cascade="all, delete-orphan")
    ledger_entries = relationship("LedgerEntry", back_populates="policy", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Policy(number='{self.policy_number}', status='{self.status}')>"


class Coverage(Base):
    __tablename__ = "erp_coverages"

    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(Integer, ForeignKey("erp_policies.id"), nullable=False)
    name = Column(String(150), nullable=False)
    limit_amount = Column(Float, default=0.0, nullable=False)
    deductible = Column(Float, default=0.0, nullable=False)
    premium_amount = Column(Float, default=0.0, nullable=False)
    status = Column(String(30), default="active", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    policy = relationship("Policy", back_populates="coverages")


class Claim(Base):
    __tablename__ = "erp_claims"

    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(Integer, ForeignKey("erp_policies.id"), nullable=False)
    claim_number = Column(String(100), unique=True, nullable=False)
    reported_date = Column(Date, nullable=False)
    occurrence_date = Column(Date, nullable=True)
    status = Column(String(30), default="open", nullable=False)
    amount = Column(Float, default=0.0, nullable=False)
    paid_amount = Column(Float, default=0.0, nullable=False)
    currency = Column(String(10), default="TND", nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    policy = relationship("Policy", back_populates="claims")


class Invoice(Base):
    __tablename__ = "erp_invoices"

    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(Integer, ForeignKey("erp_policies.id"), nullable=False)
    invoice_number = Column(String(100), unique=True, nullable=False)
    issued_date = Column(Date, nullable=False)
    due_date = Column(Date, nullable=True)
    amount = Column(Float, default=0.0, nullable=False)
    paid_amount = Column(Float, default=0.0, nullable=False)
    status = Column(String(30), default="pending", nullable=False)
    currency = Column(String(10), default="TND", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    policy = relationship("Policy", back_populates="invoices")


class LedgerEntry(Base):
    __tablename__ = "erp_ledger_entries"

    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(Integer, ForeignKey("erp_policies.id"), nullable=False)
    entry_type = Column(String(50), nullable=False)
    account_code = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    amount = Column(Float, default=0.0, nullable=False)
    currency = Column(String(10), default="TND", nullable=False)
    entry_date = Column(Date, nullable=False)
    reference = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    policy = relationship("Policy", back_populates="ledger_entries")
