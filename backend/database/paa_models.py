"""Modèles SQLAlchemy pour le moteur PAA IFRS17

Tables principales :
- paa_groups : agrégats de contrats
- paa_contracts : détail contrats individuels
- paa_period_movements : mouvements comptables par période
- paa_snapshots : états à date donnée (audit trail)
"""
from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, JSON, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database.connection import Base


class PAAGroup(Base):
    __tablename__ = "paa_groups"
    
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(String(100), unique=True, index=True, nullable=False)
    portfolio = Column(String(100), nullable=False)
    inception_date = Column(Date, nullable=False)
    
    # Agrégats financiers courants
    lrc_current = Column(Float, default=0.0)
    lic_current = Column(Float, default=0.0)
    unearned_premium = Column(Float, default=0.0)
    loss_component = Column(Float, default=0.0)
    
    # Métriques opérationnelles
    coverage_days_total = Column(Integer, default=0)
    days_earned = Column(Integer, default=0)
    onerous_flag = Column(Boolean, default=False)
    
    # Métadonnées
    revenue_pattern = Column(String(50), default="linear")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relations
    contracts = relationship("PAAContract", back_populates="group", cascade="all, delete-orphan")
    movements = relationship("PAAMovement", back_populates="group", cascade="all, delete-orphan")
    snapshots = relationship("PAASnapshot", back_populates="group", cascade="all, delete-orphan")


class PAAContract(Base):
    __tablename__ = "paa_contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("paa_groups.id"), nullable=False)
    contract_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Détails contrat
    portfolio = Column(String(100), nullable=False)
    inception = Column(Date, nullable=False)
    expiry = Column(Date, nullable=False)
    
    # Financiers
    written_premium = Column(Float, nullable=False)
    expected_claim_ratio = Column(Float, default=0.6)
    expected_expense_ratio = Column(Float, default=0.1)
    acquisition_cashflows = Column(Float, default=0.0)
    
    # État sinistres
    already_incurred_claims = Column(Float, default=0.0)
    claims_paid_to_date = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relation
    group = relationship("PAAGroup", back_populates="contracts")


class PAAMovement(Base):
    __tablename__ = "paa_movements"
    
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("paa_groups.id"), nullable=False)
    
    # Période
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    period_label = Column(String(50))  # ex: "2025-Q1", "Jan 2025"
    
    # Mouvements IFRS17
    earned_premium = Column(Float, default=0.0)
    change_in_lrc = Column(Float, default=0.0)
    claims_incurred = Column(Float, default=0.0)
    claims_paid = Column(Float, default=0.0)
    change_in_lic = Column(Float, default=0.0)
    loss_component_movement = Column(Float, default=0.0)
    
    # États de fin
    lrc_end = Column(Float, default=0.0)
    lic_end = Column(Float, default=0.0)
    unearned_premium_end = Column(Float, default=0.0)
    
    # Indicateurs
    onerous_flag = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relation
    group = relationship("PAAGroup", back_populates="movements")


class PAASnapshot(Base):
    __tablename__ = "paa_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("paa_groups.id"), nullable=False)
    
    snapshot_date = Column(Date, nullable=False)
    snapshot_type = Column(String(50), default="period_end")  # period_end, onerous_test, adjustment
    
    # État complet à date
    state_json = Column(JSON)
    
    # Audit
    created_by = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relation
    group = relationship("PAAGroup", back_populates="snapshots")
