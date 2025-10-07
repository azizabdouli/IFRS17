"""Couche de persistance PAA - SQL Storage

Bridge entre PAAService (in-memory) et base de données.
Permet migration progressive sans casser l'API existante.
"""
from typing import List, Optional, Dict
from datetime import date
from sqlalchemy.orm import Session
from backend.database.paa_models import PAAGroup, PAAContract, PAAMovement, PAASnapshot
from backend.measurement.paa import ContractInput, PAAPeriodResult, PAAInitialResult


class PAAPersistence:
    """Service de persistance pour PAA"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ---------------------- Groupes ----------------------
    def save_group_initial(self, result: PAAInitialResult, contracts: List[ContractInput]) -> PAAGroup:
        """Sauvegarde initialisation groupe"""
        # Créer groupe
        db_group = PAAGroup(
            group_id=result.group_id,
            portfolio=contracts[0].portfolio if contracts else "UNKNOWN",
            inception_date=min(c.inception for c in contracts),
            lrc_current=result.lrc_initial,
            unearned_premium=result.unearned_premium_initial,
            loss_component=result.loss_component,
            coverage_days_total=result.coverage_days,
            days_earned=0,
            onerous_flag=result.onerous_flag,
        )
        self.db.add(db_group)
        self.db.flush()
        
        # Créer contrats liés
        for c in contracts:
            db_contract = PAAContract(
                group_id=db_group.id,
                contract_id=c.contract_id,
                portfolio=c.portfolio,
                inception=c.inception,
                expiry=c.expiry,
                written_premium=c.written_premium,
                expected_claim_ratio=c.expected_claim_ratio,
                expected_expense_ratio=c.expected_expense_ratio,
                acquisition_cashflows=c.acquisition_cashflows,
                already_incurred_claims=c.already_incurred_claims,
                claims_paid_to_date=c.claims_paid_to_date,
            )
            self.db.add(db_contract)
        
        self.db.commit()
        self.db.refresh(db_group)
        return db_group
    
    def update_group_state(
        self,
        group_id: str,
        lrc: float,
        lic: float,
        unearned: float,
        loss_comp: float,
        days_earned: int,
        onerous: bool,
    ):
        """Mise à jour état groupe après période"""
        db_group = self.db.query(PAAGroup).filter(PAAGroup.group_id == group_id).first()
        if not db_group:
            raise ValueError(f"Groupe {group_id} introuvable")
        
        db_group.lrc_current = lrc
        db_group.lic_current = lic
        db_group.unearned_premium = unearned
        db_group.loss_component = loss_comp
        db_group.days_earned = days_earned
        db_group.onerous_flag = onerous
        
        self.db.commit()
    
    def get_group(self, group_id: str) -> Optional[PAAGroup]:
        """Récupère groupe par ID"""
        return self.db.query(PAAGroup).filter(PAAGroup.group_id == group_id).first()
    
    def list_groups(self, portfolio: Optional[str] = None) -> List[PAAGroup]:
        """Liste groupes (optionnel: filtré par portfolio)"""
        q = self.db.query(PAAGroup)
        if portfolio:
            q = q.filter(PAAGroup.portfolio == portfolio)
        return q.all()
    
    # ---------------------- Mouvements ----------------------
    def save_movement(
        self,
        group_id: str,
        period_result: PAAPeriodResult,
        lrc_start: float,
        lic_start: float,
        claims_incurred: float,
        claims_paid: float,
    ) -> PAAMovement:
        """Sauvegarde mouvement période"""
        db_group = self.get_group(group_id)
        if not db_group:
            raise ValueError(f"Groupe {group_id} introuvable")
        
        movement = PAAMovement(
            group_id=db_group.id,
            period_start=period_result.period_start,
            period_end=period_result.period_end,
            period_label=f"{period_result.period_start.strftime('%b %Y')}",
            earned_premium=period_result.earned_premium,
            change_in_lrc=period_result.lrc_end - lrc_start,
            claims_incurred=claims_incurred,
            claims_paid=claims_paid,
            change_in_lic=period_result.lic_end - lic_start,
            loss_component_movement=period_result.loss_component,
            lrc_end=period_result.lrc_end,
            lic_end=period_result.lic_end,
            unearned_premium_end=period_result.unearned_premium_end,
            onerous_flag=period_result.onerous_flag,
        )
        self.db.add(movement)
        self.db.commit()
        self.db.refresh(movement)
        return movement
    
    def get_movements(self, group_id: str) -> List[PAAMovement]:
        """Récupère tous les mouvements d'un groupe"""
        db_group = self.get_group(group_id)
        if not db_group:
            return []
        return self.db.query(PAAMovement).filter(PAAMovement.group_id == db_group.id).order_by(PAAMovement.period_start).all()
    
    # ---------------------- Snapshots ----------------------
    def save_snapshot(self, group_id: str, snapshot_date: date, state: Dict, notes: str = ""):
        """Sauvegarde snapshot audit"""
        db_group = self.get_group(group_id)
        if not db_group:
            raise ValueError(f"Groupe {group_id} introuvable")
        
        snapshot = PAASnapshot(
            group_id=db_group.id,
            snapshot_date=snapshot_date,
            state_json=state,
            notes=notes,
        )
        self.db.add(snapshot)
        self.db.commit()
