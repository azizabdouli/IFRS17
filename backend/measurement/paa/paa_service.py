"""Service central Premium Allocation Approach (PAA)

Implémentation initiale focalisée sur :
 - Initialisation groupe de contrats
 - Calcul LRC ~ UPR (prorata temporis)
 - LIC simple (incurred - paid)
 - Test onéreux simplifié
 - Historisation des périodes consommées

Conçue pour être étendue sans casser l'API : respecter principes DDD (bounded context IFRS17 Measurement Engine).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ContractInput(BaseModel):
    """Entrée d'un contrat individuel agrégé dans un groupe IFRS17.

    Hypothèses simplificatrices (à enrichir) :
      - Aucune actualisation (contrats < 1 an ou materialité faible)
      - Pas de Risk Adjustment explicite (peut être dérivé plus tard)
      - Ratios attendus constants (peut évoluer vers triangles / chain-ladder)
    """
    contract_id: str
    portfolio: str
    inception: date
    expiry: date
    written_premium: float
    expected_claim_ratio: float = Field(0.6, ge=0, le=3)
    expected_expense_ratio: float = Field(0.1, ge=0, le=1)
    acquisition_cashflows: float = 0.0
    already_incurred_claims: float = 0.0
    claims_paid_to_date: float = 0.0

    @property
    def coverage_days(self) -> int:
        return (self.expiry - self.inception).days + 1


class PAAPeriodResult(BaseModel):
    period_start: date
    period_end: date
    earned_premium: float
    unearned_premium_end: float
    lrc_end: float
    lic_end: float
    onerous_flag: bool
    loss_component: float


class PAAInitialResult(BaseModel):
    group_id: str
    lrc_initial: float
    unearned_premium_initial: float
    onerous_flag: bool
    loss_component: float
    coverage_days: int


class PAAConfig(BaseModel):
    revenue_recognition: str = Field("linear", pattern="^(linear|coverage_units)$")
    onerous_threshold_margin: float = 0.0  # marge minimale requise
    minimum_loss_trigger: float = 1e-6


@dataclass
class PAAGroupState:
    group_id: str
    contracts: List[ContractInput]
    lrc_current: float
    unearned_premium: float
    lic_current: float = 0.0
    loss_component: float = 0.0
    coverage_days_total: int = 0
    days_earned: int = 0
    history: List[PAAPeriodResult] = field(default_factory=list)


class PAAService:
    """Service principal PAA.

    Pattern : stateful in-memory cache + option persistance SQL.
    """

    def __init__(self, config: Optional[PAAConfig] = None, persistence=None):
        self.config = config or PAAConfig()
        self._groups: Dict[str, PAAGroupState] = {}
        self.persistence = persistence  # PAAPersistence instance (optionnel)

    # ---------------------- Initialisation ----------------------
    def initialize_group(self, group_id: str, contracts: List[ContractInput]) -> PAAInitialResult:
        if group_id in self._groups:
            raise ValueError(f"Le groupe '{group_id}' est déjà initialisé")

        if not contracts:
            raise ValueError("La liste des contrats est vide")

        total_written = sum(c.written_premium for c in contracts)
        coverage_days = sum(c.coverage_days for c in contracts)  # simplifié

        expected_claims = sum(c.written_premium * c.expected_claim_ratio for c in contracts)
        expected_expenses = sum(c.written_premium * c.expected_expense_ratio for c in contracts)
        expected_margin = total_written - (expected_claims + expected_expenses)
        onerous_flag = expected_margin < self.config.onerous_threshold_margin
        loss_component = abs(expected_margin) if onerous_flag else 0.0

        state = PAAGroupState(
            group_id=group_id,
            contracts=contracts,
            lrc_current=total_written - sum(c.acquisition_cashflows for c in contracts),
            unearned_premium=total_written,
            lic_current=sum(c.already_incurred_claims for c in contracts) - sum(c.claims_paid_to_date for c in contracts),
            loss_component=loss_component,
            coverage_days_total=coverage_days,
            days_earned=0,
        )
        self._groups[group_id] = state

        result = PAAInitialResult(
            group_id=group_id,
            lrc_initial=state.lrc_current,
            unearned_premium_initial=state.unearned_premium,
            onerous_flag=onerous_flag,
            loss_component=loss_component,
            coverage_days=coverage_days,
        )
        
        # Persistance SQL (si activée)
        if self.persistence:
            self.persistence.save_group_initial(result, contracts)
        
        return result

    # ---------------------- Période ----------------------
    def process_period(
        self,
        group_id: str,
        period_start: date,
        period_end: date,
        incurred_claims: float = 0.0,
        claims_paid: float = 0.0,
    ) -> PAAPeriodResult:
        state = self._require_group(group_id)

        if period_end < period_start:
            raise ValueError("La date de fin est antérieure à la date de début")

        period_days = (period_end - period_start).days + 1
        if period_days <= 0:
            raise ValueError("Durée de période invalide")

        remaining_days = max(state.coverage_days_total - state.days_earned, 1)
        
        # Capture états avant mouvement (pour delta)
        lrc_start = state.lrc_current
        lic_start = state.lic_current

        # Reconnaissance du revenu - pour l'instant linéaire
        earned = state.unearned_premium * (period_days / remaining_days)
        earned = min(earned, state.unearned_premium)
        state.unearned_premium -= earned
        state.lrc_current = state.unearned_premium  # Approximation PAA standard
        state.days_earned += period_days

        # LIC update (simplifié)
        state.lic_current += incurred_claims - claims_paid

        # Re-test onéreux conservateur
        onerous_flag, loss_component = self._reassess_onerous(state)
        state.loss_component = loss_component

        result = PAAPeriodResult(
            period_start=period_start,
            period_end=period_end,
            earned_premium=earned,
            unearned_premium_end=state.unearned_premium,
            lrc_end=state.lrc_current,
            lic_end=state.lic_current,
            onerous_flag=onerous_flag,
            loss_component=loss_component,
        )
        state.history.append(result)
        
        # Persistance (si activée)
        if self.persistence:
            self.persistence.save_movement(
                group_id=group_id,
                period_result=result,
                lrc_start=lrc_start,
                lic_start=lic_start,
                claims_incurred=incurred_claims,
                claims_paid=claims_paid,
            )
            self.persistence.update_group_state(
                group_id=group_id,
                lrc=state.lrc_current,
                lic=state.lic_current,
                unearned=state.unearned_premium,
                loss_comp=state.loss_component,
                days_earned=state.days_earned,
                onerous=onerous_flag,
            )
        
        return result

    # ---------------------- Consultation ----------------------
    def get_group_state(self, group_id: str) -> Dict:
        state = self._require_group(group_id)
        return {
            "group_id": state.group_id,
            "lrc_current": state.lrc_current,
            "lic_current": state.lic_current,
            "unearned_premium": state.unearned_premium,
            "loss_component": state.loss_component,
            "coverage_days_total": state.coverage_days_total,
            "days_earned": state.days_earned,
            "history": [r.model_dump() for r in state.history],
        }

    # ---------------------- Helpers internes ----------------------
    def _reassess_onerous(self, state: PAAGroupState):
        margin = state.lrc_current  # proxy très conservateur
        onerous = margin < self.config.onerous_threshold_margin
        loss_component = abs(margin) if onerous else 0.0
        return onerous, loss_component

    def _require_group(self, group_id: str) -> PAAGroupState:
        if group_id not in self._groups:
            raise ValueError(f"Groupe '{group_id}' non initialisé")
        return self._groups[group_id]
