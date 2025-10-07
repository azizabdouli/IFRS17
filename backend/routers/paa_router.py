"""Router IFRS17 PAA - Version Professionnelle

Endpoints :
- Gestion groupes (init, state)
- Mouvements périodiques
- Export IFRS17
- Stress testing
- Analytics
"""
from fastapi import APIRouter, HTTPException, Depends
from datetime import date
from typing import List, Optional
from sqlalchemy.orm import Session

from backend.measurement.paa import PAAService, ContractInput
from backend.measurement.paa.paa_persistence import PAAPersistence
from backend.database.connection import get_db

router = APIRouter(prefix="/paa", tags=["📘 IFRS17 PAA"])

# Service singleton (hybride in-memory + SQL)
paa_service = PAAService()


def get_persistence(db: Session = Depends(get_db)) -> PAAPersistence:
    """Dépendance pour accès persistance"""
    return PAAPersistence(db)


@router.post("/groups/init")
def init_group(
    group_id: str,
    contracts: List[ContractInput],
    revenue_pattern: str = "linear",
    persist: bool = True,
    db: Session = Depends(get_db),
):
    """Initialise un groupe de contrats PAA
    
    Args:
        group_id: Identifiant unique du groupe
        contracts: Liste des contrats à agréger
        revenue_pattern: Modèle de reconnaissance (linear, coverage_units)
        persist: Active la persistance SQL
    """
    try:
        # Activer persistance si demandé
        if persist:
            paa_service.persistence = PAAPersistence(db)
        
        res = paa_service.initialize_group(group_id, contracts)
        return {"status": "success", "initial": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/groups/{group_id}/period")
def process_period(
    group_id: str,
    period_start: date,
    period_end: date,
    incurred_claims: float = 0.0,
    claims_paid: float = 0.0,
    persist: bool = True,
    db: Session = Depends(get_db),
):
    """Traite une période pour un groupe
    
    Génère automatiquement les mouvements IFRS17 et met à jour les états.
    """
    try:
        if persist:
            paa_service.persistence = PAAPersistence(db)
        
        res = paa_service.process_period(
            group_id=group_id,
            period_start=period_start,
            period_end=period_end,
            incurred_claims=incurred_claims,
            claims_paid=claims_paid,
        )
        return {"status": "success", "period_result": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/groups/{group_id}")
def get_group_state(group_id: str, db: Session = Depends(get_db)):
    """Récupère l'état courant + historique d'un groupe"""
    try:
        # Essayer mémoire d'abord
        state = paa_service.get_group_state(group_id)
        return {"status": "success", "state": state, "source": "memory"}
    except:
        # Fallback: chercher en base
        persistence = PAAPersistence(db)
        db_group = persistence.get_group(group_id)
        if not db_group:
            raise HTTPException(status_code=404, detail=f"Groupe {group_id} introuvable")
        
        return {
            "status": "success",
            "state": {
                "group_id": db_group.group_id,
                "lrc_current": db_group.lrc_current,
                "lic_current": db_group.lic_current,
                "unearned_premium": db_group.unearned_premium,
                "loss_component": db_group.loss_component,
                "onerous_flag": db_group.onerous_flag,
                "days_earned": db_group.days_earned,
                "coverage_days_total": db_group.coverage_days_total,
            },
            "source": "database"
        }


@router.get("/groups/{group_id}/movements")
def get_movements(group_id: str, db: Session = Depends(get_db)):
    """Liste tous les mouvements d'un groupe (pour reporting IFRS17)"""
    try:
        persistence = PAAPersistence(db)
        movements = persistence.get_movements(group_id)
        
        return {
            "status": "success",
            "group_id": group_id,
            "movements": [
                {
                    "period_start": m.period_start.isoformat(),
                    "period_end": m.period_end.isoformat(),
                    "period_label": m.period_label,
                    "earned_premium": m.earned_premium,
                    "change_in_lrc": m.change_in_lrc,
                    "claims_incurred": m.claims_incurred,
                    "claims_paid": m.claims_paid,
                    "change_in_lic": m.change_in_lic,
                    "lrc_end": m.lrc_end,
                    "lic_end": m.lic_end,
                    "unearned_premium_end": m.unearned_premium_end,
                    "onerous_flag": m.onerous_flag,
                }
                for m in movements
            ],
            "total_movements": len(movements),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/groups")
def list_groups(portfolio: Optional[str] = None, db: Session = Depends(get_db)):
    """Liste tous les groupes PAA (optionnel: filtré par portfolio)"""
    try:
        persistence = PAAPersistence(db)
        groups = persistence.list_groups(portfolio=portfolio)
        
        return {
            "status": "success",
            "groups": [
                {
                    "group_id": g.group_id,
                    "portfolio": g.portfolio,
                    "lrc_current": g.lrc_current,
                    "lic_current": g.lic_current,
                    "unearned_premium": g.unearned_premium,
                    "onerous_flag": g.onerous_flag,
                    "created_at": g.created_at.isoformat() if g.created_at else None,
                }
                for g in groups
            ],
            "total": len(groups),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/groups/{group_id}/stress-test")
def stress_test(
    group_id: str,
    claim_ratio_shock: float = 0.0,
    expense_ratio_shock: float = 0.0,
):
    """Simule un stress test sur les ratios attendus
    
    Args:
        claim_ratio_shock: Choc en points (ex: 0.1 = +10%)
        expense_ratio_shock: Choc en points
    
    Returns:
        Impact sur marge, onerous flag, loss component
    """
    try:
        state = paa_service.get_group_state(group_id)
        
        # Recalcul simplifié marge avec chocs
        # (Version simplifiée - à enrichir avec vraie projection)
        total_premium = state["unearned_premium"]
        base_claim = total_premium * 0.6  # proxy
        base_expense = total_premium * 0.1
        
        shocked_claim = base_claim + (total_premium * claim_ratio_shock)
        shocked_expense = base_expense + (total_premium * expense_ratio_shock)
        shocked_margin = total_premium - shocked_claim - shocked_expense
        
        new_onerous = shocked_margin < 0
        new_loss_component = abs(shocked_margin) if new_onerous else 0
        
        return {
            "status": "success",
            "group_id": group_id,
            "stress_scenario": {
                "claim_ratio_shock": claim_ratio_shock,
                "expense_ratio_shock": expense_ratio_shock,
            },
            "results": {
                "base_margin": state["unearned_premium"] - base_claim - base_expense,
                "shocked_margin": shocked_margin,
                "margin_impact": shocked_margin - (state["unearned_premium"] - base_claim - base_expense),
                "onerous_flag": new_onerous,
                "loss_component": new_loss_component,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics/portfolio-summary")
def portfolio_summary(db: Session = Depends(get_db)):
    """Agrégation au niveau portfolio (dashboard PAA)"""
    try:
        persistence = PAAPersistence(db)
        groups = persistence.list_groups()
        
        total_lrc = sum(g.lrc_current for g in groups)
        total_lic = sum(g.lic_current for g in groups)
        total_unearned = sum(g.unearned_premium for g in groups)
        onerous_count = sum(1 for g in groups if g.onerous_flag)
        
        return {
            "status": "success",
            "summary": {
                "total_groups": len(groups),
                "total_lrc": total_lrc,
                "total_lic": total_lic,
                "total_unearned_premium": total_unearned,
                "onerous_groups": onerous_count,
                "onerous_ratio": onerous_count / len(groups) if groups else 0,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
