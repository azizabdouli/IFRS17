"""Module IFRS17 - Premium Allocation Approach (PAA)

Ce package fournit une implémentation incrémentale et extensible de l'approche PAA.
Objectifs :
 - Rapid prototyping (in-memory) pour validation fonctionnelle
 - Extension future vers stockage persistant, subledger et reporting
 - Séparation claire logique / API

Roadmap (évolutions prévues) :
 1. Support coverage units custom (non-linéaire)
 2. Coûts d'acquisition différés (DAC) + amortissement
 3. Risk Adjustment optionnel (PAA étendue)
 4. Subledger (mouvements IFRS17 : LRC, LIC, revenue, claims, loss component)
 5. Export audit trail (Excel/JSON)
 6. Orchestration batch multi-groupes
 7. Versioning hypothèses / scenario management
"""

from .paa_service import (
    PAAService,
    PAAConfig,
    ContractInput,
    PAAPeriodResult,
    PAAInitialResult,
    PAAGroupState
)

__all__ = [
    "PAAService",
    "PAAConfig",
    "ContractInput",
    "PAAPeriodResult",
    "PAAInitialResult",
    "PAAGroupState",
]
