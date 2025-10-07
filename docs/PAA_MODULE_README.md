# Module PAA (Premium Allocation Approach) - IFRS 17

## 🎯 Vue d'ensemble

Ce module implémente l'approche PAA (Premium Allocation Approach) conformément à la norme IFRS 17, avec une architecture professionnelle extensible et une interface utilisateur intuitive.

## 📦 Composants Backend

### 1. Couche Service (`paa_service.py`)
- **PAAService**: Moteur de calcul PAA principal
- **Initialisation** de groupes de contrats
- **Traitement périodique** avec reconnaissance de revenus
- **Test onéreux** simplifié avec loss component
- Support **in-memory + persistance SQL**

### 2. Persistance SQL (`paa_persistence.py`)
- **PAAPersistence**: Bridge vers base de données
- Sauvegarde automatique des états et mouvements
- Audit trail avec snapshots
- Requêtes optimisées pour reporting

### 3. Modèles SQLAlchemy (`paa_models.py`)
Tables créées:
- `paa_groups`: Groupes de contrats agrégés
- `paa_contracts`: Détails contrats individuels
- `paa_movements`: Mouvements IFRS 17 par période
- `paa_snapshots`: États à date (audit)

### 4. Router FastAPI (`paa_router.py`)
Endpoints exposés:
- `POST /paa/groups/init`: Initialiser un groupe
- `POST /paa/groups/{group_id}/period`: Traiter une période
- `GET /paa/groups/{group_id}`: État du groupe
- `GET /paa/groups/{group_id}/movements`: Liste des mouvements
- `GET /paa/groups`: Liste tous les groupes
- `POST /paa/groups/{group_id}/stress-test`: Simulation stress
- `GET /paa/analytics/portfolio-summary`: Agrégation portfolio

## 🎨 Interface Angular

### Composant Principal (`paa-dashboard.component.ts`)
- Vue liste/détails (master-detail pattern)
- Formulaires modaux pour initialisation
- Traitement périodique intégré
- Stress testing interactif
- Tableau de mouvements IFRS 17

### Design BNA
Palette couleurs:
- **Primary**: #d32f2f (rouge BNA)
- **Secondary**: #424242 (gris foncé)
- **Success**: #4caf50
- **Warning**: #ff9800
- **Danger**: #f44336

### Features UI
- ✅ Cards résumé portfolio (LRC, LIC, groupes onéreux)
- ✅ Liste groupes avec statuts visuels
- ✅ Détails groupe avec métriques clés
- ✅ Formulaires validation période
- ✅ Stress test avec résultats immédiats
- ✅ Tableau mouvements avec tri et filtrage
- ✅ Responsive design (desktop/tablet)

## 🚀 Utilisation

### Backend - Initialiser un groupe

```python
from backend.measurement.paa import PAAService, ContractInput
from datetime import date

service = PAAService()

contracts = [
    ContractInput(
        contract_id="C1",
        portfolio="AUTO",
        inception=date(2025, 1, 1),
        expiry=date(2025, 12, 31),
        written_premium=1200.0,
        expected_claim_ratio=0.55,
        expected_expense_ratio=0.12
    )
]

result = service.initialize_group("G1", contracts)
print(f"LRC initiale: {result.lrc_initial}")
```

### Backend - Traiter une période

```python
period_result = service.process_period(
    group_id="G1",
    period_start=date(2025, 1, 1),
    period_end=date(2025, 1, 31),
    incurred_claims=180.0,
    claims_paid=150.0
)

print(f"Prime acquise: {period_result.earned_premium}")
print(f"LRC fin: {period_result.lrc_end}")
```

### API HTTP - Exemples cURL

**Initialiser groupe:**
```bash
curl -X POST "http://127.0.0.1:8001/paa/groups/init?group_id=G1&persist=true" \
  -H "Content-Type: application/json" \
  -d '[{
    "contract_id": "C1",
    "portfolio": "AUTO",
    "inception": "2025-01-01",
    "expiry": "2025-12-31",
    "written_premium": 1200,
    "expected_claim_ratio": 0.55,
    "expected_expense_ratio": 0.12
  }]'
```

**Traiter période:**
```bash
curl -X POST "http://127.0.0.1:8001/paa/groups/G1/period?period_start=2025-01-01&period_end=2025-01-31&incurred_claims=180&claims_paid=150&persist=true"
```

**Lister groupes:**
```bash
curl "http://127.0.0.1:8001/paa/groups"
```

**Stress test:**
```bash
curl -X POST "http://127.0.0.1:8001/paa/groups/G1/stress-test?claim_ratio_shock=0.1&expense_ratio_shock=0.05"
```

### Frontend - Navigation

Accéder au dashboard PAA:
```
http://localhost:4200/paa-dashboard
```

Ou via le menu de navigation (nécessite authentification).

## 📊 Mouvements IFRS 17

Chaque période génère automatiquement:
- **Earned Premium**: Prime acquise (reconnaissance revenu)
- **Change in LRC**: Variation LRC (UPR)
- **Claims Incurred**: Sinistres encourus
- **Claims Paid**: Sinistres payés
- **Change in LIC**: Variation LIC
- **Loss Component Movement**: Ajustement loss component si onéreux

## 🧪 Tests

### Test unitaire backend
```bash
cd backend
pytest tests/test_paa.py -v
```

### Test d'intégration (avec DB)
```python
from backend.database.connection import SessionLocal
from backend.measurement.paa.paa_persistence import PAAPersistence

db = SessionLocal()
persistence = PAAPersistence(db)

# Service avec persistance
service = PAAService(persistence=persistence)

# ... utiliser service normalement
```

## 📈 Roadmap & Extensions

### Phase 1 - Complété ✅
- [x] Service PAA core
- [x] Persistance SQL
- [x] Router FastAPI complet
- [x] UI Angular professionnelle
- [x] Mouvements IFRS 17
- [x] Stress testing basique

### Phase 2 - En cours 🔄
- [ ] Coverage units non linéaires (paramétrable)
- [ ] DAC (Deferred Acquisition Costs) + amortissement
- [ ] Risk Adjustment optionnel (PAA étendue)
- [ ] Export Excel mouvements IFRS 17
- [ ] Batch orchestrator multi-groupes
- [ ] Graphiques analytics (Chart.js/D3.js)

### Phase 3 - Planifié 📅
- [ ] Subledger IFRS 17 (mapping GL accounts)
- [ ] Versioning hypothèses actuarielles
- [ ] Multi-scénarios (best/worst case)
- [ ] Audit trail complet (who/when/what)
- [ ] Intégration avec PPNA dataset (auto-initialisation)
- [ ] Reporting PDF automatisé
- [ ] API export vers outils BI externes

## 🔧 Configuration

### Variables d'environnement (optionnel)
```env
PAA_REVENUE_PATTERN=linear  # ou coverage_units
PAA_ONEROUS_THRESHOLD=0.0
PAA_PERSISTENCE_ENABLED=true
```

### Configuration service
```python
from backend.measurement.paa import PAAConfig

config = PAAConfig(
    revenue_recognition="linear",
    onerous_threshold_margin=0.0,
    minimum_loss_trigger=1e-6
)

service = PAAService(config=config)
```

## 📚 Documentation API

Accéder à la documentation interactive Swagger:
```
http://127.0.0.1:8001/docs
```

Section "📘 IFRS17 PAA" pour tous les endpoints.

## 🐛 Troubleshooting

### Erreur "Group already initialized"
**Solution**: Chaque `group_id` doit être unique. Utilisez `list_groups()` pour voir les groupes existants.

### Erreur "Table paa_groups doesn't exist"
**Solution**: Relancer le backend pour créer les tables automatiquement, ou exécuter:
```python
from backend.database.connection import engine, Base
from backend.database.paa_models import *
Base.metadata.create_all(bind=engine)
```

### UI Angular: Erreur CORS
**Solution**: Vérifier que le backend autorise `http://localhost:4200` dans CORS middleware (déjà configuré).

### Mouvements vides
**Solution**: Assurez-vous d'avoir traité au moins une période avec `process_period()` après initialisation.

## 🤝 Contribution

Pour étendre le module:
1. Service logic → `paa_service.py`
2. Persistance → `paa_persistence.py`
3. API endpoints → `paa_router.py`
4. UI → `paa-dashboard.component.*`
5. Tests → `test_paa.py`

Respecter les patterns existants (DDD, Repository pattern, Component architecture).

## 📄 License

Propriété BNA - Usage interne uniquement.

---

**Auteur**: Équipe IFRS17 BNA  
**Version**: 1.0.0  
**Date**: Octobre 2025  
**Contact**: support-ifrs17@bna.com.tn
