# 🏗️ ARCHITECTURE MODULE PAA IFRS 17

## 📐 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                     IFRS 17 APPLICATION                          │
│                   (BNA - Banque Nationale Agricole)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Angular 17)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Dashboard   │  │   ML         │  │   PAA        │         │
│  │  Component   │  │   Analytics  │  │   Dashboard  │ ← NEW   │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │      IFRS17ApiService (HTTP Client)              │          │
│  │  - initPAAGroup()                                 │          │
│  │  - processPAAPeriod()                             │          │
│  │  - getPAAMovements()                              │          │
│  │  - paaStressTest()                                │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP REST API
                              │ (JSON)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              PAA ROUTER                           │          │
│  │  POST   /paa/groups/init                         │          │
│  │  POST   /paa/groups/{id}/period                  │          │
│  │  GET    /paa/groups/{id}                         │          │
│  │  GET    /paa/groups/{id}/movements               │          │
│  │  POST   /paa/groups/{id}/stress-test             │          │
│  │  GET    /paa/analytics/portfolio-summary         │          │
│  └──────────────────────────────────────────────────┘          │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │              PAA SERVICE (Business Logic)         │          │
│  │                                                   │          │
│  │  - initialize_group()                             │          │
│  │    → Calcul LRC initiale                          │          │
│  │    → Test onéreux                                 │          │
│  │                                                   │          │
│  │  - process_period()                               │          │
│  │    → Reconnaissance revenu                        │          │
│  │    → Mise à jour LRC/LIC                          │          │
│  │    → Génération mouvements                        │          │
│  │                                                   │          │
│  │  - get_group_state()                              │          │
│  │    → État courant + historique                    │          │
│  └──────────────────────────────────────────────────┘          │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │          PAA PERSISTENCE (Data Access Layer)      │          │
│  │                                                   │          │
│  │  - save_group_initial()                           │          │
│  │  - save_movement()                                │          │
│  │  - update_group_state()                           │          │
│  │  - get_movements()                                │          │
│  │  - save_snapshot()                                │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ SQLAlchemy ORM
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATABASE (MySQL)                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ paa_groups   │  │paa_contracts │  │paa_movements │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ id           │  │ id           │  │ id           │         │
│  │ group_id*    │  │ group_id (FK)│  │ group_id (FK)│         │
│  │ portfolio    │  │ contract_id* │  │ period_start │         │
│  │ lrc_current  │  │ portfolio    │  │ period_end   │         │
│  │ lic_current  │  │ inception    │  │ earned_prem  │         │
│  │ unearned_prem│  │ expiry       │  │ change_lrc   │         │
│  │ loss_comp    │  │ written_prem │  │ claims_inc   │         │
│  │ onerous_flag │  │ claim_ratio  │  │ claims_paid  │         │
│  │ created_at   │  │ expense_rat  │  │ lrc_end      │         │
│  └──────────────┘  └──────────────┘  │ lic_end      │         │
│                                       │ onerous_flag │         │
│  ┌──────────────┐                    └──────────────┘         │
│  │paa_snapshots │                                              │
│  ├──────────────┤                                              │
│  │ id           │                                              │
│  │ group_id (FK)│                                              │
│  │ snapshot_date│                                              │
│  │ state_json   │    (Audit Trail)                            │
│  │ notes        │                                              │
│  │ created_at   │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flux de Données - Initialisation Groupe

```
User Action (UI)
      │
      │ 1. Clic "Nouveau Groupe"
      ▼
┌─────────────────┐
│ Modal Form      │
│ - Group ID      │
│ - Contracts JSON│
└─────────────────┘
      │
      │ 2. Submit
      ▼
┌─────────────────────────────────────┐
│ ifrs17-api.service.ts               │
│ initPAAGroup(groupId, contracts)    │
└─────────────────────────────────────┘
      │
      │ 3. POST /paa/groups/init
      ▼
┌─────────────────────────────────────┐
│ paa_router.py                       │
│ init_group()                        │
└─────────────────────────────────────┘
      │
      │ 4. Call service
      ▼
┌─────────────────────────────────────┐
│ paa_service.py                      │
│ initialize_group(group_id, contracts│
│                                     │
│ → Calcul LRC = Σ written_premium   │
│ → Test onéreux = margin < 0 ?      │
│ → Loss component si onéreux         │
└─────────────────────────────────────┘
      │
      │ 5. Persist (if enabled)
      ▼
┌─────────────────────────────────────┐
│ paa_persistence.py                  │
│ save_group_initial()                │
│ → Insert paa_groups                 │
│ → Insert paa_contracts (bulk)       │
└─────────────────────────────────────┘
      │
      │ 6. Return PAAInitialResult
      ▼
┌─────────────────────────────────────┐
│ Response JSON                       │
│ {                                   │
│   status: "success",                │
│   initial: {                        │
│     group_id: "AUTO_2025_Q1",       │
│     lrc_initial: 39000,             │
│     onerous_flag: false             │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘
      │
      │ 7. Update UI
      ▼
┌─────────────────────────────────────┐
│ paa-dashboard.component.ts          │
│ - Reload groups list                │
│ - Close modal                       │
│ - Show success message              │
└─────────────────────────────────────┘
```

---

## 🔄 Flux de Données - Traitement Période

```
User Action
      │
      │ 1. Select Group + "Traiter Période"
      ▼
┌─────────────────────────────────────┐
│ Period Form                         │
│ - Period Start: 2025-01-01          │
│ - Period End: 2025-01-31            │
│ - Incurred Claims: 2500             │
│ - Claims Paid: 2000                 │
└─────────────────────────────────────┘
      │
      │ 2. Submit
      ▼
┌─────────────────────────────────────┐
│ processPAAPeriod(...)               │
└─────────────────────────────────────┘
      │
      │ 3. POST /paa/groups/{id}/period
      ▼
┌─────────────────────────────────────┐
│ paa_service.process_period()        │
│                                     │
│ STEP 1: Capture états début        │
│   lrc_start = state.lrc_current     │
│   lic_start = state.lic_current     │
│                                     │
│ STEP 2: Reconnaissance revenu       │
│   period_days = 31                  │
│   remaining_days = 365              │
│   earned = UPR * (31/365)           │
│   UPR -= earned                     │
│   LRC = UPR                         │
│                                     │
│ STEP 3: Mise à jour LIC             │
│   LIC += (incurred - paid)          │
│                                     │
│ STEP 4: Re-test onéreux             │
│   onerous = LRC < 0 ?               │
│                                     │
│ STEP 5: Générer PAAPeriodResult     │
└─────────────────────────────────────┘
      │
      │ 4. Save Movement
      ▼
┌─────────────────────────────────────┐
│ paa_persistence.save_movement()     │
│ → Insert paa_movements              │
│     - earned_premium                │
│     - change_in_lrc                 │
│     - claims_incurred / paid        │
│     - lrc_end, lic_end              │
│                                     │
│ → Update paa_groups                 │
│     SET lrc_current = X             │
│         lic_current = Y             │
│         days_earned += 31           │
└─────────────────────────────────────┘
      │
      │ 5. Response
      ▼
┌─────────────────────────────────────┐
│ JSON                                │
│ {                                   │
│   period_result: {                  │
│     earned_premium: 3300,           │
│     lrc_end: 35700,                 │
│     lic_end: 500,                   │
│     onerous_flag: false             │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘
      │
      │ 6. Update UI
      ▼
┌─────────────────────────────────────┐
│ - Reload movements table            │
│ - Update group state cards          │
│ - Update portfolio summary          │
└─────────────────────────────────────┘
```

---

## 🎨 Architecture Frontend (Composant PAA)

```
┌───────────────────────────────────────────────────────────────┐
│                  paa-dashboard.component                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  STATE MANAGEMENT (RxJS)                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ groups: PAAGroup[]                                      │ │
│  │ selectedGroup: PAAGroup | null                          │ │
│  │ movements: PAAMovement[]                                │ │
│  │ portfolioSummary: PortfolioSummary                      │ │
│  │ loading / loadingGroups / loadingMovements              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ACTIONS                                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ loadData()        → Get groups + summary                │ │
│  │ selectGroup()     → Load movements                      │ │
│  │ initGroup()       → POST init                           │ │
│  │ processPeriod()   → POST period                         │ │
│  │ runStressTest()   → POST stress-test                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  RENDERING                                                    │
│  ┌──────────────────┬──────────────────────────────────────┐ │
│  │ Summary Cards    │ Portfolio KPIs (4 cards)             │ │
│  ├──────────────────┼──────────────────────────────────────┤ │
│  │ Groups Panel     │ Details Panel                        │ │
│  │ (Master)         │ (Detail)                             │ │
│  │                  │                                      │ │
│  │ • List groups    │ • Group info cards                   │ │
│  │ • Select → emit  │ • Action buttons                     │ │
│  │ • Status badges  │ • Period form                        │ │
│  │                  │ • Stress test form                   │ │
│  │                  │ • Movements table                    │ │
│  └──────────────────┴──────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Modèle de Données SQL

```sql
-- Relations

paa_groups (1) ──── (N) paa_contracts
      │
      │
      ├──── (N) paa_movements
      │
      └──── (N) paa_snapshots


-- Exemple données

paa_groups
┌──────────┬──────────────┬──────────┬────────────┬────────────┬──────────────┬──────────────┐
│ id       │ group_id     │ portfolio│ lrc_current│ lic_current│ unearned_prem│ onerous_flag │
├──────────┼──────────────┼──────────┼────────────┼────────────┼──────────────┼──────────────┤
│ 1        │ AUTO_2025_Q1 │ AUTO     │ 35700      │ 500        │ 35700        │ false        │
└──────────┴──────────────┴──────────┴────────────┴────────────┴──────────────┴──────────────┘

paa_contracts
┌──────┬──────────┬───────────────┬──────────┬────────────┬────────────┬────────────────┬──────────────┐
│ id   │ group_id │ contract_id   │ portfolio│ inception  │ expiry     │ written_premium│ claim_ratio  │
├──────┼──────────┼───────────────┼──────────┼────────────┼────────────┼────────────────┼──────────────┤
│ 1    │ 1        │ AUTO_2025_001 │ AUTO     │ 2025-01-01 │ 2025-12-31 │ 15000          │ 0.55         │
│ 2    │ 1        │ AUTO_2025_002 │ AUTO     │ 2025-01-01 │ 2025-06-30 │ 6000           │ 0.60         │
│ 3    │ 1        │ AUTO_2025_003 │ AUTO     │ 2025-02-01 │ 2026-01-31 │ 18000          │ 0.50         │
└──────┴──────────┴───────────────┴──────────┴────────────┴────────────┴────────────────┴──────────────┘

paa_movements
┌──────┬──────────┬──────────────┬──────────────┬──────────────┬──────────────┬───────────┬─────────┐
│ id   │ group_id │ period_start │ period_end   │ earned_premium│ change_in_lrc│ lrc_end   │ lic_end │
├──────┼──────────┼──────────────┼──────────────┼──────────────┼──────────────┼───────────┼─────────┤
│ 1    │ 1        │ 2025-01-01   │ 2025-01-31   │ 3300         │ -3300        │ 35700     │ 500     │
│ 2    │ 1        │ 2025-02-01   │ 2025-02-28   │ 3000         │ -3000        │ 32700     │ 900     │
└──────┴──────────┴──────────────┴──────────────┴──────────────┴──────────────┴───────────┴─────────┘
```

---

## 🧩 Patterns & Principes Appliqués

### Backend

| Pattern | Utilisation | Fichier |
|---------|-------------|---------|
| **Service Layer** | Business logic isolée | paa_service.py |
| **Repository** | Abstraction data access | paa_persistence.py |
| **Dependency Injection** | FastAPI Depends | paa_router.py |
| **DTO (Data Transfer Object)** | Pydantic models | paa_service.py |
| **Domain-Driven Design** | Bounded context PAA | measurement/paa/ |

### Frontend

| Pattern | Utilisation | Fichier |
|---------|-------------|---------|
| **Component-Based** | Modularité UI | paa-dashboard.component.ts |
| **Master-Detail** | Liste + détails | HTML template |
| **Reactive Programming** | RxJS Observables | Service + Component |
| **State Management** | BehaviorSubject | Component state |
| **Dependency Injection** | Angular services | Constructor injection |

---

## 🔐 Sécurité

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTHENTICATION FLOW                      │
└─────────────────────────────────────────────────────────────┘

User Login
    │
    │ POST /auth/signin
    ▼
┌──────────────┐
│ Auth Router  │ → Verify credentials
└──────────────┘    │
                    │ Generate JWT
                    ▼
              ┌──────────┐
              │   Token  │
              └──────────┘
                    │
                    │ Stored in LocalStorage
                    ▼
┌────────────────────────────────────────┐
│       HTTP Interceptor (Angular)       │
│  → Add Authorization: Bearer <token>   │
└────────────────────────────────────────┘
                    │
                    │ Every API call
                    ▼
┌────────────────────────────────────────┐
│       AuthGuard (Angular)              │
│  → Check token validity                │
│  → Redirect to /auth if invalid        │
└────────────────────────────────────────┘
                    │
                    │ If valid
                    ▼
┌────────────────────────────────────────┐
│       PAA Router (Backend)             │
│  → Process request                     │
│  → Return data                         │
└────────────────────────────────────────┘
```

**Features Sécurité**:
- ✅ JWT tokens (expiration configurable)
- ✅ AuthGuard sur toutes routes protégées
- ✅ CORS configuré (localhost:4200 only)
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ Input validation (Pydantic)

---

## 📊 Performance & Scalabilité

### Optimisations Actuelles

```
┌────────────────────────────────────────────────────────────┐
│                    PERFORMANCE LAYERS                      │
├────────────────────────────────────────────────────────────┤
│ Frontend (Angular)                                         │
│  → OnPush Change Detection                                 │
│  → RxJS takeUntil (unsubscribe auto)                       │
│  → Lazy loading components                                 │
│  → Virtual scrolling (si >1000 lignes)                     │
├────────────────────────────────────────────────────────────┤
│ Backend (FastAPI)                                          │
│  → Async endpoints (non-blocking)                          │
│  → Database connection pooling                             │
│  → Pydantic validation (compiled)                          │
│  → SQLAlchemy lazy loading                                 │
├────────────────────────────────────────────────────────────┤
│ Database (MySQL)                                           │
│  → Indexes: group_id, contract_id                          │
│  → Foreign keys optimisés                                  │
│  → Query optimization (EXPLAIN)                            │
└────────────────────────────────────────────────────────────┘
```

### Capacités Estimées

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Groupes simultanés | 10,000+ | Limité par RAM/DB |
| Contrats par groupe | 100,000+ | Bulk insert optimisé |
| Périodes par groupe | Illimité | Archivage conseillé > 5 ans |
| Requêtes/seconde | 500+ | FastAPI async |
| Latence moyenne | <100ms | Endpoints simples |
| Utilisateurs concurrents | 100+ | Scalable horizontalement |

---

## 🚀 Évolutions Architecture (Roadmap)

### Phase 2: Enhanced Features

```
┌────────────────────────────────────────────────────────────┐
│  + Coverage Units Engine                                   │
│    → Paramétrable (contrats, sinistres, pondération)       │
│                                                            │
│  + DAC (Deferred Acquisition Costs)                        │
│    → Amortissement patterns                                │
│                                                            │
│  + Risk Adjustment Module                                  │
│    → Calcul confidence level                               │
│    → Intégration dans LIC                                  │
└────────────────────────────────────────────────────────────┘
```

### Phase 3: Enterprise Grade

```
┌────────────────────────────────────────────────────────────┐
│  + Subledger IFRS 17                                       │
│    ┌──────────────┐        ┌──────────────┐              │
│    │ PAA Movements│ ─────→ │ GL Accounts  │              │
│    └──────────────┘        └──────────────┘              │
│         │                                                  │
│         └─→ Journal Entries (auto-generated)              │
│                                                            │
│  + Batch Orchestrator                                      │
│    → Process N groups in parallel                          │
│    → Cron jobs / scheduled tasks                           │
│                                                            │
│  + Export Engine                                           │
│    → Excel (IFRS 17 format)                                │
│    → PDF (reports)                                         │
│    → JSON/XML (API externe)                                │
└────────────────────────────────────────────────────────────┘
```

### Phase 4: Cloud Native

```
┌────────────────────────────────────────────────────────────┐
│                    CLOUD ARCHITECTURE                      │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────┐      ┌─────────────┐                     │
│  │   Load      │      │   API       │                     │
│  │   Balancer  │ ───→ │   Gateway   │                     │
│  └─────────────┘      └─────────────┘                     │
│                              │                             │
│         ┌────────────────────┼────────────────────┐       │
│         │                    │                    │       │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐   │
│    │ FastAPI │         │ FastAPI │         │ FastAPI │   │
│    │ Pod 1   │         │ Pod 2   │         │ Pod N   │   │
│    └─────────┘         └─────────┘         └─────────┘   │
│         │                    │                    │       │
│         └────────────────────┴────────────────────┘       │
│                              │                             │
│                    ┌──────────────────┐                    │
│                    │ MySQL Cluster    │                    │
│                    │ (Azure/AWS RDS)  │                    │
│                    └──────────────────┘                    │
│                                                            │
│  Monitoring: Azure Monitor / CloudWatch                    │
│  CI/CD: GitHub Actions / Azure DevOps                      │
│  Containers: Docker + Kubernetes                           │
└────────────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0  
**Date**: 6 Octobre 2025  
**Équipe**: IFRS17 Development Team BNA
