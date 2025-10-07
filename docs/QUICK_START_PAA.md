# 🚀 DÉMARRAGE RAPIDE - MODULE PAA IFRS 17

## ⚡ Quick Start (3 commandes)

### Terminal 1 - Backend API

```powershell
# Activer environnement virtuel (si applicable)
.\.venv\Scripts\Activate.ps1

# Lancer serveur FastAPI
cd backend
python main.py
```

✅ **API disponible**: http://127.0.0.1:8001  
✅ **Documentation Swagger**: http://127.0.0.1:8001/docs

---

### Terminal 2 - Frontend Angular

```powershell
cd angular-frontend
npm start
```

✅ **UI disponible**: http://localhost:4200  
✅ **PAA Dashboard**: http://localhost:4200/paa-dashboard

---

### Terminal 3 - Tests (optionnel)

```powershell
# Tests backend
pytest backend/tests/test_paa.py -v

# Test imports
python -c "from backend.measurement.paa import PAAService; print('✅ OK')"
```

---

## 📋 Workflow Utilisateur Complet

### 1. Connexion
```
URL: http://localhost:4200/auth/signin
Credentials: Utiliser compte existant ou créer nouveau
```

### 2. Navigation PAA
```
Menu → "PAA Dashboard"
OU
URL directe: http://localhost:4200/paa-dashboard
```

### 3. Premier Groupe

**Cliquer "Nouveau Groupe"** puis copier/coller ce JSON:

```json
[
  {
    "contract_id": "AUTO_2025_001",
    "portfolio": "AUTO",
    "inception": "2025-01-01",
    "expiry": "2025-12-31",
    "written_premium": 15000,
    "expected_claim_ratio": 0.55,
    "expected_expense_ratio": 0.12,
    "acquisition_cashflows": 500,
    "already_incurred_claims": 0,
    "claims_paid_to_date": 0
  },
  {
    "contract_id": "AUTO_2025_002",
    "portfolio": "AUTO",
    "inception": "2025-01-01",
    "expiry": "2025-06-30",
    "written_premium": 6000,
    "expected_claim_ratio": 0.6,
    "expected_expense_ratio": 0.1,
    "acquisition_cashflows": 200
  },
  {
    "contract_id": "AUTO_2025_003",
    "portfolio": "AUTO",
    "inception": "2025-02-01",
    "expiry": "2026-01-31",
    "written_premium": 18000,
    "expected_claim_ratio": 0.5,
    "expected_expense_ratio": 0.15,
    "acquisition_cashflows": 600
  }
]
```

**Group ID**: `AUTO_2025_Q1`  
**Portfolio**: AUTO

### 4. Traiter Première Période

- **Sélectionner** le groupe `AUTO_2025_Q1` dans la liste
- **Cliquer** "Traiter Période"
- **Remplir**:
  - Début: `2025-01-01`
  - Fin: `2025-01-31`
  - Sinistres Encourus: `2500`
  - Sinistres Payés: `2000`
- **Valider**

### 5. Voir Résultats

✅ **Cards Résumé** mis à jour automatiquement  
✅ **Tableau Mouvements** affiche la période  
✅ **Métriques Groupe** actualisées (LRC, LIC)

### 6. Stress Test (optionnel)

- **Cliquer** "Stress Test"
- **Saisir**:
  - Choc Ratio Sinistres: `0.15` (+15%)
  - Choc Ratio Frais: `0.05` (+5%)
- **Lancer Simulation**
- **Voir** impact sur marge et statut onéreux

---

## 🧪 Tests API avec cURL

### Test 1: Health Check
```bash
curl http://127.0.0.1:8001/health
```

**Attendu**: JSON avec status "healthy"

### Test 2: Lister Groupes (vide initialement)
```bash
curl http://127.0.0.1:8001/paa/groups
```

**Attendu**: `{"status": "success", "groups": [], "total": 0}`

### Test 3: Initialiser Groupe
```bash
curl -X POST "http://127.0.0.1:8001/paa/groups/init?group_id=TEST_G1&persist=true" \
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

**Attendu**: JSON avec `"status": "success"`, `lrc_initial: 1200`

### Test 4: Traiter Période
```bash
curl -X POST "http://127.0.0.1:8001/paa/groups/TEST_G1/period?period_start=2025-01-01&period_end=2025-01-31&incurred_claims=150&claims_paid=120&persist=true"
```

**Attendu**: JSON avec `earned_premium > 0`

### Test 5: Consulter Mouvements
```bash
curl http://127.0.0.1:8001/paa/groups/TEST_G1/movements
```

**Attendu**: Liste avec 1 mouvement

### Test 6: Portfolio Summary
```bash
curl http://127.0.0.1:8001/paa/analytics/portfolio-summary
```

**Attendu**: JSON avec `total_groups: 1`

---

## 🐛 Troubleshooting Rapide

### Backend ne démarre pas

**Erreur**: `ModuleNotFoundError: No module named 'fastapi'`  
**Solution**:
```bash
pip install -r requirements.txt
```

**Erreur**: `Can't connect to MySQL server`  
**Solution**:
1. Vérifier MySQL lancé (XAMPP ou service)
2. Vérifier credentials dans `backend/database/connection.py`
3. Créer base `ifrs17` si nécessaire

### Frontend erreur compilation

**Erreur**: `Cannot find module '@angular/...`  
**Solution**:
```bash
cd angular-frontend
npm install
```

**Erreur**: `Port 4200 is already in use`  
**Solution**:
```bash
# Tuer processus ou utiliser autre port
ng serve --port 4201
```

### API CORS Error

**Symptôme**: Requêtes bloquées depuis Angular  
**Solution**: Vérifier `main.py` inclut `http://localhost:4200` dans CORS (déjà fait)

### Aucun groupe affiché

**Cause**: Base de données vide  
**Solution**: Créer premier groupe via UI ou API (voir Tests API)

### Mouvements vides

**Cause**: Aucune période traitée  
**Solution**: Utiliser "Traiter Période" après initialisation groupe

---

## 📚 Documentation Complète

- **README Module**: `PAA_MODULE_README.md`
- **Rapport Transformation**: `TRANSFORMATION_PAA_COMPLETE.md`
- **Checklist**: `CHECKLIST_PAA_FINAL.md`
- **API Swagger**: http://127.0.0.1:8001/docs (après démarrage backend)

---

## 🎯 Prochaines Étapes Recommandées

1. ✅ **Démarrer backend + frontend** (commandes ci-dessus)
2. ✅ **Créer premier groupe test** (JSON fourni)
3. ✅ **Traiter 3-4 périodes** pour voir évolution
4. ✅ **Tester stress test** (vérifier détection onéreux)
5. ✅ **Explorer Swagger UI** (tester tous endpoints)
6. 📅 **Formation utilisateurs** (prévoir session)
7. 📅 **Tests UAT** (User Acceptance Testing)
8. 📅 **Déploiement test BNA**

---

## 🏁 Success Criteria

Après avoir suivi ce guide, vous devriez avoir:

- [x] Backend API fonctionnel (http://127.0.0.1:8001)
- [x] Frontend UI accessible (http://localhost:4200)
- [x] Au moins 1 groupe PAA créé
- [x] Au moins 1 période traitée
- [x] Mouvements IFRS 17 visibles
- [x] Design BNA affiché correctement

**Si tous cochés** → ✅ **Module PAA opérationnel !**

---

**Bon démarrage !** 🚀

*Support: support-ifrs17@bna.com.tn*
