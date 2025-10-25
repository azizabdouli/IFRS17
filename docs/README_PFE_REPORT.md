# PFE Report — IFRS17 Hub (BNA) • Premium Allocation Approach (PAA)

A professional, report‑ready overview of the project: vision, theory (IFRS 17 PAA), architecture, implementation, ML, validation, and results — structured for inclusion in a PFE document.

---

## Table of Contents

1. Executive Summary
2. Context and Motivation
3. Objectives and Scope
4. Methodology and Governance
5. System Architecture
6. Features and Functional Coverage
7. IFRS 17 Theoretical Foundations (PAA)
8. Data and Machine Learning
9. Implementation Details (Frontend, Backend, Database)
10. Quality Assurance and Validation
11. Security and Compliance
12. Demo Scenario and User Guide
13. Results and KPIs
14. Limitations, Risks, and Mitigations
15. Roadmap and Future Work
16. How to Run (Windows)
17. Repository Structure
18. References
19. Annexes (Figures, Screens, Endpoints)

---

## 1. Executive Summary

IFRS17 Hub is a full‑stack web application for IFRS 17 accounting with the Premium Allocation Approach (PAA). It automates PPNA, RA, LC, and LRC computations; offers dashboards, exports, and an AI/ML layer; and ensures auditability and performance suitable for production at BNA.

Key highlights:
- End‑to‑end IFRS17 PAA calculations aligned with the standard (IFRS 17.53–59)
- Secure Angular frontend + FastAPI backend + MySQL/SQLite
- Machine Learning for risk analytics and LRC prediction
- Assistant AI for data quality checks and recommendations
- Validated formulas and performance (R² LRC ≥ 0.93; profitability model R² ≈ 0.96)

---

## 2. Context and Motivation

- Before: Excel‑based processes, manual consolidation, long cycle time (3–4h/portfolio), limited traceability.
- After: Automated, reliable, traceable computations; APIs and real‑time UI; audit trail; ML for decision support.
- Business drivers: regulatory compliance (IFRS 17), faster closes, improved accuracy and comparability, better governance.

---

## 3. Objectives and Scope

- Primary objective: operationalize IFRS 17 PAA for non‑life contracts (≤ 12 months) with professional software quality.
- Secondary objectives: provide ML‑driven insights (LRC prediction, risk classification, anomaly detection); enable exports and reporting; include an assistant AI.
- Scope: PAA (PPNA, RA, LC, LRC, movements, stress tests), dashboards, APIs, ML models, authentication, and role‑based access.

Out of scope: GMM/VFA calculations; advanced discounting for >12 months; subledger postings (planned).

---

## 4. Methodology and Governance

- Iterative delivery with short sprints.
- CRISP‑DM for ML lifecycle (business → data → prep → modeling → evaluation → deployment).
- Code reviews, documentation, and test automation for quality.

---

## 5. System Architecture

- Frontend: Angular 17, TypeScript, RxJS, Bootstrap/SCSS, JWT guards.
- Backend: FastAPI (Python 3.12), SQLAlchemy ORM, Pydantic, CORS.
- Database: SQLite (dev) and MySQL 8 (prod) via SQLAlchemy.
- ML/AI: XGBoost, RandomForest, Isolation Forest, KMeans; caching and chunked loading.
- Launch: PowerShell scripts orchestrate backend (127.0.0.1:8001) then frontend (localhost:4200).

Key URLs:
- Frontend: http://localhost:4200
- Backend API: http://127.0.0.1:8001
- Swagger/OpenAPI: http://127.0.0.1:8001/docs

References: `README.md` (root), `docs/ARCHITECTURE_PAA.md`.

---

## 6. Features and Functional Coverage

- IFRS17 PAA Module: groups initialization, period processing, onerosity test, movements, stress testing, portfolio summary.
- PPNA Analytics: ingestion, validation, KPI cards, visualizations.
- ML Analytics: training and prediction endpoints; clustering; anomaly detection.
- Assistant AI: insights, data quality checks, recommendations.
- Exports: Excel/PDF (where applicable); downloadable analytics.
- Security: JWT auth, guards, input validation; environment‑based config.

References: `docs/PAA_MODULE_README.md`, `README_SCENARIO_IFRS17.md`.

---

## 7. IFRS 17 Theoretical Foundations (PAA)

Core equations (consistent with implementation):

- Liability for Remaining Coverage (LRC):
  $$ LRC = PPNA + RA + LC $$

- PPNA (prorata temporis, contract):
  $$ PPNA = PE \\times \\frac{n - t}{n} $$
  with $PE$ = written premium, $n$ = coverage days, $t$ = elapsed days.

- Risk Adjustment (Cost of Capital — simplified):
  $$ RA = PPNA \\times \\sigma \\times CoC \\times CL $$
  with typical parameters: $\\sigma=8\\%$, $CoC=6\\%$, $CL \\approx 2.0$ (~95%).

- Loss Component (onérosité):
  $$ LC = \\max\\big(0,\\; PPNA\\,(S/P + F/P) - PPNA - RA\\big) $$

- KPI (passif view):
  $$ CR_{IFRS17} \\approx \\frac{LRC}{Primes} \\times 100\\% $$

References: `README_IFRS17_THEORIE.md`, `docs/ASPECT_THEORIQUE_PROJET.md`, `docs/ACTUARIAL_VALIDATION_REPORT.md`.

---

## 8. Data and Machine Learning

- Data: PPNA dataset (Excel/CSV) with premiums, PPNA, duration, sinistres, frais, product codes, dates.
- Preprocessing: imputation, outlier capping, feature engineering (ratios, temporal features), encoding, scaling.
- Models: XGBoost (LRC, claims, profitability), RandomForest (risk classification), IsolationForest (anomalies), KMeans (clustering).
- Performance (observed): LRC R² ≈ 0.937; profitability R² ≈ 0.964; claims R² ≈ 0.732.
- Endpoints: upload/train/predict; clustering; anomalies; model summaries.

References: `README_ML.md`, `backend/routers/ml_router.py`.

---

## 9. Implementation Details (Frontend, Backend, Database)

- Frontend: Angular components for dashboard, PPNA, ML, AI; services for HTTP calls; guards for roles.
- Backend: routers for auth, ppna, ml, ai, transform, projection, paa; lifespan services; unified data structures; error‑handled query params.
- Database: SQLAlchemy models; automatic table creation; PAA persistence tables (groups, contracts, movements, snapshots).

References: `backend/main.py`, `backend/routers/*`, `backend/database/*`, `docs/TRANSFORMATION_PAA_COMPLETE.md`.

---

## 10. Quality Assurance and Validation

- Unit and integration tests (pytest) for PAA and formulas.
- Actuarial validation (RA via CoC; onerosity test; CR checks) and executive summaries.
- Performance benchmarks and caching strategies.
- Troubleshooting guidance for CORS, DB, and build environments.

References: `docs/EXECUTIVE_SUMMARY_ACTUARIAL.md`, `docs/DASHBOARD_PERFORMANCE_OPTIMIZATIONS.md`.

---

## 11. Security and Compliance

- JWT authentication, bcrypt hashing, role‑based navigation (Angular guards).
- Input validation (Pydantic), ORM to prevent SQL injection, CORS for dev.
- Secrets via environment files; minimum principle of exposure for APIs.

---

## 12. Demo Scenario and User Guide

- End‑to‑end flow: launch → login → upload data → analytics (PPNA/PAA) → ML insights → exports → closure.
- Key screens: dashboard KPIs, PAA movements, ML analytics, anomaly lists.
- Common issues and resolutions captured in docs.

Reference: `README_SCENARIO_IFRS17.md` and `docs/GUIDE_DEMO_RAPIDE.md`.

---

## 13. Results and KPIs

- Example portfolio (Auto): PPNA ≈ 60% of premiums at 40% elapsed; RA ≈ 0.5–3% of premiums; LC = 0 for non‑onerous portfolios.
- LRC totals and CR well below 100% on validated datasets (profitable).
- ML performance: LRC R² ≈ 0.937; profitability R² ≈ 0.964.

Reference: `docs/EXECUTIVE_SUMMARY_ACTUARIAL.md`.

---

## 14. Limitations, Risks, and Mitigations

- Simplifications: linear revenue recognition; RA parameters global by default; discounting not activated (≤12 months).
- UI gaps (to add): Combined Ratio gauge; onerosity alert cards; temporal evolution charts.
- Mitigations: parameter segmentation by product; roadmap to enable discounting and DAC; tests and monitoring.

---

## 15. Roadmap and Future Work

- Phase 2: coverage units, DAC amortization, optional RA, advanced patterns, export enhancements.
- Phase 3: subledger mapping, multi‑approach (PAA/VFA/GMM), BI connectors, multi‑scenario projections.
- Cloud readiness: containerization, CI/CD, APM monitoring.

---

## 16. How to Run (Windows)

Using the provided PowerShell scripts:

```powershell
# Full stack (recommended)
./start_fullstack.ps1

# Or separately
./start_backend.ps1
./start_frontend.ps1
```

If starting manually:

```powershell
# Backend (from repo root)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload

# Frontend
cd angular-frontend
npm install
npm start
```

Then open: http://localhost:4200 and API docs at http://127.0.0.1:8001/docs.

---

## 17. Repository Structure (excerpt)

```
angular-frontend/     # Angular UI (components, services, guards)
backend/              # FastAPI (routers, services, database, ml, ai)
Data/                 # Sample datasets
docs/                 # Documentation (PAA, theory, transformations, guides)
README.md             # Main project README
```

---

## 18. References

- IFRS 17 Standard (IASB): https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/
- Project theory and implementation:
  - `README_IFRS17_THEORIE.md` (root) — concise theory summary
  - `docs/ASPECT_THEORIQUE_PROJET.md` — full theoretical dossier
  - `docs/PAA_MODULE_README.md` — PAA user/API guide
  - `README_ML.md` — data/ML documentation
  - `README_SCENARIO_IFRS17.md` — end‑to‑end scenario

---

## 19. Annexes (Figures, Screens, Endpoints)

- Figures (suggested for report): architecture diagram, PAA movements table, CR gauge mockup, LRC waterfall.
- Screens: dashboard KPIs, PAA movements, ML analytics, anomalies.
- Key endpoints: `/paa/groups/init`, `/paa/groups/{id}/period`, `/ml/train-lrc-prediction`, `/ml/predict/lrc`, `/ml/anomaly-detection`, `/ml/clustering`.

(End of document)
