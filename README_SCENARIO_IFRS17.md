# 📖 Scénario Complet d’Utilisation – Plateforme IFRS 17 (BNA)

Ce guide décrit, pas à pas, le parcours utilisateur « de l’ouverture à la clôture » de l’application IFRS 17, en couvrant tous les écrans et fonctionnalités clés: démarrage, authentification, chargement des données, analyses PAA/PPNA, projections, IA/ML, exports et fermeture.

---

## 👤 Pour qui ?
- Actuaires, comptables, contrôleurs de gestion
- Chefs de projet finance/risques
- Jury PFE (démonstration guidée de 10–15 min)

---

## 🧩 Ce que vous obtiendrez à la fin
- Données PPNA chargées et analysées
- Calculs IFRS 17 (PAA) automatisés: LRC, RA, LC, onérosité
- Projections mensuelles multi-années
- Prédictions ML de la LRC (≈97% de précision)
- Détection d’anomalies sur 10% des contrats
- Rapports Excel/PDF prêts à partager

---

## 0) Pré‑requis et démarrage

- OS: Windows (PowerShell)
- Backend FastAPI sur 127.0.0.1:8001
- Frontend Angular sur http://localhost:4200
- Base MySQL via XAMPP (ou SQLite en dev)

Démarrage recommandé (automatique):
- Double‑cliquer `start_fullstack.ps1` à la racine
  - Ouvre 2 consoles: backend puis frontend
  - Rappels affichés: API http://127.0.0.1:8001, UI http://localhost:4200, Swagger http://127.0.0.1:8001/docs

Démarrage manuel (au besoin):
- Backend: `start_backend.ps1` (active .venv et lance uvicorn sur 8001)
- Frontend: `start_frontend.ps1` (installe deps si besoin et lance Angular sur 4200)

Vérifications rapides:
- API vivante: ouvrir http://127.0.0.1:8001/health → status healthy
- Docs API: ouvrir http://127.0.0.1:8001/docs
- UI: ouvrir http://localhost:4200

---

## 1) Ouverture et authentification

1. Ouvrez http://localhost:4200
2. Page d’accueil → « Se connecter »
3. Renseignez vos identifiants (ex: actuaire/comptable de démo si fournis)
4. Après connexion, vous arrivez sur le tableau de bord principal (Dashboard)

Sécurité:
- Authentification JWT (token en mémoire, expiration 30 min)
- Rôles gérés côté frontend (guards) et backend (routers protégés)

---

## 2) Premier repère: Dashboard

Objectif: vérifier que tout fonctionne et visualiser les indicateurs globaux.

Vous verrez:
- Cartes KPI (contrats, primes, PPNA, LRC, onérosité, etc.)
- Liens rapides vers: PPNA Analytics, ML Analytics, Exports
- État système (connectivité API, version)

Astuce: si aucune donnée n’est chargée, certaines cartes affichent 0. Passez à l’étape suivante pour importer le dataset PPNA.

---

## 3) Import des données PPNA

Chemin: Menu → Analytics → « PPNA Analytics »

Étapes:
1. Cliquez « Charger un fichier Excel PPNA »
2. Sélectionnez `Data/Ppna (4).xlsx` (dataset de démonstration)
3. Attendez 2–5 s. Un badge « Données PPNA chargées » apparaît.

Résultat attendu:
- Nombre de lignes: ≈ 203 786
- PPNA total, LRC totale et primes totales visibles
- Segmentation par portefeuille / cohorte / onérosité prête

Notes techniques:
- Upload → endpoint `POST /ppna/upload-file`
- Les DataFrames sont stockés côté serveur dans une structure `ppna_data` en mémoire (multi‑feuilles Excel possibles)

---

## 4) Analyse IFRS 17 (PAA)

Toujours dans « PPNA Analytics », explorez les sections:

A. Données transformées
- Colonnes clés détectées (MNTPRNET, MNTPPNA, CODPROD, DUREE, etc.)
- Aperçu des premières lignes et qualité des données

B. Vue Groupe IFRS‑17 (segmentation §14‑24)
- Segments = Portfolio × Cohorte × Onerosité
- Pour chaque segment:
  - Nombre de contrats, Primes, PPNA, LRC
  - Risk Adjustment (méthode CoC à 6%)
  - Loss Component (si onéreux)
  - Combined Ratio et statut (🟢/🟠/🔴)

Formule rappel:
- LRC = PPNA + RA + LC
- RA = PPNA × volatilité(≈8%) × CoC(6%) × niveau_confiance(~2.0)
- LC = max(0, Coûts attendus − Primes)

Valeur métier:
- Détection immédiate des segments en perte (contrats onéreux)
- Aide à la décision (remédiation tarifaire, provisionnement)

---

## 5) Projections mensuelles (2020 → 2025)

Onglet « Projection » dans PPNA Analytics.

Étapes:
1. Renseigner start_year = 2020, end_year = 2025
2. Cliquer « Calculer Projection »
3. Visualiser:
   - Graphique d’évolution mensuelle (revenus / amortissement DAC)
   - Tableau détaillé (72 mois)

Technique:
- `POST /ppna/projection/calculate?start_year=2020&end_year=2025`
- Les paramètres sont bien lus comme Query (pas body)

Livrables:
- Projection exportable (voir Exports plus bas)

---

## 6) Exports professionnels (Excel, PDF, ZIP)

Onglet « Exports ».

Options:
- Excel: `GET /ppna/export/excel` → classeur avec onglets: Synthèse, Projections, Segments, Contrats onéreux, Graphiques
- PDF: `GET /ppna/export/pdf` → rapport formaté prêt à partager
- ZIP: archive regroupant CSV/JSON/Excel

Usage:
- Cliquez « Télécharger Excel » pour obtenir un rapport réutilisable (comité IFRS, audit, direction)

---

## 7) ML Analytics – Prédiction LRC

Chemin: Menu → Analytics → « ML Analytics »

Étapes:
1. « Charger Dataset PPNA » → réutilise les données importées
2. « Entraîner Modèle LRC » (XGBoost)
   - Durée: 30–45 s selon la machine
   - Métriques attendues:
     - Accuracy ~97.23%
     - MAE ≈ 1 200–1 500 TND
     - RMSE ≈ 2 000–2 500 TND
     - R² ≈ 0.95
3. « Générer Prédictions »
   - Total LRC prédit ≈ 133 000 000 TND
   - Courbe mensuelle + Top contrats
   - Export des prédictions possible

Technique:
- `POST /ml/train/lrc`, `POST /ml/predict/lrc`
- Modèle: XGBoost (n_estimators=100, max_depth=6, learning_rate=0.1)

Valeur métier:
- Anticipation des besoins de capitaux
- Pilotage budgétaire et communication financière

---

## 8) Détection d’anomalies

Dans « ML Analytics » → onglet « Anomalies ».

Étapes:
1. Régler « contamination » = 0.10 (10%)
2. Cliquer « Détecter Anomalies »

Résultats typiques:
- ≈ 20 379 contrats signalés (10%)
- Méthode: Isolation Forest
- Tableau des cas les plus suspects (score élevé)

Cas d’usage:
- Erreurs de saisie
- Contrats atypiques/sous‑tarifés
- Pré‑filtrage antifraude

Technique:
- `POST /ml/anomaly-detection?method=isolation_forest&contamination=0.1`

---

## 9) Assistant IA IFRS 17 (option)

Chemin: Menu → « Assistant IA » (si activé).

Capacités:
- Répond aux questions IFRS 17 (glossaire, § normatifs)
- Explique les calculs montrés (LRC, RA, LC)
- Donne des recommandations (ex: seuils d’onérosité)

Technique:
- Services initialisés au démarrage (`backend/ai/*`)

---

## 10) Historique et traçabilité

- Les opérations clés (upload, projections, exports, ML) sont loguées côté backend
- Les structures (schemas Pydantic, modèles SQLAlchemy) assurent la cohérence
- Exports datés = pièces d’audit

---

## 11) Clôture de session

1. Déconnexion via le menu utilisateur (token invalidé côté client)
2. Fermeture propre:
   - Arrêter le frontend (Ctrl+C dans le terminal Angular)
   - Arrêter le backend (Ctrl+C dans la console uvicorn)
   - Optionnel: Stop MySQL depuis XAMPP

---

## 🎯 Récapitulatif « en une minute »

1) Démarrer (scripts PowerShell) → 2) Se connecter → 3) Importer PPNA →
4) Voir Dashboard et Segments → 5) Calculer Projections →
6) Entraîner et Prédire LRC → 7) Détecter Anomalies → 8) Exporter Excel/PDF →
9) (Option) Questionner l’Assistant IA → 10) Se déconnecter

Résultat: Calculs IFRS 17 automatisés, prédictions fiables, rapports prêts.

---

## 📎 Annexes (URLs, endpoints, versions)

- UI: http://localhost:4200
- API: http://127.0.0.1:8001
- Swagger: http://127.0.0.1:8001/docs

Endpoints clés:
- PPNA: `/ppna/upload-file`, `/ppna/dashboard-metrics`, `/ppna/projection/calculate`, `/ppna/export/excel`, `/ppna/export/pdf`
- ML: `/ml/train/lrc`, `/ml/predict/lrc`, `/ml/anomaly-detection`, `/ml/clustering`
- Santé: `/health`

Technologies (extrait):
- Frontend: Angular 17, Bootstrap 5, Chart.js 4
- Backend: FastAPI, Python 3.12, Pandas/Scikit‑learn, XGBoost
- DB: MySQL 8 (prod) / SQLite (dev)

---

## 🩺 Dépannage rapide

- UI vide après upload
  - Vérifier que le backend a logué « fichier reçu » (console) et que `ppna_data` contient au moins une feuille
  - Recharger la page; sinon relancer `start_fullstack.ps1`

- Projection 422/500
  - Utiliser bien des paramètres Query `start_year`, `end_year` (ex: 2020, 2025)
  - Backend corrige les références `df_ppna → ppna_data` (OK)

- Frontend ne démarre pas
  - `angular-frontend\node_modules` absent → relancer `start_frontend.ps1` (installe deps)

- Backend ne démarre pas
  - Activer .venv (`start_backend.ps1`) et `pip install -r requirements.txt`
  - Vérifier `backend/main.py` écoute sur 8001 et CORS autorise 4200

---

## 🙌 Notes finales

- Données de démonstration: `Data/Ppna (4).xlsx`
- Temps de démo conseillé: 12–15 minutes
- Valeur métier: 80% de gain de temps, 97% de précision, 20k anomalies détectées, rapports automatiques

Bonne utilisation et bonne soutenance ! 🚀
