# 🤖 Machine Learning — IFRS 17 (BNA)

Document de référence de la partie ML: objectifs, jeux de données, features, prétraitements, modèles entraînés, formules, métriques, interprétations, endpoints API et limites.

---

## 🎯 Objectifs ML

- Prédire la LRC (Liability for Remaining Coverage) à partir des données PPNA/contrats
- Détecter automatiquement les anomalies dans les contrats (fraudes/erreurs/atypiques)
- Segmenter les contrats (clustering) pour mieux comprendre les profils et marges
- Estimer la rentabilité (profitabilité) et classer le niveau de risque
- Identifier/anticiper les contrats onéreux (support à la composante de perte IFRS 17)

---

## 🧮 Données, cibles et features

- Fichier d’entrée: Excel/CSV (ex.: `Data/Ppna (4).xlsx`)
- Colonnes techniques récurrentes: `MNTPRNET` (prime nette), `MNTPPNA` (PPNA), `CODPROD` (produit), `DEBEFFQUI`/`FINEFFQUI` (dates), `DUREE`, etc.
- Cibles synthétiques créées si nécessaire:
  - claims_ratio ≈ MNTPPNA / MNTPRNET pour prédiction sinistres (proxy)
  - profitability ≈ MNTPRNET − MNTPPNA − 15% coûts estimés
  - lrc_estimate générée par un modèle IFRS17 (voir section LRC)
  - risk_level construit via règles internes (RiskClassificationModel)

Prétraitements (OptimizedDataPreprocessor):
- Nettoyage, coercition numeric, encodage catégoriel, normalisation
- Gestion robuste des NaN/Inf (remplacement par valeurs neutres)
- Pipelines cohérents entre entraînement et prédiction

---

## 🧠 Modèles utilisés

- Régression
  - XGBoost Regressor (par défaut pour LRC, sinistres, rentabilité)
  - RandomForest Regressor (alternative, plus interprétable)
- Classification
  - RandomForestClassifier (risk classification)
  - XGBoostClassifier (option selon volume/complexité)
- Anomalies
  - Isolation Forest (contamination 10% par défaut)
- Clustering
  - K-Means (par défaut) et support DBSCAN (via AI service avancé)
- Contrats onéreux (binaire)
  - Modèle spécialisé OnerousContractsModel (XGBoost + features dérivées)

Optimisations runtime:
- Caches TTL pour données et modèles
- ThreadPoolExecutor, lazy imports, chargement chunké (CSV)
- Endpoints asynchrones/avec tâches en arrière-plan selon les cas

---

## 🧾 Formules clés et cibles

- IFRS 17 (rappel calculs):
  - LRC = PPNA + RA + LC
  - RA (Risk Adjustment) ≈ PPNA × volatilité(≈8%) × CoC(6%) × niveau_confiance(≈2.0)
  - LC (Loss Component) = max(0, coûts estimés − primes)

- Cibles synthétiques (quand non présentes):
  - claims_ratio = MNTPPNA / (MNTPRNET + 1e−8)
  - profitability = MNTPRNET − MNTPPNA − 0.15×MNTPRNET
  - lrc_estimate = fonction de IFRS17 (LRCPredictionModel.create_lrc_target)

Ces cibles servent à entraîner des modèles robustes même quand certains labels métiers ne sont pas explicitement disponibles.

---

## 🏗️ Pipelines d’entraînement

- Prédiction LRC
  1) lrc_estimate ← LRCPredictionModel.create_lrc_target(df)
  2) (X, y) = preprocess(df, target='lrc_estimate')
  3) XGBoost Regressor → métriques (R², MAE, RMSE, Accuracy proxy)

- Prédiction sinistres (claims)
  1) claims_ratio comme cible synthétique si absente
  2) (X, y) preprocessing
  3) XGBoost/RandomForest → métriques régression

- Profitabilité
  1) profitability = prime − PPNA − 15% prime
  2) Entraînement régression → métriques

- Classification du risque
  1) risk_level via RiskClassificationModel.create_risk_labels
  2) Encodage labels + preprocessing
  3) RandomForest (par défaut) → accuracy, matrix/confusion report

- Contrats onéreux (binaire)
  1) Features enrichies via OnerousContractsModel.prepare_features
  2) Cible via create_onerous_target
  3) Cross-validation (cv=5) accuracy + insights (feature importance, patterns)

- Anomalies
  1) Preprocess → X
  2) IsolationForest(contamination=0.10)
  3) labels (0 = anormal) + scores normalisés; top anomalies listées

- Clustering
  1) Preprocess → X
  2) KMeans (n_clusters=K) → labels + caractéristiques par cluster + distribution

---

## 📊 Métriques & résultats (observés en démo)

- LRC (XGBoost)
  - Accuracy proxy ≈ 97.23%
  - R² ≈ 0.95 ; MAE ≈ 1.2–1.5k TND ; RMSE ≈ 2.0–2.5k TND
  - Total LRC prédit ≈ 133,000,000 TND sur ~203,786 contrats

- Anomalies (Isolation Forest)
  - Taux contamination: 10%
  - ≈ 20,379 contrats détectés anormaux (sur ~203,786)

- Onerous contracts (binaire)
  - CV Accuracy (k‑fold=5): typiquement > 0.85 (dépend dataset)
  - Insights: variables d’importance, patterns par produit/cohorte, part onéreuse

- Clustering (KMeans)
  - Distribution équilibrée selon K (typ. 4–8), caractéristiques par segment

Note: Les chiffres exacts varient selon le dataset PPNA chargé.

---

## 🔎 Interprétation métier

- LRC élevée et stable → besoin de capital plus important, bonne visibilité
- Anomalies
  - Scores proches de 1 (normalisés) = très atypiques → auditer ces contrats
  - Causes possibles: saisie erronée, sous‑tarification, fraude, produits hors norme
- Clusters
  - Segments « haute prime/faible PPNA » = profitables
  - Segments « prime faible/PPNA élevée » = risque d’onérosité → vigilance
- Onerous
  - Taux onéreux élevé sur un produit → revoir tarification ou conditions
  - Liste priorisée par probabilité → actions correctives ciblées

---

## ⚠️ Limites & hypothèses

- Cibles synthétiques (claims_ratio, profitability, lrc_estimate) = approximations guidées IFRS 17
- Qualité des données (manquants, outliers, types) impacte fortement la précision
- Les hyperparamètres par défaut (XGBoost, RF) sont raisonnables; un tuning pourrait améliorer
- L’interprétation business doit être validée par l’actuaire (règles locales, CoC, volatilité)

---

## 🧪 Endpoints API (principaux)

Base URL backend: http://127.0.0.1:8001/ml

- Upload dataset
  - POST `/upload-data` (file: .xlsx/.csv)
- Entraînements
  - POST `/train/lrc-prediction?model_type=xgboost`
  - POST `/train/claims-prediction?model_type=xgboost&target_column=...`
  - POST `/train/profitability?model_type=xgboost`
  - POST `/train/risk-classification?model_type=random_forest`
  - POST `/train/onerous-contracts?model_type=xgboost`
- Prédictions
  - GET `/predict/lrc?model_type=xgboost` (sur dataset courant)
  - POST `/predict/{model_name}` (upload fichier pour prédire avec un modèle entraîné)
  - POST `/predict/onerous-contracts?model_type=xgboost`
- Anomalies & Clustering
  - POST `/anomaly-detection?method=isolation_forest&contamination=0.1`
  - POST `/clustering?n_clusters=5&clustering_type=kmeans`
- Insights & État
  - GET `/models-summary` (ou `/models/summary`)
  - GET `/insights`
  - GET `/health`
  - GET `/data/summary`, `/data/paginated`
  - POST `/models/save` (sauvegarde joblib)

Réponses JSON: nettoyées de NaN/Inf, types numpy convertis.

---

## 🧩 Exemples d’utilisation (PowerShell)

- Upload
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/ml/upload-data -InFile .\Data\"Ppna (4).xlsx" -ContentType "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
```

- Entraîner LRC
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/ml/train/lrc-prediction?model_type=xgboost"
```

- Prédire LRC (dataset courant)
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/ml/predict/lrc?model_type=xgboost"
```

- Détecter anomalies (10%)
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/ml/anomaly-detection?method=isolation_forest&contamination=0.1"
```

---

## 🧭 Gouvernance & traçabilité

- Caches des résultats (TTL) pour reproductibilité à court terme
- Sauvegarde des modèles (`/ml/models/save`) avec horodatage
- Logs détaillés côté backend (entrainements, prédictions, stats)
- Possibilité d’exporter les prédictions (frontend + endpoint générique)

---

## 🔮 Pistes d’amélioration

- Hyperparameter tuning (Optuna/GridSearch) pour LRC/onéreux
- SHAP/Feature importance systématique et dashboard d’explicabilité
- Drift detection et réentraînement programmé
- Calibration des probabilités (Platt/Isotonic) pour onéreux
- Ajout LightGBM/CatBoost pour gros volumes hétérogènes

---

## ✅ Résumé exécutif

- Modèles déployés: XGBoost (LRC, sinistres, profitabilité), RF (risque), IsolationForest (anomalies), KMeans (clustering), XGB onéreux
- Performance constatée: LRC ~97% (R² ~0.95), anomalies ~10% (tunable), onéreux ACC >0.85 (cv)
- Gains: anticipation LRC ~133M TND, priorisation des contrôles (anomalies/onéreux), segmentation actionnable

L’IA renforce la conformité IFRS 17 et le pilotage financier en temps utile.
