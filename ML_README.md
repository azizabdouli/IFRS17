# 🤖 Module Machine Learning IFRS17

Ce module ajoute des capacités avancées de machine learning au projet IFRS17 PAA pour l'analyse et la prédiction dans le domaine de l'assurance.

## 🎯 Fonctionnalités

### 📊 Modèles Prédictifs

1. **Prédiction des Sinistres** (`ClaimsPredictionModel`)
   - Prédit le ratio sinistres/primes
   - Basé sur les caractéristiques du contrat
   - Algorithmes: XGBoost, LightGBM, Random Forest

2. **Prédiction de Rentabilité** (`ProfitabilityModel`)
   - Estime la rentabilité future des contrats
   - Calcul: Prime - PPNA - Coûts estimés
   - Optimisé pour les données IFRS17

3. **Classification des Risques** (`RiskClassificationModel`)
   - Classe les contrats en 3 niveaux: Faible/Moyen/Élevé
   - Basé sur des métriques composites
   - Aide à la tarification et au provisionnement

4. **Prédiction LRC** (`LRCPredictionModel`)
   - Liability for Remaining Coverage (IFRS17)
   - Estimation des passifs de couverture restants
   - Modèles ensemble pour plus de précision

### 🔍 Analyse Exploratoire

1. **Clustering de Contrats** (`ContractClusteringModel`)
   - Segmentation automatique du portefeuille
   - K-Means et DBSCAN
   - Caractérisation des segments

2. **Détection d'Anomalies** (`AnomalyDetectionModel`)
   - Isolation Forest, Local Outlier Factor, One-Class SVM
   - Détection de fraude et contrôle qualité
   - Identification des contrats atypiques

### 🛠️ Preprocessing

- **Nettoyage automatique des données**
- **Création de features temporelles et financières**
- **Encodage des variables catégorielles**
- **Gestion des valeurs manquantes**
- **Normalisation et scaling**

## 🚀 Installation

1. **Installer les dépendances ML:**
```bash
pip install scikit-learn seaborn matplotlib joblib xgboost lightgbm tensorflow keras
```

2. **Structure créée:**
```
backend/ml/
├── __init__.py
├── data_preprocessing.py    # Preprocessing des données
├── ml_service.py           # Service principal ML
└── models/
    ├── __init__.py
    ├── base_model.py       # Classe de base
    └── insurance_models.py # Modèles spécialisés
```

## 💻 Utilisation

### Via l'API FastAPI

1. **Démarrer le backend:**
```bash
uvicorn backend.main:app --reload
```

2. **Endpoints disponibles:**
- `POST /ml/upload-data` - Upload des données
- `POST /ml/train/claims-prediction` - Entraîner modèle sinistres
- `POST /ml/train/profitability` - Entraîner modèle rentabilité
- `POST /ml/train/risk-classification` - Entraîner classification risques
- `POST /ml/clustering` - Clustering des contrats
- `POST /ml/anomaly-detection` - Détection d'anomalies
- `GET /ml/models/summary` - Résumé des modèles
- `GET /ml/insights` - Insights sur les données

### Via l'Interface Streamlit

1. **Lancer l'interface ML:**
```bash
streamlit run frontend/ml_interface.py
```

2. **Sections disponibles:**
- 🏠 Accueil - Vue d'ensemble
- 📊 Upload & Insights - Chargement et analyse
- 🎯 Modèles Prédictifs - Entraînement des modèles
- 🔍 Clustering - Segmentation du portefeuille
- ⚠️ Détection d'Anomalies - Identification des outliers
- 📈 Résultats - Performance et résultats

### Via le Code Python

```python
from backend.ml.ml_service import MLService

# Initialisation
ml_service = MLService()

# Chargement des données
df = ml_service.load_and_preprocess_data("Data/Ppna (4).xlsx")

# Entraînement des modèles
claims_results = ml_service.train_claims_prediction_model(df)
profit_results = ml_service.train_profitability_model(df)

# Clustering
clustering_results = ml_service.perform_contract_clustering(df, n_clusters=5)

# Détection d'anomalies
anomaly_results = ml_service.detect_anomalies(df)

# Sauvegarde
ml_service.save_all_models("models")
```

## 🧪 Démonstration

Exécuter le script de démonstration:

```bash
python ml_demo.py
```

Deux modes disponibles:
1. **Complet** - Avec le fichier Excel réel
2. **Simple** - Avec des données synthétiques

## 📊 Métriques et Évaluation

### Modèles de Régression
- **R²** - Coefficient de détermination
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MSE** - Mean Square Error

### Modèles de Classification
- **Accuracy** - Précision globale
- **Precision** - Précision par classe
- **Recall** - Rappel par classe
- **F1-Score** - Score F1 pondéré

### Clustering
- **Distribution des clusters**
- **Caractéristiques moyennes par cluster**
- **Taille des segments**

### Détection d'Anomalies
- **Taux d'anomalies détectées**
- **Scores d'anomalie**
- **Contrats flaggés comme anormaux**

## 🎨 Visualisations

Le module génère automatiquement:
- Graphiques de distribution des clusters
- Métriques de performance des modèles
- Tableaux de caractéristiques
- Insights business automatisés

## 🔒 Bonnes Pratiques

1. **Validation croisée** - Tous les modèles utilisent une validation k-fold
2. **Preprocessing robuste** - Gestion automatique des données manquantes
3. **Sauvegarde automatique** - Modèles persistés avec joblib
4. **Logging complet** - Traçabilité de tous les processus
5. **API RESTful** - Interface standardisée
6. **Documentation** - Code documenté et typé

## 🚀 Extensions Possibles

1. **Modèles Deep Learning** - Réseaux de neurones pour patterns complexes
2. **Time Series** - Prédiction temporelle des tendances
3. **AutoML** - Optimisation automatique des hyperparamètres
4. **Explainability** - SHAP values et LIME pour l'interprétabilité
5. **Real-time Scoring** - API de scoring en temps réel
6. **A/B Testing** - Framework de test des modèles

## 🎯 Applications Métier

### Tarification
- Segmentation risk-based
- Prédiction des sinistres futurs
- Optimisation des primes

### Provisionnement IFRS17
- Estimation automatique du LRC
- Détection des contrats onéreux
- Projection des cash-flows

### Gestion des Risques
- Classification automatique des risques
- Détection précoce des anomalies
- Monitoring du portefeuille

### Business Intelligence
- Segmentation client avancée
- Insights automatisés
- Tableaux de bord prédictifs

## 📞 Support

Pour toute question ou amélioration, le code est modulaire et extensible. Chaque composant peut être utilisé indépendamment ou en combinaison selon les besoins métier.