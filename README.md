# Système ML IFRS17 - Analyse Prédictive des Contrats d'Assurance

## 📋 Description

Ce projet implémente un système complet d'analyse prédictive pour les contrats d'assurance selon la norme IFRS17. Il utilise des techniques de Machine Learning avancées pour analyser, classifier et prédire les comportements des contrats d'assurance.

## 🏗️ Architecture

### Backend (FastAPI)
- **API REST** pour l'entraînement et la prédiction des modèles ML
- **Preprocessing** robuste des données IFRS17
- **4 modèles ML spécialisés** :
  - 💰 **Modèle de Rentabilité** (R² > 0.96)
  - ⚠️ **Classification des Risques** (Accuracy > 0.8)
  - 🔍 **Prédiction des Sinistres** (R² > 0.73)
  - 📈 **Prédiction LRC** (R² > 0.93)

### Frontend (Streamlit)
- **Interface utilisateur intuitive** pour l'upload et l'analyse
- **Visualisations interactives** des résultats
- **Export** des rapports en PDF/Excel

### Fonctionnalités Avancées
- 🎯 **Clustering** des contrats
- 🔍 **Détection d'anomalies**
- 📊 **Tableaux de bord** de monitoring
- 🔄 **Pipeline ML** automatisé

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Structure du projet
```
├── backend/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── ml/                  # Modules ML
│   │   ├── ml_service.py    # Service principal ML
│   │   ├── data_preprocessing.py  # Preprocessing des données
│   │   └── models/          # Modèles ML spécialisés
│   ├── routers/             # Routes API
│   └── services/            # Services métier
├── frontend/
│   ├── app.py              # Interface principale
│   └── ml_interface.py     # Interface ML spécialisée
├── Data/                   # Données IFRS17
└── models/                 # Modèles sauvegardés
```

## 🔧 Utilisation

### 1. Démarrer l'API Backend
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

### 2. Démarrer l'Interface ML
```bash
streamlit run frontend/ml_interface.py --server.port 8504
```

### 3. Accéder aux interfaces
- **API Documentation** : http://127.0.0.1:8001/docs
- **Interface ML** : http://127.0.0.1:8504

## 📊 Utilisation de l'API

### Upload de données
```bash
POST /ml/upload-data
Content-Type: multipart/form-data
```

### Entraînement des modèles
```bash
# Modèle de rentabilité
POST /ml/train-profitability?model_type=xgboost

# Classification des risques
POST /ml/train-risk-classification?model_type=random_forest

# Prédiction des sinistres
POST /ml/train-claims-prediction?model_type=xgboost

# Prédiction LRC
POST /ml/train-lrc-prediction?model_type=xgboost
```

### Clustering
```bash
POST /ml/clustering?n_clusters=5&clustering_type=kmeans
```

### Détection d'anomalies
```bash
POST /ml/anomaly-detection?method=isolation_forest&contamination=0.1
```

## 🔍 Format des Données

Le système accepte les fichiers Excel (.xlsx) ou CSV (.csv) avec les colonnes IFRS17 standard :
- `MNTPRNET` : Montant prime nette
- `MNTPPNA` : Montant PPNA
- `DUREE` : Durée du contrat
- `CODPROD` : Code produit
- `DEBEFFQUI`, `FINEFFQUI` : Dates d'effet
- Et autres colonnes métier...

## 📈 Performance des Modèles

| Modèle | Métrique | Performance |
|--------|----------|-------------|
| Rentabilité | R² Score | 0.964 |
| Classification Risques | Accuracy | 0.8+ |
| Prédiction Sinistres | R² Score | 0.732 |
| Prédiction LRC | R² Score | 0.937 |

## 🛠️ Technologies Utilisées

- **Backend** : FastAPI, Python 3.8+
- **Frontend** : Streamlit
- **ML/IA** : XGBoost, Random Forest, scikit-learn, LightGBM
- **Data** : Pandas, NumPy
- **Visualisation** : Plotly, Matplotlib

## 📦 Dépendances Principales

Voir `requirements.txt` pour la liste complète des dépendances.

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

Pour toute question ou support, contactez l'équipe de développement.

---

**Développé avec ❤️ pour l'analyse prédictive IFRS17**