# IFRS17 Hub - BNA (Banque Nationale Agricole)# IFRS17 Insurance Application - Version Production 🎉



Application professionnelle de gestion IFRS 17 avec approche PAA (Premium Allocation Approach).## 📋 Aperçu du Projet

Application web full-stack pour la comptabilité d'assurance IFRS17 avec interface moderne Angular et backend FastAPI sécurisé.

## 🚀 Démarrage Rapide

## 🏗️ Architecture

### Option 1: Full Stack (Recommandé)- **Frontend**: Angular 17 avec design glassmorphism et authentification JWT

```powershell- **Backend**: FastAPI avec SQLAlchemy ORM et services IFRS17

.\start_fullstack.ps1- **Base de données**: SQLite (dev) / MySQL (prod) avec XAMPP

```- **Authentification**: JWT tokens avec bcrypt password hashing

- **Services IA**: Assistant IA et analytiques ML intégrés

### Option 2: Séparé

**Backend:**## 📁 Structure du Projet

```powershell

.\start_backend.ps1```

```📦 IFRS17-Application/

├── 🅰️ angular-frontend/          # Application Angular 17

**Frontend:**│   ├── src/

```powershell│   │   ├── app/

.\start_frontend.ps1│   │   │   ├── components/       # Composants UI

```│   │   │   │   ├── auth/         # Authentification

│   │   │   │   ├── dashboard/    # Tableau de bord

## 🌐 URLs│   │   │   │   ├── ppna-analytics/   # Analyses PPNA

│   │   │   │   ├── ml-analytics/     # Analytics ML

- **Frontend**: http://localhost:4200│   │   │   │   └── ai-assistant/     # Assistant IA

- **Backend API**: http://127.0.0.1:8001│   │   │   ├── services/         # Services HTTP

- **Documentation API**: http://127.0.0.1:8001/docs│   │   │   ├── guards/           # Guards de route

│   │   │   └── models/           # Modèles TypeScript

## 📚 Documentation│   │   └── environments/         # Configuration environnements

│   ├── package.json              # Dépendances Node.js

Consultez le dossier `docs/` pour:│   └── angular.json              # Configuration Angular

- `QUICK_START_PAA.md` - Démarrage rapide module PAA├── 🐍 backend/                   # API FastAPI

- `PAA_MODULE_README.md` - Guide utilisateur complet│   ├── auth/                     # Services d'authentification

- `ARCHITECTURE_PAA.md` - Architecture technique│   │   ├── security.py           # Sécurité JWT

- `TRANSFORMATION_PAA_COMPLETE.md` - Rapport de transformation│   │   └── user_service.py       # Service utilisateur

│   ├── database/                 # Couche de données

## 🛠️ Prérequis│   │   ├── models.py             # Modèles SQLAlchemy

│   │   ├── schemas.py            # Schémas Pydantic

- **Python** 3.11+│   │   └── connection.py         # Configuration DB

- **Node.js** 18+│   ├── routers/                  # Routes API

- **MySQL** 8.0+│   │   ├── auth_router.py        # Routes authentification

│   │   ├── ppna_router.py        # Routes PPNA

## 📦 Installation│   │   ├── ml_router.py          # Routes ML

│   │   └── ai_router.py          # Routes IA

### Backend│   ├── services/                 # Services métier

```bash│   │   ├── ppna_service.py       # Service PPNA IFRS17

python -m venv .venv│   │   ├── data_mapper.py        # Mapping de données

.venv\Scripts\Activate.ps1│   │   └── export_service.py     # Export de données

pip install -r requirements.txt│   ├── ai/                       # Services IA

```│   │   ├── ifrs17_ai_assistant.py    # Assistant IA

│   │   └── predictive_ai_service.py  # IA prédictive

### Frontend│   ├── ml/                       # Machine Learning

```bash│   │   ├── ml_service.py         # Services ML

cd angular-frontend│   │   └── models/               # Modèles ML

npm install│   └── main.py                   # Point d'entrée principal

```├── 📊 Data/                      # Données IFRS17

│   └── Ppna (4).xlsx             # Données PPNA

### Base de données├── ⚙️ Configuration

1. Démarrer MySQL (XAMPP ou service Windows)│   ├── .env                      # Variables d'environnement

2. Créer la base: `CREATE DATABASE ifrs17_db;`│   ├── .env.example              # Exemple configuration

3. Le backend créera automatiquement les tables au démarrage│   ├── requirements.txt          # Dépendances Python

│   └── .gitignore                # Git ignore

## 🎯 Fonctionnalités├── 🚀 Déploiement

│   ├── simple_auth_server.py     # Serveur auth standalone

### Modules Principaux│   └── start_full_stack.bat      # Lanceur production

- ✅ **Dashboard IFRS17** - Vue d'ensemble et KPIs└── 📖 README.md                  # Documentation

- ✅ **Module PAA** - Premium Allocation Approach (IFRS 17.53-59)```

- ✅ **PPNA Analytics** - Analyse des données PPNA

- ✅ **ML Analytics** - Machine Learning prédictif## ⚡ Démarrage Rapide

- ✅ **Assistant IA** - Assistant IFRS17 intelligent

### 1. Prérequis

### Fonctionnalités Techniques- **Python 3.12+** avec pip

- ✅ Authentification sécurisée (JWT)- **Node.js 18+** avec npm/Angular CLI

- ✅ API REST documentée (Swagger/OpenAPI)- **XAMPP** (pour MySQL en production)

- ✅ Export Excel/PDF

- ✅ Tests automatisés### 2. Installation

- ✅ Base de données MySQL

- ✅ Architecture modulaire#### Backend Python

```bash

## 📁 Structure du Projet# Activer l'environnement virtuel

.venv\Scripts\activate

```

.# Installer les dépendances

├── backend/                  # Backend FastAPIpip install -r requirements.txt

│   ├── main.py              # Point d'entrée```

│   ├── routers/             # Routes API

│   ├── services/            # Logique métier#### Frontend Angular

│   ├── database/            # Modèles et connexion```bash

│   ├── measurement/paa/     # Module PAAcd angular-frontend

│   ├── ml/                  # Machine Learningnpm install

│   ├── ai/                  # Services IA```

│   └── tests/               # Tests unitaires

│### 3. Configuration

├── angular-frontend/         # Frontend Angular 17

│   └── src/app/#### Variables d'environnement

│       ├── components/      # Composants UICopier `.env.example` vers `.env` et configurer :

│       ├── services/        # Services HTTP```env

│       ├── guards/          # Guards de navigationDATABASE_URL=sqlite:///./ifrs17.db

│       └── models/          # Interfaces TypeScriptSECRET_KEY=your-secret-key-here

│ALGORITHM=HS256

├── docs/                     # Documentation complèteACCESS_TOKEN_EXPIRE_MINUTES=30

├── Data/                     # Données de test```

└── requirements.txt          # Dépendances Python

```#### Base de données

- **Développement**: SQLite (automatique)

## 🧪 Tests- **Production**: MySQL avec XAMPP



### Backend### 4. Lancement

```bash

# Activer l'environnement virtuel#### Option 1: Lancement automatique (recommandé)

.venv\Scripts\Activate.ps1```bash

start_full_stack.bat

# Lancer les tests```

pytest backend/tests/ -v

#### Option 2: Lancement manuel

# Avec coverage```bash

pytest backend/tests/ --cov=backend --cov-report=html# Terminal 1 - Backend

```python simple_auth_server.py



### Frontend# Terminal 2 - Frontend

```bashcd angular-frontend

cd angular-frontendng serve

npm test```

```

### 5. Accès

## 🔧 Configuration- **Frontend**: http://localhost:4200

- **Backend API**: http://localhost:8001

### Backend (.env)- **Documentation API**: http://localhost:8001/docs

```env

DATABASE_URL=mysql+pymysql://root:@localhost/ifrs17_db## 🔐 Authentification

SECRET_KEY=your-secret-key-here

ALGORITHM=HS256### Comptes par défaut

ACCESS_TOKEN_EXPIRE_MINUTES=30- **Actuaire**: `actuaire@bna.tn` / `password123`

```- **Comptable**: `comptable@bna.tn` / `password123`



### Frontend (environment.ts)### Fonctionnalités

```typescript- JWT Token Authentication

export const environment = {- Hachage bcrypt des mots de passe

  production: false,- Contrôle d'accès basé sur les rôles

  apiUrl: 'http://127.0.0.1:8001'- Guards Angular pour la sécurité des routes

};

```## 📊 Fonctionnalités IFRS17



## 📊 Module PAA - Guide Rapide### Services PPNA

- Calculs PAA (Premium Allocation Approach)

### 1. Créer un groupe de contrats- Gestion des passifs d'assurance

```json- Projections financières TND

POST /paa/groups/init- Export de données Excel

{

  "group_id": "AUTO_2025_Q1",### Analytics ML

  "contracts": [- Prédictions basées sur l'IA

    {- Analyse de tendances

      "contract_id": "C1",- Optimisation des réserves

      "portfolio": "AUTO",- Modèles d'apprentissage automatique

      "inception": "2025-01-01",

      "expiry": "2025-12-31",### Assistant IA

      "written_premium": 15000,- Support conversationnel IFRS17

      "expected_claim_ratio": 0.55,- Aide à la décision

      "expected_expense_ratio": 0.12- Explications des calculs

    }- Recommandations personnalisées

  ]

}## 🎨 Interface Utilisateur

```

### Design

### 2. Traiter une période- **Style**: Glassmorphism moderne

```json- **Responsive**: Support mobile et desktop

POST /paa/groups/{group_id}/period- **Thème**: Violet/blanc professionnel

{- **UX**: Navigation intuitive

  "period_start": "2025-01-01",

  "period_end": "2025-01-31",### Composants

  "incurred_claims": 2500,- Dashboard interactif

  "claims_paid": 2000- Formulaires réactifs Angular

}- Tableaux de données avancés

```- Graphiques et visualisations



### 3. Consulter les mouvements## 🔧 Technologies

```

GET /paa/groups/{group_id}/movements### Frontend

```- **Angular 17**: Framework principal

- **TypeScript**: Langage typé

## 🔒 Sécurité- **RxJS**: Programmation réactive

- **Angular Material**: Composants UI

- Authentification JWT avec expiration- **SCSS**: Styles avancés

- CORS configuré (localhost uniquement en dev)

- Protection SQL injection (SQLAlchemy ORM)### Backend

- Validation des entrées (Pydantic)- **FastAPI**: Framework API moderne

- Variables d'environnement pour les secrets- **SQLAlchemy**: ORM Python

- **Pydantic**: Validation de données

## 🚢 Déploiement- **Bcrypt**: Hachage de mots de passe

- **PyJWT**: Gestion JWT

### Production

1. Configurer `environment.prod.ts`### Base de données

2. Build frontend: `npm run build`- **SQLite**: Développement

3. Configurer variables d'environnement backend- **MySQL**: Production

4. Utiliser Gunicorn/Uvicorn pour le backend- **XAMPP**: Serveur local

5. Servir le frontend via nginx

## 📈 Déploiement Production

## 🐛 Dépannage

### Préparation

### Backend ne démarre pas1. Configurer MySQL dans XAMPP

- Vérifier que MySQL est démarré2. Mettre à jour les variables d'environnement

- Vérifier le fichier `.env`3. Construire la version de production Angular

- Vérifier les dépendances: `pip install -r requirements.txt`4. Optimiser les performances



### Frontend ne charge pas### Commandes

- Vérifier que le backend est démarré```bash

- Vérifier `environment.ts` (apiUrl)# Build Angular pour production

- Réinstaller: `npm install`cd angular-frontend

ng build --configuration production

### Dashboard lent

- Vérifier la connexion MySQL# Lancer le serveur complet

- Optimiser les requêtes (voir logs backend)start_full_stack.bat

- Activer le cache si disponible```



## 👥 Équipe## 🛠️ Développement



- **Développeur**: Abdouli Aziz### Structure de développement

- **Organisation**: BNA (Banque Nationale Agricole)- Code organisé par domaines métier

- **Version**: 2.0.0- Services réutilisables

- **Date**: Octobre 2025- Typage strict TypeScript/Python

- Tests unitaires intégrés

## 📄 Licence

### Bonnes pratiques

Propriétaire - BNA © 2025  - Architecture modulaire

Tous droits réservés.- Injection de dépendances

- Gestion d'erreurs centralisée

## 🔗 Liens Utiles- Logging structuré



- [FastAPI Documentation](https://fastapi.tiangolo.com/)## � Support

- [Angular Documentation](https://angular.io/docs)

- [IFRS 17 Standard](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/)### Documentation

- [PAA Approach Explained](https://www.ifrs.org/projects/work-plan/ifrs-17-implementation/)- **API**: http://localhost:8001/docs

- **Frontend**: Navigation dans l'application

---- **IFRS17**: Guides intégrés dans l'assistant IA



**Note**: Ce projet implémente la norme IFRS 17 avec l'approche PAA (Premium Allocation Approach) conformément aux paragraphes 53-59 de la norme.### Maintenance

- Logs d'application disponibles
- Monitoring des performances
- Sauvegarde automatique des données
- Mises à jour sécurisées

---

## 🏆 Statut du Projet

✅ **Application complète et fonctionnelle**  
✅ **Authentification sécurisée implémentée**  
✅ **Services IFRS17 opérationnels**  
✅ **Interface moderne déployée**  
✅ **Base de données configurée**  
✅ **Prêt pour la production**  

---

**Développé pour BNA - Banque Nationale Agricole**  
*Application IFRS17 - Comptabilité d'Assurance Moderne*
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos paramètres
```

### 3. Configuration du Frontend
```bash
cd angular-frontend
npm install
```

### 4. Démarrage des serveurs

#### Backend (Terminal 1)
```bash
python simple_auth_server.py
# Serveur disponible sur http://127.0.0.1:8001
```

#### Frontend (Terminal 2)
```bash
cd angular-frontend
npm start
# Application disponible sur http://localhost:4200
```

## 📱 Utilisation

1. **Accès à l'application** : Ouvrir http://localhost:4200
2. **Inscription** : Créer un compte avec rôle "actuaire" ou "comptable"
3. **Connexion** : Se connecter avec les identifiants créés
4. **Dashboard** : Accéder au tableau de bord IFRS17

## 🛠️ API Endpoints

### Authentification
- `POST /auth/register` - Inscription utilisateur
- `POST /auth/login` - Connexion
- `GET /auth/verify` - Vérification token
- `GET /auth/me` - Profil utilisateur
- `POST /auth/logout` - Déconnexion

### Système
- `GET /` - Page d'accueil de l'API
- `GET /health` - Statut de santé de l'API
- `GET /docs` - Documentation Swagger

## 📚 Structure du Projet

```
├── angular-frontend/          # Application Angular
│   ├── src/app/
│   │   ├── components/        # Composants UI
│   │   ├── services/          # Services Angular
│   │   ├── guards/            # Guards de navigation
│   │   └── models/            # Modèles TypeScript
│   └── package.json
├── backend/                   # API FastAPI
│   ├── auth/                  # Authentification
│   ├── database/              # Base de données
│   ├── routers/               # Routes API
│   ├── services/              # Services métier
│   └── utils/                 # Utilitaires
├── Data/                      # Données de test
├── .env                       # Configuration
├── requirements.txt           # Dépendances Python
├── simple_auth_server.py      # Serveur d'authentification
└── README.md                  # Cette documentation
```

## 🔧 Configuration

### Variables d'environnement (.env)
```env
DATABASE_TYPE=sqlite
DATABASE_URL=sqlite:///./ifrs17_auth.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Base de données
- **Développement** : SQLite (automatique)
- **Production** : PostgreSQL/MySQL (configurable)
- **Tables** : Créées automatiquement au démarrage

## � Sécurité

- **JWT** pour l'authentification
- **bcrypt** pour le hachage des mots de passe
- **CORS** configuré pour le développement
- **Validation** des entrées utilisateur
- **Protection** des routes sensibles

## � Tests

```bash
# Test de l'API d'authentification
python test_auth_api.py
```

## 🚀 Déploiement

### Production
1. Configurer PostgreSQL ou MySQL
2. Mettre à jour les variables d'environnement
3. Builder l'application Angular : `ng build --prod`
4. Déployer avec gunicorn ou uvicorn

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## � Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

---

**Développé pour la gestion comptable IFRS17 - Système sécurisé pour actuaires et comptables**

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