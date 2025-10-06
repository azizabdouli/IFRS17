# 🚀 **GUIDE D'EXÉCUTION - PROJET IFRS17 TRANSFORMÉ**

## ⚡ **DÉMARRAGE RAPIDE (Recommandé)**

### 🎯 **Option 1 : Lanceur Automatique**
```bash
# Exécuter le fichier batch pour démarrer tout automatiquement
start_full_stack.bat
```

### 🔧 **Option 2 : Démarrage Manuel (3 Terminals)**

#### **Terminal 1 - Backend Python (FastAPI)**
```bash
# Naviguer vers le dossier principal
cd "C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main"

# Activer l'environnement virtuel Python
.venv\Scripts\activate

# Installer/Mettre à jour les dépendances
pip install -r requirements.txt

# Démarrer le serveur FastAPI
cd backend
uvicorn main:app --reload --port 8001 --host 0.0.0.0
```

#### **Terminal 2 - Frontend Angular**
```bash
# Naviguer vers le dossier Angular
cd "C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main\angular-frontend"

# Installer les dépendances Node.js
npm install

# Démarrer le serveur Angular
npm start
# ou
ng serve --port 4200 --host 0.0.0.0
```

#### **Terminal 3 - Base de Données (Optionnel)**
```bash
# Si vous utilisez XAMPP pour MySQL
# Démarrer Apache et MySQL depuis XAMPP Control Panel

# Ou utiliser SQLite (par défaut)
# Aucune action requise - SQLite s'initialise automatiquement
```

---

## 🌐 **ACCÈS AUX APPLICATIONS**

### 📊 **Frontend Angular**
- **URL** : http://localhost:4200
- **Interface** : Dashboard Unifié Analyste IFRS17
- **Fonctionnalités** : Authentification, Dashboard, Analytics

### 🔧 **Backend API**
- **URL** : http://localhost:8001
- **Documentation** : http://localhost:8001/docs (Swagger UI)
- **Redoc** : http://localhost:8001/redoc
- **Health Check** : http://localhost:8001/health

---

## 👤 **COMPTES DE TEST DISPONIBLES**

### 🔐 **Utilisateurs par Défaut**
Le système crée automatiquement des comptes de test :

#### **Analyste IFRS17 Principal**
- **Email** : `analyste@bna.tn`
- **Mot de passe** : `password123`
- **Rôle** : Analyste IFRS17
- **Entreprise** : BNA
- **Département** : Assurance

#### **Expert Senior**
- **Email** : `expert@bna.tn`
- **Mot de passe** : `password123`
- **Rôle** : Analyste IFRS17
- **Entreprise** : BNA
- **Département** : Assurance

---

## 🎯 **WORKFLOW DE TEST DU SYSTÈME TRANSFORMÉ**

### **1. Connexion**
1. Ouvrir http://localhost:4200
2. Se connecter avec `analyste@bna.tn` / `password123`
3. Observer l'interface unifiée (plus de sélection de rôle)

### **2. Dashboard Unifié**
1. **En-tête utilisateur** avec progression et niveau
2. **Alertes contextuelles** intelligentes
3. **KPIs unifiés** (PPNA, Contrats Onéreux, Conformité, Précision)
4. **Actions recommandées** personnalisées
5. **Résumé hebdomadaire** avec badges

### **3. Système de Gamification**
1. **Points** : Attribués automatiquement pour les actions
2. **Niveaux** : Débutant → Intermédiaire → Expert → Maître IFRS17
3. **Badges** : Récompenses pour achievements spécifiques
4. **Progression** : Barre de progression vers le niveau suivant

### **4. Fonctionnalités IFRS17**
1. **Analytics PPNA** : http://localhost:4200/ppna-analytics
2. **ML Analytics** : http://localhost:4200/ml-analytics
3. **Transformations** : http://localhost:4200/data-transformations
4. **Assistant IA** : http://localhost:4200/ai-assistant

---

## 🔧 **RÉSOLUTION DE PROBLÈMES**

### **Erreur Backend**
```bash
# Vérifier que le port 8001 est libre
netstat -ano | findstr :8001

# Redémarrer avec un autre port si nécessaire
uvicorn main:app --reload --port 8002 --host 0.0.0.0
```

### **Erreur Frontend**
```bash
# Nettoyer le cache npm
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Démarrer sur un autre port si nécessaire
ng serve --port 4201
```

### **Problème de CORS**
Le backend est configuré pour accepter les requêtes depuis :
- `http://localhost:4200`
- `http://127.0.0.1:4200`

### **Base de Données**
```python
# La base SQLite se crée automatiquement
# Fichier : backend/ifrs17.db

# Pour reset la base de données
rm backend/ifrs17.db
# Redémarrer le backend pour recréer
```

---

## 📊 **ENDPOINTS API TRANSFORMÉS**

### **Authentification**
- `POST /auth/login` - Connexion unifiée
- `POST /auth/register` - Inscription analyste IFRS17
- `GET /auth/me` - Profil utilisateur avec progression

### **Dashboard Unifié**
- `GET /dashboard/` - Dashboard complet
- `POST /dashboard/award-points/{points}` - Attribution points
- `GET /dashboard/user-progress` - Progression détaillée
- `GET /dashboard/recommended-actions` - Actions recommandées
- `GET /dashboard/alerts` - Alertes contextuelles

### **IFRS17 Existantes**
- `GET /ppna/*` - Analytics PPNA
- `GET /ml/*` - Machine Learning
- `GET /ai/*` - Assistant IA
- `GET /transform/*` - Transformations

---

## ✅ **VÉRIFICATION DU SYSTÈME**

### **Checklist de Démarrage**
- [ ] Backend démarré sur http://localhost:8001
- [ ] Frontend démarré sur http://localhost:4200
- [ ] Swagger accessible sur http://localhost:8001/docs
- [ ] Connexion réussie avec comptes test
- [ ] Dashboard unifié affiché
- [ ] KPIs et alertes visibles
- [ ] Actions recommandées fonctionnelles

### **Tests de Gamification**
- [ ] Attribution de points fonctionne
- [ ] Progression utilisateur mise à jour
- [ ] Badges attribués correctement
- [ ] Niveaux calculés automatiquement

---

## 🎉 **SYSTÈME PRÊT !**

Votre application IFRS17 transformée est maintenant opérationnelle avec :

🎯 **Rôle unifié** : Analyste IFRS17 uniquement
📊 **Dashboard intelligent** : KPIs contextuels et alertes
🏆 **Gamification** : Points, badges, niveaux
🧠 **Recommandations** : Actions personnalisées
⚡ **Performance** : Workflow optimisé 15-20 min

**Bon test du nouveau système ! 🚀**