# 🔍 GUIDE DE DIAGNOSTIC - PROBLÈME D'INSCRIPTION

## ✅ Services Vérifiés
- Backend: http://localhost:8001 ✅ FONCTIONNE
- Frontend: http://localhost:4200 ✅ FONCTIONNE
- Base de données: SQLite ✅ ACTIVE

## 🧪 TESTS À EFFECTUER

### Test 1: Formulaire HTML Simple
1. Ouvrir: `test_inscription.html` dans le navigateur
2. Remplir le formulaire
3. Cliquer sur "Créer le Compte"
4. **Si ça marche** → Le backend fonctionne, problème dans Angular
5. **Si ça ne marche pas** → Problème backend ou CORS

### Test 2: Console du Navigateur (F12)
1. Ouvrir http://localhost:4200
2. Appuyer sur F12
3. Aller dans l'onglet "Console"
4. Cliquer sur "Inscription"
5. Remplir le formulaire
6. Cliquer sur "Créer mon Compte"
7. **Regarder les messages dans la console:**
   - "📝 Tentative d'inscription..." → Le clic fonctionne
   - "Formulaire valide: false" → Le formulaire a des erreurs
   - "❌ Formulaire invalide" → Voir quels champs sont invalides

### Test 3: Vérifier les Champs Requis

**Champs obligatoires:**
- ✅ Prénom (min 2 caractères)
- ✅ Nom (min 2 caractères)
- ✅ Email (format valide)
- ✅ Mot de passe (min 6 caractères)
- ✅ Confirmation mot de passe (identique)
- ✅ **Accepter les conditions** ← IMPORTANT!

**Champs optionnels:**
- Société (valeur par défaut: BNA)
- Département (valeur par défaut: Assurance)
- Téléphone
- ID Employé

## 🔧 SOLUTIONS RAPIDES

### Solution 1: Si le bouton est grisé
**Cause:** Formulaire invalide
**Actions:**
1. Vérifier que TOUS les champs requis sont remplis
2. **COCHER la case "J'accepte les conditions"** ← Souvent oublié!
3. Vérifier que les mots de passe correspondent
4. Vérifier le format de l'email

### Solution 2: Si le bouton clique mais rien ne se passe
**Cause:** Erreur JavaScript
**Actions:**
1. Ouvrir la Console (F12)
2. Chercher les erreurs en rouge
3. Noter le message d'erreur
4. Vérifier l'onglet "Network" pour voir si la requête est envoyée

### Solution 3: Si erreur 401/403
**Cause:** Problème d'authentification ou CORS
**Actions:**
1. Vérifier que le backend est démarré
2. Vérifier les headers CORS dans le backend

### Solution 4: Si erreur 400
**Cause:** Données invalides envoyées au backend
**Actions:**
1. Vérifier le format des données dans la console
2. Vérifier que tous les champs requis par le backend sont présents

## 📞 MESSAGE D'ERREUR TYPIQUES

### "Formulaire invalide"
→ Un ou plusieurs champs ne respectent pas les règles
→ Vérifier chaque champ un par un
→ COCHER la case des conditions!

### "Erreur de connexion"
→ Backend pas démarré ou mauvaise URL
→ Vérifier http://localhost:8001/docs

### "Email déjà existant"
→ Utiliser un autre email ou supprimer ifrs17_auth.db

### "password cannot be longer than 72 bytes"
→ Utiliser un mot de passe plus court (< 72 caractères)

## 🎯 CHECKLIST RAPIDE

Avant de cliquer sur "Créer mon Compte", vérifier:

[ ] Prénom rempli (≥ 2 caractères)
[ ] Nom rempli (≥ 2 caractères)  
[ ] Email valide (ex: test@bna.tn)
[ ] Mot de passe ≥ 6 caractères
[ ] Confirmation mot de passe identique
[ ] ☑️ Case "J'accepte les conditions" COCHÉE ← TRÈS IMPORTANT!

Si tout est coché et le bouton reste grisé:
→ Ouvrir F12 → Console
→ Taper: `signupForm.valid` dans la console
→ Si false, taper: `signupForm.errors`
