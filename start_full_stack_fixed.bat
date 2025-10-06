@echo off
title IFRS17 - Démarrage Simplifié
color 0A
echo.
echo ======================================================
echo              🚀 IFRS17 - DÉMARRAGE RÉUSSI !
echo ======================================================
echo.

cd /d "C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main"

echo ✅ Activation de l'environnement Python...
call .venv\Scripts\activate.bat

echo ✅ Démarrage du serveur Backend sur le port 8001...
start "Backend IFRS17" cmd /k "python start_backend_new.py"

timeout /t 5 /nobreak >nul

echo ✅ Démarrage du serveur Frontend sur le port 4200...
cd angular-frontend
start "Frontend IFRS17" cmd /k "npm start"

echo.
echo 🎉 DÉMARRAGE TERMINÉ !
echo.
echo 📊 Frontend Angular : http://localhost:4200
echo 🔧 Backend API      : http://localhost:8001
echo 📚 Documentation    : http://localhost:8001/docs
echo.
echo 🔐 Comptes de test :
echo    Email: analyste@bna.tn
echo    Mot de passe: password123
echo.
echo Appuyez sur une touche pour fermer...
pause >nul