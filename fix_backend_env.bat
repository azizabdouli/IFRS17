@echo off
title IFRS17 - Réparation Environnement Backend
color 0C
echo.
echo ======================================================
echo           🔧 RÉPARATION ENVIRONNEMENT BACKEND
echo ======================================================
echo.

cd /d "C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main"

echo ❌ Suppression de l'ancien environnement virtuel...
if exist .venv rmdir /s /q .venv

echo ✅ Création d'un nouvel environnement virtuel...
python -m venv .venv

echo ✅ Activation de l'environnement...
call .venv\Scripts\activate.bat

echo ✅ Mise à jour de pip...
python -m pip install --upgrade pip

echo ✅ Installation des dépendances...
pip install -r requirements.txt

echo.
echo ✅ Environnement réparé avec succès !
echo.
echo Pour démarrer le backend :
echo   1. .venv\Scripts\activate
echo   2. cd backend
echo   3. uvicorn main:app --reload --port 8001
echo.
pause