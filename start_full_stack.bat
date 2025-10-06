@echo off
title IFRS17 - Lanceur de Serveurs
color 0E
echo.
echo ======================================================
echo               🚀 IFRS17 - FULL STACK
echo ======================================================
echo.
echo Démarrage automatique des serveurs:
echo   📡 Backend (FastAPI) - Port 8001
echo   🎨 Frontend (Angular) - Port 4200
echo.

cd /d "C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main"

echo ✅ Lancement du serveur Backend...
start "Backend IFRS17" start_backend_server.bat

timeout /t 3 /nobreak >nul

echo ✅ Lancement du serveur Frontend...
start "Frontend IFRS17" start_frontend_server.bat

echo.
echo 🎉 Les deux serveurs sont en cours de démarrage!
echo.
echo 📋 URLs d'accès:
echo   🔐 API Backend: http://127.0.0.1:8001
echo   📖 Documentation API: http://127.0.0.1:8001/docs
echo   🌐 Application Frontend: http://localhost:4200
echo.
echo ⚠️  Gardez cette fenêtre ouverte pour surveiller les serveurs
echo ⏹️  Pour arrêter tout: fermez les fenêtres ou Ctrl+C
echo.

pause