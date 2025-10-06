@echo off
echo ========================================
echo   IFRS17 - Demarrage Complet
echo ========================================
echo.
echo 1. Ouverture du Backend...
start "Backend IFRS17" cmd /k "%~dp0start_backend_persistent.bat"

timeout /t 5 /nobreak >nul

echo 2. Ouverture du Frontend...
start "Frontend Angular" cmd /k "%~dp0start_frontend.bat"

echo.
echo ========================================
echo   Application demarree !
echo   Backend:  http://127.0.0.1:8001
echo   Frontend: http://localhost:4200
echo ========================================
echo.
pause
