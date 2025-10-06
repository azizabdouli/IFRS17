@echo off
echo ========================================
echo   Demarrage Frontend Angular
echo ========================================

cd /d "%~dp0\angular-frontend"

echo Frontend demarre sur http://localhost:4200
echo.
ng serve --open

pause
