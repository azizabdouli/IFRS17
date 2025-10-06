@echo off
echo ========================================
echo   Demarrage Backend IFRS17
echo ========================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Backend demarre sur http://127.0.0.1:8001
echo.
uvicorn backend.main:app --host 127.0.0.1 --port 8001

pause
