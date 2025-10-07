Write-Host "Demarrage Backend IFRS17..." -ForegroundColor Cyan

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
    Write-Host "Environnement virtuel active" -ForegroundColor Green
} else {
    Write-Host "Erreur: Environnement virtuel non trouve" -ForegroundColor Red
    Write-Host "Creez-le avec: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Backend API: http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "Documentation: http://127.0.0.1:8001/docs" -ForegroundColor Green
Write-Host ""

# Lancer depuis la racine pour que les imports backend.* fonctionnent
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
