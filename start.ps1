# Script de démarrage PowerShell pour le système ML IFRS17
# Usage: .\start.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🎯 SYSTÈME ML IFRS17 - DÉMARRAGE" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

# Vérification de l'environnement virtuel
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activation de l'environnement virtuel..." -ForegroundColor Green
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️ Environnement virtuel non trouvé (.venv)" -ForegroundColor Yellow
    Write-Host "💡 Créez un environnement virtuel avec: python -m venv .venv" -ForegroundColor Blue
}

# Menu de sélection
Write-Host "`n📋 Services disponibles:" -ForegroundColor White
Write-Host "1. API FastAPI (port 8001)" -ForegroundColor Cyan
Write-Host "2. Interface ML Streamlit (port 8504)" -ForegroundColor Cyan
Write-Host "3. Les deux services" -ForegroundColor Cyan
Write-Host "4. Quitter" -ForegroundColor Red

$choice = Read-Host "`n🔧 Votre choix (1-4)"

switch ($choice) {
    "1" {
        Write-Host "🚀 Démarrage de l'API FastAPI..." -ForegroundColor Green
        uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
    }
    "2" {
        Write-Host "🖥️ Démarrage de l'interface Streamlit..." -ForegroundColor Green
        streamlit run frontend/ml_interface.py --server.port 8504
    }
    "3" {
        Write-Host "🚀 Démarrage des deux services..." -ForegroundColor Green
        
        # Démarrer l'API en arrière-plan
        Start-Job -ScriptBlock {
            Set-Location $using:PWD
            uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
        } -Name "IFRS17-API"
        
        Write-Host "✅ API démarrée en arrière-plan" -ForegroundColor Green
        Start-Sleep -Seconds 3
        
        # Ouvrir les URLs
        Write-Host "🌐 Ouverture des interfaces..." -ForegroundColor Blue
        Start-Process "http://127.0.0.1:8001/docs"
        Start-Process "http://127.0.0.1:8504"
        
        # Démarrer Streamlit
        streamlit run frontend/ml_interface.py --server.port 8504
        
        # Arrêter l'API quand Streamlit se ferme
        Get-Job -Name "IFRS17-API" | Stop-Job
        Get-Job -Name "IFRS17-API" | Remove-Job
    }
    "4" {
        Write-Host "👋 Au revoir!" -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host "❌ Choix invalide" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✅ Script terminé" -ForegroundColor Green