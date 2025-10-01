# Script de démarrage pour l'interface Angular IFRS17
# Exécute le frontend Angular et le backend FastAPI

Write-Host "🏛️ DÉMARRAGE INTERFACE IFRS17 - COMPTABILITÉ ASSURANCE" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Blue

# Vérification de Node.js et Angular CLI
Write-Host "🔍 Vérification des prérequis..." -ForegroundColor Yellow
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js n'est pas installé ou non accessible dans le PATH" -ForegroundColor Red
    exit 1
}

if (!(Get-Command ng -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Angular CLI n'est pas installé. Installation..." -ForegroundColor Red
    npm install -g @angular/cli
}

# Démarrage du backend FastAPI
Write-Host "🚀 Démarrage du backend FastAPI..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-Command", "cd '$PSScriptRoot'; python backend/main.py" -WindowStyle Minimized

# Attendre 3 secondes pour le démarrage du backend
Start-Sleep -Seconds 3

# Démarrage du frontend Angular
Write-Host "🌟 Démarrage du frontend Angular..." -ForegroundColor Green
Set-Location "$PSScriptRoot\angular-frontend"

# Installation des dépendances si nécessaire
if (!(Test-Path "node_modules")) {
    Write-Host "📦 Installation des dépendances npm..." -ForegroundColor Yellow
    npm install
}

# Lancement du serveur de développement
Write-Host "🎯 Lancement de l'interface IFRS17..." -ForegroundColor Cyan
Write-Host "URL: http://localhost:4200" -ForegroundColor Green
Write-Host "API Backend: http://localhost:8001" -ForegroundColor Green
Write-Host ""
Write-Host "✨ Interface prête ! Fonctionnalités disponibles:" -ForegroundColor Magenta
Write-Host "   • Dashboard temps réel IFRS17" -ForegroundColor White
Write-Host "   • Métriques LRC, LIC, CSM, Risk Adjustment" -ForegroundColor White
Write-Host "   • Analyses actuarielles et ML" -ForegroundColor White
Write-Host "   • Assistant IA pour comptabilité d'assurance" -ForegroundColor White
Write-Host "   • Exports Excel et PDF" -ForegroundColor White
Write-Host ""

ng serve --open --port 4200