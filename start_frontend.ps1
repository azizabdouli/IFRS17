Write-Host "Demarrage Frontend IFRS17..." -ForegroundColor Cyan

Set-Location angular-frontend

if (-not (Test-Path "node_modules")) {
    Write-Host "Installation des dependances npm..." -ForegroundColor Yellow
    npm install
}

Write-Host ""
Write-Host "Frontend: http://localhost:4200" -ForegroundColor Green
Write-Host ""
npm start
