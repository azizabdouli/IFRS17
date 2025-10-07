Write-Host "======================================" -ForegroundColor Cyan
Write-Host "    LANCEMENT FULL STACK IFRS17       " -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Demarrage Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", "start_backend.ps1"
Start-Sleep -Seconds 3

Write-Host "2. Demarrage Frontend..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-File", "start_frontend.ps1"

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "    APPLICATION LANCEE AVEC SUCCES    " -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "URLs disponibles:" -ForegroundColor Cyan
Write-Host "  Backend API:    http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  Frontend:       http://localhost:4200" -ForegroundColor White
Write-Host "  API Docs:       http://127.0.0.1:8001/docs" -ForegroundColor White
Write-Host ""
