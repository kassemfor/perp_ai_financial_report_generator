# Automated setup script for Windows
# Usage: .\setup.ps1

Write-Host "🚀 AI Financial Report Generator - Setup (Windows)" -ForegroundColor Green
Write-Host "=================================================="

# Check Python
Write-Host "✓ Checking Python installation..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3 required" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "✓ Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate venv
Write-Host "✓ Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "✓ Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
Write-Host "✓ Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "config" | Out-Null

# Run tests
Write-Host "✓ Running automated tests..." -ForegroundColor Cyan
pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate venv: .\venv\Scripts\Activate.ps1"
Write-Host "2. Start Ollama (if using): ollama serve"
Write-Host "3. Run web UI: streamlit run src\frontend\streamlit_app.py"
Write-Host "4. Or run CLI: python main.py --file data.csv --output report.pdf"
Write-Host ""
