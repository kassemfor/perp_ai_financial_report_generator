#!/bin/bash
# Automated setup script for AI Financial Report Generator

echo "🚀 AI Financial Report Generator - Setup"
echo "========================================"

# Check Python version
echo "✓ Checking Python installation..."
python3 --version || { echo "Python 3 required"; exit 1; }

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "✓ Creating directories..."
mkdir -p uploads outputs logs config templates

# Copy config template
echo "✓ Setting up configuration..."
touch .env

# Run tests
echo "✓ Running automated tests..."
pytest tests/ -v || { echo "Tests failed"; exit 1; }

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Ollama (if using): ollama serve"
echo "3. Run web UI: streamlit run src/frontend/streamlit_app.py"
echo "4. Or run CLI: python main.py --file data.csv --output report.pdf"
echo ""
