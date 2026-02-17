# 🚀 Quick Start Guide

## 1. Installation (2 minutes)

### Linux/Mac
```bash
cd ai_financial_report_generator
chmod +x setup.sh
./setup.sh
```

### Windows
```powershell
cd ai_financial_report_generator
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\setup.ps1
```

## 2. Configure LLM Backend (1 minute)

### Option A: Ollama (Recommended - Local)
```bash
# Install from https://ollama.ai
ollama pull mistral:latest
ollama serve  # Runs on http://localhost:11434
```

### Option B: OpenAI
```bash
export LLM_BACKEND=openai
export LLM_API_KEY=sk-...
```

### Option C: Gemini
```bash
export LLM_BACKEND=gemini
export LLM_API_KEY=your-api-key
```

## 3. Run Application (30 seconds)

### Web Interface (Recommended)
```bash
source venv/bin/activate  # or: .\venv\Scripts\Activate.ps1
streamlit run src/frontend/streamlit_app.py
```
Then open http://localhost:8501

### Command Line
```bash
python main.py --file data.csv --output report.pdf
```

## 4. Upload & Generate Report (1 minute)

1. Open web UI
2. Upload financial file (CSV, PDF, DOCX, XLSX, TXT)
3. Customize report options in sidebar
4. Click "Generate Report"
5. Download PDF

## Examples

### Via Web UI
- Upload `financial_data.csv`
- Select "Include Charts" and "Include KPIs"
- Set company branding colors
- Click Generate
- Download as PDF

### Via CLI
```bash
python main.py \
  --file data.csv \
  --output report.pdf \
  --company-name "Acme Corp" \
  --primary-color "#0066cc" \
  --chart-types line,bar
```

### Via Python API
```python
from src.agents.coordinator_agent import CoordinatorAgent

coordinator = CoordinatorAgent()
task = {
    "task_id": "report_001",
    "file_path": "data.csv",
    "include_charts": True,
    "chart_types": ["line", "bar"]
}
result = coordinator.execute(task)
print(result.data)
```

## Troubleshooting

### LLM Connection Failed
```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Or switch backend
export LLM_BACKEND=openai
export LLM_API_KEY=sk-...
```

### Missing Dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use
```bash
# Change Streamlit port
streamlit run src/frontend/streamlit_app.py --server.port 8502
```

### File Format Not Supported
- Supported: CSV, TXT, PDF, DOCX, XLSX
- Ensure file is not corrupted
- Check file permissions

## Performance Tips

- Use Ollama for fastest local inference
- For large files (>10MB), consider cloud API
- Disable "Include Charts" if slow on older machines
- Run tests first: `pytest tests/ -v`

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501

# View logs
docker logs ai-financial-report-gen

# Stop
docker-compose down
```

## Next Steps

1. Test with sample data
2. Customize branding in sidebar
3. Configure LLM backend for your use case
4. Deploy to production (local, Docker, or cloud)
5. Add custom agents if needed

---

**For full documentation, see README.md**
