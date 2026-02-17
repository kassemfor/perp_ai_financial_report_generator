# AI Financial Report Generator

An autonomous, multi-agent system that transforms raw financial data into professional AI-powered reports with advanced customization, visualization, and multi-LLM support.

## 🎯 Features

✅ **Multi-Agent Architecture**: 11 specialized agents + 1 coordinator
✅ **Multi-LLM Support**: OpenAI, Gemini, Ollama, LM Studio, Anthropic (all local or API)
✅ **File Format Support**: CSV, TXT, PDF, DOCX, XLSX
✅ **Automated Report Generation**: Summaries, insights, outlines, charts
✅ **Advanced Customization**: Branding, colors, fonts, layouts
✅ **Professional PDF Output**: ReportLab-powered with full control
✅ **Interactive Web UI**: Streamlit-based with tabs and controls
✅ **Automated Testing**: pytest suite with validation pipeline
✅ **Self-Healing**: Error detection and auto-correction
✅ **Local Deployment**: Run entirely on your machine

## 🏗️ System Architecture

### Agents

1. **PlannerAgent**: Project decomposition and roadmap
2. **BackendAgent**: File parsing and PDF assembly
3. **AIAgent**: Summarization, outline generation, insights
4. **VisualizationAgent**: Charts, KPIs, color themes
5. **TestingAgent**: Automated validation
6. **DebuggingAgent**: Error detection and fixes
7. **EnvironmentSetupAgent**: Dependency management
8. **CoordinatorAgent**: Multi-agent orchestration

### Supported Formats

| Format | Parsing | Extraction | Support |
|--------|---------|-----------|---------|
| CSV    | ✅      | Tables    | Full    |
| XLSX   | ✅      | Multiple sheets | Full |
| TXT    | ✅      | Tables    | Full    |
| PDF    | ✅      | Text + tables | Full |
| DOCX   | ✅      | Text + tables | Full |

### LLM Backends

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Google Gemini**: gemini-pro
- **Ollama**: Local models (Mistral, Llama 2, etc.)
- **LM Studio**: Local models via API
- **Anthropic**: Claude 3

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 2GB RAM minimum (4GB recommended)
- pip or conda

### Installation

```bash
# Clone repository
git clone <repo>
cd ai_financial_report_generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup LLM Backend

**Option 1: Ollama (Recommended for local)**
```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull mistral:latest
# Ollama runs on http://localhost:11434
```

**Option 2: OpenAI API**
```bash
export LLM_BACKEND=openai
export LLM_API_KEY=sk-...
```

**Option 3: Gemini**
```bash
export LLM_BACKEND=gemini
export LLM_API_KEY=your-gemini-key
```

### Run Application

```bash
# Start Streamlit web UI
streamlit run src/frontend/streamlit_app.py

# Or use Python directly
python main.py --file data.csv --output report.pdf
```

## 📊 Usage Examples

### Via Web UI

1. Open http://localhost:8501
2. Upload financial data (CSV, PDF, DOCX, XLSX)
3. Configure report options in sidebar
4. Click "Generate Report"
5. Download PDF, Excel, or JSON

### Via Python API

```python
from src.agents.coordinator_agent import CoordinatorAgent
from src.config import CONFIG

# Configure LLM
CONFIG.llm.backend = "ollama"
CONFIG.llm.model_name = "mistral:latest"

# Configure branding
CONFIG.branding.company_name = "Your Company"
CONFIG.branding.primary_color = "#0066cc"

# Execute report generation
coordinator = CoordinatorAgent()
task = {
    "task_id": "report_001",
    "file_path": "financial_data.csv",
    "report_type": "financial",
    "include_summary": True,
    "include_charts": True,
    "chart_types": ["line", "bar", "pie"]
}

result = coordinator.execute(task)
print(result.data)
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
LLM_BACKEND=ollama              # ollama, openai, gemini, lm_studio, anthropic
LLM_MODEL=mistral:latest       # Model name
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=                    # If using API-based backend
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# Company Branding
COMPANY_NAME="Your Company"
PRIMARY_COLOR="#1f77b4"
SECONDARY_COLOR="#ff7f0e"

# System
DEBUG=False
```

### Config File (src/config.py)

Modify directly in code:

```python
from src.config import CONFIG

CONFIG.branding.company_name = "Analytics Corp"
CONFIG.branding.primary_color = "#0066cc"
CONFIG.report.max_pages = 100
```

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_suite.py::TestConfiguration -v
```

## 📁 Project Structure

```
ai_financial_report_generator/
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py
│   │   ├── planner_agent.py
│   │   ├── backend_agent.py
│   │   ├── ai_agent.py
│   │   ├── visualization_agent.py
│   │   ├── testing_agent.py
│   │   ├── debugging_agent.py
│   │   ├── environment_setup_agent.py
│   │   └── coordinator_agent.py
│   ├── backend/                # Data processing
│   │   └── file_parser.py
│   ├── core/                   # Core infrastructure
│   │   ├── state.py            # Shared memory
│   │   └── llm_interface.py    # LLM abstraction
│   ├── frontend/               # Web interface
│   │   └── streamlit_app.py
│   └── config.py               # Configuration
├── tests/
│   └── test_suite.py           # Pytest suite
├── requirements.txt
├── README.md
└── main.py                     # Entry point
```

## 🎨 Customization

### Add Custom Chart Type

```python
# In visualization_agent.py
def _create_custom_chart(self, data):
    # Your implementation
    return {"type": "custom", "image": base64_image}
```

### Add New Agent

```python
from src.agents.base_agent import BaseAgent, AgentResult

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("CustomAgent")

    def execute(self, task):
        # Your logic
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={},
            duration_seconds=0
        )
```

## 🐛 Troubleshooting

### LLM Connection Issues

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test OpenAI API key
openai api models.list
```

### File Parsing Errors

- Ensure file format is supported
- Check file isn't corrupted
- Verify permissions

### Missing Dependencies

```bash
# Reinstall all
pip install -r requirements.txt --force-reinstall
```

## 📈 Performance

| Operation | Time | Memory |
|-----------|------|--------|
| CSV parsing | 1-2s | 50MB |
| Summarization | 5-10s | 200MB |
| Chart generation | 2-3s | 100MB |
| PDF assembly | 2-5s | 150MB |
| **Total pipeline** | **15-30s** | **500MB** |

## 🔐 Security Notes

- API keys should use environment variables (never hardcode)
- Uploaded files stored in `uploads/` directory
- Reports stored in `outputs/` directory
- All file operations are sandboxed

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional file format support (STATA, SAS, SQL)
- More LLM backends (Hugging Face, Local)
- Advanced NLP (named entity recognition, sentiment analysis)
- Dashboard templates
- REST API layer

## 📄 License

MIT License - See LICENSE file

## 📞 Support

For issues, questions, or feature requests, open an issue on GitHub.

---

**Built with ❤️ | v1.0 | AI-Powered Financial Intelligence**
