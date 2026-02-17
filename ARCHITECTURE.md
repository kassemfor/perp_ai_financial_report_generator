# System Architecture

## Multi-Agent Orchestration

```
User Input (File + Config)
         |
         v
┌─────────────────────────────┐
│   CoordinatorAgent          │
│ (High-level Controller)     │
│ - Manages execution flow    │
│ - Coordinates agents        │
│ - Maintains shared state    │
└─────────────────────────────┘
         |
         ├─────────────────────────────────────────┐
         |                                         |
         v                                         v
    ┌─────────────┐                        ┌──────────────┐
    │PlannerAgent │                        │EnvironmentA  │
    │             │                        │SetupAgent    │
    │- Decompose  │                        │              │
    │- Roadmap    │                        │- Check env   │
    │- Dependencies│                       │- Verify pkgs │
    └─────────────┘                        └──────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 1: Data Ingestion                        │
    ├─────────────────────────────────────────────────┤
    │  BackendAgent                                   │
    │  ├─ FileParser (CSV, PDF, DOCX, XLSX, TXT)     │
    │  ├─ Data Validation                            │
    │  └─ Financial Data Extraction                  │
    └─────────────────────────────────────────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 2: AI Analysis                           │
    ├─────────────────────────────────────────────────┤
    │  AIAgent (powered by LLMInterface)              │
    │  ├─ Summarization                              │
    │  ├─ Insight Extraction                         │
    │  ├─ Outline Generation                         │
    │  └─ Keyword Extraction                         │
    └─────────────────────────────────────────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 3: Visualization                         │
    ├─────────────────────────────────────────────────┤
    │  VisualizationAgent                             │
    │  ├─ Line Charts (trends)                        │
    │  ├─ Bar Charts (comparisons)                    │
    │  ├─ Pie Charts (composition)                    │
    │  └─ KPI Cards                                  │
    └─────────────────────────────────────────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 4: Report Assembly                       │
    ├─────────────────────────────────────────────────┤
    │  BackendAgent (PDF Generation)                  │
    │  ├─ ReportLab PDF creation                      │
    │  ├─ Branding application                        │
    │  └─ Output validation                           │
    └─────────────────────────────────────────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 5: Testing & Validation                  │
    ├─────────────────────────────────────────────────┤
    │  TestingAgent                                   │
    │  ├─ Unit tests                                 │
    │  ├─ Integration tests                          │
    │  └─ Output validation                          │
    └─────────────────────────────────────────────────┘
         |
         v
    ┌─────────────────────────────────────────────────┐
    │  Phase 6: Error Handling                        │
    ├─────────────────────────────────────────────────┤
    │  DebuggingAgent                                 │
    │  ├─ Error detection                            │
    │  ├─ Root cause analysis                        │
    │  └─ Fix suggestions                            │
    └─────────────────────────────────────────────────┘
         |
         v
    Professional PDF Report + JSON Metadata

```

## Data Flow

```
Raw Input Files
    ↓
File Parser (multi-format support)
    ↓
Parsed Data + Metadata
    ↓
Shared Memory (thread-safe)
    ↓
AI Processing (via LLM)
    ↓
Summary + Insights + Outline
    ↓
Visualization Generation
    ↓
Chart Assets (base64 encoded)
    ↓
PDF Assembly (ReportLab)
    ↓
PDF Output + Metadata
```

## LLM Interface Architecture

```
┌──────────────────────────────────────┐
│    LLMInterface (Unified)            │
└──────────────────────────────────────┘
         |
    ┌────┼────┬────────┬──────────┐
    |    |    |        |          |
    v    v    v        v          v
 OpenAI Gemini Ollama LMStudio Anthropic
 (Cloud) (Cloud) (Local) (Local)  (Cloud)
```

## Shared State Management

```
┌──────────────────────────────────┐
│    SharedMemory (Thread-safe)    │
├──────────────────────────────────┤
│ _memory (dict)                   │
│ ├─ agent_data                    │
│ ├─ execution_state               │
│ └─ reports                       │
│                                  │
│ _execution_history (list)        │
│ ├─ Task 1: {...}                │
│ ├─ Task 2: {...}                │
│ └─ ...                          │
│                                  │
│ Thread locks for safety          │
└──────────────────────────────────┘
```

## Agent Communication Pattern

```
CoordinatorAgent
    |
    +→ Agent.execute(task)
         |
         +→ log_step()
         +→ save_to_state()
         +→ get_from_state()
         |
         └→ return AgentResult
             (success, data, error, duration)
    |
    └→ next_agent...

All agents share same MEMORY object
for inter-agent communication
```

## Configuration Hierarchy

```
.env file
    ↓
os.environ
    ↓
CONFIG.from_env()
    ↓
LLMConfig, BrandingConfig, ReportConfig, FileParsingConfig
    ↓
SystemConfig (aggregated)
    ↓
Used by: LLMInterface, FileParser, VisualizationAgent, BackendAgent
```

## Deployment Architecture

### Local Deployment
```
Machine
├─ Python 3.8+
├─ Virtual Environment
├─ Dependencies (pip)
├─ Ollama or API key
└─ Streamlit Web UI (localhost:8501)
```

### Docker Deployment
```
docker-compose.yml
├─ ai-financial-report-gen service
│  └─ Streamlit (port 8501)
└─ ollama service
   └─ LLM inference (port 11434)
```

### Cloud Deployment
```
Container Registry
    ↓
Cloud Platform (AWS/GCP/Azure)
    ↓
Load Balancer
    ↓
App Replicas (auto-scaling)
```

## Error Handling Flow

```
Agent Execution
    |
    +→ Exception occurs
         |
         +→ Caught in try/except
         |
         +→ Logged to MEMORY
         |
         +→ DebuggingAgent.execute()
              |
              +→ Error classification
              |
              +→ Suggestion generation
              |
              +→ Retry or manual intervention
```

## Performance Profile

| Component | Time | Memory |
|-----------|------|--------|
| File parsing | 1-2s | 50MB |
| LLM call (5-10s avg) | 5-10s | 200MB |
| Visualization | 2-3s | 100MB |
| PDF generation | 2-5s | 150MB |
| **Total** | **15-30s** | **500MB** |

---

**For detailed implementation, see source code in src/agents/**
