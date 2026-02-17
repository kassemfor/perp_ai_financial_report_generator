"""
System configuration: LLM backends, branding, defaults.
"""

import os
from typing import Literal
from dataclasses import dataclass, field
from enum import Enum

class LLMBackend(Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    ANTHROPIC = "anthropic"

@dataclass
class LLMConfig:
    """LLM configuration."""
    backend: LLMBackend = LLMBackend.OLLAMA
    model_name: str = "mistral:latest"
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60

    @classmethod
    def from_env(cls):
        """Load from environment variables."""
        backend_str = os.getenv("LLM_BACKEND", "ollama").lower()
        backend = LLMBackend[backend_str.upper()] if backend_str.upper() in LLMBackend.__members__ else LLMBackend.OLLAMA

        config_map = {
            LLMBackend.OPENAI: {"model_name": "gpt-4", "base_url": "https://api.openai.com/v1"},
            LLMBackend.GEMINI: {"model_name": "gemini-pro", "base_url": "https://generativelanguage.googleapis.com"},
            LLMBackend.OLLAMA: {"model_name": "mistral:latest", "base_url": "http://localhost:11434"},
            LLMBackend.LM_STUDIO: {"model_name": "local-model", "base_url": "http://localhost:1234"},
            LLMBackend.ANTHROPIC: {"model_name": "claude-3-opus", "base_url": "https://api.anthropic.com"},
        }

        defaults = config_map.get(backend, {})
        return cls(
            backend=backend,
            model_name=os.getenv("LLM_MODEL", defaults.get("model_name")),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", defaults.get("base_url")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

@dataclass
class BrandingConfig:
    """PDF branding configuration."""
    company_name: str = "Financial Analytics Corp"
    logo_path: str = ""
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    accent_color: str = "#2ca02c"
    font_name: str = "Helvetica"
    footer_text: str = "Confidential - AI Generated Report"
    include_watermark: bool = True

@dataclass
class ReportConfig:
    """Report generation configuration."""
    include_summary: bool = True
    include_charts: bool = True
    include_kpis: bool = True
    include_trends: bool = True
    chart_types: list = field(default_factory=lambda: ["line", "bar", "pie"])
    max_pages: int = 50
    include_raw_data_appendix: bool = True

@dataclass
class FileParsingConfig:
    """File parsing configuration."""
    supported_formats: list = field(
        default_factory=lambda: ["csv", "txt", "pdf", "docx", "xlsx", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]
    )
    max_file_size_mb: int = 100
    extract_tables: bool = True
    ocr_enabled: bool = False

@dataclass
class SystemConfig:
    """Master system configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    file_parsing: FileParsingConfig = field(default_factory=FileParsingConfig)

    # Paths
    upload_dir: str = "uploads"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # Runtime
    debug_mode: bool = bool(os.getenv("DEBUG", "False").lower() == "true")
    max_workers: int = 4

    @classmethod
    def from_env(cls):
        """Load complete config from environment."""
        return cls(
            llm=LLMConfig.from_env(),
            branding=BrandingConfig(
                company_name=os.getenv("COMPANY_NAME", "Financial Analytics Corp"),
                primary_color=os.getenv("PRIMARY_COLOR", "#1f77b4"),
                secondary_color=os.getenv("SECONDARY_COLOR", "#ff7f0e"),
            ),
        )

# Global config instance
CONFIG = SystemConfig.from_env()
