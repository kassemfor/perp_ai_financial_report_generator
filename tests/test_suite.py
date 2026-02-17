"""
Automated test suite - pytest based.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestConfiguration:
    """Test configuration loading."""

    def test_config_loading(self):
        from src.config import CONFIG
        assert CONFIG is not None
        assert CONFIG.llm is not None

    def test_llm_backends(self):
        from src.config import LLMBackend
        assert LLMBackend.OPENAI.value == "openai"
        assert LLMBackend.OLLAMA.value == "ollama"

class TestFileParser:
    """Test file parsing."""

    def test_parser_initialization(self):
        from src.backend.file_parser import FileParser
        parser = FileParser()
        assert "csv" in parser.supported_formats
        assert "pdf" in parser.supported_formats
        assert "xlsx" in parser.supported_formats
        assert "png" in parser.supported_formats

class TestLLMInterface:
    """Test LLM interface."""

    def test_llm_initialization(self):
        from src.core.llm_interface import llm
        assert llm is not None
        assert llm.backend is not None

class TestAgents:
    """Test agent initialization."""

    def test_planner_agent(self):
        from src.agents.planner_agent import PlannerAgent
        agent = PlannerAgent()
        assert agent.name == "PlannerAgent"

    def test_backend_agent(self):
        from src.agents.backend_agent import BackendAgent
        agent = BackendAgent()
        assert agent.name == "BackendAgent"

    def test_ai_agent(self):
        from src.agents.ai_agent import AIAgent
        agent = AIAgent()
        assert agent.name == "AIAgent"

    def test_visualization_agent(self):
        from src.agents.visualization_agent import VisualizationAgent
        agent = VisualizationAgent()
        assert agent.name == "VisualizationAgent"

    def test_coordinator_agent(self):
        from src.agents.coordinator_agent import CoordinatorAgent
        coordinator = CoordinatorAgent()
        assert coordinator.name == "CoordinatorAgent"
        assert "planner" in coordinator.agents

class TestSharedMemory:
    """Test shared state management."""

    def test_memory_operations(self):
        from src.core.state import SharedMemory
        mem = SharedMemory()
        mem.set("test_key", "test_value")
        assert mem.get("test_key") == "test_value"

    def test_memory_list_operations(self):
        from src.core.state import SharedMemory
        mem = SharedMemory()
        mem.append_to_list("test_list", "item1")
        mem.append_to_list("test_list", "item2")
        assert len(mem.get("test_list", [])) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
