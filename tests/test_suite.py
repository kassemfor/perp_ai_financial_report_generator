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

    def test_ifrs_specific_code_classification(self):
        from src.backend.file_parser import FileParser

        parser = FileParser()
        assert parser._classify_rfs_code("Right-of-use asset") == "ROU_ASSET"
        assert parser._classify_rfs_code("Lease liability") == "LEASE_LIABILITY"
        assert parser._classify_rfs_code("Deferred revenue") == "DEFERRED_REVENUE"
        assert parser._classify_rfs_code("Expected credit loss allowance") == "ECL_PROVISION"

    def test_rfs_statement_includes_ifrs_tags(self):
        from src.backend.file_parser import FileParser

        parser = FileParser()
        text = (
            "Revenue 2024 1200\n"
            "Deferred revenue 2024 300\n"
            "Lease liability 2024 180\n"
            "Right-of-use asset 2024 170\n"
            "Revenue 2023 1100\n"
        )
        statement = parser._build_rfs_statement(text=text, source="plain_text", confidence=0.92)

        assert statement["schema_version"] == "1.2"
        assert statement["summary"]["line_items_detected"] >= 4
        assert statement["summary"]["comparative_periods_detected"] >= 2
        standards = statement["document_profile"].get("recognized_ifrs_standards", [])
        assert "IFRS 15" in standards
        assert "IFRS 16" in standards

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


class TestFinancialStandardsKnowledge:
    """Test packaged standards knowledge."""

    def test_standards_bundle_contains_ifrs_map(self):
        from src.utils.financial_standards import standards_bundle

        bundle = standards_bundle()
        assert bundle["ifrs_standards_map"]
        assert any("IFRS 18" in row.get("standard", "") for row in bundle["ifrs_standards_map"])
        assert bundle["industry_kpi_rows"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
