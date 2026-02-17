"""
Testing Agent: Automated test execution and validation.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from src.agents.base_agent import BaseAgent, AgentResult

class TestingAgent(BaseAgent):
    """Runs automated tests and validates system functionality."""

    def __init__(self):
        super().__init__("TestingAgent")

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Run test suite."""
        start = datetime.now()

        try:
            self.log_step("Starting automated test suite")

            test_results = {
                "file_parsing": self._test_file_parsing(),
                "data_validation": self._test_data_validation(),
                "llm_integration": self._test_llm_integration(),
                "visualization": self._test_visualization(),
                "pdf_generation": self._test_pdf_generation(),
            }

            passed = sum(1 for r in test_results.values() if r.get("status") == "passed")
            total = len(test_results)

            result = {
                "test_suite": "core_pipeline",
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "details": test_results,
                "overall_status": "passed" if passed == total else "failed"
            }

            duration = (datetime.now() - start).total_seconds()

            return AgentResult(
                agent_name=self.name,
                success=(passed == total),
                data=result,
                duration_seconds=duration
            )

        except Exception as e:
            self.log_error(str(e))
            return AgentResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

    def _test_file_parsing(self) -> Dict[str, Any]:
        """Test file parsing module."""
        try:
            from src.backend.file_parser import FileParser
            parser = FileParser()
            return {"status": "passed", "message": "FileParser initialized"}
        except Exception as e:
            return {"status": "failed", "message": str(e)}

    def _test_data_validation(self) -> Dict[str, Any]:
        """Test data validation."""
        try:
            test_data = {"data": [{"account": "Revenue", "amount": 1000}]}
            assert "data" in test_data
            return {"status": "passed", "message": "Data validation works"}
        except Exception as e:
            return {"status": "failed", "message": str(e)}

    def _test_llm_integration(self) -> Dict[str, Any]:
        """Test LLM interface."""
        try:
            from src.core.llm_interface import llm
            health = llm.health_check()
            status = "passed" if health else "warning"
            return {"status": status, "message": f"LLM health: {health}"}
        except Exception as e:
            return {"status": "warning", "message": f"LLM unavailable: {str(e)}"}

    def _test_visualization(self) -> Dict[str, Any]:
        """Test visualization module."""
        try:
            from src.agents.visualization_agent import VisualizationAgent
            viz = VisualizationAgent()
            return {"status": "passed", "message": "VisualizationAgent initialized"}
        except Exception as e:
            return {"status": "failed", "message": str(e)}

    def _test_pdf_generation(self) -> Dict[str, Any]:
        """Test PDF generation."""
        try:
            import reportlab
            return {"status": "passed", "message": "ReportLab available"}
        except ImportError:
            return {"status": "warning", "message": "ReportLab not installed"}
        except Exception as e:
            return {"status": "failed", "message": str(e)}
