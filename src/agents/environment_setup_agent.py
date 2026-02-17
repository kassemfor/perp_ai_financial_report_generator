"""
Environment Setup Agent: Dependencies and configuration.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from src.agents.base_agent import BaseAgent, AgentResult

class EnvironmentSetupAgent(BaseAgent):
    """Sets up runtime environment."""

    def __init__(self):
        super().__init__("EnvironmentSetupAgent")

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Setup environment."""
        start = datetime.now()

        try:
            self.log_step("Checking environment")

            # Verify Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            self.log_step(f"Python {python_version}")

            # Check required packages
            packages = {
                "pandas": "pandas",
                "numpy": "numpy",
                "matplotlib": "matplotlib",
                "plotly": "plotly",
                "reportlab": "reportlab",
                "python-docx": "docx",
                "pypdf": "pypdf",
                "streamlit": "streamlit",
                "requests": "requests",
                "pytesseract": "pytesseract",
                "pypdfium2": "pypdfium2",
            }

            installed = []
            missing = []

            for package_name, module_name in packages.items():
                try:
                    __import__(module_name)
                    installed.append(package_name)
                except ImportError:
                    missing.append(package_name)

            result = {
                "python_version": python_version,
                "installed_packages": len(installed),
                "missing_packages": missing,
                "environment_status": "ready" if not missing else "partial"
            }

            duration = (datetime.now() - start).total_seconds()

            return AgentResult(
                agent_name=self.name,
                success=True,
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
