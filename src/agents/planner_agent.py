"""
Planner Agent: Breaks project into phases and creates execution roadmap.
"""

import json
from typing import Dict, Any
from datetime import datetime
from src.agents.base_agent import BaseAgent, AgentResult
from src.core.llm_interface import llm

class PlannerAgent(BaseAgent):
    """Plans project execution and manages task sequence."""

    def __init__(self):
        super().__init__("PlannerAgent")

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Create execution plan for financial report generation."""
        start = datetime.now()

        try:
            self.log_step("Analyzing project requirements")

            # Determine project phases
            file_format = task.get("file_format", "mixed")
            report_type = task.get("report_type", "financial")

            plan = {
                "project_id": task.get("task_id"),
                "phases": [
                    {
                        "phase": 1,
                        "name": "Data Ingestion",
                        "agents": ["BackendAgent"],
                        "tasks": ["Parse file", "Validate data", "Extract tables"],
                        "duration_est": "5 min"
                    },
                    {
                        "phase": 2,
                        "name": "Content Analysis",
                        "agents": ["AIAgent"],
                        "tasks": ["Summarize content", "Extract insights", "Generate outline"],
                        "duration_est": "10 min"
                    },
                    {
                        "phase": 3,
                        "name": "Visualization",
                        "agents": ["VisualizationAgent"],
                        "tasks": ["Create charts", "Generate KPI visualizations"],
                        "duration_est": "8 min"
                    },
                    {
                        "phase": 4,
                        "name": "Report Generation",
                        "agents": ["BackendAgent"],
                        "tasks": ["Assemble PDF", "Apply branding", "Validate output"],
                        "duration_est": "5 min"
                    }
                ],
                "dependencies": {
                    "phase_2": ["phase_1"],
                    "phase_3": ["phase_1", "phase_2"],
                    "phase_4": ["phase_2", "phase_3"]
                },
                "resource_allocation": {
                    "max_workers": 4,
                    "memory_requirement_mb": 512
                }
            }

            self.save_to_state("execution_plan", plan)
            self.log_step(f"Created plan with {len(plan['phases'])} phases")

            duration = (datetime.now() - start).total_seconds()

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=plan,
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
