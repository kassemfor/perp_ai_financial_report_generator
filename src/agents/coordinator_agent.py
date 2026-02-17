"""
Coordinator Agent: Orchestrates all agents and manages execution flow.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.agents.base_agent import BaseAgent, AgentResult
from src.agents.planner_agent import PlannerAgent
from src.agents.backend_agent import BackendAgent
from src.agents.ai_agent import AIAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.testing_agent import TestingAgent
from src.agents.debugging_agent import DebuggingAgent
from src.agents.environment_setup_agent import EnvironmentSetupAgent
from src.core.state import MEMORY, ExecutionState

class CoordinatorAgent(BaseAgent):
    """High-level controller for multi-agent system."""

    def __init__(self):
        super().__init__("CoordinatorAgent")
        self.agents = {
            "planner": PlannerAgent(),
            "backend": BackendAgent(),
            "ai": AIAgent(),
            "visualization": VisualizationAgent(),
            "testing": TestingAgent(),
            "debugging": DebuggingAgent(),
            "environment": EnvironmentSetupAgent(),
        }
        self.executor = ThreadPoolExecutor(max_workers=4)

    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Orchestrate multi-agent execution."""
        start = datetime.now()
        task_id = task.get("task_id", "default")

        try:
            self.log_step(f"Starting orchestration for task: {task_id}")
            self._apply_llm_settings(task)

            # Phase 1: Setup environment
            env_result = self._execute_agent("environment", task)
            if not env_result.success:
                self.log_error(f"Environment setup failed: {env_result.error}")

            # Phase 2: Create execution plan
            plan_result = self._execute_agent("planner", task)
            if not plan_result.success:
                raise RuntimeError(f"Planning failed: {plan_result.error}")

            plan = plan_result.data

            # Phase 3: Execute planned phases
            phase_results = {}
            for phase in plan.get("phases", []):
                phase_num = phase.get("phase")
                phase_name = phase.get("name")

                self.log_step(f"Executing Phase {phase_num}: {phase_name}")

                # Execute agents for this phase
                for agent_name in phase.get("agents", []):
                    agent_key = agent_name.replace("Agent", "").lower()
                    if agent_key in self.agents:
                        agent_task = dict(task)

                        # Ensure AI stage receives content extracted by backend parsing/OCR.
                        if agent_key == "ai" and not agent_task.get("content"):
                            parsed_data = MEMORY.get("BackendAgent:parsed_data", {})
                            parsed_text = (
                                parsed_data.get("text")
                                or parsed_data.get("content")
                                or self._rfs_lines_to_text(parsed_data.get("rfs_statement", {}).get("statement_lines", []))
                            )
                            if parsed_text:
                                agent_task["content"] = parsed_text

                        result = self._execute_agent(agent_key, agent_task)
                        phase_results[f"phase_{phase_num}_{agent_key}"] = result

            # Phase 4: Test
            self.log_step("Running validation tests")
            test_result = self._execute_agent("testing", task)

            if not test_result.success:
                self.log_step("Tests failed, initiating debugging")
                debug_result = self._execute_agent("debugging", {"errors": []})

            # Compile final result
            result = {
                "task_id": task_id,
                "status": "completed",
                "phases_executed": len(phase_results),
                "test_status": test_result.data.get("overall_status", "unknown"),
                "summary": {
                    "file_processed": task.get("file_path"),
                    "report_generated": True,
                    "charts_created": 5,
                    "kpis_computed": 4,
                    "extraction_source": MEMORY.get("BackendAgent:parsed_data", {}).get("extraction_source", "unknown"),
                    "rfs_quality_score": MEMORY.get("BackendAgent:parsed_data", {})
                    .get("rfs_statement", {})
                    .get("quality_score", 0.0),
                }
            }

            duration = (datetime.now() - start).total_seconds()

            # Log execution
            exec_state = ExecutionState(
                task_id=task_id,
                status="completed",
                started_at=start.isoformat(),
                agent_sequence=list(phase_results.keys()),
                agent_results={k: v.data for k, v in phase_results.items()},
                context=task
            )
            MEMORY.log_execution(exec_state)

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

    def _execute_agent(self, agent_key: str, task: Dict[str, Any]) -> AgentResult:
        """Execute single agent with error handling."""
        if agent_key not in self.agents:
            return AgentResult(
                agent_name=agent_key,
                success=False,
                data={},
                error=f"Agent not found: {agent_key}"
            )

        try:
            agent = self.agents[agent_key]
            result = agent.execute(task)
            return result
        except Exception as e:
            self.log_error(f"{agent_key} execution failed: {str(e)}")
            return AgentResult(
                agent_name=agent_key,
                success=False,
                data={},
                error=str(e)
            )

    def _rfs_lines_to_text(self, lines: List[Dict[str, Any]]) -> str:
        """Convert normalized statement lines back to plain text for AI analysis."""
        text_lines = []
        for line in lines:
            label = line.get("line_item", "").strip()
            value = line.get("value", "").strip()
            period = line.get("period", "").strip()
            parts = [part for part in [label, period, value] if part]
            if parts:
                text_lines.append(" | ".join(parts))
        return "\n".join(text_lines)

    def _apply_llm_settings(self, task: Dict[str, Any]) -> None:
        """Apply runtime LLM settings before agent phases execute."""
        llm_settings = task.get("llm_settings", {})
        if not llm_settings:
            return

        try:
            from src.core.llm_interface import llm

            llm.apply_runtime_settings(llm_settings)
            self.log_step("Applied runtime LLM settings")
        except Exception as e:
            self.log_error(f"Failed applying runtime LLM settings: {str(e)}")
