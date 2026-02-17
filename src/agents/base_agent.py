"""
Base agent class for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from src.core.state import MEMORY, ExecutionState

@dataclass
class AgentResult:
    """Result from agent execution."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float = 0

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.state = MEMORY

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute the agent's primary task."""
        pass

    def log_step(self, message: str):
        """Log execution step."""
        self.logger.info(f"[{self.name}] {message}")

    def log_error(self, message: str):
        """Log error."""
        self.logger.error(f"[{self.name}] {message}")

    def save_to_state(self, key: str, value: Any):
        """Save data to shared state."""
        self.state.set(f"{self.name}:{key}", value)

    def get_from_state(self, key: str, default=None):
        """Get data from shared state."""
        return self.state.get(f"{self.name}:{key}", default)
