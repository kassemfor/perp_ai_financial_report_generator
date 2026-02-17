"""
Shared state and memory management for agents.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from datetime import datetime
import json
from pathlib import Path
import threading

@dataclass
class ExecutionState:
    """Tracks execution progress and state."""
    task_id: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: str = ""
    completed_at: str = ""
    agent_sequence: List[str] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

class SharedMemory:
    """Thread-safe shared memory for multi-agent system."""

    def __init__(self):
        self._memory = {}
        self._lock = threading.RLock()
        self._execution_history = []

    def set(self, key: str, value: Any):
        """Store value in memory."""
        with self._lock:
            self._memory[key] = value

    def get(self, key: str, default=None):
        """Retrieve value from memory."""
        with self._lock:
            return self._memory.get(key, default)

    def update(self, key: str, updates: Dict[str, Any]):
        """Update nested dictionary."""
        with self._lock:
            if key not in self._memory:
                self._memory[key] = {}
            self._memory[key].update(updates)

    def append_to_list(self, key: str, item: Any):
        """Append to list in memory."""
        with self._lock:
            if key not in self._memory:
                self._memory[key] = []
            self._memory[key].append(item)

    def log_execution(self, state: ExecutionState):
        """Log execution state."""
        with self._lock:
            state.completed_at = datetime.now().isoformat()
            self._execution_history.append(state.to_dict())

    def get_history(self):
        """Get execution history."""
        with self._lock:
            return list(self._execution_history)

    def clear(self):
        """Clear all memory."""
        with self._lock:
            self._memory.clear()
            self._execution_history.clear()

# Global shared memory instance
MEMORY = SharedMemory()
