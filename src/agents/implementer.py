"""Implementer — writes code for a single task."""

from src.agent_base import AgentBase


class ImplementerAgent(AgentBase):
    agent_role = "implementer"
    system_prompt_path = "implementer.txt"
