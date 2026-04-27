"""Repair — fixes bugs found by Verifier or user."""

from src.agent_base import AgentBase


class RepairAgent(AgentBase):
    agent_role = "repair"
    system_prompt_path = "repair.txt"
