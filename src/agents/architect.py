"""Architect — decomposes user request into a design + task list."""

import json
import re
from src.agent_base import AgentBase

try:
    from pydantic import BaseModel, Field, ValidationError
    _HAS_PYDANTIC = True
except Exception:
    _HAS_PYDANTIC = False
    BaseModel = object
    def Field(*a, **kw):
        return None


if _HAS_PYDANTIC:
    class TaskItem(BaseModel):
        id: str = Field(default="")
        description: str = Field(default="")
        type: str = Field(default="create")
        target_files: list[str] = Field(default_factory=list)
        dependencies: list[str] = Field(default_factory=list)
        acceptance: str = Field(default="")

    class DesignOutput(BaseModel):
        design: dict = Field(default_factory=dict)
        tasks: list[TaskItem] = Field(default_factory=list)


class ArchitectAgent(AgentBase):
    agent_role = "architect"
    system_prompt_path = "architect.txt"

    async def run(self, task: dict) -> dict:
        result = await super().run(task)
        if result["status"] != "complete":
            return result

        design, tasks = self._parse_output(result["result"])
        result["design"] = design
        result["tasks"] = tasks
        self.logger.info(f"[{self.agent_role}] Parsed {len(tasks)} tasks from design")
        return result

    def _parse_output(self, text: str) -> tuple[dict, list]:
        data = self._extract_any_json(text)

        # Pydantic validation path
        if _HAS_PYDANTIC and isinstance(data, dict):
            try:
                validated = DesignOutput(**data)
                return validated.design.model_dump(), [t.model_dump() for t in validated.tasks]
            except ValidationError as exc:
                self.logger.warning(f"Pydantic validation failed: {exc}")
            except Exception:
                pass

        # Manual fallback
        design = data.get("design", {}) if isinstance(data, dict) else {}
        tasks = self._normalize_tasks(
            data.get("tasks", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        )
        return design, tasks

    def _extract_any_json(self, text: str) -> dict | list:
        # 1. fenced block
        m = re.search(r"```(?:json)?\s*\n([{\[][^`]*[}\]])\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 2. any object / array
        for pat in (r"(\{.*\})", r"(\[.*\])"):
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        # 3. whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        return {}

    def _normalize_tasks(self, raw: list) -> list[dict]:
        out = []
        for i, t in enumerate(raw):
            if isinstance(t, dict):
                out.append({
                    "id": t.get("id") or f"task_{i+1}",
                    "description": t.get("description", ""),
                    "type": t.get("type", "create"),
                    "target_files": t.get("target_files", []),
                    "dependencies": t.get("dependencies", []),
                    "acceptance": t.get("acceptance", ""),
                })
        return out

    def _parse_numbered_tasks(self, text: str) -> list[dict]:
        pattern = re.compile(r"(?:^|\n)\s*(?:\d+[.)]\s+|[-*]\s+)(.+?)(?=\n\s*(?:\d+[.)]\s+|[-*]\s+)|\Z)", re.DOTALL)
        matches = pattern.findall(text)
        tasks = []
        for i, desc in enumerate(matches):
            desc = desc.strip()
            if desc:
                tasks.append({
                    "id": f"task_{i+1}",
                    "description": desc,
                    "type": "create",
                    "target_files": [],
                    "dependencies": [f"task_{i}"] if i > 0 else [],
                    "acceptance": "",
                })
        return tasks
