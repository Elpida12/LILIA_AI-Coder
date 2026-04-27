"""Tool definitions, schemas, and dispatch."""

import inspect
from dataclasses import dataclass


# OpenAI-format tool definitions
_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read contents of a file relative to project root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit an existing file by replacing old_text with new_text. Use exact text from read_file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories in the project.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "default": "."}
            },
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_project",
        "description": "Search for text across all project files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_memory",
        "description": "Update a section of shared project memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "content": {"type": "object"},
            },
            "required": ["section", "content"],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal task completion with a verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "verdict": {"type": "string", "enum": ["pass", "issues", "fail", "complete"]},
                "details": {"type": "object"},
            },
            "required": ["summary", "verdict"],
        },
    },
]

_ROLE_TOOLS = {
    "architect": frozenset({"read_file", "list_files", "search_project", "update_memory", "task_complete"}),
    "implementer": frozenset({"read_file", "write_file", "edit_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
    "verifier": frozenset({"read_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
    "repair": frozenset({"read_file", "write_file", "edit_file", "list_files", "search_project", "run_command", "update_memory", "task_complete"}),
}


class ToolRegistry:
    def __init__(self, fs, shell, git, memory, logger):
        self.fs = fs
        self.shell = shell
        self.git = git
        self.memory = memory
        self.logger = logger

    def get_schemas(self, role: str | None = None) -> list[dict]:
        allowed = _ROLE_TOOLS.get(role)
        tools = _TOOL_DEFINITIONS if allowed is None else [
            t for t in _TOOL_DEFINITIONS if t["name"] in allowed
        ]
        return [{"type": "function", "function": t} for t in tools]

    def execute(self, name: str, arguments: dict, agent_name: str = "unknown") -> str:
        allowed = _ROLE_TOOLS.get(agent_name)
        if allowed is not None and name not in allowed:
            return f"Error: '{agent_name}' cannot use '{name}'. Allowed: {sorted(allowed)}"
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            sig = inspect.signature(handler)
            valid = {k: v for k, v in arguments.items() if k in sig.parameters}
            result = handler(**valid)
            self.logger.tool_call(agent_name, name, arguments, result, success=True)
            return str(result)
        except Exception as exc:
            msg = f"Error: {exc}"
            self.logger.tool_call(agent_name, name, arguments, msg, success=False)
            return msg

    def _tool_read_file(self, path: str) -> str:
        return self.fs.read(path)

    def _tool_write_file(self, path: str, content: str) -> str:
        return self.fs.write(path, content)

    def _tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        return self.fs.edit(path, old_text, new_text)

    def _tool_list_files(self, directory: str = ".") -> str:
        return self.fs.list_files(directory)

    def _tool_run_command(self, command: str, timeout: int = 120) -> str:
        # Flush staged writes so shell commands can see current files
        self.fs.commit()
        return self.shell.run(command, timeout)

    def _tool_search_project(self, query: str) -> str:
        return self.fs.search_project(query)

    def _tool_update_memory(self, section: str, content: dict) -> str:
        result = self.memory.update_section(section, content)
        # Verify persistence (read-back check)
        if not self.memory._path.exists():
            return f"Error: Memory file not found after update at {self.memory._path}"
        return result

    def _tool_task_complete(self, summary: str, verdict: str = "complete", details: dict | None = None) -> str:
        return f"TASK_COMPLETE [{verdict}]: {summary}"
