"""Structured JSONL + plain console logger."""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class Logger:
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    def __init__(self, logs_dir: str, level: str = "INFO", log_to_terminal: bool = True):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.level = self.LEVELS.get(level.upper(), 1)
        self.log_to_terminal = log_to_terminal
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.logs_dir / f"session_{ts}.jsonl"
        self.session_log.touch()

    def _write(self, event_type: str, **data: Any):
        entry = {"timestamp": datetime.now().isoformat(), "event": event_type, **data}
        with open(self.session_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def _should_log(self, level: str) -> bool:
        return self.LEVELS.get(level.upper(), 1) >= self.level

    def _print(self, level: str, msg: str):
        if not self.log_to_terminal:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        stream = sys.stderr if level == "ERROR" else sys.stdout
        print(f"[{ts}] [{level}] {msg}", file=stream)

    def debug(self, msg: str, **data: Any):
        if self._should_log("DEBUG"):
            self._write("debug", message=msg, **data)
            self._print("DEBUG", msg)

    def info(self, msg: str, **data: Any):
        if self._should_log("INFO"):
            self._write("info", message=msg, **data)
            self._print("INFO", msg)

    def warning(self, msg: str, **data: Any):
        if self._should_log("WARNING"):
            self._write("warning", message=msg, **data)
            self._print("WARNING", msg)

    def error(self, msg: str, exception: Exception | None = None, **data: Any):
        if self._should_log("ERROR"):
            entry = {"message": msg, **data}
            if exception:
                entry["exception"] = str(exception)
                entry["traceback"] = traceback.format_exc()
            self._write("error", **entry)
            self._print("ERROR", msg)

    def llm_call(self, agent: str, messages: list, response: str, thinking: str = "",
                 tokens_prompt: int = 0, tokens_completion: int = 0, duration: float = 0.0,
                 finish_reason: str = ""):
        self._write(
            "llm_call",
            agent=agent,
            message_count=len(messages),
            response_preview=response[:500],
            thinking_preview=thinking[:500],
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            duration_seconds=round(duration, 2),
            finish_reason=finish_reason,
        )
        self._print("INFO", f"LLM [{agent}] {tokens_prompt}+{tokens_completion} tok, {duration:.1f}s, finish={finish_reason}")

    def tool_call(self, agent: str, tool_name: str, arguments: dict, result: str, success: bool = True):
        self._write(
            "tool_call",
            agent=agent,
            tool=tool_name,
            arguments=arguments,
            result_preview=result[:500],
            success=success,
        )
        status = "OK" if success else "FAIL"
        self._print("INFO", f"Tool [{agent}] {tool_name}() -> {status}")

    def rule(self, title: str, char: str = "="):
        self._write("rule", title=title)
        if self.log_to_terminal:
            print(f"\n{char * 60}\n  {title}\n{char * 60}")
