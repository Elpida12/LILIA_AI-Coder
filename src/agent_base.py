"""Base agent - ReAct loop with native tool calling only."""

import asyncio
import json
from pathlib import Path


class AgentBase:
    """Think -> Tool call(s) -> Observe -> Repeat -> task_complete."""

    agent_role: str = "base"
    system_prompt_path: str = "base.txt"

    def __init__(self, llm, tools, memory, logger, prompts_dir: str,
                 max_iterations: int = 15, thinking_budget: int = 2048):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.max_iterations = max_iterations
        self.thinking_budget = thinking_budget
        self._consecutive_truncations = 0  # Fix #1E: Track truncation streaks

        prompt_file = Path(prompts_dir) / self.system_prompt_path
        if prompt_file.exists():
            self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()
        else:
            self.system_prompt = f"You are the {self.agent_role} agent."
            logger.warning(f"Prompt file missing: {prompt_file}")

    def build_system_message(self) -> str:
        parts = [self.system_prompt]
        parts.append("\n\n--- PROJECT MEMORY ---")
        parts.append(self.memory.get_compact())
        parts.append("---")
        parts.append("\n\n--- AVAILABLE TOOLS ---")
        for t in self.tools.get_schemas(self.agent_role):
            fn = t["function"]
            params = ", ".join(fn["parameters"]["properties"].keys())
            parts.append(f"  {fn['name']}({params}) — {fn['description']}")
        parts.append("---")
        return "\n".join(parts)

    def build_initial_messages(self, task: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.build_system_message()},
            {"role": "user", "content": self._format_task(task)},
        ]

    def _format_task(self, task: dict) -> str:
        project = self.memory.data.get("project", {}).get("name", "this project")
        lines = [
            f"You are working on project: {project}",
            "",
            f"Task: {task.get('description', '')}",
        ]
        if task.get("type"):
            lines.append(f"Type: {task['type']}")
        if task.get("workspace_hint"):
            lines.append(f"\n--- WORKSPACE ---\n{task['workspace_hint']}\n---")
        if task.get("observation"):
            lines.append(f"\n--- OBSERVED STATE ---\n{task['observation']}\n---")
        if task.get("target_files"):
            lines.append(f"Target files: {', '.join(task['target_files'])}")
        if task.get("acceptance"):
            lines.append(f"Acceptance: {task['acceptance']}")
        return "\n".join(lines)

    async def run(self, task: dict) -> dict:
        """Main ReAct loop. Returns dict with status, result, verdict, etc."""
        messages = self.build_initial_messages(task)
        files_changed: set[str] = set()
        empty_retries = 0

        # Fix #2: Track useful iterations separately from total LLM calls
        iteration = 0
        useful_iterations = 0

        # Fix #1E: Reset truncation counter at start of run
        self._consecutive_truncations = 0

        self.logger.info(f"[{self.agent_role}] Starting task: {task.get('description', '')[:60]}...")

        while useful_iterations < self.max_iterations:
            iteration += 1

            # Fix #4: Context window management — trim if approaching limit
            total_chars = sum(len(json.dumps(m, default=str)) for m in messages)
            estimated_tokens = total_chars // 3  # Conservative: ~3 chars per token
            max_prompt_tokens = int(self.llm.context_size * 0.65)  # Reserve 35% for completion

            if estimated_tokens > max_prompt_tokens:
                self.logger.warning(
                    f"[{self.agent_role}] Context approaching limit "
                    f"({estimated_tokens} estimated tokens). Trimming history."
                )
                # Keep: system message + last 6 messages (3 tool-result rounds)
                MIN_HISTORY = 7
                if len(messages) > MIN_HISTORY:
                    system_msg = messages[0]
                    recent = messages[-6:]
                    dropped = len(messages) - MIN_HISTORY
                    messages = [system_msg] + [
                        {
                            "role": "user",
                            "content": f"[{dropped} earlier messages trimmed to fit context window]"
                        }
                    ] + recent
                    self.logger.info(
                        f"[{self.agent_role}] Trimmed {dropped} messages from history"
                    )

            try:
                response = await self.llm.achat(
                    messages=messages,
                    tools=self.tools.get_schemas(self.agent_role),
                    agent_name=self.agent_role,
                    thinking_budget=self.thinking_budget,
                )
            except Exception as exc:
                self.logger.error(f"[{self.agent_role}] LLM call failed: {exc}")
                return self._result("error", f"LLM failure: {exc}", files_changed, iteration)

            content = response["content"]
            tool_calls = response.get("tool_calls", [])

            # --- Fix #1E: Truncation detection ---
            truncated = response.get("truncated", False)
            finish_reason = response.get("finish_reason", "stop")

            if truncated and tool_calls:
                # Response was truncated mid-generation and contains incomplete
                # tool calls. Do NOT execute them — inject a message instead.
                self._consecutive_truncations += 1
                self.logger.warning(
                    f"[{self.agent_role}] Response truncated (finish_reason=length). "
                    f"Skipping {len(tool_calls)} tool call(s). "
                    f"Consecutive truncations: {self._consecutive_truncations}"
                )

                # If we've hit 3+ consecutive truncations, abort early
                if self._consecutive_truncations >= 3:
                    return self._result(
                        "truncation_loop",
                        f"Aborted after {self._consecutive_truncations} consecutive truncated responses. "
                        f"Increase max_tokens or reduce thinking_budget in config.",
                        files_changed, iteration,
                    )

                # Tell the LLM to try again with shorter output
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was truncated because it exceeded the "
                        "maximum token limit. Please try again with a SHORTER response. "
                        "If writing a file, break it into smaller pieces or use edit_file "
                        "instead of write_file for large content."
                    ),
                })
                continue

            # Also check for malformed tool calls (JSON parse failed but not truncated)
            has_malformed = any(tc.get("_malformed") for tc in tool_calls)
            if has_malformed:
                self._consecutive_truncations += 1
                self.logger.warning(
                    f"[{self.agent_role}] Malformed tool call arguments detected. "
                    f"Skipping execution. Consecutive: {self._consecutive_truncations}"
                )
                if self._consecutive_truncations >= 3:
                    return self._result(
                        "truncation_loop",
                        "Aborted after 3 consecutive malformed tool calls.",
                        files_changed, iteration,
                    )
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous tool call had malformed arguments (JSON parse error). "
                        "This usually happens when the response is too long. Please try "
                        "again with a SHORTER response. Break large files into smaller pieces."
                    ),
                })
                continue

            # Reset truncation counter on successful response
            self._consecutive_truncations = 0

            if not tool_calls and not content.strip():
                empty_retries += 1
                if empty_retries >= 3:
                    return self._result("error", "Repeated empty responses", files_changed, iteration)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Your response was empty. Please use a tool or provide content."})
                continue
            empty_retries = 0

            # If no tool calls, it's a final answer
            if not tool_calls:
                return self._result("complete", content, files_changed, iteration)

            # Log assistant with tool calls
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in tool_calls
                ]
            messages.append(msg)

            # Execute all tool calls and build tool responses
            completion_args = None
            tool_results = []

            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                result = self.tools.execute(name, args, agent_name=self.agent_role)
                tool_results.append({"id": tc["id"], "name": name, "result": result})

                if name in ("write_file", "edit_file") and args.get("path"):
                    files_changed.add(args["path"])

                if name == "task_complete":
                    completion_args = args

            # Add tool results as tool-role messages
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["id"],
                    "content": tr["result"],
                })

            if completion_args:
                verdict = completion_args.get("verdict", "complete")
                summary = completion_args.get("summary", content)
                details = completion_args.get("details", {})
                return {
                    "status": "complete",
                    "result": summary,
                    "verdict": verdict,
                    "details": details,
                    "files_changed": sorted(files_changed),
                    "iterations": iteration,
                }

            # Fix #2: Only increment useful_iterations for productive iterations
            useful_iterations += 1

        # Fix #2: Report useful vs total iteration counts
        return self._result("max_iterations",
            f"Hit {self.max_iterations} useful iterations ({iteration} total calls)",
            files_changed, iteration)

    def _result(self, status: str, result: str, files_changed: set, iterations: int) -> dict:
        # Fix #6B: For truncation_loop status, use a clear verdict
        verdict = "truncation_loop" if status == "truncation_loop" else status
        return {
            "status": status,
            "result": result,
            "verdict": verdict,
            "details": {},
            "files_changed": sorted(files_changed),
            "iterations": iterations,
        }