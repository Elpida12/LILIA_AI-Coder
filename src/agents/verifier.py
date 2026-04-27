"""Verifier — reviews code and runs tests, returns a verdict."""

import ast
import importlib.util
import re
import subprocess
from pathlib import Path
from src.agent_base import AgentBase


class VerifierAgent(AgentBase):
    agent_role = "verifier"
    system_prompt_path = "verifier.txt"

    async def run(self, task: dict) -> dict:
        # Pre-run deterministic checks
        project_root = task.get("project_root")
        target_files = task.get("target_files", [])
        precheck = self._run_prechecks(project_root, target_files)

        # Inject precheck into task observation
        task["observation"] = f"{task.get('observation', '')}\n\n--- PRE-CHECKS ---\n{precheck}"

        result = await super().run(task)
        result["precheck"] = precheck
        return result

    def _run_prechecks(self, project_root: str | None, target_files: list[str]) -> str:
        if not project_root or not target_files:
            return "No files to pre-check."

        root = Path(project_root)
        py_files = [f for f in target_files if f.endswith(".py")]
        if not py_files:
            return "No Python files to pre-check."

        lines = []
        errors = []

        # 1. Syntax checks
        for rel in py_files:
            fpath = root / rel
            if not fpath.exists():
                errors.append(f"  {rel}: file not found")
                continue
            try:
                ast.parse(fpath.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errors.append(f"  {rel}: line {exc.lineno}: {exc.msg}")
            except Exception as exc:
                errors.append(f"  {rel}: {exc}")

        # 2. Import resolution
        import_errors = self._check_imports(root, py_files)
        if import_errors:
            errors.extend(import_errors)

        # 3. Ruff (optional)
        ruff_out = self._run_tool_subprocess("ruff", ["check", *py_files], root)
        if ruff_out:
            errors.extend(ruff_out)

        # 4. MyPy (optional)
        mypy_out = self._run_tool_subprocess("mypy", ["--ignore-missing-imports", *py_files], root)
        if mypy_out:
            errors.extend(mypy_out)

        if errors:
            lines.append(f"FAIL: {len(errors)} issue(s) found in pre-checks:")
            lines.extend(errors)
            lines.append("You MUST fix these before task_complete.")
        else:
            lines.append(f"PASS: {len(py_files)} file(s) passed all pre-checks.")

        return "\n".join(lines)

    @staticmethod
    def _check_imports(root: Path, py_files: list[str]) -> list[str]:
        """Flag imports that are not local and not resolvable in environment."""
        errors = []
        local_modules: set[str] = set()
        for p in root.rglob("*.py"):
            rel = p.relative_to(root)
            parts = rel.with_suffix("").parts
            for i in range(len(parts)):
                local_modules.add(".".join(parts[:i+1]))

        for rel in py_files:
            try:
                tree = ast.parse((root / rel).read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name.split(".")[0]
                        if mod not in local_modules and importlib.util.find_spec(mod) is None:
                            errors.append(f"  {rel}: unresolved import '{alias.name}'")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod = node.module.split(".")[0]
                        if mod not in local_modules and importlib.util.find_spec(mod) is None:
                            errors.append(f"  {rel}: unresolved import '{node.module}'")
        return errors

    @staticmethod
    def _run_tool_subprocess(tool_name: str, args: list[str], cwd: Path) -> list[str]:
        try:
            proc = subprocess.run(
                [tool_name, *args],
                cwd=cwd, capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0 and (proc.stdout.strip() or proc.stderr.strip()):
                out = (proc.stdout + "\n" + proc.stderr).strip()[:800]
                return [f"  {tool_name.upper()} ERRORS:", f"    {out}"]
        except FileNotFoundError:
            return [f"  {tool_name.upper()}: tool not installed, skipping."]
        except Exception as exc:
            return [f"  {tool_name.upper()}: failed to run: {exc}"]
        return []
