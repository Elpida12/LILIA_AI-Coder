"""Shell command execution within project directory."""

import subprocess
import sys
from pathlib import Path


class ShellRunner:
    def __init__(self, cwd: Path, venv_path: Path | None = None):
        self.cwd = cwd.resolve()
        self.venv_path = venv_path.resolve() if venv_path else None

    def _venv_python(self) -> str | None:
        if not self.venv_path:
            return None
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        exe = "python.exe" if sys.platform == "win32" else "python"
        candidate = self.venv_path / bin_dir / exe
        return str(candidate) if candidate.exists() else None

    def run(self, command: str, timeout: int = 120) -> str:
        venv_python = self._venv_python()
        if venv_python and command.strip().startswith("python "):
            command = f"{venv_python} {command.strip()[7:]}"

        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.cwd,
                capture_output=True, text=True, timeout=timeout,
            )
            lines = [f"Command: {command}", f"Exit code: {proc.returncode}"]
            if proc.stdout:
                lines.append(f"STDOUT:\n{proc.stdout}")
            if proc.stderr:
                lines.append(f"STDERR:\n{proc.stderr}")
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as exc:
            return f"Command failed: {exc}"

    @staticmethod
    def parse_exit_code(output: str) -> int:
        """Parse exit code from ShellRunner.run() output string.

        Returns the exit code if found, or -1 if unable to determine.
        """
        for line in output.split("\n"):
            if line.startswith("Exit code:"):
                try:
                    return int(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    return -1
        return -1
