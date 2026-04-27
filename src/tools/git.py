"""Git checkpointing for rollback support."""

import subprocess
from pathlib import Path


class GitCheckpoint:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve()

    def _git(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.root,
            capture_output=True, text=True,
        )

    def is_repo(self) -> bool:
        return (self.root / ".git").exists()

    def init(self):
        if not self.is_repo():
            self._git("init")
            self._git("config", "user.email", "agent@local")
            self._git("config", "user.name", "AI Coder")

    def checkpoint(self, message: str) -> str:
        if not self.is_repo():
            return "Git not initialized"
        self._git("add", "-A")
        r = self._git("commit", "-m", message, "--allow-empty")
        return f"Checkpoint: {message} (rc={r.returncode})"

    def restore_head(self) -> str:
        if not self.is_repo():
            return "Git not initialized"
        self._git("checkout", "HEAD", "--", ".")
        # Note: We deliberately do NOT run `git clean -fd` here because it
        # would permanently delete untracked files (e.g., newly created source
        # files that haven't been committed yet). Instead, we rely on
        # checkout to restore tracked files and the FileSandbox overlay revert
        # to discard staged-but-not-committed writes.
        return "Restored to HEAD"
