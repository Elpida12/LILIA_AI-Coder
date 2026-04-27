"""Sandboxed filesystem operations with staging overlay."""

import difflib
from pathlib import Path


class FileSandbox:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve()
        self._overlay: dict[str, str] = {}

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be a non-empty relative string")
        p = Path(path)
        if p.is_absolute():
            raise ValueError(f"Absolute paths not allowed: {path!r}")
        resolved = (self.root / p).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes project root: {path!r}")
        return resolved

    def _read_real(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_text(encoding="utf-8")

    def read(self, path: str) -> str:
        if path in self._overlay:
            return self._overlay[path]
        return self._read_real(path)

    def write(self, path: str, content: str) -> str:
        self._resolve(path)
        self._overlay[path] = content
        return f"STAGED write {path} ({len(content)} chars)"

    def edit(self, path: str, old_text: str, new_text: str) -> str:
        current = self.read(path)
        if old_text in current:
            updated = current.replace(old_text, new_text, 1)
            return self.write(path, updated)
        # Fuzzy fallback: old_text doesn't match exactly, but is very similar
        # to a region in current. Find the best-matching window of current
        # that resembles old_text, then replace that window with new_text.
        #
        # Strategy: slide a window of old_text's length (with some margin)
        # across current, computing SequenceMatcher ratio for each window.
        # Pick the window with the highest ratio >= 0.95 and replace it.
        matcher_full = difflib.SequenceMatcher(None, current, old_text)
        if matcher_full.ratio() < 0.95:
            raise ValueError(
                f"Text not found in {path}. "
                f"Fuzzy confidence: {matcher_full.ratio():.0%} (need >=95%)."
            )

        # If old_text is approximately the same length as current, it's
        # a whole-file diff — replace the entire content.
        len_ratio = min(len(old_text), len(current)) / max(len(old_text), len(current))
        if len_ratio >= 0.8:
            updated = new_text
            return self.write(path, updated)

        # old_text is a smaller fragment — find the best-matching window
        best_start = 0
        best_end = len(current)
        best_ratio = 0.0
        # Slide a window across current. Use step size proportional to
        # old_text length to avoid excessive comparisons.
        win_len = len(old_text)
        step = max(1, win_len // 10)
        for start in range(0, len(current), step):
            end = min(start + win_len + win_len // 4, len(current))
            window = current[start:end]
            r = difflib.SequenceMatcher(None, window, old_text).ratio()
            if r > best_ratio:
                best_ratio = r
                best_start = start
                best_end = end

        if best_ratio >= 0.95:
            updated = current[:best_start] + new_text + current[best_end:]
            return self.write(path, updated)

        raise ValueError(
            f"Text not found in {path}. "
            f"Fuzzy confidence: {matcher_full.ratio():.0%} (need >=95%)."
        )

    def list_files(self, directory: str = ".") -> str:
        target = self._resolve(directory)
        if not target.is_dir():
            return f"Error: {directory} is not a directory"
        lines = []
        for p in sorted(target.rglob("*")):
            rel = p.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            marker = "[DIR]" if p.is_dir() else "[FILE]"
            lines.append(f"{marker} {rel}")
        # Overlay files not yet committed
        for staged_path in sorted(self._overlay):
            if not any(part.startswith(".") for part in Path(staged_path).parts):
                lines.append(f"[STAGED] {staged_path}")
        return "\n".join(lines) if lines else "(empty directory)"

    def search_project(self, query: str) -> str:
        import re
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        # Real files
        for p in sorted(self.root.rglob("*")):
            if p.is_dir() or any(part.startswith(".") for part in p.relative_to(self.root).parts):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matches = list(pattern.finditer(text))
            if matches:
                rel = p.relative_to(self.root)
                for m in matches[:3]:
                    line_no = text[:m.start()].count(chr(10)) + 1
                    snippet = text[max(0, m.start()-30):min(len(text), m.end()+30)].replace(chr(10), " ")
                    results.append(f"  {rel}:{line_no}: ...{snippet}...")
        # Staged files
        for sp, content in self._overlay.items():
            matches = list(pattern.finditer(content))
            if matches:
                for m in matches[:3]:
                    line_no = content[:m.start()].count(chr(10)) + 1
                    snippet = content[max(0, m.start()-30):min(len(content), m.end()+30)].replace(chr(10), " ")
                    results.append(f"  [STAGED] {sp}:{line_no}: ...{snippet}...")
        if not results:
            return f"No matches for '{query}'"
        return f"Found {len(results)} match(es):\n" + "\n".join(results[:20])

    def commit(self, paths: list[str] | None = None):
        """Persist staged overlay to disk."""
        candidate_paths = paths if paths is not None else list(self._overlay)
        items = {p: self._overlay[p] for p in candidate_paths if p in self._overlay}
        for path, content in items.items():
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        # Clear committed from overlay
        for p in items:
            self._overlay.pop(p, None)
        return f"Committed {len(items)} file(s)"

    def revert(self, paths: list[str] | None = None):
        """Discard staged changes without writing to disk."""
        if paths is None:
            count = len(self._overlay)
            self._overlay.clear()
            return f"Reverted {count} staged file(s)"
        for p in paths:
            self._overlay.pop(p, None)
        return f"Reverted {len(paths)} staged file(s)"
