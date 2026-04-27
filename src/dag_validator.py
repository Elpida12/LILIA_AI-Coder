"""DAG validation — topological sort, cycle detection, and root-cause analysis."""

from collections import defaultdict


class DAGValidator:
    """
    Validates and schedules task dependency graphs.

    Guarantees:
    - Detects cycles before execution starts
    - Prioritizes source files over test files
    - Flags deadlock clusters where only test tasks are runnable
    """

    def __init__(self, tasks: list[dict]):
        self.tasks = {t["id"]: t for t in tasks}
        self._graph = self._build_graph()

    def _build_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        for tid, t in self.tasks.items():
            for dep in t.get("dependencies", []):
                graph[tid].add(dep)
        return graph

    def detect_cycles(self) -> list[list[str]]:
        """Return all simple cycles in the dependency graph."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: list[str] = []
        stack_set: set[str] = set()

        def dfs(node: str):
            visited.add(node)
            stack.append(node)
            stack_set.add(node)
            for dep in self._graph.get(node, set()):
                if dep not in visited:
                    dfs(dep)
                elif dep in stack_set:
                    # Found a cycle
                    cycle_start = stack.index(dep)
                    cycle = stack[cycle_start:] + [dep]
                    cycles.append(cycle)
            stack.pop()
            stack_set.remove(node)

        for tid in self.tasks:
            if tid not in visited:
                dfs(tid)
        return cycles

    def topological_order(self) -> tuple[list[str], list[str]]:
        """
        Return (ordered_task_ids, cycle_breaker_removals).
        Raises ValueError with detailed message if irreducible cycles exist.
        Tasks with missing dependencies are excluded from ordering (they can
        never be scheduled). Call validate_or_fail() first to catch these.
        """
        # Prune dependencies that reference non-existent task IDs
        # so they don't block scheduling (validate_or_fail catches this)
        effective_graph: dict[str, set[str]] = {}
        for tid, deps in self._graph.items():
            effective_graph[tid] = deps & set(self.tasks)

        cycles = self.detect_cycles()
        if not cycles:
            # Kahn's algorithm
            in_degree = {tid: 0 for tid in self.tasks}
            for tid in self.tasks:
                for dep in effective_graph.get(tid, set()):
                    in_degree[tid] = in_degree.get(tid, 0) + 1

            # Initialize: tasks with no dependencies
            queue = [tid for tid in self.tasks if in_degree[tid] == 0]
            # Sort queue by priority: source files first, test files last
            queue.sort(key=self._task_priority)

            ordered: list[str] = []
            while queue:
                node = queue.pop(0)
                ordered.append(node)
                # Decrement in-degree for dependents
                for tid in self.tasks:
                    if node in effective_graph.get(tid, set()):
                        in_degree[tid] -= 1
                        if in_degree[tid] == 0:
                            queue.append(tid)
                            queue.sort(key=self._task_priority)
            return ordered, []

        # Attempt greedy cycle breaking: remove "test" tasks from clusters
        removals: list[str] = []
        for cycle in cycles:
            # Find the test task(s) in the cycle and suggest removing their deps
            test_tasks = [tid for tid in cycle if self._is_test_task(tid)]
            if test_tasks:
                # Suggest removing dependencies pointing into the test task
                for tt in test_tasks:
                    removals.append(tt)
        
        if removals:
            return [], removals

        # Irreducible cycle
        cycle_desc = "\n".join(
            f"  Cycle {i+1}: {' -> '.join(c)}" for i, c in enumerate(cycles)
        )
        raise ValueError(
            f"Circular dependencies detected in task graph. Execution cannot proceed.\n"
            f"{cycle_desc}\n"
            f"Please revise the design so each task depends only on earlier tasks."
        )

    def validate_or_fail(self) -> str:
        """
        Validate the graph. Returns empty string on success.
        On failure, returns a descriptive error message.
        """
        # Check: are all dependency references valid?
        all_ids = set(self.tasks)
        all_deps = {dep for t in self.tasks.values() for dep in t.get("dependencies", [])}
        missing = all_deps - all_ids
        if missing:
            return (
                f"Tasks reference non-existent dependency IDs: {sorted(missing)}. "
                f"Each dependency must match a task ID in the design."
            )

        try:
            ordered, removals = self.topological_order()
        except ValueError as exc:
            return str(exc)

        if removals:
            rem_list = ", ".join(removals)
            return (
                f"Dependency graph contains cycle(s) that can be broken by removing "
                f"test tasks: {rem_list}. Suggest removing their dependencies "
                f"so they run after source tasks complete."
            )

        # Check: are ALL root tasks (in-degree 0) test tasks?
        # A valid design must have at least one source task with no dependencies.
        root_tasks = [tid for tid in self.tasks if not self._graph.get(tid)]
        if root_tasks and all(self._is_test_task(tid) for tid in root_tasks):
            test_ids = ", ".join(root_tasks)
            return (
                f"WARNING: Only test tasks are runnable at start (e.g., {test_ids}). "
                f"No source-file tasks appear before them. The design must include "
                f"source implementation tasks with no dependencies."
            )

        # Also warn if no root tasks exist at all (everything has dependencies)
        if not root_tasks and self.tasks:
            return (
                "WARNING: No tasks have zero dependencies — all tasks depend on "
                "other tasks. This may indicate a cycle or missing source tasks."
            )

        return ""

    def _task_has_test_files(self, tid: str) -> bool:
        task = self.tasks.get(tid, {})
        targets = [f.lower() for f in task.get("target_files", [])]
        return any("test" in tf for tf in targets)

    def schedule_source_first(self) -> list[str]:
        """
        Return task IDs in dependency order with source tasks prioritized.
        Raises ValueError on irreducible cycle.
        """
        ordered, removals = self.topological_order()
        if removals:
            raise ValueError(
                f"Cannot schedule: cycle break needed for {removals}. "
                f"Please fix dependencies in design."
            )
        return ordered

    @staticmethod
    def _is_test_task(t: dict | str) -> bool:
        """Heuristic: task IDs, descriptions, or target_files mentioning test are test tasks."""
        if isinstance(t, dict):
            tid = t.get("id", "").lower()
            desc = t.get("description", "").lower()
            targets = [f.lower() for f in t.get("target_files", [])]
            return "test" in tid or "test" in desc or any("test" in tf for tf in targets)
        return "test" in str(t).lower()

    def _task_priority(self, tid: str) -> int:
        """Sort key: lower = higher priority. Source tasks score lower than test tasks."""
        task = self.tasks.get(tid, {})
        return 1 if self._is_test_task(task) else 0


def validate_design_tasks(design: dict) -> dict:
    """
    Validate a design document and its tasks.
    Returns {"valid": bool, "error": str, "ordered_tasks": list[str]}.
    """
    tasks = design.get("tasks", [])
    if not tasks:
        return {"valid": False, "error": "Design has no tasks.", "ordered_tasks": []}

    try:
        validator = DAGValidator(tasks)
        error = validator.validate_or_fail()
        if error:
            return {"valid": False, "error": error, "ordered_tasks": []}
        ordered = validator.schedule_source_first()
        return {"valid": True, "error": "", "ordered_tasks": ordered}
    except ValueError as exc:
        return {"valid": False, "error": str(exc), "ordered_tasks": []}
