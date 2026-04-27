"""State-machine coordinator — async with DAG parallel execution."""

import asyncio
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.llm_backend import LLMBackend
from src.memory import ProjectMemory
from src.logger import Logger
from src.tools.fs import FileSandbox
from src.tools.shell import ShellRunner
from src.tools.git import GitCheckpoint
from src.tools.registry import ToolRegistry
from src.agents import ArchitectAgent, ImplementerAgent, VerifierAgent, RepairAgent
from src.dag_validator import DAGValidator


class TaskDAG:
    """Manages dependency-aware task ordering."""

    def __init__(self, tasks: list[dict]):
        self.tasks = {t["id"]: t for t in tasks}
        self.completed: set[str] = set()
        self.failed: set[str] = set()

    def ready(self) -> list[dict]:
        """Return tasks ready to run, sorted: source tasks before test tasks."""
        ready_tasks = [
            t for tid, t in self.tasks.items()
            if tid not in self.completed | self.failed
            and all(dep in self.completed for dep in t.get("dependencies", []))
        ]
        # Sort: non-test tasks first
        def _is_test(t: dict) -> bool:
            tid = t.get("id", "").lower()
            desc = t.get("description", "").lower()
            targets = [f.lower() for f in t.get("target_files", [])]
            return "test" in tid or "test" in desc or any("test" in tf for tf in targets)
        ready_tasks.sort(key=lambda t: (1 if _is_test(t) else 0, t.get("id", "")))
        return ready_tasks

    def mark(self, tid: str, ok: bool):
        (self.completed if ok else self.failed).add(tid)

    def all_done(self) -> bool:
        return self.completed | self.failed >= set(self.tasks)


class Coordinator:
    """
    Async state machine for project generation:
        IDLE -> DESIGN   (Architect produces design + task list)
        DESIGN -> [for each task or batch of independent tasks]
            IMPLEMENT -> VERIFY
        -> DONE
    """

    SEMAPHORE_VALUE = 3  # Limit concurrent LLM calls

    def __init__(self, config: dict):
        self.cfg = config
        self.logger = Logger(
            logs_dir=config["workspace"]["logs_dir"],
            level=config["logging"]["level"],
            log_to_terminal=config["logging"]["log_to_terminal"],
        )

        llm_cfg = config["llm"]
        self.llm = LLMBackend(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["model"],
            api_key=llm_cfg.get("api_key", ""),
            context_size=llm_cfg["context_size"],
            reasoning_format=llm_cfg["reasoning_format"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            logger=self.logger,
        )

        self.workspace = Path(config["workspace"]["root"]).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.agents_cfg = config.get("agents", {})
        self.features = config.get("features", {})
        self.prompts_dir = str(Path(__file__).parent / "prompts")
        self._semaphore = asyncio.Semaphore(self.features.get("concurrent_llm_limit", self.SEMAPHORE_VALUE))

    async def run(self, user_prompt: str, project_name: str | None = None) -> dict:
        if not project_name:
            words = user_prompt.split()[:4]
            project_name = "_".join(w.lower() for w in words if w.isalnum())
            if not project_name:
                project_name = f"project_{int(time.time())}"
        else:
            if "/" in project_name or "\\" in project_name or ".." in project_name:
                raise ValueError(f"Unsafe project_name: {project_name!r}")

        project_root = self.workspace / project_name
        project_root.mkdir(parents=True, exist_ok=True)

        self.logger.rule(f"PROJECT: {project_name}", char="=")
        self.logger.info(f"Root: {project_root}")
        self.logger.info(f"Prompt: {user_prompt}")

        # --- memory ---
        memory = ProjectMemory(project_name, str(self.workspace))
        memory.load()
        memory.update_section("project", {"name": project_name, "description": user_prompt})

        # --- venv ---
        venv_path = project_root / ".venv"
        if self.features.get("venv_auto", True):
            await self._setup_venv(venv_path)

        # --- tools ---
        fs = FileSandbox(project_root)
        shell = ShellRunner(project_root, venv_path)
        git = GitCheckpoint(project_root)
        git.init()
        tools = ToolRegistry(fs, shell, git, memory, self.logger)

        # --- PHASE 1: DESIGN ---
        self.logger.rule("PHASE 1: DESIGN", char="-")
        tasks = await self._run_design(tools, memory, user_prompt)
        if not tasks:
            return self._fail("design", "No tasks produced by architect", project_root)

        # --- VALIDATE TASK GRAPH ---
        validator = DAGValidator(tasks)
        error = validator.validate_or_fail()
        if error:
            self.logger.error(f"Task graph validation failed: {error}")
            return self._fail("design", f"Invalid task graph: {error}", project_root)
        ordered_tasks = validator.schedule_source_first()
        self.logger.info(f"Task schedule order: {ordered_tasks}")

        memory.update_section("tasks", tasks)
        git.checkpoint("design complete")
        fs.commit()  # commit architect writes

        # --- PHASE 2: EXECUTE TASKS (DAG parallel) ---
        self.logger.rule("PHASE 2: IMPLEMENTATION", char="-")
        dag = TaskDAG(tasks)
        failed_tasks: list[str] = []

        # Track retry error patterns for escalation
        retry_errors: dict[str, str] = {}

        stop_early = False
        with ThreadPoolExecutor(max_workers=4) as pool:
            loop = asyncio.get_running_loop()
            while not dag.all_done() and not stop_early:
                batch = dag.ready()
                if not batch:
                    break
                coros = [
                    self._run_task_cycle_semaphore(tid, task, tools, memory, git, project_root, pool, retry_errors)
                    for task in batch
                    if (tid := task["id"])
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for task, res in zip(batch, results):
                    tid = task["id"]
                    if isinstance(res, Exception):
                        self.logger.error(f"[{tid}] Exception: {res}")
                        dag.mark(tid, False)
                        failed_tasks.append(tid)
                    elif res:
                        dag.mark(tid, True)
                    else:
                        dag.mark(tid, False)
                        failed_tasks.append(tid)
                        if self.features.get("stop_on_first_failure", False):
                            # Mark all remaining tasks in this batch as failed
                            for remaining_task in batch:
                                rid = remaining_task["id"]
                                if rid != tid and rid not in dag.completed and rid not in dag.failed:
                                    dag.mark(rid, False)
                                    failed_tasks.append(rid)
                            stop_early = True
                            break

        # --- PHASE 3: FINAL CHECK ---
        self.logger.rule("PHASE 3: FINAL CHECK", char="-")
        entry = memory.data.get("project", {}).get("entry_point", "")

        def _is_test_task_count(tid: str) -> bool:
            """Consistent test-task heuristic for counting (same as TaskDAG/DAGValidator)."""
            task_info = dag.tasks.get(tid, {})
            low_id = tid.lower()
            desc = task_info.get("description", "").lower()
            targets = [f.lower() for f in task_info.get("target_files", [])]
            return "test" in low_id or "test" in desc or any("test" in tf for tf in targets)

        source_tasks_attempted = len([t for t in tasks if not _is_test_task_count(t["id"])])
        source_tasks_completed = len([
            tid for tid in dag.completed
            if not _is_test_task_count(tid)
        ])

        if entry and (project_root / entry).exists():
            proc = shell.run(f"python {entry}", timeout=30)  # Fix #3: Increased from 10 to 30 seconds
            self.logger.info(f"Entry module check:\n{proc[:500]}")
            # Fix #3: Parse and check exit code
            exit_code = ShellRunner.parse_exit_code(proc)
            if exit_code != 0:
                self.logger.error(
                    f"Entry module '{entry}' failed with exit code {exit_code}"
                )
                failed_tasks.append("_entry_point")
        elif entry:
            self.logger.error(f"Entry point '{entry}' is missing.")

        # If no source tasks were attempted/completed and only tests exist, treat as failure
        if source_tasks_attempted == 0 and len(tasks) > 0:
            self.logger.error("No source-file tasks were attempted. Only test tasks were scheduled.")
            self.logger.rule("PROJECT PARTIAL FAILURE", char="!")
            return {
                "status": "failed",
                "project_root": str(project_root),
                "tasks_total": len(tasks),
                "tasks_completed": len(dag.completed),
                "failed_tasks": failed_tasks,
                "task_history": memory.data.get("task_history", []),
            }

        # --- SUMMARY ---
        total = len(tasks)
        completed = len(dag.completed)
        if failed_tasks:
            self.logger.rule("PROJECT PARTIAL FAILURE", char="!")
            return {
                "status": "partial",
                "project_root": str(project_root),
                "tasks_total": total,
                "tasks_completed": completed,
                "failed_tasks": failed_tasks,
                "task_history": memory.data.get("task_history", []),
            }

        self.logger.rule("PROJECT COMPLETE", char="=")
        return {
            "status": "complete",
            "project_root": str(project_root),
            "tasks_total": total,
            "tasks_completed": completed,
            "task_history": memory.data.get("task_history", []),
        }

    async def _run_design(self, tools: ToolRegistry, memory: ProjectMemory, prompt: str) -> list[dict]:
        agent = ArchitectAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        result = await agent.run({"description": prompt, "type": "design"})
        if result["status"] != "complete":
            self.logger.error(f"Architect failed: {result['result']}")
            return []

        design = result.get("design", {})
        if design:
            memory.update_section("design", design)
            if "entry_point" in design:
                memory.update_section("project", {"entry_point": design["entry_point"]})

        tasks = result.get("tasks", [])
        if not tasks:
            self.logger.error("Architect produced no tasks")
            return []
        return tasks

    async def _run_task_cycle_semaphore(self, tid: str, task: dict, tools: ToolRegistry,
                                         memory: ProjectMemory, git: GitCheckpoint,
                                         project_root: Path, pool: ThreadPoolExecutor,
                                         retry_errors: dict[str, str] | None = None) -> bool:
        async with self._semaphore:
            return await self._run_task_cycle(tid, task, tools, memory, git, project_root, pool, retry_errors)

    async def _run_task_cycle(self, tid: str, task: dict, tools: ToolRegistry,
                               memory: ProjectMemory, git: GitCheckpoint,
                               project_root: Path, pool: ThreadPoolExecutor,
                               retry_errors: dict[str, str] | None = None) -> bool:
        max_retries = self.features.get("max_task_retries", 5)
        task_start = time.time()
        files_changed: list[str] = []
        attempts = 0
        prev_error_signature = ""
        retry_errors = retry_errors or {}

        # Use a copy of the task dict to avoid mutating shared references
        # across concurrent task cycles and retry iterations.
        task = {**task}

        for attempt in range(1, max_retries + 1):
            attempts += 1
            lessons_learned = ""

            if attempt > 1:
                self.logger.info(f"[{tid}] Retry {attempt}/{max_retries}")

                # Fix #6: Differentiate between truncation/exhaustion failures and
                # logic failures. For truncation, preserve partial work.
                prev_err = retry_errors.get(tid, "")
                is_truncation_failure = (
                    "truncation_loop" in prev_err
                    or ("Hit" in prev_err and "iterations" in prev_err)
                )

                if is_truncation_failure:
                    # Preserve staged changes — partial work is valid
                    tools.fs.commit()
                    git.checkpoint(f"retry {tid} attempt {attempt} (partial work preserved)")
                    self.logger.info(f"[{tid}] Preserving partial work from truncated attempt")
                else:
                    # Full rollback for genuine logic failures
                    git.restore_head()
                    git.checkpoint(f"retry {tid} attempt {attempt}")
                    tools.fs.revert()  # discard staged changes
                    self.logger.info(f"[{tid}] Full rollback on logic failure")

                # Build lessons learned from previous attempts
                prev_error = retry_errors.get(tid, "")
                if prev_error:
                    lessons_learned = (
                        f"\n\n--- PREVIOUS ATTEMPT ERROR ---\n"
                        f"{prev_error}\n"
                        f"---\n\n"
                        f"Instruction: Do NOT repeat the same mistake. Try a fundamentally "
                        f"different approach. If a test keeps failing with the same error, "
                        f"consider fixing the source code instead of the test."
                    )

            # Inject workspace root and lessons into task context
            task["project_root"] = str(project_root)
            task["workspace_hint"] = (
                f"Project workspace is at: {project_root}\n"
                f"All file paths are relative to this directory.\n"
                f"Do NOT use 'cd /testbed' or similar hardcoded paths.\n"
                f"Run commands directly (e.g. 'python -m pytest tests/')."
            )
            if lessons_learned:
                task["observation"] = task.get("observation", "") + lessons_learned

            task["type"] = "implement"
            impl_result = await self._run_implementer(tools, memory, task)
            # Fix #6B: Log truncation_loop status specifically
            if impl_result["status"] == "truncation_loop":
                self.logger.warning(
                    f"[{tid}] Implementer hit truncation loop. Will preserve partial work on retry."
                )
            elif impl_result["status"] != "complete":
                self.logger.error(f"[{tid}] Implementer failed: {impl_result['result']}")
            retry_errors[tid] = f"Implementer error: {impl_result['result']}"
            if impl_result["status"] != "complete":
                continue

            current_files = impl_result.get("files_changed", [])
            files_changed = list(dict.fromkeys(files_changed + current_files))
            tools.fs.commit(current_files)
            git.checkpoint(f"{tid} implemented")

            # VERIFY
            verify_result = await self._run_verifier(tools, memory, task, current_files, project_root)
            if verify_result["status"] != "complete":
                self.logger.error(f"[{tid}] Verifier did not complete")
                retry_errors[tid] = f"Verifier error: {verify_result.get('result', 'unknown')}"
                continue

            verdict = verify_result.get("verdict", "pass")
            details = verify_result.get("details", {})
            issues = details.get("issues", [])

            if verdict == "pass":
                memory.add_task(
                    task_id=tid, description=task["description"],
                    status="complete", files_changed=files_changed,
                    summary=impl_result["result"],
                    duration=time.time() - task_start, attempts=attempt,
                )
                self.logger.info(f"[{tid}] PASS")
                return True

            # Compute error signature for escalation check
            error_sig = ""
            if issues:
                error_sig = " | ".join(str(i) for i in issues[:3])
            elif verify_result.get("result"):
                error_sig = verify_result["result"][:200]

            # ESCALATION: same signature as last attempt -> escalate to repair immediately
            if attempt >= 2 and error_sig and error_sig == prev_error_signature:
                self.logger.info(f"[{tid}] Repeated same error on retry. Escalating to RepairAgent...")
                repair_ok = await self._run_repair(tools, memory, task, issues or [error_sig], current_files, project_root)
                if repair_ok:
                    tools.fs.commit(current_files)
                    git.checkpoint(f"{tid} repaired-escalated")
                    verify2 = await self._run_verifier(tools, memory, task, current_files, project_root)
                    if verify2.get("verdict") == "pass":
                        memory.add_task(
                            task_id=tid, description=task["description"],
                            status="complete", files_changed=files_changed,
                            summary="Repair succeeded (escalated)",
                            duration=time.time() - task_start, attempts=attempt,
                        )
                        self.logger.info(f"[{tid}] PASS after escalated repair")
                        return True
                self.logger.info(f"[{tid}] Escalated repair/re-verify failed, rolling back...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Escalated repair failed. Error: {error_sig}"
                prev_error_signature = error_sig
                continue

            if verdict == "issues" and issues:
                self.logger.info(f"[{tid}] Issues found, dispatching repair...")
                repair_ok = await self._run_repair(tools, memory, task, issues, current_files, project_root)
                if repair_ok:
                    tools.fs.commit(current_files)
                    git.checkpoint(f"{tid} repaired")
                    verify2 = await self._run_verifier(tools, memory, task, current_files, project_root)
                    if verify2.get("verdict") == "pass":
                        memory.add_task(
                            task_id=tid, description=task["description"],
                            status="complete", files_changed=files_changed,
                            summary="Repair succeeded",
                            duration=time.time() - task_start, attempts=attempt,
                        )
                        self.logger.info(f"[{tid}] PASS after repair")
                        return True
                self.logger.info(f"[{tid}] Repair/re-verify failed, rolling back...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Repair failed. Issues: {issues}"
                prev_error_signature = error_sig
                continue

            if verdict == "fail":
                self.logger.info(f"[{tid}] Verdict=fail, rolling back to retry...")
                tools.fs.revert(current_files)
                retry_errors[tid] = f"Verifier fail: {verify_result.get('result', 'unknown')}"
                prev_error_signature = error_sig
                continue

            self.logger.warning(f"[{tid}] Unknown verdict '{verdict}', retrying...")
            tools.fs.revert(current_files)
            retry_errors[tid] = f"Unknown verdict: {verdict}"

        memory.add_task(
            task_id=tid, description=task["description"],
            status="failed", files_changed=files_changed,
            summary=f"Failed after {max_retries} attempts",
            duration=time.time() - task_start, attempts=attempts,
        )
        return False

    async def _run_implementer(self, tools: ToolRegistry, memory: ProjectMemory, task: dict) -> dict:
        agent = ImplementerAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        return await agent.run(task)

    async def _run_verifier(self, tools: ToolRegistry, memory: ProjectMemory,
                             task: dict, target_files: list[str],
                             project_root: Path) -> dict:
        verify_task = {
            "description": f"Review and test: {task['description']}",
            "target_files": target_files,
            "project_root": str(project_root),
            "type": "verify",
        }
        agent = VerifierAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        return await agent.run(verify_task)

    async def _run_repair(self, tools: ToolRegistry, memory: ProjectMemory,
                           task: dict, issues: list, target_files: list[str],
                           project_root: Path) -> bool:
        repair_task = {
            "description": (
                f"Fix issues in task {task['id']}: {task['description']}\n\n"
                f"Issues: {chr(10).join(f'- {i}' for i in issues)}"
            ),
            "target_files": target_files,
            "project_root": str(project_root),
            "type": "repair",
        }
        agent = RepairAgent(
            llm=self.llm, tools=tools, memory=memory, logger=self.logger,
            prompts_dir=self.prompts_dir,
            max_iterations=self.agents_cfg.get("max_iterations", 15),
            thinking_budget=self.agents_cfg.get("thinking_budget", 8192),
        )
        result = await agent.run(repair_task)
        return result["status"] == "complete" and result.get("verdict") != "fail"

    async def _setup_venv(self, venv_path: Path):
        if sys.platform == "win32":
            pip = str(venv_path / "Scripts\\pip.exe")
        else:
            pip = str(venv_path / "bin" / "pip")
        loop = asyncio.get_running_loop()
        try:
            if not venv_path.exists():
                self.logger.info(f"Creating venv: {venv_path}")
                await loop.run_in_executor(None, lambda: subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, capture_output=True, text=True,
                ))
                await loop.run_in_executor(None, lambda: subprocess.run(
                    [pip, "install", "--upgrade", "pip"],
                    check=True, capture_output=True, text=True,
                ))
                self.logger.info("venv created")
            await loop.run_in_executor(None, lambda: subprocess.run(
                [pip, "install", "pytest", "pytest-timeout"],
                check=True, capture_output=True, text=True,
            ))
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"venv setup failed: {exc}")
        except FileNotFoundError:
            self.logger.error("venv pip not found — corrupted or cross-platform issue")
        except Exception as exc:
            self.logger.error(f"venv setup unexpected error: {exc}")

    def _fail(self, phase: str, error: str, project_root: Path) -> dict:
        self.logger.error(f"Failed in {phase}: {error}")
        return {
            "status": "failed",
            "phase": phase,
            "error": error,
            "project_root": str(project_root),
        }
