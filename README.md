<p align="center">LILIA</p>
<p align="center">Local Intent-Driven Language Implementation Architecture</p>

> A fully-local, multi-agent coding assistant that turns a natural-language
> project description into a complete, runnable Python codebase — without
> sending a single token to the cloud.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)
![LLM](https://img.shields.io/badge/LLM-llama--server-informational)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline](#pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Layout](#project-layout)
- [How It Works](#how-it-works)
- [Logs & Outputs](#logs--outputs)
- [Resilience & Safety](#resilience--safety)

---

## Overview

**LILIA** orchestrates four specialised LLM agents — an *architect*,
*implementer*, *verifier*, and *repair* agent — around a shared memory store,
a sandboxed tool-calling layer, and a DAG-based task scheduler. You hand it a
prompt like *"build me a terminal snake game"* and it produces a fully-structured,
tested project directory.

All inference runs against a local
[`llama-server`](https://github.com/ggerganov/llama.cpp) endpoint
(OpenAI-compatible `/v1/chat/completions`), so no code, prompts, or
intermediate reasoning ever leave your machine.

## Features

- **DAG-based parallel execution** — the architect produces a dependency-aware
  task graph; independent tasks run concurrently (bounded by a configurable
  concurrency semaphore).
- **ReAct loop per agent** — *think → act (tool call) → observe → repeat*, with a
  configurable iteration budget and context-window trimming.
- **Deterministic safety gates** — every `.py` file is `ast.parse`-checked after
  edits; the verifier also runs `ruff` and `mypy` (if available) and flags
  unresolved imports before handoff to the LLM.
- **DAG validation before execution** — cycle detection, missing-dependency
  checks, and a source-first scheduling heuristic prevent malformed task graphs
  from entering the pipeline.
- **Crash recovery via git checkpoints** — every agent phase is git-committed;
  failed tasks roll back to the last checkpoint. Staged-but-uncommitted writes
  are tracked in a virtual overlay so they can be reverted atomically.
- **Per-project virtualenv** — each generated project gets its own `.venv` so
  its dependencies stay isolated, and `pytest` + `pytest-timeout` are
  auto-installed.
- **Lessons-learned injection** — on retry, the previous error is injected into
  the task context with an instruction to try a fundamentally different approach.
- **Structured JSONL logs** — session events, raw LLM calls, and tool calls go
  to timestamped log files for post-hoc inspection.
- **LLM response caching** — in-memory and on-disk SHA-256-keyed cache avoids
  redundant inference for identical prompts.

## Pipeline

| Phase | Agent | Responsibility | Output |
|-------|-------|----------------|--------|
| 1. Design | `ArchitectAgent` | Decompose the prompt into a design document + dependency-ordered task list | `tasks.json` (in memory) |
| 2. Implement (per task) | `ImplementerAgent` | Write/modify files for one task | File edits |
| 3. Verify (per task) | `VerifierAgent` | Run deterministic pre-checks, then LLM review + test execution | Verdict: pass / issues / fail |
| 4. Repair (if needed) | `RepairAgent` | Fix issues reported by verifier | File edits |
| 5. Final Check | `Coordinator` | Smoke-test entry module; count source vs. test task completion | Go / partial / no-go |

Every agent shares the same `ProjectMemory` JSON document (compact project
state, task history, bug journal, architectural decisions, file registry) and
talks to the filesystem through a `ToolRegistry` backed by a `FileSandbox` that
enforces safety rules (relative paths only, read-before-edit, path-escape
prevention).

## Prerequisites

- **Python 3.10+** (the code uses PEP 604 / PEP 585 generics — `str | None`, `list[dict]`)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server`** running locally, serving an OpenAI-compatible endpoint
- **A reasoning-capable instruct model** (tested with Qwen3 + DeepSeek-style `<thinking>` tags)
- **Git** (used for checkpoint/rollback — must be on PATH)
- *(Optional)* **ruff** and **mypy** — the verifier auto-detects and runs them if available

## Installation

```bash
# 1. Clone
git clone https://github.com/Elpida12/LILIA.git
cd LILIA

# 2. Create & activate a venv for the coordinator itself
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** each *generated* project gets its own isolated `.venv` inside
> `workspace/<project_name>/.venv`, so the coordinator's venv only needs the
> packages in `requirements.txt`.

## Configuration

All runtime behaviour is driven by [`config.yaml`](./config.yaml).

```yaml
llm:
  base_url: "http://127.0.0.1:8080/v1"
  model: "qwen3"
  api_key: "no-key-needed"
  context_size: 40000
  reasoning_format: "deepseek"
  temperature: 0.3
  max_tokens: 16384
  cache_dir: "./logs/llm_cache"

agents:
  max_iterations: 20          # ReAct loop budget per agent
  thinking_budget: 4096       # <thinking> token budget (combined with max_tokens)

workspace:
  root: "./workspace"         # where generated projects live
  logs_dir: "./logs"          # where JSONL logs go

logging:
  level: "INFO"                # DEBUG | INFO | WARNING | ERROR
  log_to_terminal: true

features:
  git_checkpoints: true       # enable git commit after each phase
  venv_auto: true             # auto-create per-project venv
  max_task_retries: 5         # retry budget per task
  stop_on_first_failure: false # abort pipeline on first task failure
  concurrent_llm_limit: 3     # max parallel LLM calls
```

| Key | Purpose |
|-----|---------|
| `llm.*` | Where to reach the local chat server, model name, context size, reasoning format. |
| `llm.cache_dir` | Directory for on-disk LLM response cache. Set to `null` to disable. |
| `agents.max_iterations` | Hard cap on productive tool-calling turns per agent before forced stop. |
| `agents.thinking_budget` | Per-agent `<thinking>` budget passed to llama-server (honoured when `reasoning_format: deepseek`). Note: `max_tokens` is the TOTAL budget for thinking + content combined. |
| `features.max_task_retries` | How many full implement → verify → repair cycles a single task gets. |
| `features.concurrent_llm_limit` | Semaphore controlling how many LLM calls run in parallel. |
| `features.git_checkpoints` | Commit project state after each agent phase for rollback. |
| `features.stop_on_first_failure` | If true, abort the entire pipeline on the first task failure. |

## Usage

### Interactive mode

```bash
python main.py
```

You'll be prompted for a project description (multi-line, finish with an
empty line) and an optional project name.

```
============================================================
  LOCAL AGENTIC AI CODER
============================================================

Describe your project (empty line to submit):
Build a terminal snake game in curses with configurable speed and a high-score file.

Project name (Enter to auto-generate): snake
```

### CLI mode

Pass the prompt as arguments; use `--name` to set the project directory:

```bash
python main.py --name snake "Build a terminal snake game in curses"
```

The generated project lands in `workspace/<project_name>/`:

```
workspace/snake/
├── .venv/                   # isolated project venv
├── .memory.json             # shared agent memory (crash-recoverable)
├── main.py
├── snake.py
├── config.py
├── tests/
│   ├── test_snake.py
│   └── test_config.py
└── requirements.txt
```

### Resuming a crashed run

If a run fails or is interrupted, simply rerun the same command with the
same `--name`. The coordinator reads `.memory.json` from the project
directory, and the git checkpoint history allows it to restore the last
known-good state and resume from there.

## Project Layout

```
.
├── main.py                  # entry point — config load, CLI, banner, summary
├── config.yaml              # all runtime knobs
├── requirements.txt         # coordinator dependencies
├── src/
│   ├── __init__.py
│   ├── coordinator.py       # async state machine — task DAG, agent dispatch, retry logic
│   ├── agent_base.py        # shared ReAct loop (think → act → observe)
│   ├── llm_backend.py       # async httpx client for llama-server + caching + circuit breaker
│   ├── memory.py            # shared JSON document, atomic persistence
│   ├── logger.py            # JSONL logs + plain terminal output
│   ├── dag_validator.py     # topological sort, cycle detection, source-first scheduling
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── architect.py     # decomposes prompt → design + task list (with Pydantic validation)
│   │   ├── implementer.py   # writes code for one task
│   │   ├── verifier.py      # deterministic pre-checks (ast, ruff, mypy) + LLM review
│   │   └── repair.py        # fixes issues raised by verifier
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py      # tool catalogue, role-scoped permissions, dispatch
│   │   ├── fs.py            # sandboxed file read/write/edit with staging overlay
│   │   ├── shell.py         # shell command execution (venv-aware)
│   │   └── git.py           # git init / checkpoint / restore for rollback
│   └── prompts/
│       ├── architect.txt    # system prompt for architect agent
│       ├── implementer.txt  # system prompt for implementer agent
│       ├── verifier.txt     # system prompt for verifier agent
│       └── repair.txt       # system prompt for repair agent
├── workspace/               # generated projects live here
└── logs/                    # JSONL logs + LLM cache per run
```

## How It Works

### 1. DAG-validated task graph

The architect produces a list of tasks with `id`, `dependencies`,
`target_files`, and an `acceptance` criterion. Before any code is written,
the `DAGValidator` checks for:

- **Cycles** — detected via DFS; the pipeline aborts with a clear error
  listing every cycle found.
- **Missing dependency IDs** — references to non-existent task IDs are caught.
- **Source-first ordering** — tasks targeting source files are always
  scheduled before test tasks, preventing the "only tests are runnable"
  deadlock.

### 2. Staged filesystem with rollback

All file writes go through a `FileSandbox` overlay — they are staged in
memory and only committed to disk after the implementer or repair agent
completes a successful verify pass. If verification fails, the overlay is
reverted (discarded) and the git HEAD is restored, rolling the project back
to the last known-good checkpoint. This means a failed agent never leaves
half-written files on disk.

### 3. Deterministic gates around LLM judgement

An LLM saying *"looks good to me"* is not proof of a working project.
The verifier runs *deterministic* checks *before* the LLM review:

- **Syntax gate** — `ast.parse` on every Python target file.
- **Import resolution** — flags imports that are neither local nor available
  in the current environment.
- **Linter gates** — `ruff check` and `mypy` (if installed on the host).

These results are injected into the verifier's observation so the LLM can
factor them into its verdict. A failing deterministic gate forces a
verdict of `issues` or `fail` regardless of the LLM's opinion.

### 4. Truncation-aware agent loop

When the LLM server returns `finish_reason=length`, the response was
truncated mid-generation. The agent loop detects this and **refuses to
execute** any tool calls from the truncated response (they may be
incomplete). Instead, it tells the model to produce a shorter output.
After 3 consecutive truncations, the agent returns a `truncation_loop`
status, which the coordinator uses to preserve partial work on retry
rather than rolling back (partial progress is still valuable).

### 5. Role-scoped tool permissions

Each agent role sees only the tools it needs:

| Tool | Architect | Implementer | Verifier | Repair |
|------|:---------:|:-----------:|:--------:|:------:|
| `read_file` | ✓ | ✓ | ✓ | ✓ |
| `write_file` | | ✓ | | ✓ |
| `edit_file` | | ✓ | | ✓ |
| `list_files` | ✓ | ✓ | ✓ | ✓ |
| `search_project` | ✓ | ✓ | ✓ | ✓ |
| `run_command` | | ✓ | ✓ | ✓ |
| `update_memory` | ✓ | ✓ | ✓ | ✓ |
| `task_complete` | ✓ | ✓ | ✓ | ✓ |

The architect cannot write files; the verifier cannot modify code. This
prevents agents from accidentally exceeding their mandate.

## Logs & Outputs

Each run writes a timestamped JSONL file to `logs/`:

| File | Contents |
|------|----------|
| `session_YYYYMMDD_HHMMSS.jsonl` | All events — phase transitions, task lifecycle, agent dispatch, tool calls, LLM calls, errors |

At the end of a run, `main.py` renders a summary indicating project status
(complete / partial / failed), tasks completed, and any failed task IDs.

## Resilience & Safety

| Mechanism | What it protects against |
|-----------|------------------------|
| `FileSandbox` path-escape check | Agent writing outside project root |
| `ast.parse` post-write | Syntactically invalid Python files |
| `GitCheckpoint` + `FileSandbox.revert()` | Partial/failed agent runs corrupting the workspace |
| Circuit breaker (`_consecutive_failures` ≥ 5) | Repeated requests to a downed LLM server |
| `DAGValidator` cycle detection | Unrunnable circular task dependencies |
| `TaskDAG` source-first scheduling | Test tasks blocking on unwritten source files |
| Truncation loop detection (3 consecutive) | Infinite retry when `max_tokens` is too small |
| Error-signature escalation | Same bug retried endlessly with the same approach |
| Lessons-learned injection on retry | Agent repeating the exact same mistake |

---

