#!/usr/bin/env python3
"""new_AI_CODER — local multi-agent AI programmer."""

import asyncio
import sys
import yaml
from pathlib import Path

from src.coordinator import Coordinator
from src.llm_backend import LLMError


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "config.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def main():
    config = load_config()
    project_name = None

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if "--name" in args:
            idx = args.index("--name")
            if idx + 1 < len(args):
                project_name = args[idx + 1]
                del args[idx:idx + 2]
        user_prompt = " ".join(args)
    else:
        print("\nDescribe your project (empty line to submit):")
        lines = []
        while True:
            line = input()
            if lines and line == "":
                break
            if line == "":
                continue
            lines.append(line)
        user_prompt = "\n".join(lines).strip()
        if not user_prompt:
            print("No prompt. Exiting.")
            sys.exit(1)
        try:
            project_name = input("Project name (Enter to auto-generate): ").strip() or None
        except EOFError:
            pass

    coord = Coordinator(config)

    try:
        result = await coord.run(user_prompt, project_name=project_name)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except LLMError as exc:
        print(f"\nLLM Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFatal: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        await coord.llm.close()

    # Print summary
    print("\n" + "=" * 60)
    if result["status"] == "complete":
        print("  PROJECT COMPLETE")
        print(f"  Root : {result['project_root']}")
        print(f"  Tasks: {result['tasks_completed']}/{result['tasks_total']}")
    else:
        print(f"  PROJECT {result['status'].upper()}")
        if result.get("phase"):
            print(f"  Phase: {result['phase']}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        if result.get("failed_tasks"):
            print(f"  Failed tasks: {', '.join(result['failed_tasks'])}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
