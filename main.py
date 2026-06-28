"""Single root entrypoint that dispatches to workflow tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys


TASK_MODULES = {
    "classify-base-model": "classification_workflow.classify_base_model",
    "train-finetuned-model": "classification_workflow.run_model_finetuning",
    "classify-finetuned-model": "classification_workflow.classify_finetuned_model",
    "generate-sentiment-scores": "classification_workflow.generate_sentiment_scores",
}

TASK_SEQUENCE = (
    "classify-base-model",
    "train-finetuned-model",
    "classify-finetuned-model",
    "generate-sentiment-scores",
)


def run_task(task_name: str, task_args: list[str] | None = None) -> None:
    module_name = TASK_MODULES[task_name]
    command = [sys.executable, "-m", module_name, *(task_args or [])]

    print(f"\n>>> Running task: {task_name}")
    print("Command:", " ".join(command))
    sys.stdout.flush()

    subprocess.run(command, check=True)


def run_all_tasks() -> None:
    print("Running the full workflow with default task arguments.")
    for task_name in TASK_SEQUENCE:
        run_task(task_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full sentiment workflow or dispatch an individual task."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=("all", *TASK_MODULES.keys()),
        default=None,
        help="Workflow task to run. Defaults to the main pipeline.",
    )
    parser.add_argument(
        "-t", "--task",
        dest="task_flag",
        choices=("all", *TASK_MODULES.keys()),
        help="Workflow task to run (alternative to positional argument).",
    )
    args, remaining_args = parser.parse_known_args()
    
    # Handle both positional and flag-based task specification
    task_name = (
        args.task_flag 
        if args.task_flag is not None 
        else args.task if args.task is not None 
        else "all"
    )
    
    if task_name == "all":
        if remaining_args:
            parser.error("Additional arguments are only supported for individual tasks.")
        try:
            run_all_tasks()
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc
        return

    try:
        run_task(task_name, remaining_args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
