"""Command-line entry point shared by desktop and terminal users."""

import argparse
from pathlib import Path
import sys

from .config import MODEL_CHOICES, TrainConfig, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="webslm-trainer", description="Small-model LoRA trainer")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("desktop", help="Open the desktop app")
    for name in ("check", "train"):
        command = commands.add_parser(name, help="Validate a dataset" if name == "check" else "Train an adapter")
        command.add_argument("--model", default=MODEL_CHOICES[0], help="Hugging Face model ID")
        command.add_argument("--dataset", required=True, type=Path, help="Conversational JSONL file")
        command.add_argument("--output", required=True, type=Path, help="New or empty adapter directory")
        command.add_argument("--steps", type=int, default=30, help="Training steps (default: 30)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "desktop":
        from .desktop import main as desktop_main
        desktop_main()
        return 0
    config = TrainConfig(args.model, args.dataset, args.output, args.steps)
    try:
        rows = validate_config(config)
        if args.command == "check":
            print(f"Valid: {rows} training examples; {config.model}; output {config.output}")
        else:
            from .training import train
            train(config)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
