"""Command-line entry point shared by desktop and terminal users."""

import argparse
from pathlib import Path
import sys
from urllib.error import URLError

from .catalog import format_parameters, search_hub_models, starter_models
from .config import MODEL_CHOICES, TrainConfig, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="webslm-trainer", description="Small-model LoRA trainer")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("desktop", help="Open the desktop app")
    models = commands.add_parser("models", help="List starter SLMs or search Hugging Face")
    models.add_argument("--search", help="Search public models with 4B parameters or fewer")
    models.add_argument("--limit", type=int, default=40, help="Maximum search results (default: 40)")
    models.add_argument("--mode", choices=("text", "audio", "video"), help="Filter search results by task")
    for name in ("check", "train"):
        command = commands.add_parser(name, help="Validate a dataset" if name == "check" else "Train an adapter")
        command.add_argument("--model", default=MODEL_CHOICES[0], help="Hugging Face model ID")
        command.add_argument("--dataset", required=True, type=Path, help="Conversational JSONL file")
        command.add_argument("--output", required=True, type=Path, help="New or empty adapter directory")
        command.add_argument("--steps", type=int, default=30, help="Training steps (default: 30)")
    for name in ("check-media", "train-media"):
        command = commands.add_parser(name, help="Validate audio/video data" if name == "check-media" else "Train an audio/video adapter")
        command.add_argument("--modality", required=True, choices=("audio", "video"))
        command.add_argument("--model", required=True, help="Multimodal Hugging Face model ID")
        command.add_argument("--dataset", required=True, type=Path, help="JSONL with audio/video paths")
        command.add_argument("--output", required=True, type=Path, help="New or empty run directory")
        command.add_argument("--steps", type=int, default=30)
        command.add_argument("--template", default="", help="LLaMA-Factory template (auto for starter models)")
        command.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
        command.add_argument("--backend-python", default="", help="Python executable in a separate LLaMA-Factory environment")
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
    if args.command == "models":
        try:
            entries = search_hub_models(args.search, args.limit, args.mode) if args.search else starter_models(args.mode)
        except (ValueError, OSError, URLError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        for item in entries:
            print(f"{format_parameters(item.parameters):>5}  {item.modes:<18}  {item.model_id}  [{item.license}]")
        return 0
    if args.command in ("check-media", "train-media"):
        from .media import MediaTrainConfig, read_media_dataset, train_media
        config = MediaTrainConfig(args.modality, args.model, args.dataset, args.output,
                                  args.steps, args.template, args.precision, args.backend_python)
        try:
            if args.command == "check-media":
                print(f"Valid: {len(read_media_dataset(config))} {config.modality} examples; {config.model}")
            else:
                train_media(config)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
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
