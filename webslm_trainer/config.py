"""Model presets and dataset validation. No ML dependencies are imported here."""

from dataclasses import dataclass
import json
from pathlib import Path

from .catalog import starter_models

MODEL_CHOICES = tuple(entry.model_id for entry in starter_models("text"))


@dataclass(frozen=True)
class TrainConfig:
    model: str
    dataset: Path
    output: Path
    steps: int = 30


def validate_dataset(path: Path) -> int:
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        raise ValueError("Dataset must be an existing .jsonl file.")
    count = 0
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
                raise ValueError(f"Line {line_number}: use only prompt and completion fields.")
            validate_conversation(row, line_number)
            count += 1
    if not count:
        raise ValueError("Dataset has no examples.")
    return count


def validate_conversation(row: dict, line_number: int) -> None:
    prompt, completion = row["prompt"], row["completion"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Line {line_number}: prompt must be a nonempty list of messages.")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"Line {line_number}: completion must contain one assistant message.")
    for message in prompt + completion:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"Line {line_number}: each message needs role and content.")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"Line {line_number}: unsupported message role.")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"Line {line_number}: message content cannot be empty.")
    if completion[0]["role"] != "assistant" or prompt[-1]["role"] != "user":
        raise ValueError(f"Line {line_number}: end the prompt with a user and completion with an assistant.")


def validate_config(config: TrainConfig) -> int:
    if not config.model.strip():
        raise ValueError("Choose a model ID.")
    if not isinstance(config.steps, int) or not 1 <= config.steps <= 100000:
        raise ValueError("Steps must be between 1 and 100000.")
    if config.output.resolve() == config.dataset.resolve():
        raise ValueError("Output directory cannot be the dataset file.")
    if config.output.exists() and (not config.output.is_dir() or any(config.output.iterdir())):
        raise ValueError("Output directory must be new or empty to protect existing files.")
    return validate_dataset(config.dataset)
