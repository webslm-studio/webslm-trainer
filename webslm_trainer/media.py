"""Audio/video LoRA path through a separately installed LLaMA-Factory backend."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from .catalog import known_template
from .config import validate_conversation


@dataclass(frozen=True)
class MediaTrainConfig:
    modality: str
    model: str
    dataset: Path
    output: Path
    steps: int = 30
    template: str = ""
    precision: str = "bf16"
    backend_python: str = ""


def read_media_dataset(config: MediaTrainConfig) -> list[dict]:
    if config.modality not in ("audio", "video"):
        raise ValueError("Media mode must be audio or video.")
    if not config.model.strip():
        raise ValueError("Choose a model ID.")
    if not isinstance(config.steps, int) or not 1 <= config.steps <= 100000:
        raise ValueError("Steps must be between 1 and 100000.")
    if config.precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("Precision must be bf16, fp16, or fp32.")
    if not config.dataset.is_file() or config.dataset.suffix.lower() != ".jsonl":
        raise ValueError("Dataset must be an existing .jsonl file.")
    if config.output.exists() and (not config.output.is_dir() or any(config.output.iterdir())):
        raise ValueError("Output directory must be new or empty to protect existing files.")
    if not (config.template or known_template(config.model, config.modality)):
        raise ValueError("Enter the LLaMA-Factory template name for this model.")
    key = "audios" if config.modality == "audio" else "videos"
    rows = []
    with config.dataset.open("r", encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {number}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(row, dict) or set(row) != {"prompt", "completion", key}:
                raise ValueError(f"Line {number}: use prompt, completion, and {key} fields.")
            validate_conversation(row, number)
            media = row[key]
            if not isinstance(media, list) or not media or not all(isinstance(item, str) and item for item in media):
                raise ValueError(f"Line {number}: {key} must contain file paths.")
            placeholders = sum(message["content"].count(f"<{config.modality}>") for message in row["prompt"])
            if placeholders != len(media):
                raise ValueError(f"Line {number}: each {key} file needs one <{config.modality}> marker in the prompt.")
            resolved = []
            for item in media:
                path = Path(item)
                path = path if path.is_absolute() else config.dataset.parent / path
                if not path.is_file():
                    raise ValueError(f"Line {number}: media file not found: {path}")
                resolved.append(str(path.resolve()))
            rows.append({"messages": row["prompt"] + row["completion"], key: resolved})
    if not rows:
        raise ValueError("Dataset has no examples.")
    return rows


def backend_command(config: MediaTrainConfig, run_file: Path) -> list[str]:
    if config.backend_python:
        executable = config.backend_python
        if not Path(executable).is_file() and not shutil.which(executable):
            raise RuntimeError(f"LLaMA-Factory Python was not found: {executable}")
        return [executable, "-m", "llamafactory.cli", "train", str(run_file)]
    executable = shutil.which("llamafactory-cli")
    if not executable:
        raise RuntimeError("Install LLaMA-Factory in a separate environment and set its Python path with --backend-python.")
    return [executable, "train", str(run_file)]


def prepare_media_run(config: MediaTrainConfig) -> tuple[Path, list[str]]:
    rows = read_media_dataset(config)
    run_dir = config.output / "run"
    config_file = run_dir / "train.json"
    command = backend_command(config, config_file)
    run_dir.mkdir(parents=True, exist_ok=False)
    key = "audios" if config.modality == "audio" else "videos"
    (run_dir / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    dataset_info = {"webslm_media": {"file_name": "dataset.json", "formatting": "sharegpt",
                    "columns": {"messages": "messages", key: key},
                    "tags": {"role_tag": "role", "content_tag": "content",
                             "user_tag": "user", "assistant_tag": "assistant"}}}
    (run_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
    training = {
        "model_name_or_path": config.model,
        "trust_remote_code": config.model in {"Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-Omni-3B"},
        "stage": "sft", "do_train": True, "finetuning_type": "lora",
        "lora_rank": 8, "lora_target": "all", "dataset": "webslm_media",
        "dataset_dir": str(run_dir.resolve()), "media_dir": str(config.dataset.parent.resolve()),
        "template": config.template or known_template(config.model, config.modality),
        "cutoff_len": 2048, "max_steps": config.steps,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4, "logging_steps": 1, "save_strategy": "no",
        "output_dir": str((config.output / "adapter").resolve()),
        "report_to": "none", "dataloader_num_workers": 0,
        "bf16": config.precision == "bf16", "fp16": config.precision == "fp16",
    }
    if config.modality == "video":
        # LLaMA-Factory defaults video_min_pixels to 16*16 (256); keep the
        # cap above that floor while limiting frame memory use.
        training.update(video_max_pixels=16384, video_min_pixels=256, video_fps=1.0, video_maxlen=32)
    config_file.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} {config.modality} examples in {run_dir}", flush=True)
    return config_file, command


def train_media(config: MediaTrainConfig) -> Path:
    run_file, command = prepare_media_run(config)
    print(f"Running LLaMA-Factory with {run_file}", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"LLaMA-Factory stopped with exit code {result.returncode}.")
    adapter = config.output / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"Training ended without a LoRA adapter in {adapter}.")
    print(f"Saved LoRA adapter to {adapter}", flush=True)
    return adapter
