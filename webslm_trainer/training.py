"""The complete training path: JSONL -> TRL/PEFT -> LoRA adapter."""

import json
from pathlib import Path

from .config import TrainConfig, validate_config
from . import liquid


def train(config: TrainConfig) -> Path:
    examples = validate_config(config)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training packages are missing. Install GPU-compatible PyTorch, then run "
            "python -m pip install -e '.[train]'."
        ) from exc

    print(f"Inspecting {config.model}", flush=True)
    model_config = AutoConfig.from_pretrained(config.model)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if not tokenizer.chat_template:
        raise ValueError("This model has no chat template. Choose a conversational text model.")
    is_liquid = liquid.supports(model_config.model_type)
    recipe = "Liquid LFM2" if is_liquid else "generic causal LM"
    modules = liquid.target_modules() if is_liquid else "all-linear"
    print(f"Recipe: {recipe} (architecture: {model_config.model_type})", flush=True)
    print(f"Loading {examples} examples from {config.dataset}", flush=True)
    dataset = load_dataset("json", data_files=str(config.dataset), split="train")
    precision = (
        {"bf16": True, "model_init_kwargs": {"dtype": torch.bfloat16}}
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else {"fp16": True, "model_init_kwargs": {"dtype": torch.float16}}
        if torch.cuda.is_available()
        else {}
    )
    args = SFTConfig(
        output_dir=str(config.output),
        max_steps=config.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_length=512,
        completion_only_loss=True,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        **precision,
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )
    print(f"Model: {config.model}", flush=True)
    print(f"LoRA targets: {lora.target_modules}", flush=True)
    trainer = SFTTrainer(
        model=config.model,
        args=args,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(config.output))
    metadata = {
        "base_model": config.model,
        "dataset_rows": examples,
        "steps": config.steps,
        "format": "conversational prompt/completion",
        "recipe": recipe,
        "architecture": model_config.model_type,
    }
    (config.output / "webslm_training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved LoRA adapter to {config.output}", flush=True)
    return config.output
