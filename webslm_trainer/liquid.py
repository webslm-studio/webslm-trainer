"""LFM2/LFM2.5-specific text fine-tuning recipe.

Liquid's conv-attention hybrid uses projection names that differ from the
usual Llama/Qwen names. Keep this recipe separate from the generic SLM path.
"""

LFM_MODEL_TYPE = "lfm2"
LFM_LORA_TARGETS = ("w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "out_proj", "in_proj")


def supports(model_type: str) -> bool:
    return model_type == LFM_MODEL_TYPE


def target_modules() -> list[str]:
    return list(LFM_LORA_TARGETS)
