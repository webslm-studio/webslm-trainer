"""Small verified starter catalog plus live Hugging Face SLM discovery."""

from dataclasses import dataclass
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SLM_PARAMETER_LIMIT = 4_000_000_000


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    parameters: int
    modes: str
    license: str
    backend: str
    template: str = ""


STARTER_MODELS = (
    ModelEntry("Qwen/Qwen2.5-0.5B-Instruct", 500_000_000, "text", "apache-2.0", "trl"),
    ModelEntry("Qwen/Qwen3-0.6B", 600_000_000, "text", "apache-2.0", "trl"),
    ModelEntry("HuggingFaceTB/SmolLM2-360M-Instruct", 360_000_000, "text", "apache-2.0", "trl"),
    ModelEntry("LiquidAI/LFM2.5-230M", 230_000_000, "text", "lfm1.0", "trl-liquid"),
    ModelEntry("LiquidAI/LFM2.5-350M", 350_000_000, "text", "lfm1.0", "trl-liquid"),
    ModelEntry("LiquidAI/LFM2.5-350M-Base", 350_000_000, "text", "lfm1.0", "trl-liquid"),
    ModelEntry("LiquidAI/LFM2.5-1.2B-Instruct", 1_200_000_000, "text", "lfm1.0", "trl-liquid"),
    ModelEntry("Qwen/Qwen2.5-Omni-3B", 5_537_120_672, "audio,video", "qwen-research", "llamafactory", "qwen2_omni"),
    ModelEntry("Qwen/Qwen3-VL-2B-Instruct", 2_000_000_000, "video", "apache-2.0", "llamafactory", "qwen3_vl_nothink"),
)


def starter_models(mode: str | None = None) -> list[ModelEntry]:
    return [entry for entry in STARTER_MODELS if mode is None or mode in entry.modes.split(",")]


def known_template(model_id: str, mode: str) -> str:
    return next((item.template for item in starter_models(mode) if item.model_id == model_id), "")


def search_hub_models(query: str, limit: int = 40, mode: str | None = None) -> list[ModelEntry]:
    """Search public <=4B repositories; results are candidates, not compatibility claims."""
    if not query.strip():
        raise ValueError("Enter a model name or organization to search.")
    if not 1 <= limit <= 200:
        raise ValueError("Search limit must be between 1 and 200.")
    tasks = {"text": {"text-generation"},
             "audio": {"any-to-any", "audio-text-to-text", "automatic-speech-recognition"},
             "video": {"any-to-any", "image-text-to-text", "video-text-to-text"}}
    if mode is not None and mode not in tasks:
        raise ValueError("Mode must be text, audio, or video.")
    params = [("search", query.strip()), ("num_parameters", "max:4B"),
              ("sort", "downloads"), ("direction", "-1"), ("limit", str(limit))]
    params += [("expand", key) for key in ("safetensors", "cardData", "gated", "pipeline_tag", "library_name")]
    url = "https://huggingface.co/api/models?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "webslm-trainer/0.1"})
    with urlopen(request, timeout=15) as response:
        records = json.load(response)
    entries = []
    for record in records:
        total = (record.get("safetensors") or {}).get("total")
        if not isinstance(total, int) or not 0 < total <= SLM_PARAMETER_LIMIT:
            continue
        if record.get("gated") or record.get("library_name") not in (None, "transformers"):
            continue
        model_id = record.get("id")
        if not isinstance(model_id, str) or "/" not in model_id:
            continue
        if re.search(r"(?:gguf|gptq|awq|4bit|8bit|fp8|w4a16|quantized|mlx-community|bnb-4bit)",
                     model_id, flags=re.IGNORECASE):
            continue
        card = record.get("cardData") or {}
        license_name = card.get("license_name") if card.get("license") == "other" else card.get("license")
        license_name = license_name or "check model card"
        pipeline = record.get("pipeline_tag") or "unknown"
        if mode is not None and pipeline not in tasks[mode]:
            continue
        entries.append(ModelEntry(model_id, total, pipeline, license_name, "candidate"))
    return entries


def format_parameters(count: int) -> str:
    return f"{count / 1_000_000_000:.2f}B" if count >= 1_000_000_000 else f"{count / 1_000_000:.0f}M"
