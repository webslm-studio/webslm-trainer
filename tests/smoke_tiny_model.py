"""Optional CPU end-to-end smoke test; needs the [train] extra and a tokenizer download."""

import json
from pathlib import Path
import tempfile

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from webslm_trainer.config import TrainConfig
from webslm_trainer.training import train


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    model_dir = root / "tiny-model"
    model_dir.mkdir()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}{{ eos_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128,
    ))
    model.save_pretrained(model_dir)
    dataset = root / "tiny.jsonl"
    dataset.write_text(json.dumps({
        "prompt": [{"role": "user", "content": "Hi"}],
        "completion": [{"role": "assistant", "content": "Hello"}],
    }) + "\n", encoding="utf-8")
    output = train(TrainConfig(str(model_dir), dataset, root / "adapter", 1))
    assert (output / "adapter_config.json").is_file()
    assert (output / "webslm_training.json").is_file()
    print("CPU end-to-end smoke test passed")
