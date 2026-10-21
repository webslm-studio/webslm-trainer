"""Exercise both recipe routes without downloading model weights."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from webslm_trainer.config import TrainConfig
from webslm_trainer.training import train


class TrainingTests(unittest.TestCase):
    def test_liquid_architecture_routes_to_its_own_recipe(self):
        for model_id, architecture, expected in (
            ("some-org/renamed-liquid", "lfm2", ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]),
            ("some-org/other-slm", "qwen2", "all-linear"),
        ):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.jsonl"
                path.write_text(json.dumps({
                    "prompt": [{"role": "user", "content": "Hi"}],
                    "completion": [{"role": "assistant", "content": "Hello"}],
                }), encoding="utf-8")
                output = Path(directory) / "adapter"
                captured = {}

                class FakeLoraConfig:
                    def __init__(self, **kwargs):
                        self.target_modules = kwargs["target_modules"]
                        captured["modules"] = self.target_modules

                class FakeTrainer:
                    def __init__(self, **kwargs):
                        self.model = SimpleNamespace(print_trainable_parameters=lambda: None)
                        captured["tokenizer"] = kwargs["processing_class"]

                    def train(self):
                        captured["trained"] = True

                    def save_model(self, folder):
                        Path(folder).mkdir()
                        (Path(folder) / "adapter_config.json").write_text("{}", encoding="utf-8")

                fake_torch = ModuleType("torch")
                fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
                fake_datasets = ModuleType("datasets")
                fake_datasets.load_dataset = lambda *args, **kwargs: ["row"]
                fake_peft = ModuleType("peft")
                fake_peft.LoraConfig = FakeLoraConfig
                fake_trl = ModuleType("trl")
                fake_trl.SFTConfig = lambda **kwargs: kwargs
                fake_trl.SFTTrainer = FakeTrainer
                fake_transformers = ModuleType("transformers")
                fake_transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(model_type=architecture))
                fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(chat_template="template"))
                modules = {"torch": fake_torch, "datasets": fake_datasets, "peft": fake_peft,
                           "trl": fake_trl, "transformers": fake_transformers}

                with patch.dict(sys.modules, modules):
                    result = train(TrainConfig(model_id, path, output, 1))
                self.assertEqual(result, output)
                self.assertEqual(captured["modules"], expected)
                self.assertTrue(captured["trained"])
                self.assertEqual(captured["tokenizer"].chat_template, "template")
                metadata = json.loads((output / "webslm_training.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["architecture"], architecture)


if __name__ == "__main__":
    unittest.main()
