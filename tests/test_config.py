import json
from pathlib import Path
import tempfile
import unittest

from webslm_trainer.config import TrainConfig, validate_config
from webslm_trainer import liquid


VALID_ROW = {
    "prompt": [{"role": "user", "content": "Question?"}],
    "completion": [{"role": "assistant", "content": "Answer."}],
}


class ConfigTests(unittest.TestCase):
    def test_valid_conversation_and_liquid_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.jsonl"
            dataset.write_text(json.dumps(VALID_ROW) + "\n", encoding="utf-8")
            config = TrainConfig("LiquidAI/LFM2.5-350M", dataset, Path(directory) / "adapter")
            self.assertEqual(validate_config(config), 1)
            self.assertTrue(liquid.supports("lfm2"))
            self.assertFalse(liquid.supports("qwen2"))
            self.assertEqual(liquid.target_modules(), list(liquid.LFM_LORA_TARGETS))

    def test_rejects_non_conversational_data(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.jsonl"
            dataset.write_text(json.dumps({"prompt": "Question?", "completion": "Answer."}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt must be"):
                validate_config(TrainConfig("model", dataset, Path(directory) / "adapter"))

    def test_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.jsonl"
            dataset.write_text(json.dumps(VALID_ROW), encoding="utf-8")
            output = Path(directory) / "adapter"
            output.mkdir()
            (output / "important.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                validate_config(TrainConfig("model", dataset, output))


if __name__ == "__main__":
    unittest.main()
