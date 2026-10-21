import io
import json
import unittest
from unittest.mock import patch

from webslm_trainer.catalog import search_hub_models, starter_models


class CatalogTests(unittest.TestCase):
    def test_search_keeps_small_trainable_candidates_for_mode(self):
        records = [
            {"id": "LiquidAI/LFM2.5-350M", "safetensors": {"total": 354000000},
             "pipeline_tag": "text-generation", "library_name": "transformers", "gated": False,
             "cardData": {"license": "other", "license_name": "lfm1.0"}},
            {"id": "example/Large-8B-AWQ-4bit", "safetensors": {"total": 3000000000},
             "pipeline_tag": "text-generation", "library_name": "transformers", "gated": False},
            {"id": "example/Audio-1B", "safetensors": {"total": 1000000000},
             "pipeline_tag": "audio-text-to-text", "library_name": "transformers", "gated": False},
        ]
        response = io.BytesIO(json.dumps(records).encode("utf-8"))
        with patch("webslm_trainer.catalog.urlopen", return_value=response):
            matches = search_hub_models("LFM", mode="text")
        self.assertEqual([entry.model_id for entry in matches], ["LiquidAI/LFM2.5-350M"])
        self.assertEqual(matches[0].license, "lfm1.0")

    def test_media_starters_are_visible(self):
        self.assertIn("Qwen/Qwen2.5-Omni-3B", [entry.model_id for entry in starter_models("audio")])
        self.assertIn("Qwen/Qwen3-VL-2B-Instruct", [entry.model_id for entry in starter_models("video")])


if __name__ == "__main__":
    unittest.main()
