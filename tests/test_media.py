import json
from pathlib import Path
import sys
import tempfile
import unittest

from webslm_trainer.media import MediaTrainConfig, prepare_media_run, read_media_dataset


class MediaTests(unittest.TestCase):
    def test_audio_and_video_runs_prepare_backend_inputs(self):
        for mode, model in (("audio", "Qwen/Qwen2.5-Omni-3B"),
                            ("video", "Qwen/Qwen3-VL-2B-Instruct")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                media = root / f"clip.{ 'wav' if mode == 'audio' else 'mp4'}"
                media.write_bytes(b"sample media bytes")
                data = root / "train.jsonl"
                data.write_text(json.dumps({
                    "prompt": [{"role": "user", "content": f"<{mode}>Describe the clip."}],
                    "completion": [{"role": "assistant", "content": "A short clip."}],
                    f"{mode}s": [media.name],
                }) + "\n", encoding="utf-8")
                config = MediaTrainConfig(mode, model, data, root / "output", 2,
                                          backend_python=sys.executable)
                run_file, command = prepare_media_run(config)
                self.assertEqual(command[:4], [sys.executable, "-m", "llamafactory.cli", "train"])
                run = json.loads(run_file.read_text(encoding="utf-8"))
                self.assertEqual(run["model_name_or_path"], model)
                self.assertEqual(run["max_steps"], 2)
                self.assertTrue(run["template"])
                if mode == "video":
                    self.assertGreaterEqual(run["video_max_pixels"], run["video_min_pixels"])
                rows = json.loads((run_file.parent / "dataset.json").read_text(encoding="utf-8"))
                self.assertEqual(rows[0][f"{mode}s"], [str(media.resolve())])

    def test_rejects_missing_media_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").write_bytes(b"data")
            data = root / "train.jsonl"
            data.write_text(json.dumps({
                "prompt": [{"role": "user", "content": "Describe it."}],
                "completion": [{"role": "assistant", "content": "A clip."}],
                "audios": ["clip.wav"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker"):
                read_media_dataset(MediaTrainConfig("audio", "Qwen/Qwen2.5-Omni-3B", data, root / "out"))


if __name__ == "__main__":
    unittest.main()
