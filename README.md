# WebSLM Trainer

WebSLM Trainer is a small desktop and command-line front end for supervised LoRA fine-tuning of **small language models (SLMs)**. It supports text directly with [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft). Audio and video runs use a separate [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) installation. [Liquid LFM2/LFM2.5](https://github.com/Liquid4All/docs/blob/main/guides/migration-guide.mdx) text models get an architecture-specific recipe.

This repository contains WebSLM's own UI, validation, recipes, sample data, and launch configuration. It does not bundle model weights or copy Unsloth, LLaMA-Factory, or Liquid's LEAP-Finetune source.

## What counts as an SLM here?

There is no universal parameter cutoff. For **this product**, an SLM has a **language backbone of roughly 4 billion parameters or fewer** and performs a language task, including a model that reads text, audio, or video and produces text. Live Hugging Face search filters repositories to at most 4B reported parameters; the curated multimodal starters may have additional encoder or speech-generator weights. For example, Qwen2.5-Omni-3B has a 3B language backbone but its model repository reports about 5.5B total parameters. The size label is not a guarantee of low memory use. You can type another model ID, including one outside the catalog; the selected backend and model architecture must still support fine-tuning it.

Fine-tuning is useful when a compact model needs to follow a repeatable task, vocabulary, response format, or domain policy using your own examples. It does not give a model reliable facts that are absent from the data, and a small sample corpus does not prove improved accuracy. Keep held-out examples to compare the base model and adapter before deploying.

## What is available

| Data | Training path | Starter models | Dataset | Verification |
| --- | --- | --- | --- | --- |
| Text | TRL + PEFT LoRA | Qwen 0.5B/0.6B, SmolLM2 360M, Liquid LFM2.5 230M/350M/1.2B | Conversational JSONL | One-step CPU training and adapter save tested with a tiny local model and Liquid LFM2.5-230M |
| Audio input → text | LLaMA-Factory LoRA | [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | JSONL with `<audio>` markers and audio file paths | Corpus validation and backend configuration tested; full GPU training requires a compatible machine |
| Video input → text | LLaMA-Factory LoRA | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), Qwen2.5-Omni-3B | JSONL with `<video>` markers and video file paths | Corpus validation and backend configuration tested; full GPU training requires a compatible machine |
| Biological or molecular sequences | Pending a defined data type and task | — | — | Not implemented |

The desktop app has a model picker and live Hugging Face search. The CLI lists starters with `python -m webslm_trainer models` and searches public models at or below 4B with `python -m webslm_trainer models --search Qwen --mode text` (or `audio` / `video`). Search results are **candidates**, not a promise that every architecture, tokenizer, license, and hardware combination works. The text backend requires a Transformers causal LM with a chat template. Audio/video require a LLaMA-Factory template; starter models fill that in automatically.

## Install and open the desktop app

Use Python 3.10+ for WebSLM Trainer. On Windows, from this repository:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[train]"
.venv\Scripts\python -m webslm_trainer desktop
```

For GPU training, install the [PyTorch build for your hardware](https://pytorch.org/get-started/locally/) in the environment **before** the `pip install` step. The desktop interface uses Tkinter from Python; some Linux distributions package Tk separately. Model weights download from Hugging Face on first use.

### Audio and video backend

LLaMA-Factory has its own dependency versions, so install it in a **separate** Python 3.11–3.13 environment. Follow its [installation instructions](https://github.com/hiyouga/LlamaFactory#installation) and GPU PyTorch instructions. A Windows example is:

```powershell
py -3.11 -m venv .venv-media
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git ..\LLaMA-Factory
.venv-media\Scripts\python -m pip install -e ..\LLaMA-Factory
```

In the desktop app, choose **audio** or **video** and set **LLaMA-Factory Python** to `.venv-media\Scripts\python.exe`. If `llamafactory-cli` is already on your `PATH`, leave that field blank. Audio/video training needs a suitable GPU and more memory than the text examples; choose `bf16` on supported GPUs or `fp16` otherwise. The user supplies model weights and must review each model's license. The Qwen2.5-Omni-3B card labels its license `qwen-research`; Liquid weights use `lfm1.0`.

## A concrete text use case

[`examples/corpus/harbor_bike_faq.md`](examples/corpus/harbor_bike_faq.md) defines a **fictional** bike repair shop's policy. [`harbor_bike_train.jsonl`](examples/corpus/harbor_bike_train.jsonl) contains 24 original question/answer examples. They teach a small model to answer consistently about hours, flat repairs, appointments, and e-bike limits, and to avoid inventing prices or giving unsafe ride advice. [`harbor_bike_eval.jsonl`](examples/corpus/harbor_bike_eval.jsonl) contains five held-out questions to compare before and after training. These examples demonstrate the data format and workflow; they are too small to claim a quality improvement.

```powershell
.venv\Scripts\python -m webslm_trainer check --dataset examples/corpus/harbor_bike_train.jsonl --output outputs/harbor-text
.venv\Scripts\python -m webslm_trainer train --model LiquidAI/LFM2.5-350M --dataset examples/corpus/harbor_bike_train.jsonl --output outputs/harbor-text --steps 30
```

For a model with another architecture, replace the `--model` value. The trainer inspects `AutoConfig.model_type`; `lfm2` selects [Liquid's LoRA targets](https://github.com/Liquid4All/docs/blob/main/guides/migration-guide.mdx), including for renamed LFM checkpoints. Other causal LMs use PEFT's `all-linear` selection. TRL applies the **selected model's chat template** and trains on the assistant completion. The result is a small PEFT adapter in the output directory; the base model is still needed to run it.

Each text JSONL row has a conversational prompt ending in a user message and one assistant completion:

```json
{"prompt":[{"role":"user","content":"Do you repair e-bike batteries?"}],"completion":[{"role":"assistant","content":"No. We accept mechanical e-bike repairs but do not diagnose or repair batteries or chargers."}]}
```

## Audio and video corpus format

The included synthetic clips are tiny **format checks**. They show how paths and labels connect; they are not a useful model training corpus. Replace them with representative media and high-quality answers for your task. The original clips can be regenerated with [`examples/media/generate.py`](examples/media/generate.py) using `ffmpeg` or `imageio-ffmpeg`.

```powershell
.venv\Scripts\python -m webslm_trainer check-media --modality audio --model Qwen/Qwen2.5-Omni-3B --dataset examples/corpus/audio_train.jsonl --output outputs/audio-run
.venv\Scripts\python -m webslm_trainer train-media --modality audio --model Qwen/Qwen2.5-Omni-3B --dataset examples/corpus/audio_train.jsonl --output outputs/audio-run --steps 30 --backend-python .venv-media\Scripts\python.exe

.venv\Scripts\python -m webslm_trainer check-media --modality video --model Qwen/Qwen3-VL-2B-Instruct --dataset examples/corpus/video_train.jsonl --output outputs/video-run
.venv\Scripts\python -m webslm_trainer train-media --modality video --model Qwen/Qwen3-VL-2B-Instruct --dataset examples/corpus/video_train.jsonl --output outputs/video-run --steps 30 --backend-python .venv-media\Scripts\python.exe
```

An audio row contains an `<audio>` marker and matching `audios` path list; a video row uses `<video>` and `videos`. Relative paths resolve from the JSONL file's directory. WebSLM checks that files exist, converts the rows to LLaMA-Factory's documented ShareGPT layout, writes the exact training config under the output directory, and launches its CLI. The adapter is expected under `output/adapter`. For another LLaMA-Factory-compatible model, pass its model ID and `--template TEMPLATE_NAME` after checking its official model/template documentation.

```json
{"prompt":[{"role":"user","content":"<audio>What is happening?"}],"completion":[{"role":"assistant","content":"Two tones rise in pitch."}],"audios":["../media/rising_tones.wav"]}
```

Outputs must be new or empty directories so an earlier adapter cannot be overwritten. An interrupted media run leaves its configuration and partial output in place for inspection.

## Architecture and limits

The installed WebSLM package remains small. Text training uses its direct TRL/PEFT path; audio/video launch LLaMA-Factory only when selected. Liquid text uses a distinct recipe because its hybrid convolution/attention architecture names LoRA modules differently. [LEAP-Finetune](https://github.com/Liquid4All/liquid-finetune) is a useful larger Liquid-specific option for distributed and vision-language work; [Unsloth](https://github.com/unslothai/unsloth) focuses on accelerated text fine-tuning. We use their documented interfaces and recipes, not their source trees.

Current WebSLM runs are supervised LoRA fine-tuning only. Quantization, distributed training, evaluation automation, model merging, image-only workflows, speech generation, and biological/molecular sequence objectives are not part of this version. A custom model ID is possible, but support depends on the backend, training objective, chat or media template, device memory, and model license. The repository is MIT licensed; model weights and external backends have their own terms.

Run the lightweight tests with `python -m unittest discover -s tests`. An optional one-step CPU integration check is `python tests/smoke_tiny_model.py` after installing `[train]`; it downloads the GPT-2 tokenizer but creates and trains a tiny random model locally.
