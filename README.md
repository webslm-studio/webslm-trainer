# WebSLM Trainer

A small, local desktop app and CLI for **supervised LoRA fine-tuning of text SLMs**. Choose a model, select a JSONL dataset, train, and save an adapter. The app adds a narrow workflow around [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft); it does not copy or bundle Unsloth, LLaMA-Factory, or Liquid's LEAP-Finetune code.

The presets are [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), plus Liquid's [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M), [350M](https://huggingface.co/LiquidAI/LFM2.5-350M), [350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base), and [1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct). You can also type another Hugging Face model ID into the desktop model box or pass it through `--model`. This supports text causal models compatible with Transformers, TRL, and PEFT; no single trainer can promise every SLM architecture or license works. The trainer inspects the model architecture, uses a dedicated [Liquid LFM2 recipe](webslm_trainer/liquid.py) with [Liquid AI's LoRA target modules](https://github.com/Liquid4All/docs/blob/main/guides/migration-guide.mdx), and uses PEFT's `all-linear` target for other models. TRL applies the selected model's chat template to the conversational dataset.

## Install

Use Python 3.10+ in a virtual environment. Install [PyTorch for your hardware](https://pytorch.org/get-started/locally/) first, then install the training dependencies:

```sh
python -m pip install -e ".[train]"
```

The desktop UI uses Python's built-in Tkinter. Some Python distributions require a separate Tk package. Training is possible on a CPU but will be slow; a compatible GPU is strongly recommended. Model weights download from Hugging Face on the first run. Neither the repository nor the Python package includes model weights.

## Train

Open the desktop app:

```sh
python -m webslm_trainer desktop
```

Or use the CLI:

```sh
python -m webslm_trainer check --dataset examples/tiny.jsonl --output outputs/my-adapter
python -m webslm_trainer train --model LiquidAI/LFM2.5-350M --dataset examples/tiny.jsonl --output outputs/my-adapter --steps 30
```

The output directory must be new or empty. Training writes a PEFT LoRA adapter there, plus a small `webslm_training.json` record. The base model is still needed to use the adapter. Stop the desktop run with **Stop**; partial output from an interrupted run is not considered a finished adapter.

## Dataset

Use UTF-8 JSONL, one example per line. Each row has a conversational `prompt` ending in a user turn and a single assistant `completion`. This lets TRL use the selected model's own chat template and train only on the assistant completion.

```json
{"prompt":[{"role":"user","content":"What is an SLM?"}],"completion":[{"role":"assistant","content":"An SLM is a small language model."}]}
```

See [`examples/tiny.jsonl`](examples/tiny.jsonl) for a small format example. It only checks the workflow; useful tuning needs a larger, high-quality dataset. The trainer currently uses a 512-token sequence limit, batch size 1 with four accumulation steps, LoRA rank 8, and no intermediate checkpoints.

## Scope and licenses

This first version supports text causal models, supervised fine-tuning, and adapter export. It has no quantization, distributed training, model merging, evaluation suite, or multimodal pipeline. LLaMA-Factory and Liquid's [LEAP-Finetune](https://github.com/Liquid4All/liquid-finetune) offer those broader capabilities if they become necessary.

WebSLM Trainer's own code is MIT licensed. Model weights have separate licenses. In particular, Liquid LFM2.5 models use the [LFM 1.0 license](https://huggingface.co/LiquidAI/LFM2.5-350M) rather than this repository's MIT license; review the model terms before use or redistribution.
