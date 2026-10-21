"""Small Tk desktop app for text, audio, and video LoRA runs."""

import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .catalog import format_parameters, known_template, search_hub_models, starter_models
from .config import TrainConfig, validate_config
from .media import MediaTrainConfig, read_media_dataset


class ModelBrowser:
    def __init__(self, parent: tk.Tk, mode: str, on_select):
        self.window = tk.Toplevel(parent)
        self.window.title("Browse small models")
        self.window.geometry("790x430")
        self.on_select = on_select
        self.mode = mode
        self.results: queue.Queue[tuple[list, str]] = queue.Queue()
        self.query = tk.StringVar()
        bar = ttk.Frame(self.window, padding=10)
        bar.pack(fill="x")
        ttk.Label(bar, text="Search Hugging Face (up to 4B parameters)").pack(side="left")
        ttk.Entry(bar, textvariable=self.query).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(bar, text="Search", command=self.search).pack(side="left")
        self.status = tk.StringVar(value="Starter models; double-click to select. Search results are candidates.")
        ttk.Label(self.window, textvariable=self.status, padding=(10, 0)).pack(anchor="w")
        self.tree = ttk.Treeview(self.window, columns=("size", "task", "license"), show="tree headings")
        self.tree.heading("#0", text="Model ID")
        self.tree.column("#0", width=390)
        for key, title, width in (("size", "Size", 65), ("task", "Task", 160), ("license", "License", 130)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree.bind("<Double-1>", self.select)
        self.show(starter_models(mode))
        self.window.after(100, self.drain)

    def show(self, models) -> None:
        self.tree.delete(*self.tree.get_children())
        for entry in models:
            self.tree.insert("", "end", text=entry.model_id,
                             values=(format_parameters(entry.parameters), entry.modes, entry.license))

    def search(self) -> None:
        query = self.query.get().strip()
        if not query:
            self.status.set("Enter a model name or organization.")
            return
        self.status.set("Searching Hugging Face...")

        def worker() -> None:
            try:
                self.results.put((search_hub_models(query, 100, self.mode), ""))
            except (ValueError, OSError) as exc:
                self.results.put(([], str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def drain(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            models, error = self.results.get_nowait()
            self.show(models)
            self.status.set(error or f"{len(models)} candidates found. Check each model card and training compatibility.")
        except queue.Empty:
            pass
        self.window.after(100, self.drain)

    def select(self, _event) -> None:
        selected = self.tree.selection()
        if selected:
            self.on_select(self.tree.item(selected[0], "text"))
            self.window.destroy()


class TrainerWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.closed = False
        root.title("WebSLM Trainer")
        root.minsize(780, 570)
        self.mode = tk.StringVar(value="text")
        self.model = tk.StringVar(value=starter_models("text")[0].model_id)
        self.dataset = tk.StringVar()
        self.output = tk.StringVar(value=str(Path.cwd() / "outputs" / "my-adapter"))
        self.steps = tk.StringVar(value="30")
        self.template = tk.StringVar()
        self.precision = tk.StringVar(value="bf16")
        self.backend_python = tk.StringVar()
        self.events: queue.Queue[str | None] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(9, weight=1)

        ttk.Label(frame, text="Data type").grid(row=0, column=0, sticky="w", pady=5)
        mode_box = ttk.Combobox(frame, textvariable=self.mode, values=("text", "audio", "video"), state="readonly")
        mode_box.grid(row=0, column=1, sticky="ew", pady=5)
        mode_box.bind("<<ComboboxSelected>>", self.on_mode)
        ttk.Label(frame, text="Model ID (editable)").grid(row=1, column=0, sticky="w", pady=5)
        self.model_box = ttk.Combobox(frame, textvariable=self.model, values=[m.model_id for m in starter_models("text")])
        self.model_box.grid(row=1, column=1, sticky="ew", pady=5)
        self.model_box.bind("<<ComboboxSelected>>", self.on_model)
        ttk.Button(frame, text="Browse SLMs", command=self.browse_models).grid(row=1, column=2, padx=(8, 0), pady=5)
        ttk.Label(frame, text="Dataset (.jsonl)").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.dataset).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Browse", command=self.pick_dataset).grid(row=2, column=2, padx=(8, 0), pady=5)
        ttk.Label(frame, text="Run output").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output).grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Browse", command=self.pick_output).grid(row=3, column=2, padx=(8, 0), pady=5)
        ttk.Label(frame, text="Steps").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.steps, width=10).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Media template").grid(row=5, column=0, sticky="w", pady=5)
        self.template_entry = ttk.Entry(frame, textvariable=self.template)
        self.template_entry.grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Label(frame, text="Precision").grid(row=6, column=0, sticky="w", pady=5)
        self.precision_box = ttk.Combobox(frame, textvariable=self.precision,
                                           values=("bf16", "fp16", "fp32"), state="readonly", width=10)
        self.precision_box.grid(row=6, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="LLaMA-Factory Python").grid(row=7, column=0, sticky="w", pady=5)
        self.backend_entry = ttk.Entry(frame, textvariable=self.backend_python)
        self.backend_entry.grid(row=7, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Browse", command=self.pick_backend).grid(row=7, column=2, padx=(8, 0), pady=5)
        controls = ttk.Frame(frame)
        controls.grid(row=8, column=0, columnspan=3, sticky="w", pady=10)
        self.start_button = ttk.Button(controls, text="Start training", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.log = tk.Text(frame, height=12, wrap="word", state="disabled")
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew")
        self.on_mode()
        root.after(100, self.drain_events)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def on_mode(self, _event=None) -> None:
        mode = self.mode.get()
        choices = starter_models(mode)
        self.model_box.configure(values=[entry.model_id for entry in choices])
        self.model.set(choices[0].model_id)
        self.on_model()
        state = "normal" if mode != "text" else "disabled"
        self.template_entry.configure(state=state)
        self.precision_box.configure(state="readonly" if mode != "text" else "disabled")
        self.backend_entry.configure(state=state)

    def on_model(self, _event=None) -> None:
        self.template.set(known_template(self.model.get(), self.mode.get()))

    def browse_models(self) -> None:
        def choose(model_id: str) -> None:
            self.model.set(model_id)
            self.on_model()
        ModelBrowser(self.root, self.mode.get(), choose)

    def pick_dataset(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("JSON Lines", "*.jsonl")])
        if selected:
            self.dataset.set(selected)

    def pick_output(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            parent = Path(selected)
            candidate = parent / "webslm-run"
            index = 2
            while candidate.exists():
                candidate = parent / f"webslm-run-{index}"
                index += 1
            self.output.set(str(candidate))

    def pick_backend(self) -> None:
        selected = filedialog.askopenfilename(title="Python in the LLaMA-Factory environment")
        if selected:
            self.backend_python.set(selected)

    def append(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self) -> None:
        try:
            steps = int(self.steps.get())
            model = self.model.get().strip()
            dataset = Path(self.dataset.get().strip())
            output = Path(self.output.get().strip())
            mode = self.mode.get()
            if mode == "text":
                config = TrainConfig(model, dataset, output, steps)
                count = validate_config(config)
                command = [sys.executable, "-u", "-m", "webslm_trainer", "train",
                           "--model", model, "--dataset", str(dataset),
                           "--output", str(output), "--steps", str(steps)]
            else:
                config = MediaTrainConfig(mode, model, dataset, output, steps,
                                          self.template.get().strip(), self.precision.get(),
                                          self.backend_python.get().strip())
                count = len(read_media_dataset(config))
                command = [sys.executable, "-u", "-m", "webslm_trainer", "train-media",
                           "--modality", mode, "--model", model, "--dataset", str(dataset),
                           "--output", str(output), "--steps", str(steps),
                           "--precision", config.precision]
                if config.template:
                    command += ["--template", config.template]
                if config.backend_python:
                    command += ["--backend-python", config.backend_python]
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check training setup", str(exc))
            return
        try:
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            bufsize=1, creationflags=flags,
                                            start_new_session=os.name != "nt")
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc))
            return
        self.append(f"Starting {count} {mode} examples with {model}...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self.read_output, daemon=True).start()

    def read_output(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        for line in process.stdout:
            self.events.put(line.rstrip("\n"))
        code = process.wait()
        self.events.put(f"Training {'finished' if code == 0 else 'stopped or failed'} (exit {code}).")
        self.events.put(None)

    def drain_events(self) -> None:
        if self.closed:
            return
        try:
            while True:
                item = self.events.get_nowait()
                if item is None:
                    self.process = None
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                else:
                    self.append(item)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            self.append("Stopping training...")

    def close(self) -> None:
        self.stop()
        self.closed = True
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrainerWindow(root)
    root.mainloop()
