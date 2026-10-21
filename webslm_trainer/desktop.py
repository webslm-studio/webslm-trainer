"""Dependency-free Tk desktop form. Training runs in a separate process."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import MODEL_CHOICES, TrainConfig, validate_config


class TrainerWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WebSLM Trainer")
        root.minsize(640, 440)
        self.model = tk.StringVar(value=MODEL_CHOICES[0])
        self.dataset = tk.StringVar()
        self.output = tk.StringVar(value=str(Path.cwd() / "outputs" / "my-adapter"))
        self.steps = tk.StringVar(value="30")
        self.events: queue.Queue[str | None] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Model ID (editable)").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(frame, textvariable=self.model, values=MODEL_CHOICES).grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)
        ttk.Label(frame, text="Dataset (.jsonl)").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.dataset).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Browse", command=self.pick_dataset).grid(row=1, column=2, padx=(8, 0), pady=5)
        ttk.Label(frame, text="Adapter output").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Browse", command=self.pick_output).grid(row=2, column=2, padx=(8, 0), pady=5)
        ttk.Label(frame, text="Steps").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.steps, width=10).grid(row=3, column=1, sticky="w", pady=5)
        controls = ttk.Frame(frame)
        controls.grid(row=4, column=0, columnspan=3, sticky="w", pady=10)
        self.start_button = ttk.Button(controls, text="Start training", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.log = tk.Text(frame, height=14, wrap="word", state="disabled")
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew")
        root.after(100, self.drain_events)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def pick_dataset(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("JSON Lines", "*.jsonl")])
        if selected:
            self.dataset.set(selected)

    def pick_output(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            parent = Path(selected)
            candidate = parent / "webslm-adapter"
            index = 2
            while candidate.exists():
                candidate = parent / f"webslm-adapter-{index}"
                index += 1
            self.output.set(str(candidate))

    def append(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self) -> None:
        try:
            config = TrainConfig(self.model.get().strip(), Path(self.dataset.get().strip()),
                                 Path(self.output.get().strip()), int(self.steps.get()))
            rows = validate_config(config)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check training setup", str(exc))
            return
        command = [sys.executable, "-u", "-m", "webslm_trainer", "train",
                   "--model", config.model, "--dataset", str(config.dataset),
                   "--output", str(config.output), "--steps", str(config.steps)]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc))
            return
        self.append(f"Starting {rows} examples with {config.model}...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self.read_output, daemon=True).start()

    def read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.events.put(line.rstrip("\n"))
        code = self.process.wait()
        self.events.put(f"Training {'finished' if code == 0 else 'stopped or failed'} (exit {code}).")
        self.events.put(None)

    def drain_events(self) -> None:
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
            self.process.terminate()
            self.append("Stopping training...")

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrainerWindow(root)
    root.mainloop()
