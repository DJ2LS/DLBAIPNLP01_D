"""Runs 5_train_bert.py on a Modal cloud GPU and saves the results to results/.
Usage: modal run 5_train_bert_modal.py (runs only while the terminal stays open)"""

import subprocess
import sys
from pathlib import Path

import modal

GPUS = ["H100", "A100-40GB", "A10"]

app = modal.App("bert-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas==3.0.6", "torch==2.14.1", "transformers==5.20.2", "accelerate==1.15.0")
    .add_local_dir("data/processed", "/root/data/processed")
    .add_local_file("5_train_bert.py", "/root/5_train_bert.py")
)


@app.function(image=image, gpu=GPUS, cpu=8, memory=32768, timeout=6 * 60 * 60)
def train():
    Path("/root/results").mkdir(exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    subprocess.run([sys.executable, "5_train_bert.py"], cwd="/root", check=True)
    return {f.name: f.read_bytes() for f in Path("/root/results").iterdir()}


@app.local_entrypoint()
def main():
    for name, content in train.remote().items():
        (Path("results") / name).write_bytes(content)
        print(f"Saved: results/{name}")
