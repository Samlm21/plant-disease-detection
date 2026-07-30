# ==========================================================
# File: src/utils.py
# ==========================================================

"""
General utility functions used throughout the project.

Functions
---------
✓ Set random seeds
✓ Save JSON
✓ Load JSON
✓ Save YAML
✓ Load YAML
✓ Count model parameters
✓ Calculate classification accuracy
✓ Format elapsed time
✓ Create directories safely
✓ Get current timestamp
✓ Select computation device
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ==========================================================
# Random Seed
# ==========================================================

def seed_everything(seed: int = 42) -> None:
    """
    Make experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Device Selection
# ==========================================================

def get_device() -> torch.device:
    """
    Automatically select CUDA if available.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ==========================================================
# Directory Utilities
# ==========================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


# ==========================================================
# JSON
# ==========================================================

def save_json(data: dict, filepath: str | Path) -> None:

    filepath = Path(filepath)

    ensure_dir(filepath.parent)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str | Path) -> dict:

    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# YAML
# ==========================================================

def save_yaml(data: dict, filepath: str | Path):

    filepath = Path(filepath)

    ensure_dir(filepath.parent)

    with open(filepath, "w") as f:
        yaml.safe_dump(data, f)


def load_yaml(filepath: str | Path):

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


# ==========================================================
# Time Formatting
# ==========================================================

def format_time(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS.
    """

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"


# ==========================================================
# Timestamp
# ==========================================================

def timestamp() -> str:
    """
    Current timestamp for experiment folders.
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================================
# Accuracy
# ==========================================================

def calculate_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate batch accuracy.
    """

    predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == labels).sum().item()

    return correct / labels.size(0)


# ==========================================================
# Parameter Counting
# ==========================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Human Readable Parameters
# ==========================================================

def readable_parameters(model: torch.nn.Module) -> str:

    params = count_parameters(model)

    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f} M"

    if params >= 1_000:
        return f"{params / 1_000:.2f} K"

    return str(params)


# ==========================================================
# Timer
# ==========================================================

class Timer:
    """
    Simple timer.

    Example
    -------
    timer = Timer()

    ...

    print(timer.elapsed())
    """

    def __init__(self):

        self.start = time.time()

    def reset(self):

        self.start = time.time()

    def elapsed(self):

        return time.time() - self.start

    def elapsed_str(self):

        return format_time(self.elapsed())


# ==========================================================
# Average Meter
# ==========================================================

class AverageMeter:
    """
    Computes running averages.

    Useful for tracking

    loss

    accuracy

    precision

    recall
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):

        self.sum += value * n
        self.count += n

        self.avg = self.sum / self.count


# ==========================================================
# Model Summary
# ==========================================================

def model_summary(model: torch.nn.Module):

    print("=" * 60)

    print(model)

    print("=" * 60)

    print(f"Trainable Parameters : {readable_parameters(model)}")

    print("=" * 60)


# ==========================================================
# Save Model Summary
# ==========================================================

def save_model_summary(
    model: torch.nn.Module,
    filepath: str | Path,
):

    filepath = Path(filepath)

    ensure_dir(filepath.parent)

    with open(filepath, "w", encoding="utf-8") as f:

        f.write(str(model))
        f.write("\n\n")
        f.write("=" * 60)
        f.write("\n")
        f.write(f"Trainable Parameters : {count_parameters(model)}")
        f.write("\n")