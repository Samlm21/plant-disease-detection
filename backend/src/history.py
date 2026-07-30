# ==========================================================
# File: src/history.py
# ==========================================================

"""
History Manager

Tracks and stores training history.

Features
--------
✓ Training loss
✓ Validation loss
✓ Training accuracy
✓ Validation accuracy
✓ Learning rate
✓ Epoch time
✓ Save to JSON
✓ Load history
✓ Resume training
"""

from __future__ import annotations

from pathlib import Path

from src.utils import ensure_dir, load_json, save_json


class History:
    """
    Stores training metrics.

    Example
    -------
    history = History()

    history.update(...)

    history.save("models/history.json")
    """

    def __init__(self):

        self.data = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    # ------------------------------------------------------
    # Update
    # ------------------------------------------------------

    def update(
        self,
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
        learning_rate,
        epoch_time,
    ):

        self.data["train_loss"].append(float(train_loss))
        self.data["val_loss"].append(float(val_loss))

        self.data["train_accuracy"].append(float(train_accuracy))
        self.data["val_accuracy"].append(float(val_accuracy))

        self.data["learning_rate"].append(float(learning_rate))
        self.data["epoch_time"].append(float(epoch_time))

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    def save(self, filepath):

        filepath = Path(filepath)

        ensure_dir(filepath.parent)

        save_json(self.data, filepath)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    def load(self, filepath):

        self.data = load_json(filepath)

    # ------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------

    def state_dict(self):

        return self.data

    # ------------------------------------------------------
    # Resume
    # ------------------------------------------------------

    def load_state_dict(self, state_dict):

        self.data = state_dict

    # ------------------------------------------------------
    # Latest Metrics
    # ------------------------------------------------------

    @property
    def latest(self):

        if len(self.data["train_loss"]) == 0:
            return None

        return {
            key: values[-1]
            for key, values in self.data.items()
        }

    # ------------------------------------------------------
    # Number of Epochs
    # ------------------------------------------------------

    @property
    def epochs(self):

        return len(self.data["train_loss"])

    # ------------------------------------------------------
    # Best Validation Accuracy
    # ------------------------------------------------------

    @property
    def best_accuracy(self):

        if self.epochs == 0:
            return 0.0

        return max(self.data["val_accuracy"])

    # ------------------------------------------------------
    # Best Validation Loss
    # ------------------------------------------------------

    @property
    def best_loss(self):

        if self.epochs == 0:
            return float("inf")

        return min(self.data["val_loss"])

    # ------------------------------------------------------
    # Reset
    # ------------------------------------------------------

    def reset(self):

        for key in self.data:
            self.data[key] = []

    # ------------------------------------------------------
    # String Representation
    # ------------------------------------------------------

    def __repr__(self):

        return (
            f"History("
            f"epochs={self.epochs}, "
            f"best_acc={self.best_accuracy:.4f}, "
            f"best_loss={self.best_loss:.4f})"
        )