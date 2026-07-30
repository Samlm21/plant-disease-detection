# ==========================================================
# File: src/checkpoint.py
# ==========================================================

"""
Checkpoint Manager

Handles

✓ Best model saving
✓ Latest checkpoint saving
✓ Resume training
✓ Optimizer state
✓ Scheduler state
✓ AMP scaler state
✓ Training history
✓ Metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.utils import ensure_dir


class CheckpointManager:
    """
    Production checkpoint manager.

    Example
    -------
    manager = CheckpointManager("models/saved_models")

    manager.save(...)

    checkpoint = manager.load("latest_model.pth")
    """

    def __init__(self, checkpoint_dir: str | Path):

        self.checkpoint_dir = ensure_dir(checkpoint_dir)

    # ------------------------------------------------------
    # Private Helper
    # ------------------------------------------------------

    def _checkpoint_dict(
        self,
        epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=None,
        best_score=None,
        metadata=None,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict":
                scheduler.state_dict() if scheduler else None,
            "scaler_state_dict":
                scaler.state_dict() if scaler else None,
            "history": history,
            "best_score": best_score,
            "metadata": metadata,
        }

        return checkpoint

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    def save(
        self,
        filename,
        epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=None,
        best_score=None,
        metadata=None,
    ):

        checkpoint = self._checkpoint_dict(
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            history,
            best_score,
            metadata,
        )

        path = self.checkpoint_dir / filename

        torch.save(checkpoint, path)

        print(f"[Checkpoint Saved] {path}")

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    def save_best(
        self,
        epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=None,
        best_score=None,
        metadata=None,
    ):

        self.save(
            filename="best_model.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_score=best_score,
            metadata=metadata,
        )

    # ------------------------------------------------------
    # Latest
    # ------------------------------------------------------

    def save_latest(
        self,
        epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=None,
        best_score=None,
        metadata=None,
    ):

        self.save(
            filename="latest_model.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_score=best_score,
            metadata=metadata,
        )

    # ------------------------------------------------------
    # Epoch Checkpoint
    # ------------------------------------------------------

    def save_epoch(
        self,
        epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=None,
        best_score=None,
        metadata=None,
    ):

        filename = f"checkpoint_epoch_{epoch}.pth"

        self.save(
            filename=filename,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_score=best_score,
            metadata=metadata,
        )

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    def load(
        self,
        filename,
        device="cpu",
    ):

        path = self.checkpoint_dir / filename

        if not path.exists():
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        print(f"[Checkpoint Loaded] {path}")

        return checkpoint

    # ------------------------------------------------------
    # Resume Training
    # ------------------------------------------------------

    def resume(
        self,
        filename,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu",
    ):

        checkpoint = self.load(filename, device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer and checkpoint["optimizer_state_dict"]:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if scheduler and checkpoint["scheduler_state_dict"]:

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if scaler and checkpoint["scaler_state_dict"]:

            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        return {
            "epoch": checkpoint["epoch"],
            "history": checkpoint.get("history"),
            "best_score": checkpoint.get("best_score"),
            "metadata": checkpoint.get("metadata"),
        }

    # ------------------------------------------------------
    # Exists
    # ------------------------------------------------------

    def exists(self, filename):

        return (self.checkpoint_dir / filename).exists()

    # ------------------------------------------------------
    # Delete
    # ------------------------------------------------------

    def delete(self, filename):

        path = self.checkpoint_dir / filename

        if path.exists():
            path.unlink()

    # ------------------------------------------------------
    # List
    # ------------------------------------------------------

    def list_checkpoints(self):

        return sorted(self.checkpoint_dir.glob("*.pth"))