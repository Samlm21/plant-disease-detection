import os
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from src import config
from src.preprocessing import get_dataloaders
from src.model import get_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, verbose: bool = True, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        """Saves model state dict when validation loss decreases."""
        if self.verbose:
            logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model weights to {path}...")
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> Tuple[float, float]:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap dataloader with tqdm
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update tqdm description
        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Validates the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_loss = running_loss / total
            current_acc = correct / total
            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    device: torch.device = config.DEVICE
) -> Dict[str, List[float]]:
    """Runs the training and validation loops, saving the best weights and training history."""
    logger.info(f"Starting training on device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    # Define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Define early stopping
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_best.pth")
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch} Summary -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Log to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Step LR scheduler
        scheduler.step(val_loss)
        
        # Check early stopping
        early_stopping(val_loss, model, model_path)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Training stopped.")
            break

    # Save history as JSON file
    history_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection models.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18", 
        choices=["baseline", "resnet18", "both"],
        help="Model type to train: 'baseline', 'resnet18', or 'both'"
    )
    args = parser.parse_args()

    # 1. Load Data
    logger.info("Loading datasets and creating DataLoaders...")
    train_loader, val_loader, _, class_names = get_dataloaders()
    num_classes = len(class_names)
    logger.info(f"Loaded dataset with {num_classes} classes: {class_names}")
    
    # Save class names mapping for inference consistency
    class_mapping_path = os.path.join(config.SAVED_MODELS_DIR, "class_names.json")
    with open(class_mapping_path, "w") as f:
        json.dump(class_names, f, indent=4)
    logger.info(f"Class names mapping saved to {class_mapping_path}")

    models_to_train = []
    if args.model in ["baseline", "both"]:
        models_to_train.append("baseline")
    if args.model in ["resnet18", "both"]:
        models_to_train.append("resnet18")

    for model_name in models_to_train:
        logger.info(f"==================================================")
        logger.info(f"TRAINING MODEL: {model_name.upper()}")
        logger.info(f"==================================================")
        
        if model_name == "baseline":
            model = get_model("baseline", num_classes=num_classes)
        else: # resnet18
            model = get_model("resnet18", num_classes=num_classes, pretrained=True)
            # Transfer learning: freeze the resnet backbone layers initially, only train the fc head
            # This is standard transfer learning practice.
            # To do full training or fine-tuning, we can unfreeze after.
            logger.info("Freezing ResNet backbone for feature extraction...")
            model.freeze_backbone(freeze=True)
            
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name
        )


if __name__ == "__main__":
    main()
