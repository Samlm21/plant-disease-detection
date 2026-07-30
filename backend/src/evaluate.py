import os
import argparse
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from src import config
from src.preprocessing import get_dataloaders, PlantDataset, get_val_test_transforms
from src.model import get_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_learning_curves(history_path: str, model_name: str, save_path: str) -> None:
    """Plots training and validation accuracy and loss curves from history JSON file."""
    if not os.path.exists(history_path):
        logger.warning(f"History file {history_path} does not exist. Skipping curve plotting.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], 'bo-', label='Training Loss')
    ax1.plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
    ax1.set_title(f'{model_name.upper()} - Loss curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy Curve
    ax2.plot(epochs, history["train_acc"], 'bo-', label='Training Acc')
    ax2.plot(epochs, history["val_acc"], 'ro-', label='Validation Acc')
    ax2.set_title(f'{model_name.upper()} - Accuracy curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Learning curves saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_name: str, save_path: str) -> None:
    """Plots confusion matrix as a Seaborn heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {save_path}")


def evaluate_model(model_name: str, test_loader: DataLoader, class_names: List[str], device: torch.device) -> None:
    """Evaluates the selected model on the test dataset and generates diagnostic reports and plots."""
    num_classes = len(class_names)
    
    # 1. Load Model and Weights
    model = get_model(model_name, num_classes=num_classes)
    model_weights_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_best.pth")
    
    if not os.path.exists(model_weights_path):
        logger.error(f"Trained weights not found at: {model_weights_path}. Train the model first.")
        return

    logger.info(f"Loading weights from {model_weights_path}...")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # 2. Run Inference on Test Set
    logger.info(f"Evaluating {model_name} on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    logger.info(f"--- TEST SET PERFORMANCE ({model_name.upper()}) ---")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    # Generate classification report
    clf_report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info(f"\nClassification Report:\n{clf_report}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 4. Save results to disk
    report_file_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_evaluation.txt")
    with open(report_file_path, "w") as f:
        f.write(f"Model Evaluation: {model_name.upper()}\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    logger.info(f"Evaluation report written to {report_file_path}")

    # Plot Confusion Matrix
    cm_plot_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, model_name, cm_plot_path)

    # Plot Learning Curves
    history_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_history.json")
    learning_curves_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_learning_curves.png")
    plot_learning_curves(history_path, model_name, learning_curves_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained plant disease models.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18", 
        choices=["baseline", "resnet18", "both"],
        help="Model type to evaluate: 'baseline', 'resnet18', or 'both'"
    )
    args = parser.parse_args()

    # Load test split DataLoader
    logger.info("Initializing test dataloader...")
    _, _, test_loader, class_names = get_dataloaders()
    
    models_to_eval = []
    if args.model in ["baseline", "both"]:
        models_to_eval.append("baseline")
    if args.model in ["resnet18", "both"]:
        models_to_eval.append("resnet18")

    for model_name in models_to_eval:
        logger.info(f"==================================================")
        logger.info(f"EVALUATING MODEL: {model_name.upper()}")
        logger.info(f"==================================================")
        evaluate_model(model_name, test_loader, class_names, config.DEVICE)


if __name__ == "__main__":
    main()
