import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from src import config

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for PyTorch models.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        # Use register_full_backward_hook to avoid deprecation warnings in modern PyTorch
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates the 2D Grad-CAM heatmap for a given input tensor.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor shape (1, 3, H, W)
            class_idx (int, optional): Index of the target class. If None, uses predicted class.
            
        Returns:
            np.ndarray: Grayscale heatmap normalized between 0 and 1 (H_target, W_target)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or Activations were not captured by hooks. Verify target_layer is correct.")
            
        # Extract gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]  # Shape: (C, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # Shape: (C, H, W)
        
        # Compute neuron importance weights alpha (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)
        
        # Compute weighted combination of forward activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # Shape: (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to focus only on features that positively contribute to the target class
        cam = np.maximum(cam, 0)
        
        # Normalize heatmap to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam

    def remove_hooks(self) -> None:
        """Removes the registered forward and backward hooks from the model."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(original_image_path: str, heatmap: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays the Grad-CAM heatmap on the original image.
    
    Args:
        original_image_path (str): Path to original image.
        heatmap (np.ndarray): Normalized Grad-CAM heatmap (0-1).
        alpha (float): Opacity of the heatmap overlay (0.0 to 1.0).
        
    Returns:
        Tuple: (overlay_rgb, heatmap_colored_rgb)
    """
    # Load original image with OpenCV (loads as BGR)
    orig_img = cv2.imread(original_image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image at {original_image_path}")
        
    height, width, _ = orig_img.shape
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Convert heatmap from single-channel normalized float to 8-bit image
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap to convert grayscale to RGB heatmap representation
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # Superimpose heatmap onto original image
    overlay = cv2.addWeighted(orig_img, 1.0 - alpha, heatmap_colored, alpha, 0)
    
    # Convert to RGB (for plotting/Pillow compatibility)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb, heatmap_colored_rgb


def save_gradcam_visualization(
    original_image_path: str,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    save_path: str,
    predicted_class: str,
    confidence: float
) -> None:
    """Creates and saves a side-by-side plot comparing original, heatmap and overlay images."""
    orig_img = Image.open(original_image_path).convert("RGB")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Original
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Leaf Image")
    axes[0].axis('off')
    
    # Plot Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Class: {predicted_class} | Conf: {confidence:.2%})")
    axes[2].axis('off')
    
    plt.suptitle("Grad-CAM Explanation of Disease Detection", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Grad-CAM visualization saved successfully to {save_path}")
