import os
import json
import logging
import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, Tuple, Optional
import cv2

from src import config
from src.preprocessing import get_val_test_transforms
from src.model import get_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PlantDiseasePredictor:
    """Production-grade inference class for predicting plant diseases."""
    
    def __init__(self, model_name: str = "resnet18"):
        self.model_name = model_name
        self.device = config.DEVICE
        
        # Load class names mapping
        class_mapping_path = os.path.join(config.SAVED_MODELS_DIR, "class_names.json")
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(
                f"Class names mapping not found at {class_mapping_path}. "
                "Ensure you have run training first or that the file is in the models/saved_models directory."
            )
            
        with open(class_mapping_path, "r") as f:
            self.class_names = json.load(f)
            
        self.num_classes = len(self.class_names)
        logger.info(f"Loaded class mapping: {self.class_names}")
        
        # Load Model
        self.model = get_model(self.model_name, num_classes=self.num_classes)
        model_weights_path = os.path.join(config.SAVED_MODELS_DIR, f"{self.model_name}_best.pth")
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Trained weights not found at: {model_weights_path}. Train the model first.")
            
        logger.info(f"Loading weights from {model_weights_path}...")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = get_val_test_transforms()

    def validate_image_quality(self, image_path: str) -> Tuple[bool, str]:
        """
        Validates the image for quality, corruption, and blur.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # 1. Check if file is readable as an image using PIL
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            return False, f"Invalid or corrupted image format. Could not verify: {e}"

        # 2. Check quality details using OpenCV
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            return False, "Failed to load image via OpenCV. Corrupted or unsupported format."

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # Check image resolution (too small check)
        h, w = gray.shape
        if h < 64 or w < 64:
            return False, f"Image resolution is too low ({w}x{h}). Minimum required is 64x64."

        # Check blur using Variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 10.0:  # Threshold for blurry images
            return False, f"Image is too blurry (Laplacian variance: {laplacian_var:.2f}). Please upload a clear image."

        # Check low contrast
        contrast = gray.max() - gray.min()
        if contrast < 40:  # Low contrast threshold
            return False, f"Image has very low contrast (contrast range: {contrast}). Please upload a better-lit image."

        return True, ""

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predicts disease and confidence for a given image.
        
        Args:
            image_path (str): Path to input image file.
            
        Returns:
            Dict: Prediction outputs (class, disease, confidence, recommendations, etc.)
        """
        # Validate Image
        is_valid, err_msg = self.validate_image_quality(image_path)
        if not is_valid:
            logger.warning(f"Image validation failed: {err_msg}")
            return {
                "success": False,
                "error": err_msg
            }
            
        # Load and preprocess
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {e}"
            }
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dim [1, 3, 224, 224]
        
        # Run Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Extract highest scoring class
        conf, pred_idx = torch.max(probabilities, dim=0)
        confidence = conf.item()
        predicted_class = self.class_names[pred_idx.item()]
        
        # Extract disease name (split by ___ and replace underscore with spaces)
        parts = predicted_class.split("___")
        plant = parts[0].replace("_", " ").title()
        disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "healthy"
        
        if disease_name == "healthy":
            disease_pretty = f"{plant} (Healthy)"
        else:
            disease_pretty = f"{plant} with {disease_name.title()}"
            
        # Get Recommendation
        recommendation = config.DISEASE_RECOMMENDATIONS.get(
            predicted_class, 
            "No specific recommendation available. Consult an agricultural specialist."
        )
        
        return {
            "success": True,
            "class_raw": predicted_class,
            "disease": disease_pretty,
            "confidence": float(confidence),
            "recommendation": recommendation,
            "model_used": self.model_name
        }
