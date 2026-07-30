import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from typing import Dict, Any

from src import config
from src.predict import PlantDiseasePredictor
from src.explainability import GradCAM, overlay_heatmap, save_gradcam_visualization

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection System API",
    description="Production-ready API for identifying plant health status, diseases, and generating treatment recommendations with Grad-CAM explainability.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
STATIC_DIR = os.path.join(config.BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static directory to serve Grad-CAM images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Lazy loading of Predictor to allow application to start fast
predictor = None

def get_predictor() -> PlantDiseasePredictor:
    global predictor
    if predictor is None:
        try:
            # We default to resnet18 for production deployment
            predictor = PlantDiseasePredictor(model_name="resnet18")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize disease predictor: {e}")
    return predictor

@app.get("/")
def read_root() -> Dict[str, Any]:
    """Returns API metadata, loaded models, and class definitions."""
    try:
        pred = get_predictor()
        classes = pred.class_names
        model_name = pred.model_name
        device = str(pred.device)
    except Exception as e:
        classes = []
        model_name = "Not loaded"
        device = "None"
        
    return {
        "app": "Plant Disease Detection System API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model_name,
        "device": device,
        "classes_supported": classes,
        "endpoints": {
            "GET /": "Metadata and health check",
            "POST /predict": "Upload leaf image for diagnosis"
        }
    }

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Generate a Grad-CAM activation heatmap overlay")
) -> Dict[str, Any]:
    """
    Diagnoses plant leaf disease from an uploaded image.
    
    Inputs:
    - **file**: Binary image file (JPEG, PNG, etc.)
    - **explain**: Boolean flag to enable Grad-CAM explainability heatmap.
    
    Returns:
        JSON response with prediction details and optional static URL for the Grad-CAM visualization.
    """
    pred_engine = get_predictor()
    
    # Verify file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{file_ext}'. Please upload an image (.jpg, .jpeg, .png, .bmp)."
        )
        
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(STATIC_DIR, f"temp_upload{file_ext}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload stream: {e}")

    # Validate image quality (resolution, corruption, blur, contrast)
    is_valid, error_msg = pred_engine.validate_image_quality(temp_file_path)
    if not is_valid:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=error_msg)

    # Perform prediction
    prediction = pred_engine.predict(temp_file_path)
    if not prediction["success"]:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=prediction["error"])

    response_data = {
        "success": True,
        "disease": prediction["disease"],
        "confidence": prediction["confidence"],
        "recommendation": prediction["recommendation"],
        "class_raw": prediction["class_raw"],
        "model_used": prediction["model_used"]
    }

    # Generate Grad-CAM visualization if requested
    if explain:
        try:
            # We perform Grad-CAM on the ResNet18 model
            # The target layer in torchvision ResNet18 is model.backbone.layer4
            target_layer = pred_engine.model.backbone.layer4
            gradcam = GradCAM(model=pred_engine.model, target_layer=target_layer)
            
            # Load and preprocess image for Grad-CAM tensor input
            from PIL import Image
            pil_img = Image.open(temp_file_path).convert("RGB")
            input_tensor = pred_engine.transform(pil_img).unsqueeze(0).to(pred_engine.device)
            
            # Get prediction index to backpropagate gradients for that class
            pred_class_idx = pred_engine.class_names.index(prediction["class_raw"])
            
            # Compute heatmap
            heatmap = gradcam.generate_heatmap(input_tensor, class_idx=pred_class_idx)
            
            # Generate overlay
            overlay, colored_heatmap = overlay_heatmap(temp_file_path, heatmap)
            
            # Save visual explanation
            explanation_filename = f"gradcam_explanation_{file.filename.split('.')[0]}.png"
            explanation_path = os.path.join(STATIC_DIR, explanation_filename)
            
            save_gradcam_visualization(
                original_image_path=temp_file_path,
                heatmap=heatmap,
                overlay=overlay,
                save_path=explanation_path,
                predicted_class=prediction["disease"],
                confidence=prediction["confidence"]
            )
            
            # Remove hooks to avoid memory leaks
            gradcam.remove_hooks()
            
            # Add explainability URL to the response
            response_data["explanation_url"] = f"/static/{explanation_filename}"
            
        except Exception as e:
            response_data["explainability_error"] = f"Failed to compute Grad-CAM heatmap: {e}"

    # Clean up temporary upload file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        
    return response_data


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard() -> HTMLResponse:
    """Serves the interactive web dashboard for plant disease detection."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detection Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0B0F19;
            --bg-surface: rgba(22, 30, 49, 0.7);
            --bg-surface-hover: rgba(30, 41, 67, 0.8);
            --border-glow: rgba(16, 185, 129, 0.2);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
            --primary: #10B981;
            --primary-gradient: linear-gradient(135deg, #10B981 0%, #06B6D4 100%);
            --error-gradient: linear-gradient(135deg, #EF4444 0%, #F59E0B 100%);
            --accent: #6366F1;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 40%);
        }

        header {
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            z-index: 10;
        }

        .header-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-pill {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: var(--primary);
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--primary);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--primary);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        }

        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 3rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 250px;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.03);
            box-shadow: inset 0 0 12px rgba(16, 185, 129, 0.05);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
            filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.2));
        }

        .upload-area:hover .upload-icon {
            transform: translateY(-5px);
        }

        .upload-text {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .upload-hint {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.3);
        }

        #file-input {
            display: none;
        }

        /* Image Preview */
        .preview-container {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            display: none;
            background-color: #0d121f;
        }

        .preview-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .preview-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(11, 15, 25, 0.9), transparent);
            padding: 1.5rem 1rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .preview-filename {
            font-size: 0.85rem;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 60%;
        }

        .change-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-primary);
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .change-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Options */
        .options-group {
            margin: 1.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .checkbox-container {
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 0.9rem;
            user-select: none;
            color: var(--text-secondary);
        }

        .checkbox-container input {
            display: none;
        }

        .checkmark {
            width: 18px;
            height: 18px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            margin-right: 0.5rem;
            display: inline-block;
            position: relative;
            transition: all 0.2s;
        }

        .checkbox-container:hover .checkmark {
            border-color: var(--primary);
        }

        .checkbox-container input:checked + .checkmark {
            background: var(--primary);
            border-color: var(--primary);
        }

        .checkbox-container input:checked + .checkmark::after {
            content: "";
            position: absolute;
            left: 5px;
            top: 2px;
            width: 4px;
            height: 8px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }

        /* Buttons */
        .analyze-btn {
            width: 100%;
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
            filter: brightness(1.1);
        }

        .analyze-btn:active {
            transform: translateY(0);
        }

        .analyze-btn:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.2);
            box-shadow: none;
            cursor: not-allowed;
        }

        /* Results Pane (Right side) */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 350px;
            justify-content: center;
        }

        .placeholder-state {
            text-align: center;
            color: var(--text-secondary);
            padding: 4rem 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .placeholder-icon {
            font-size: 4rem;
            opacity: 0.3;
            animation: float 4s ease-in-out infinite;
        }

        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        .loading-state {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            gap: 1.5rem;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(16, 185, 129, 0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-state {
            display: none;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Diagnostic details */
        .diagnosis-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }

        .diagnosis-title {
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--text-primary);
        }

        .diagnosis-label-badge {
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .badge-healthy {
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .badge-diseased {
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }

        /* Score Circle */
        .metrics-grid {
            display: flex;
            align-items: center;
            gap: 2rem;
            margin-bottom: 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .score-circle {
            position: relative;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .score-circle svg {
            width: 80px;
            height: 80px;
            transform: rotate(-90deg);
        }

        .score-circle circle {
            fill: none;
            stroke-width: 6;
        }

        .score-circle .bg-circle {
            stroke: rgba(255, 255, 255, 0.05);
        }

        .score-circle .progress-circle {
            stroke: var(--primary);
            stroke-dasharray: 226;
            stroke-dashoffset: 226;
            stroke-linecap: round;
            transition: stroke-dashoffset 1s ease-out;
        }

        .score-value {
            position: absolute;
            font-size: 1.1rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--text-primary);
        }

        .meta-details {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .meta-row {
            display: flex;
            gap: 0.5rem;
            font-size: 0.9rem;
        }

        .meta-key {
            color: var(--text-secondary);
        }

        .meta-val {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Recommendations */
        .recommendation-card {
            background: rgba(16, 185, 129, 0.05);
            border-left: 4px solid var(--primary);
            padding: 1.2rem;
            border-radius: 0 12px 12px 0;
            margin-bottom: 1.5rem;
        }

        .recommendation-card.diseased {
            background: rgba(239, 68, 68, 0.05);
            border-left-color: #EF4444;
        }

        .recommendation-title {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: var(--text-primary);
        }

        .recommendation-text {
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-secondary);
        }

        /* Explanation visualization */
        .visualization-area {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.2);
        }

        .visualization-title {
            padding: 0.8rem 1.2rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .gradcam-img {
            width: 100%;
            display: block;
            object-fit: contain;
            background: #0d121f;
        }

        /* Alerts */
        .error-alert {
            display: none;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #F87171;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
            align-items: center;
            gap: 0.5rem;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: auto;
            background: rgba(11, 15, 25, 0.4);
        }
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <h1>🌿 Plant Disease Detection</h1>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>System Online</span>
            </div>
        </div>
    </header>

    <main>
        <!-- Diagnosis Upload (Left) -->
        <section class="card">
            <h2>📷 Select Leaf Image</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem; line-height: 1.4;">
                Upload a clear close-up picture of a plant leaf to obtain an immediate diagnosis, confidence metrics, and treatment advice.
            </p>

            <div class="error-alert" id="error-box">
                ⚠️ <span id="error-message"></span>
            </div>

            <div class="upload-area" id="drop-zone">
                <span class="upload-icon">📁</span>
                <p class="upload-text">Drag & drop your leaf image here or click to browse</p>
                <p class="upload-hint">Supports JPEG, PNG, BMP (min 64x64px)</p>
                
                <div class="preview-container" id="preview-box">
                    <img src="" class="preview-image" id="img-preview" alt="Upload preview">
                    <div class="preview-overlay">
                        <span class="preview-filename" id="filename-label">leaf.jpg</span>
                        <button type="button" class="change-btn" id="change-image-btn">Change</button>
                    </div>
                </div>
            </div>
            
            <input type="file" id="file-input" accept="image/jpeg,image/png,image/bmp">

            <div class="options-group">
                <label class="checkbox-container">
                    <input type="checkbox" id="explain-checkbox" checked>
                    <span class="checkmark"></span>
                    Enable Grad-CAM Activation Heatmap Overlay (Explainable AI)
                </label>
            </div>

            <button class="analyze-btn" id="analyze-btn" disabled>
                <span>🔍</span> Diagnose Leaf
            </button>
        </section>

        <!-- Diagnosis Results (Right) -->
        <section class="card">
            <h2>📊 Diagnostic Output</h2>
            
            <div class="results-container">
                <!-- Placeholder State -->
                <div class="placeholder-state" id="res-placeholder">
                    <span class="placeholder-icon">🔬</span>
                    <p style="font-weight: 500;">Awaiting Input</p>
                    <p style="font-size: 0.85rem;">Upload a leaf image on the left and click "Diagnose Leaf" to see results here.</p>
                </div>

                <!-- Loading State -->
                <div class="loading-state" id="res-loading">
                    <div class="spinner"></div>
                    <p style="font-weight: 500;">Running Neural Network Inference...</p>
                    <p style="font-size: 0.85rem; color: var(--text-secondary);">Calculating predictions and generating Grad-CAM overlays.</p>
                </div>

                <!-- Results State -->
                <div class="result-state" id="res-output">
                    <div class="diagnosis-header">
                        <div>
                            <p class="diagnosis-title" id="disease-name-label">Tomato (Healthy)</p>
                            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.2rem;" id="raw-class-label">Tomato___healthy</p>
                        </div>
                        <span class="diagnosis-label-badge badge-healthy" id="health-status-badge">Healthy</span>
                    </div>

                    <div class="metrics-grid">
                        <div class="score-circle">
                            <svg>
                                <circle class="bg-circle" cx="40" cy="40" r="36" />
                                <circle class="progress-circle" id="progress-indicator" cx="40" cy="40" r="36" />
                            </svg>
                            <span class="score-value" id="confidence-value-label">97%</span>
                        </div>
                        <div class="meta-details">
                            <div class="meta-row">
                                <span class="meta-key">Confidence:</span>
                                <span class="meta-val" id="confidence-text-label">97.0%</span>
                            </div>
                            <div class="meta-row">
                                <span class="meta-key">Inference Model:</span>
                                <span class="meta-val" id="model-name-label">resnet18</span>
                            </div>
                        </div>
                    </div>

                    <div class="recommendation-card" id="recommendation-box">
                        <p class="recommendation-title">💊 Suggested Action & Care</p>
                        <p class="recommendation-text" id="recommendation-text-label">
                            Your tomato plant looks healthy! Keep following standard watering practices and monitoring for pests.
                        </p>
                    </div>

                    <div class="visualization-area" id="explanation-container" style="display: none;">
                        <div class="visualization-title">👁️ Grad-CAM Visual Explanation</div>
                        <img src="" class="gradcam-img" id="explanation-image" alt="Grad-CAM analysis">
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer>
        <p>Plant Disease Detection System • Powered by PyTorch & FastAPI</p>
    </footer>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const previewBox = document.getElementById('preview-box');
        const imgPreview = document.getElementById('img-preview');
        const filenameLabel = document.getElementById('filename-label');
        const changeImageBtn = document.getElementById('change-image-btn');
        const analyzeBtn = document.getElementById('analyze-btn');
        const explainCheckbox = document.getElementById('explain-checkbox');
        const errorBox = document.getElementById('error-box');
        const errorMessage = document.getElementById('error-message');

        const resPlaceholder = document.getElementById('res-placeholder');
        const resLoading = document.getElementById('res-loading');
        const resOutput = document.getElementById('res-output');

        const diseaseNameLabel = document.getElementById('disease-name-label');
        const rawClassLabel = document.getElementById('raw-class-label');
        const healthStatusBadge = document.getElementById('health-status-badge');
        const progressIndicator = document.getElementById('progress-indicator');
        const confidenceValueLabel = document.getElementById('confidence-value-label');
        const confidenceTextLabel = document.getElementById('confidence-text-label');
        const modelNameLabel = document.getElementById('model-name-label');
        const recommendationBox = document.getElementById('recommendation-box');
        const recommendationTextLabel = document.getElementById('recommendation-text-label');
        const explanationContainer = document.getElementById('explanation-container');
        const explanationImage = document.getElementById('explanation-image');

        let selectedFile = null;

        // Trigger file input
        dropZone.addEventListener('click', (e) => {
            if (e.target !== changeImageBtn && !previewBox.contains(e.target)) {
                fileInput.click();
            }
        });

        changeImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        // Drag and drop handlers
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.match('image.*')) {
                showError("Please upload an image file (JPEG, PNG, BMP).");
                return;
            }
            
            selectedFile = file;
            filenameLabel.textContent = file.name;
            errorBox.style.display = 'none';

            const reader = new FileReader();
            reader.onload = function(e) {
                imgPreview.src = e.target.result;
                previewBox.style.display = 'block';
                analyzeBtn.disabled = false;
            }
            reader.readAsDataURL(file);
        }

        function showError(msg) {
            errorMessage.textContent = msg;
            errorBox.style.display = 'flex';
            analyzeBtn.disabled = true;
        }

        // Diagnose leaf submission
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            // Update UI state
            resPlaceholder.style.display = 'none';
            resOutput.style.display = 'none';
            resLoading.style.display = 'flex';
            errorBox.style.display = 'none';
            analyzeBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);

            const explain = explainCheckbox.checked;

            try {
                const response = await fetch(`/predict?explain=${explain}`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || "Server error occurred during prediction.");
                }

                if (data.success) {
                    displayResult(data, explain);
                } else {
                    throw new Error(data.error || "Unknown prediction failure.");
                }

            } catch (err) {
                showError(err.message);
                resPlaceholder.style.display = 'flex';
                resLoading.style.display = 'none';
                resOutput.style.display = 'none';
            } finally {
                analyzeBtn.disabled = false;
            }
        });

        function displayResult(data, explain) {
            resLoading.style.display = 'none';
            resOutput.style.display = 'block';

            // Set labels
            diseaseNameLabel.textContent = data.disease;
            rawClassLabel.textContent = data.class_raw;
            modelNameLabel.textContent = data.model_used;
            recommendationTextLabel.textContent = data.recommendation;

            // Set status badge and card theme
            const isHealthy = data.class_raw.toLowerCase().includes('healthy');
            if (isHealthy) {
                healthStatusBadge.textContent = 'Healthy';
                healthStatusBadge.className = 'diagnosis-label-badge badge-healthy';
                recommendationBox.className = 'recommendation-card';
            } else {
                healthStatusBadge.textContent = 'Diseased';
                healthStatusBadge.className = 'diagnosis-label-badge badge-diseased';
                recommendationBox.className = 'recommendation-card diseased';
            }

            // Set confidence metrics
            const confPercent = Math.round(data.confidence * 100);
            confidenceValueLabel.textContent = `${confPercent}%`;
            confidenceTextLabel.textContent = `${(data.confidence * 100).toFixed(1)}%`;

            // Animate progress circle
            const circumference = 226;
            const offset = circumference - (data.confidence * circumference);
            progressIndicator.style.strokeDashoffset = offset;

            // Change progress circle color depending on health
            if (isHealthy) {
                progressIndicator.style.stroke = '#10B981';
            } else {
                progressIndicator.style.stroke = '#EF4444';
            }

            // Set visual explanation
            if (explain && data.explanation_url) {
                explanationImage.src = data.explanation_url + '?t=' + new Date().getTime(); // Prevent caching
                explanationContainer.style.display = 'block';
            } else {
                explanationContainer.style.display = 'none';
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content, status_code=200)
