# 🌿 Plant Disease Detection System

A production-quality **Plant Disease Detection System** using Convolutional Neural Networks (CNNs) built with PyTorch, FastAPI, and Grad-CAM explainability. Diagnoses plant diseases from leaf images with high accuracy and actionable agricultural recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Results](#results)

---

## 🔍 Overview

This system receives a plant leaf image and predicts:

- ✅ Whether the plant is **healthy or diseased**
- 🦠 The specific **disease category** (38 classes across 14 crops)
- 📊 A **confidence score** for the prediction
- 💊 A tailored **agricultural recommendation**

---

## 🏗️ Architecture

# 🌿 Plant Disease Detection System

A production-quality **Plant Disease Detection System** featuring a modern React dashboard and Convolutional Neural Networks (CNNs) built with PyTorch and FastAPI. Diagnoses plant diseases from leaf images with high accuracy, Grad-CAM explainability, and actionable agricultural recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Results](#results)

---

## 🔍 Overview

This full-stack system allows users to upload a plant leaf image via a sleek web interface and predicts:

- ✅ Whether the plant is **healthy or diseased**
- 🦠 The specific **disease category** (38 classes across 14 crops)
- 📊 A **confidence score** for the prediction
- 🗺️ A **visual Grad-CAM heatmap** showing the affected regions
- 💊 A tailored **agricultural recommendation**

---

## 🏗️ Architecture

```
[ React / Vite Frontend ] (Glassmorphism Dashboard)
        │
        ▼ POST /predict (Image)
[ FastAPI Backend ]
        │
        ▼
  Preprocessing (Resize 224×224, Normalize)
        │
        ▼
 ┌──────────────────────────────────┐
 │  MODEL 1: Custom CNN Baseline    │   OR   ┌─────────────────────────────────┐
 │  4× ConvBlock(Conv+BN+ReLU+Pool) │        │ MODEL 2: ResNet-18 (Transfer)   │
 │  → FC(512) → Dropout → FC(38)   │        │ Frozen Backbone → Custom Head   │
 └──────────────────────────────────┘        └─────────────────────────────────┘
        │
        ▼
  Softmax → Class + Confidence
        │
        ▼
  Grad-CAM Heatmap (optional) + Recommendation
        │
        ▼ JSON Response
[ React UI Displays Results & History ]

---

Framework Choice: PyTorch was selected for its dynamic computation graph, clean nn.Module OOP design, and first-class support for Grad-CAM gradient hooks.

## 📊 Dataset

**Source:** [BrandonFors/Plant-Diseases-PlantVillage-Dataset](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) (Hugging Face)

| Metric | Value |
|--------|-------|
| Total Images | ~54,305 |
| Classes | 38 (14 plant species × healthy + diseased) |
| Image Format | RGB JPEG |
| Input Resolution | 224 × 224 (resized) |

**Splits:**

| Set | Ratio | Purpose |
|-----|-------|---------|
| Train | 70% | Model optimization |
| Validation | 15% | Early stopping, LR scheduling |
| Test | 15% | Final unbiased evaluation |

---

## 📁 Project Structure

```text
plant-disease-detection/
├── frontend/                    # React / Vite SPA Dashboard
│   ├── src/                     # React components, pages, and API services
│   ├── package.json
│   └── tailwind.config.js
├── backend/                     # FastAPI / PyTorch API & ML Operations
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── data/
│   │   ├── raw/                 # Downloaded images per class
│   │   └── processed/           # Train / Val / Test splits
│   ├── models/
│   │   └── saved_models/        # .pth weights, history JSON, reports
│   ├── notebooks/
│   │   ├── data_analysis.ipynb  # EDA (class distribution, image sizes)
│   │   └── model_experiments.ipynb # Prototype experiments
│   ├── src/                     
│   │   ├── config.py            # Global paths, hyperparameters
│   │   ├── data_loader.py       # HF download + train/val/test split
│   │   ├── preprocessing.py     # PyTorch Dataset, transforms, DataLoader
│   │   ├── model.py             # Custom CNN + ResNet-18 transfer model
│   │   ├── train.py             # Training loop, early stopping, checkpoints
│   │   ├── evaluate.py          # Metrics, confusion matrix, learning curves
│   │   ├── predict.py           # Inference engine with image validation
│   │   └── explainability.py    # Grad-CAM hooks and overlay generation
│   └── tests/
│       ├── test_data.py         # Dataset / transform unit tests
│       ├── test_model.py        # Model output shape unit tests
│       └── test_api.py          # FastAPI integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

1. Backend & ML Setup
Bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI backend
uvicorn app.main:app --reload --port 8000
2. Frontend Setup
Open a new terminal and start the React app:

Bash
cd frontend
npm install
npm run dev
The dashboard will be available at http://localhost:5174
---

## 🗃️ Prepare Dataset

Downloads and splits the PlantVillage dataset automatically:

```bash
# Full dataset (38 classes, ~54K images)
python -m src.data_loader

# Prototype mode (5 classes, 150 images each — enabled by default in config.py)
# Set PROTOTYPE_MODE = True in src/config.py (already the default)
python -m src.data_loader
```

---

## 🏋️ Training

Train one or both models:

```bash
# Train ResNet-18 Transfer Learning model (recommended)
python -m src.train --model resnet18

# Train Custom CNN Baseline
python -m src.train --model baseline

# Train both and compare
python -m src.train --model both
```

Training logs checkpoints to `models/saved_models/` with:
- `{model}_best.pth` — best weights by validation loss
- `{model}_history.json` — per-epoch metrics
- `class_names.json` — class index mapping

---

## 📈 Evaluation

```bash
# Evaluate ResNet-18
python -m src.evaluate --model resnet18

# Evaluate Baseline CNN
python -m src.evaluate --model baseline

# Evaluate both
python -m src.evaluate --model both
```

Generates:
- Classification report (accuracy, precision, recall, F1)
- Confusion matrix heatmap PNG
- Training/validation learning curves PNG

---

## 🔍 Grad-CAM Explainability

Grad-CAM highlights which regions of the leaf influenced the disease prediction:

```python
from src.predict import PlantDiseasePredictor
from src.explainability import GradCAM, overlay_heatmap, save_gradcam_visualization

predictor = PlantDiseasePredictor(model_name="resnet18")
result = predictor.predict("path/to/leaf.jpg")
print(result)
```

Or via the API with `?explain=true` (see below).

---

## 🚀 API Usage

### Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger docs available at: **http://localhost:8000/docs**

### Endpoints

#### `GET /` — Health check
```json
{
  "app": "Plant Disease Detection System API",
  "status": "online",
  "model_loaded": "resnet18",
  "classes_supported": ["Apple___Apple_scab", "...", "Tomato___healthy"]
}
```

#### `POST /predict` — Diagnose disease

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@leaf.jpg"
```

**Response:**
```json
{
  "success": true,
  "disease": "Tomato with Late Blight",
  "confidence": 0.943,
  "recommendation": "Treat with copper-based fungicides immediately...",
  "class_raw": "Tomato___Late_blight",
  "model_used": "resnet18"
}
```

#### `POST /predict?explain=true` — With Grad-CAM

```bash
curl -X POST "http://localhost:8000/predict?explain=true" \
  -F "file=@leaf.jpg"
```

Adds `explanation_url` to the response pointing to the Grad-CAM overlay image served from `/static/`.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t plant-disease-api .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  plant-disease-api
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Results

> *Results below are representative benchmarks. Run evaluation after training to get your exact numbers.*

| Model | Test Accuracy | F1-Score | Parameters |
|-------|--------------|----------|------------|
| Custom CNN Baseline | ~85-90% | ~0.87 | ~13M |
| ResNet-18 Transfer | ~95-98% | ~0.96 | ~11M |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
Frontend UI	| React 18, Vite, Tailwind CSS, Framer Motion
| Deep Learning | PyTorch 2.x + torchvision |
| Computer Vision | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Explainability | Custom Grad-CAM (PyTorch hooks) |
| API | FastAPI + Uvicorn |
| Testing | pytest |
| Deployment | Docker |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://plantvillage.psu.edu/) — Hughes & Salathé, 2015
- [BrandonFors HuggingFace Mirror](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset)
- [PyTorch](https://pytorch.org/) | [FastAPI](https://fastapi.tiangolo.com/)
