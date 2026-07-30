import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Ensure critical directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, SAVED_MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

# Dataset Download Details
# PlantVillage unaugmented color dataset from Mendeley Data
DATASET_URL = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
ZIP_FILENAME = "Plant_leaf_diseases_dataset_without_augmentation.zip"
RAW_EXTRACT_SUBDIR = "Plant_leave_diseases_dataset" # This is the subfolder extracted from the zip

# Prototyping Settings for Quick Iteration
# Set PROTOTYPE_MODE to True to subsample the dataset for faster training and validation.
PROTOTYPE_MODE = True
PROTOTYPE_CLASSES = [
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]
PROTOTYPE_SAMPLE_FRACTION = 0.15 # Use 15% of images per class in prototype mode (fits in seconds/minutes)

# Core Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Small batch size to accommodate GTX 1650 4GB VRAM
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5

# System Settings
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 # Set to 0 for Windows compatibility and debugging safety

# Agricultural Recommendations
DISEASE_RECOMMENDATIONS = {
    # Apple
    "Apple___Apple_scab": "Apple Scab detected. Apply sulfur or copper-based fungicides in early spring. Rake and burn fallen leaves to reduce overwintering spores.",
    "Apple___Black_rot": "Apple Black Rot detected. Prune out dead wood and cankers during winter. Spray with labeled fungicides when symptoms persist.",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust detected. Remove nearby juniper/cedar trees if possible. Apply preventive fungicides in early spring.",
    "Apple___healthy": "Apple tree looks healthy! Keep pruning annually and monitor for seasonal pests.",
    # Blueberry
    "Blueberry___healthy": "Blueberry plant is healthy! Maintain soil acidity (pH 4.5-5.5) and adequate moisture.",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "Cherry Powdery Mildew detected. Apply sulfur or potassium bicarbonate sprays. Ensure good air circulation with proper pruning.",
    "Cherry_(including_sour)___healthy": "Cherry tree is healthy! Keep pruning to improve airflow and sunlight penetration.",
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray Leaf Spot detected. Use resistant hybrids, practice crop rotation, and apply fungicides if economically justified.",
    "Corn_(maize)___Common_rust_": "Corn Common Rust detected. Apply foliar fungicides (e.g., triazoles) at early infection stages. Use resistant varieties for future plantings.",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight detected. Apply fungicides at tasseling. Rotate crops and use resistant hybrids.",
    "Corn_(maize)___healthy": "Corn plant is healthy! Maintain fertility levels and monitor for pests.",
    # Grape
    "Grape___Black_rot": "Grape Black Rot detected. Apply copper fungicides. Prune vines for ventilation, and clean up mummified grapes.",
    "Grape___Esca_(Black_Measles)": "Grape Esca (Black Measles) detected. Remove and destroy infected wood. Apply pruning wound protectants. No effective chemical cure exists.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight detected. Apply copper-based fungicides. Improve canopy airflow through pruning.",
    "Grape___healthy": "Grapevine is healthy! Maintain trellis system and prune regularly.",
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus Greening (HLB) detected. This is incurable — remove and destroy infected trees immediately. Control Asian citrus psyllid vectors with insecticides.",
    # Peach
    "Peach___Bacterial_spot": "Peach Bacterial Spot detected. Spray with copper bactericide. Plant resistant varieties and avoid overhead watering.",
    "Peach___healthy": "Peach tree is healthy! Ensure good airflow through pruning and feed appropriately.",
    # Pepper
    "Pepper,_bell___Bacterial_spot": "Pepper Bacterial Spot detected. Use copper sprays. Rotate crops and avoid working in wet foliage.",
    "Pepper,_bell___healthy": "Pepper plant is healthy! Keep soil moist but not waterlogged.",
    # Potato
    "Potato___Early_blight": "Potato Early Blight detected. Use fungicides like copper oxychloride. Ensure proper plant spacing and clean up crop debris after harvest.",
    "Potato___Late_blight": "Potato Late Blight detected. Apply mancozeb or chlorothalonil fungicides. Destroy infected vines, harvest dry potatoes, and avoid overhead irrigation.",
    "Potato___healthy": "Potato plant is healthy! Continue monitoring soil moisture and guard against pests.",
    # Raspberry
    "Raspberry___healthy": "Raspberry canes are healthy! Prune out old canes after harvest and maintain proper row spacing.",
    # Soybean
    "Soybean___healthy": "Soybean plant is healthy! Maintain soil nutrients and monitor for aphids and whiteflies.",
    # Squash
    "Squash___Powdery_mildew": "Squash Powdery Mildew detected. Spray with horticultural oils or potassium bicarbonate. Water plants at the base to keep leaves dry.",
    # Strawberry
    "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch detected. Remove infected leaves and avoid high nitrogen fertilizer. Maintain clean rows.",
    "Strawberry___healthy": "Strawberry plant is healthy! Ensure mulching to prevent direct berry-to-soil contact.",
    # Tomato
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot detected. Apply copper-based bactericides. Use disease-free seeds and avoid working with wet plants.",
    "Tomato___Early_blight": "Tomato Early Blight detected. Apply chlorothalonil or copper fungicides. Remove lower leaves to prevent soil splash-up, and rotate crops annually.",
    "Tomato___Late_blight": "Tomato Late Blight detected. Treat with copper-based fungicides immediately. Prune and destroy infected leaves; water soil directly to keep foliage dry.",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold detected. Improve greenhouse ventilation. Apply copper or chlorothalonil fungicides and remove infected leaves.",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot detected. Remove and destroy infected leaves. Apply copper or chlorothalonil fungicides and mulch around base of plants.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites detected on Tomato. Use miticides or neem oil. Keep plants well-watered as drought stress worsens infestations.",
    "Tomato___Target_Spot": "Tomato Target Spot detected. Apply chlorothalonil fungicide. Improve air circulation through pruning and staking.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus detected. Remove and destroy infected plants immediately. Control whitefly vectors with reflective mulches and insecticides.",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus detected. Remove infected plants. Disinfect tools and hands; avoid tobacco use around tomatoes as it can spread TMV.",
    "Tomato___healthy": "Tomato plant looks healthy! Maintain regular watering and ensure adequate sunlight.",
}
