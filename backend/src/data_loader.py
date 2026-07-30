import os
import shutil
import random
import logging
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple
import cv2

from src import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_and_extract_hf(dest_raw_dir: str, prototype_mode: bool, prototype_classes: List[str]) -> List[str]:
    """
    Downloads the PlantVillage dataset from Hugging Face and saves images to raw class directories.
    In prototype mode, only saves images for the target prototype classes, up to 150 images per class.
    """
    logger.info("Loading PlantVillage dataset from Hugging Face...")
    try:
        # Use BrandonFors mirror which contains actual PIL images + ClassLabel
        dataset_dict = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        raise e

    # Extract class names mapping from metadata
    class_names = dataset_dict["train"].features["label"].names
    logger.info(f"Hugging Face dataset loaded. Found {len(class_names)} classes.")

    # Create raw directories
    os.makedirs(dest_raw_dir, exist_ok=True)
    for c_name in class_names:
        if not prototype_mode or c_name in prototype_classes:
            os.makedirs(os.path.join(dest_raw_dir, c_name), exist_ok=True)

    # Class sample counters for prototype mode
    class_counters = Counter()
    max_proto_samples = 150  # Cap images per class in prototype mode for super fast runs

    # Iterate over all splits (train/test) in Hugging Face dataset and save images
    logger.info("Saving images to data/raw class directories...")
    
    # We combine splits if there are multiple to have full control over splitting
    splits = list(dataset_dict.keys())
    
    for split in splits:
        split_data = dataset_dict[split]
        for i, item in enumerate(tqdm(split_data, desc=f"Processing split: {split}")):
            img = item["image"]          # PIL Image
            label_idx = item["label"]   # integer index
            class_name = class_names[label_idx]

            # If in prototype mode, skip non-target classes or over-limit classes
            if prototype_mode:
                if class_name not in prototype_classes:
                    continue
                if class_counters[class_name] >= max_proto_samples:
                    continue
                class_counters[class_name] += 1

            # Save the image file
            img_filename = f"{split}_{i}.jpg"
            img_path = os.path.join(dest_raw_dir, class_name, img_filename)

            # Avoid re-saving if it already exists
            if not os.path.exists(img_path):
                img.convert("RGB").save(img_path, quality=95)

    logger.info(f"Finished saving raw images. Summary of raw directories:")
    for c_name in os.listdir(dest_raw_dir):
        c_path = os.path.join(dest_raw_dir, c_name)
        if os.path.isdir(c_path):
            num_files = len(os.listdir(c_path))
            if num_files > 0:
                logger.info(f"  - {c_name}: {num_files} images")
                
    return class_names


def split_and_organize_data(
    raw_dir: str, 
    processed_dir: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
) -> Tuple[Dict[str, int], List[str]]:
    """Splits raw images into train/val/test directories stratified by class."""
    
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Get classes with non-empty raw directories
    target_classes = [
        d for d in os.listdir(raw_dir) 
        if os.path.isdir(os.path.join(raw_dir, d)) and len(os.listdir(os.path.join(raw_dir, d))) > 0
    ]

    # Clean existing processed directory if it exists to avoid contamination
    if os.path.exists(processed_dir):
        logger.info(f"Cleaning existing processed directory: {processed_dir}")
        shutil.rmtree(processed_dir)
        
    # Setup subdirectories
    splits = ["train", "val", "test"]
    for split in splits:
        for c in target_classes:
            os.makedirs(os.path.join(processed_dir, split, c), exist_ok=True)

    split_counts = {split: 0 for split in splits}
    random.seed(config.SEED)

    logger.info(f"Splitting and copying images for {len(target_classes)} classes...")
    for c in target_classes:
        class_path = os.path.join(raw_dir, c)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        random.shuffle(images)
        
        total_imgs = len(images)
        train_idx = int(total_imgs * train_ratio)
        val_idx = train_idx + int(total_imgs * val_ratio)
        
        train_imgs = images[:train_idx]
        val_imgs = images[train_idx:val_idx]
        test_imgs = images[val_idx:]
        
        def copy_files(img_list: List[str], split_name: str):
            for img_name in img_list:
                src_path = os.path.join(class_path, img_name)
                dest_path = os.path.join(processed_dir, split_name, c, img_name)
                shutil.copy2(src_path, dest_path)
                split_counts[split_name] += 1
                
        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")
        copy_files(test_imgs, "test")
        
        logger.info(f"Class '{c}': Split {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

    logger.info("Dataset split and organization finished.")
    logger.info(f"Final Counts -> Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}")
    return split_counts, target_classes


def main():
    # Setup folders
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    # 1. Download and Extract via HF
    download_and_extract_hf(
        dest_raw_dir=config.DATA_RAW_DIR, 
        prototype_mode=config.PROTOTYPE_MODE,
        prototype_classes=config.PROTOTYPE_CLASSES
    )
    
    # 2. Split and Organize Dataset
    split_and_organize_data(
        raw_dir=config.DATA_RAW_DIR,
        processed_dir=config.DATA_PROCESSED_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

if __name__ == "__main__":
    main()
