"""
Unit tests for data pipeline: preprocessing transforms, Dataset class, split ratios.
"""
import os
import pytest
import torch
from torchvision import transforms
from PIL import Image
import tempfile
import shutil

from src.preprocessing import get_train_transforms, get_val_test_transforms, PlantDataset

class TestTransforms:
    """Tests for transform pipelines."""

    def test_train_transforms_output_shape(self):
        """Train transforms should produce a (3, 224, 224) tensor."""
        transform = get_train_transforms()
        img = Image.new("RGB", (256, 256), color=(128, 200, 50))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"

    def test_val_transforms_output_shape(self):
        """Validation transforms should produce a (3, 224, 224) tensor."""
        transform = get_val_test_transforms()
        img = Image.new("RGB", (512, 400), color=(100, 100, 100))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_train_transforms_normalised(self):
        """After normalization, pixel values should deviate from raw [0,1] range."""
        transform = get_val_test_transforms()
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        tensor = transform(img)
        # ImageNet mean subtraction means all-white image gives values around 2.25
        assert tensor.max().item() > 1.0 or tensor.min().item() < 0.0


class TestPlantDataset:
    """Tests for the custom PlantDataset class."""

    def setup_method(self):
        """Create a temporary directory with fake class folders and images."""
        self.temp_dir = tempfile.mkdtemp()
        self.classes = ["Tomato___healthy", "Tomato___Late_blight"]
        for cls in self.classes:
            cls_dir = os.path.join(self.temp_dir, cls)
            os.makedirs(cls_dir)
            for i in range(5):
                img = Image.new("RGB", (256, 256), color=(i * 40, i * 20, 10))
                img.save(os.path.join(cls_dir, f"img_{i}.jpg"))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_length(self):
        """Dataset length should match total number of images created."""
        dataset = PlantDataset(self.temp_dir, transform=get_val_test_transforms())
        assert len(dataset) == 10

    def test_dataset_classes(self):
        """Dataset should correctly detect class directories."""
        dataset = PlantDataset(self.temp_dir, transform=get_val_test_transforms())
        assert set(dataset.classes) == set(self.classes)

    def test_dataset_item(self):
        """__getitem__ should return (tensor, int_label) pairs."""
        dataset = PlantDataset(self.temp_dir, transform=get_val_test_transforms())
        tensor, label = dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < len(self.classes)

    def test_class_to_idx_consistency(self):
        """class_to_idx and idx_to_class should be mutual inverses."""
        dataset = PlantDataset(self.temp_dir, transform=get_val_test_transforms())
        for cls, idx in dataset.class_to_idx.items():
            assert dataset.idx_to_class[idx] == cls
