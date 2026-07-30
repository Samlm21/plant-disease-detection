"""
Unit tests for model architectures: output dimensions, forward passes, and parameter counts.
"""
import pytest
import torch
from src.model import PlantBaselineCNN, PlantResNet18Transfer, get_model


NUM_CLASSES = 5
BATCH = 2
INPUT = torch.randn(BATCH, 3, 224, 224)


class TestBaselineCNN:
    """Tests for the custom PlantBaselineCNN model."""

    def test_output_shape(self):
        """Forward pass should produce (batch_size, num_classes) logits."""
        model = PlantBaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(INPUT)
        assert out.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH},{NUM_CLASSES}), got {out.shape}"

    def test_parameters_exist(self):
        """Model should have trainable parameters."""
        model = PlantBaselineCNN(num_classes=NUM_CLASSES)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0, "Model has no trainable parameters"

    def test_no_nan_in_output(self):
        """Forward pass output should not contain NaN values."""
        model = PlantBaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(INPUT)
        assert not torch.isnan(out).any(), "Output contains NaN values"


class TestResNet18Transfer:
    """Tests for the pretrained ResNet18 transfer learning model."""

    def test_output_shape(self):
        """Forward pass should produce (batch_size, num_classes) logits."""
        model = PlantResNet18Transfer(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(INPUT)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_freeze_backbone(self):
        """Frozen backbone should have no grad on feature layers, only on fc head."""
        model = PlantResNet18Transfer(num_classes=NUM_CLASSES, pretrained=False)
        model.freeze_backbone(freeze=True)
        # fc head params should still require grad
        for name, param in model.backbone.named_parameters():
            if "fc" in name:
                assert param.requires_grad, f"fc param {name} should require grad"
            else:
                assert not param.requires_grad, f"backbone param {name} should be frozen"

    def test_unfreeze_backbone(self):
        """Unfreezing backbone should restore grad for all layers."""
        model = PlantResNet18Transfer(num_classes=NUM_CLASSES, pretrained=False)
        model.freeze_backbone(freeze=True)
        model.freeze_backbone(freeze=False)
        for param in model.parameters():
            assert param.requires_grad, "All params should require grad after unfreezing"


class TestGetModel:
    """Tests for the get_model factory function."""

    def test_get_baseline(self):
        model = get_model("baseline", num_classes=3)
        assert isinstance(model, PlantBaselineCNN)

    def test_get_resnet18(self):
        model = get_model("resnet18", num_classes=3, pretrained=False)
        assert isinstance(model, PlantResNet18Transfer)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            get_model("vgg16", num_classes=3)
