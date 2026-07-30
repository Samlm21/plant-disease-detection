import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

class PlantBaselineCNN(nn.Module):
    """
    Custom Convolutional Neural Network baseline model.
    Designed for leaf disease classification with 4 convolutional blocks.
    Input image dimensions: (3, 224, 224)
    """
    def __init__(self, num_classes: int):
        super(PlantBaselineCNN, self).__init__()
        
        # Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Block 4: 28x28 -> 14x14
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Classifier head
        # Spatial size is 14x14 after 4 MaxPool layers (224 / 2^4 = 14)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu_fc = nn.ReLU()
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        
        x = self.flatten(x)
        x = self.drop_fc(self.relu_fc(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        return x


class PlantResNet18Transfer(nn.Module):
    """
    Transfer learning model leveraging pretrained ResNet-18.
    The final classification layer is replaced with a custom head.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(PlantResNet18Transfer, self).__init__()
        
        # Load pretrained ResNet-18 model
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
            
        # Get input feature dimension of the original fc layer
        num_features = self.backbone.fc.in_features
        
        # Replace the original classification head with our custom multi-layer classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freezes or unfreezes backbone parameters to support fine-tuning vs feature extraction.
        
        Args:
            freeze (bool): If True, freezes the weights. If False, makes them trainable.
        """
        # Freeze/unfreeze all layers of the resnet backbone EXCEPT the new fc layer
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(model_name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """
    Model factory helper function.
    
    Args:
        model_name (str): 'baseline' or 'resnet18'
        num_classes (int): Number of target classes
    """
    if model_name.lower() == 'baseline':
        return PlantBaselineCNN(num_classes=num_classes)
    elif model_name.lower() == 'resnet18':
        pretrained = kwargs.get('pretrained', True)
        return PlantResNet18Transfer(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'baseline' or 'resnet18'")
