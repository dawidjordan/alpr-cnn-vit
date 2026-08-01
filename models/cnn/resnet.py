import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights

from models.shared.base_model import BaseModel


class ResNetClassifier(BaseModel):
    

    VARIANTS = {
        "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__(num_classes=num_classes, pretrained=pretrained)

        if variant not in self.VARIANTS:
            raise ValueError(f"Nieznany wariant: {variant}. Dostępne: {list(self.VARIANTS)}")

        model_fn, weights, feature_dim = self.VARIANTS[variant]
        self.variant = variant
        self.feature_dim = feature_dim

        
        base = model_fn(weights=weights if pretrained else None)
        self._backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1, 1)

        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self._backbone(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def model_info(self) -> dict:
        return {
            "name": f"ResNet-{self.variant[-2:]}",
            "family": "CNN",
            "variant": self.variant,
            "input_size": (3, 224, 224),
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
        }
