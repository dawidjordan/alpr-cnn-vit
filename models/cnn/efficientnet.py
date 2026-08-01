import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights, EfficientNet_B7_Weights

from models.shared.base_model import BaseModel


class EfficientNetClassifier(BaseModel):
    

    VARIANTS = {
        "efficientnet_b4": (
            models.efficientnet_b4,
            EfficientNet_B4_Weights.IMAGENET1K_V1,
            1792,
            380,
        ),
        "efficientnet_b7": (
            models.efficientnet_b7,
            EfficientNet_B7_Weights.IMAGENET1K_V1,
            2560,
            600,
        ),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__(num_classes=num_classes, pretrained=pretrained)

        if variant not in self.VARIANTS:
            raise ValueError(f"Nieznany wariant: {variant}. Dostępne: {list(self.VARIANTS)}")

        model_fn, weights, feature_dim, img_size = self.VARIANTS[variant]
        self.variant = variant
        self.feature_dim = feature_dim
        self.recommended_img_size = img_size

        base = model_fn(weights=weights if pretrained else None)

        # EfficientNet ma strukturę: features → avgpool → classifier
        self._backbone = base.features
        self.avgpool = base.avgpool

        # Zastępujemy oryginalny classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
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
            "name": f"EfficientNet-{self.variant.split('_')[-1].upper()}",
            "family": "CNN",
            "variant": self.variant,
            "input_size": (3, self.recommended_img_size, self.recommended_img_size),
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
        }
