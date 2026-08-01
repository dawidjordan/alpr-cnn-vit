import torch
import torch.nn as nn

from models.shared.base_model import BaseModel


class SwinClassifier(BaseModel):
    

    VARIANTS = {
        "swin_tiny_patch4_window7_224":  768,
        "swin_small_patch4_window7_224": 768,
        "swin_base_patch4_window7_224":  1024,
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(num_classes=num_classes, pretrained=pretrained)

        try:
            import timm
        except ImportError:
            raise ImportError("Zainstaluj: pip install timm")

        if variant not in self.VARIANTS:
            raise ValueError(f"Nieznany wariant: {variant}. Dostępne: {list(self.VARIANTS)}")

        self.variant = variant
        self.feature_dim = self.VARIANTS[variant]

        self._backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def model_info(self) -> dict:
        size_tag = self.variant.split("_")[1]  # tiny / small / base
        return {
            "name": f"Swin-{size_tag.capitalize()}",
            "family": "ViT (Swin)",
            "variant": self.variant,
            "input_size": (3, 224, 224),
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
        }
