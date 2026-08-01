import torch
import torch.nn as nn

from models.shared.base_model import BaseModel


class ViTClassifier(BaseModel):
    

    VARIANTS = {
        "vit_small_patch16_224": 384,
        "vit_base_patch16_224":  768,
        "vit_large_patch16_224": 1024,
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
        img_size: int = 224,
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
        self.img_size = img_size

        
        self._backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            drop_rate=dropout,
            dynamic_img_size=True,
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
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
       
        return self._backbone(x) 

    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        
        attention_maps = []

        def hook_fn(module, input, output):
            
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps.append(output[1].detach())

        
        blocks = list(self._backbone.blocks.children())
        target_block = blocks[layer_idx]
        handle = target_block.attn.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self._backbone(x)

        handle.remove()

        if not attention_maps:
            raise RuntimeError(
                "Nie udało się pobrać map uwagi. "
                "Upewnij się, że model timm obsługuje attention output."
            )
        return attention_maps[0]

    def model_info(self) -> dict:
        return {
            "name": self.variant.replace("_", "-").upper(),
            "family": "ViT",
            "variant": self.variant,
            "input_size": (3, self.img_size, self.img_size),
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
        }
