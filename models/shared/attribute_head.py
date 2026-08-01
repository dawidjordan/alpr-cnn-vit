import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AttributePrediction:
    
    color_logits: torch.Tensor     # (num_colors,)
    type_logits:  torch.Tensor     # (num_types,)
    make_logits:  torch.Tensor     # (num_makes,)

    @property
    def color_idx(self) -> int:
        return self.color_logits.argmax().item()

    @property
    def type_idx(self) -> int:
        return self.type_logits.argmax().item()

    @property
    def make_idx(self) -> int:
        return self.make_logits.argmax().item()

    def confidences(self) -> dict:
        return {
            "color": torch.softmax(self.color_logits, dim=-1).max().item(),
            "type":  torch.softmax(self.type_logits,  dim=-1).max().item(),
            "make":  torch.softmax(self.make_logits,  dim=-1).max().item(),
        }


class AttributeHead(nn.Module):
    

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VehicleAttributeModel(nn.Module):
    

    def __init__(
        self,
        backbone,
        num_colors: int = 15,
        num_types:  int = 12,
        num_makes:  int = 108,
        hidden_dim: int = 256,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.backbone  = backbone
        self.num_colors = num_colors
        self.num_types  = num_types
        self.num_makes  = num_makes

        feature_dim = backbone.feature_dim

        self.head_color = AttributeHead(feature_dim, num_colors, hidden_dim, dropout)
        self.head_type  = AttributeHead(feature_dim, num_types,  hidden_dim, dropout)
        self.head_make  = AttributeHead(feature_dim, num_makes,  hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        features = self.backbone.get_features(x)
        return (
            self.head_color(features),
            self.head_type(features),
            self.head_make(features),
        )

    def predict(self, x: torch.Tensor) -> list[AttributePrediction]:
        
        with torch.no_grad():
            color_logits, type_logits, make_logits = self(x)

        return [
            AttributePrediction(
                color_logits=color_logits[i],
                type_logits=type_logits[i],
                make_logits=make_logits[i],
            )
            for i in range(x.size(0))
        ]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }
