
import torch
import torch.nn as nn


class OCRHead(nn.Module):
    

    NUM_PLATE_CHARS = 7

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

       
        if hidden_dim > 0:
            self.shared_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
            )
            head_input_dim = hidden_dim
        else:
            self.shared_proj = nn.Identity()
            head_input_dim = feature_dim

        
        self.heads = nn.ModuleList([
            nn.Linear(head_input_dim, num_classes)
            for _ in range(self.NUM_PLATE_CHARS)
        ])

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
       
        x = self.shared_proj(features)
        return [head(x) for head in self.heads]


class OCRModel(nn.Module):
    

    def __init__(
        self,
        backbone,           
        num_classes: int,   
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        feature_dim = backbone.feature_dim

        self.ocr_head = OCRHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        
        features = self.backbone.get_features(x)
        return self.ocr_head(features)

    def count_parameters(self) -> dict:
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }
