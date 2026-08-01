
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import time
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._backbone: Optional[nn.Module] = None  # każda subklasa ustawia backbone

   
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ...

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        
        ...

    @abstractmethod
    def model_info(self) -> dict:
        
        ...

    

    def count_parameters(self, trainable_only: bool = True) -> dict:
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }

    @torch.no_grad()
    def measure_inference_time(
        self,
        input_size: tuple = (1, 3, 224, 224),
        n_warmup: int = 10,
        n_runs: int = 100,
        device: str = "cuda",
    ) -> dict:
        
        self.eval()
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        dummy = torch.randn(*input_size, device=dev)

       
        for _ in range(n_warmup):
            _ = self(dummy)

       
        if dev.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = self(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

        import statistics
        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times)
        fps = 1000.0 / mean_ms * input_size[0] 

        return {
            "device": str(dev),
            "batch_size": input_size[0],
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "fps": round(fps, 1),
        }



    def freeze_backbone(self) -> None:
        
        if self._backbone is None:
            raise AttributeError("Subklasa musi ustawić self._backbone")
        for param in self._backbone.parameters():
            param.requires_grad = False
        print(f"[{self.__class__.__name__}] Backbone zamrożony.")

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        
        if self._backbone is None:
            raise AttributeError("Subklasa musi ustawić self._backbone")

        layers = list(self._backbone.children())
        start = from_layer if from_layer is not None else 0

        for layer in layers[start:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self._backbone.parameters() if p.requires_grad)
        print(
            f"[{self.__class__.__name__}] Backbone odmrożony od warstwy {start}. "
            f"Trenowalne parametry backbone: {trainable:,}"
        )

    
    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        
        checkpoint = {
            "state_dict": self.state_dict(),
            "model_info": self.model_info(),
            "parameters": self.count_parameters(),
        }
        if metadata:
            checkpoint["metadata"] = metadata

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[{self.__class__.__name__}] Checkpoint zapisany → {path}")

    def load(self, path: str, strict: bool = True) -> dict:
       
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        print(f"[{self.__class__.__name__}] Wagi wczytane z {path}")
        return checkpoint.get("metadata", {})

    

    def __repr__(self) -> str:
        info = self.model_info()
        params = self.count_parameters()
        return (
            f"{info['name']} ({info['family']})\n"
            f"  Klasy:            {self.num_classes}\n"
            f"  Parametry total:  {params['total_M']} M\n"
            f"  Parametry trenow: {params['trainable_M']} M\n"
            f"  Wejście:          {info.get('input_size', 'N/A')}\n"
        )
