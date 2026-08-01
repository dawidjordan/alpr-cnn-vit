from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time
import numpy as np


@dataclass
class Detection:
    
    bbox: list                         
    confidence: float
    class_id: int
    class_name: str
    area: float = 0.0
    crop: Optional[np.ndarray] = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.area = max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": round(self.area, 1),
        }


@dataclass
class DetectionResult:
    
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: tuple = (0, 0)        # (H, W)
    model_name: str = ""
    error: Optional[str] = None

    @property
    def best(self) -> Optional[Detection]:
        
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def success(self) -> bool:
        return len(self.detections) > 0 and self.error is None

    def filtered(self, min_conf: float = 0.0, min_area: float = 0.0) -> list[Detection]:
        
        return [
            d for d in self.detections
            if d.confidence >= min_conf and d.area >= min_area
        ]


class BaseDetector(ABC):
   

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        verbose: bool = False,
    ):
        
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.verbose = verbose
        self.model = None
        self._load_model()

    

    @abstractmethod
    def _load_model(self) -> None:
        
        ...

    @abstractmethod
    def _run_inference(self, image: np.ndarray) -> list[Detection]:
       
        ...

    @abstractmethod
    def target_classes(self) -> list[str]:
        
        ...

    

    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = False,
        min_area: float = 0.0,
    ) -> DetectionResult:
        
        if image is None or image.size == 0:
            return DetectionResult(error="Pusty lub nieprawidłowy obraz wejściowy")

        t0 = time.perf_counter()
        result = DetectionResult(
            image_shape=image.shape[:2],
            model_name=self.__class__.__name__,
        )

        try:
            raw_detections = self._run_inference(image)

            
            filtered = [
                d for d in raw_detections
                if d.confidence >= self.confidence_threshold
                and d.area >= min_area
            ]

            
            if return_crops:
                for det in filtered:
                    det.crop = self._crop_image(image, det.bbox)

            result.detections = filtered

        except Exception as e:
            result.error = f"{self.__class__.__name__} inference error: {e}"

        result.inference_time_ms = (time.perf_counter() - t0) * 1000
        return result


    @staticmethod
    def _crop_image(image: np.ndarray, bbox: list, padding: int = 0) -> np.ndarray:
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _resolve_device(device: str) -> str:
        
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def compute_iou(bbox1: list, bbox2: list) -> float:
        
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def warmup(self, input_size: tuple = (640, 640), n: int = 3) -> None:
       
        dummy = np.zeros((*input_size, 3), dtype=np.uint8)
        for _ in range(n):
            self._run_inference(dummy)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  model:      {self.model_path}\n"
            f"  device:     {self.device}\n"
            f"  conf_thr:   {self.confidence_threshold}\n"
            f"  iou_thr:    {self.iou_threshold}\n"
            f"  classes:    {self.target_classes()}\n"
            f")"
        )
