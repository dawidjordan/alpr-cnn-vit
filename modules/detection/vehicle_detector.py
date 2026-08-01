from pathlib import Path
from typing import Optional
import numpy as np

from modules.detection.base_detector import BaseDetector, Detection, DetectionResult



VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector(BaseDetector):
    

    PRETRAINED_MODELS = {
        "nano":   "yolov8n.pt",   #  3.2M param |  6 GFLOPs | najszybszy
        "small":  "yolov8s.pt",   # 11.2M param | 28 GFLOPs
        "medium": "yolov8m.pt",   # 25.9M param | 79 GFLOPs | dobry balans
        "large":  "yolov8l.pt",   # 43.7M param | 165 GFLOPs
        "xlarge": "yolov8x.pt",   # 68.2M param | 258 GFLOPs | najdokładniejszy
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "medium",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_vehicle_types: Optional[list[str]] = None,
        device: str = "auto",
        verbose: bool = False,
    ):
    
        if model_path is None:
            if variant not in self.PRETRAINED_MODELS:
                raise ValueError(
                    f"Nieznany wariant '{variant}'. "
                    f"Dostępne: {list(self.PRETRAINED_MODELS)}"
                )
            model_path = self.PRETRAINED_MODELS[variant]

        self.variant = variant
        self._target_types = set(
            target_vehicle_types
            if target_vehicle_types is not None
            else VEHICLE_CLASS_IDS.values()
        )

        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
            verbose=verbose,
        )

    

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Zainstaluj ultralytics: pip install ultralytics"
            )

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        if not self.verbose:
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())

                # Pomijamy klasy spoza słownika pojazdów
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                class_name = VEHICLE_CLASS_IDS[class_id]

                # Pomijamy typy poza wybraną listą
                if class_name not in self._target_types:
                    continue

                bbox = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
                conf = float(box.conf.item())

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                ))

        return detections

    def target_classes(self) -> list[str]:
        return sorted(self._target_types)

    

    def detect_largest(self, image: np.ndarray) -> Optional[Detection]:
        
        result = self.detect(image, return_crops=True)
        if not result.success:
            return None
        return max(result.detections, key=lambda d: d.area)

    def detect_closest_to_center(self, image: np.ndarray) -> Optional[Detection]:
        
        result = self.detect(image, return_crops=True)
        if not result.success:
            return None

        h, w = image.shape[:2]
        img_center = (w / 2, h / 2)

        def dist_to_center(det: Detection) -> float:
            cx, cy = det.center
            return ((cx - img_center[0]) ** 2 + (cy - img_center[1]) ** 2) ** 0.5

        return min(result.detections, key=dist_to_center)

    def finetune_config(self) -> dict:
        
        return {
            "model": self.PRETRAINED_MODELS.get(self.variant, "yolov8m.pt"),
            "data": "data/processed/vehicle_dataset.yaml",  # własny dataset
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "mosaic": 1.0,
            "augment": True,
            "device": self.device,
            "project": "outputs/vehicle_detector",
            "name": f"yolov8{self.variant[0]}_finetuned",
            "save_period": 10,
        }
