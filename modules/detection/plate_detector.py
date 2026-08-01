from pathlib import Path
from typing import Optional
import numpy as np

from modules.detection.base_detector import BaseDetector, Detection, DetectionResult


class PlateDetector(BaseDetector):
   
    
    PLATE_CLASS_NAME = "license_plate"
    PLATE_CLASS_ID = 0   

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        crop_padding: int = 8,
        min_plate_area: int = 500,
        rectify_perspective: bool = False,
        device: str = "auto",
        verbose: bool = False,
    ):
       
        self.crop_padding = crop_padding
        self.min_plate_area = min_plate_area
        self.rectify_perspective = rectify_perspective

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
            raise ImportError("Zainstaluj ultralytics: pip install ultralytics")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tablicy nie znaleziony: {model_path}\n"
                f"Wytrenuj model: python scripts/train_plate_detector.py\n"
                f"lub pobierz pretrenowany: patrz docs/dataset_guide.md"
            )

        self.model = YOLO(str(model_path))
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
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                class_id = int(box.cls.item())

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=self.PLATE_CLASS_NAME,
                ))

        return detections

    def target_classes(self) -> list[str]:
        return [self.PLATE_CLASS_NAME]

    

    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = True,
        min_area: float = None,
    ) -> DetectionResult:
        
        min_area = min_area if min_area is not None else self.min_plate_area
        result = super().detect(image, return_crops=False, min_area=min_area)

        if return_crops and result.success:
            for det in result.detections:
                crop = self._crop_image(image, det.bbox, padding=self.crop_padding)
                if self.rectify_perspective:
                    crop = self._rectify(image, det.bbox)
                det.crop = crop

        return result



    def _rectify(self, image: np.ndarray, bbox: list) -> np.ndarray:
        
        try:
            import cv2
        except ImportError:
            return self._crop_image(image, bbox, self.crop_padding)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1

        
        target_w = max(w, 200)
        target_h = int(target_w / 4.5)

        src_pts = np.float32([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ])
        dst_pts = np.float32([
            [0, 0], [target_w, 0], [target_w, target_h], [0, target_h]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (target_w, target_h))

    def get_plate_aspect_ratio(self, detection: Detection) -> float:
        
        return detection.width / detection.height if detection.height > 0 else 0.0

   

    @staticmethod
    def training_config(
        dataset_yaml: str = "data/processed/ccpd/dataset.yaml",
        base_model: str = "yolov8m.pt",
    ) -> dict:
       
        return {
            "model": base_model,
            "data": dataset_yaml,
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            # Augmentacje pomocne dla tablic:
            "degrees": 5.0,        # Małe obroty — tablice rzadko bardzo pochylone
            "perspective": 0.001,  # Symuluje kąt kamery
            "mosaic": 0.5,
            "mixup": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "flipud": 0.0,         # Tablice nie są odwrócone pionowo
            "fliplr": 0.5,
            "device": "auto",
            "project": "outputs/plate_detector",
            "name": "yolov8m_ccpd",
            "save_period": 10,
            "val": True,
            "plots": True,
        }
