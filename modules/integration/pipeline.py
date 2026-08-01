import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ALPRResult:
    
    
    vehicle_bbox: Optional[list] = None          
    vehicle_confidence: float = 0.0

    plate_bbox: Optional[list] = None            
    plate_confidence: float = 0.0

    
    plate_text: Optional[str] = None            
    plate_text_confidence: float = 0.0
    plate_chars: list = field(default_factory=list)  

    
    vehicle_color: Optional[str] = None
    vehicle_color_confidence: float = 0.0

    vehicle_type: Optional[str] = None           
    vehicle_type_confidence: float = 0.0

    vehicle_make: Optional[str] = None          
    vehicle_make_confidence: float = 0.0

   
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        
        return self.plate_text is not None and len(self.plate_text) > 0

    def to_dict(self) -> dict:
        return {
            "plate_text": self.plate_text,
            "plate_confidence": round(self.plate_text_confidence, 3),
            "vehicle": {
                "color": self.vehicle_color,
                "type": self.vehicle_type,
                "make": self.vehicle_make,
            },
            "bboxes": {
                "vehicle": self.vehicle_bbox,
                "plate": self.plate_bbox,
            },
            "processing_time_ms": round(self.processing_time_ms, 1),
            "success": self.success,
        }


class ALPRPipeline:
   

    def __init__(
        self,
        vehicle_detector=None,
        plate_detector=None,
        ocr=None,
        attribute_classifier=None,
        min_vehicle_conf: float = 0.5,
        min_plate_conf: float = 0.4,
        device: str = "cuda",
    ):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr = ocr
        self.attribute_classifier = attribute_classifier
        self.min_vehicle_conf = min_vehicle_conf
        self.min_plate_conf = min_plate_conf
        self.device = device

    def process(self, image: np.ndarray) -> ALPRResult:
        
        result = ALPRResult()
        t_start = time.perf_counter()

        try:
        
            if self.vehicle_detector is not None:
                vehicle_detections = self.vehicle_detector.detect(image)
                if not vehicle_detections:
                    result.error = "Nie wykryto pojazdu"
                    return result

                
                best = max(vehicle_detections, key=lambda d: d["confidence"])
                if best["confidence"] < self.min_vehicle_conf:
                    result.error = f"Pewność detekcji pojazdu zbyt niska: {best['confidence']:.2f}"
                    return result

                result.vehicle_bbox = best["bbox"]
                result.vehicle_confidence = best["confidence"]
                vehicle_crop = self._crop(image, best["bbox"])
            else:
                vehicle_crop = image  

            
            if self.plate_detector is not None:
                plate_detections = self.plate_detector.detect(vehicle_crop)
                if not plate_detections:
                    result.error = "Nie wykryto tablicy rejestracyjnej"
                    return result

                best_plate = max(plate_detections, key=lambda d: d["confidence"])
                if best_plate["confidence"] < self.min_plate_conf:
                    result.error = f"Pewność detekcji tablicy zbyt niska: {best_plate['confidence']:.2f}"
                    return result

                result.plate_bbox = best_plate["bbox"]
                result.plate_confidence = best_plate["confidence"]
                plate_crop = self._crop(vehicle_crop, best_plate["bbox"])
            else:
                plate_crop = vehicle_crop

            
            if self.ocr is not None:
                ocr_result = self.ocr.recognize(plate_crop)
                result.plate_text = ocr_result["text"]
                result.plate_text_confidence = ocr_result["confidence"]
                result.plate_chars = ocr_result.get("chars", [])

            
            if self.attribute_classifier is not None:
                attrs = self.attribute_classifier.classify(vehicle_crop)
                result.vehicle_color = attrs.get("color")
                result.vehicle_color_confidence = attrs.get("color_conf", 0.0)
                result.vehicle_type = attrs.get("type")
                result.vehicle_type_confidence = attrs.get("type_conf", 0.0)
                result.vehicle_make = attrs.get("make")
                result.vehicle_make_confidence = attrs.get("make_conf", 0.0)

        except Exception as e:
            result.error = f"Błąd pipeline: {str(e)}"

        finally:
            result.processing_time_ms = (time.perf_counter() - t_start) * 1000

        return result

    def process_batch(self, images: list) -> list:
        
        return [self.process(img) for img in images]

    @staticmethod
    def _crop(image: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]
