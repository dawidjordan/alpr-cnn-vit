import numpy as np
import cv2
from dataclasses import dataclass
from typing import Callable


@dataclass
class DegradationConfig:
    
    name: str
    severity: int       # 1 (łagodne) – 5 (silne)
    transform: Callable


class ImageDegradation:
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        sigma = severity
        ksize = 2 * int(3 * sigma) + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)

    @staticmethod
    def motion_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
       
        kernels = [3, 7, 11, 15, 20]
        k = kernels[min(severity - 1, 4)]
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(image, -1, kernel)

    
    @staticmethod
    def gaussian_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        stds = [5, 10, 20, 35, 50]
        std = stds[min(severity - 1, 4)]
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_pepper_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
       
        amounts = [0.005, 0.01, 0.02, 0.04, 0.08]
        amount = amounts[min(severity - 1, 4)]

        result = image.copy()
        n_pixels = int(amount * image.size / image.shape[2])

        # Sól (białe piksele)
        coords = [np.random.randint(0, d, n_pixels) for d in image.shape[:2]]
        result[coords[0], coords[1]] = 255

        # Pieprz (czarne piksele)
        coords = [np.random.randint(0, d, n_pixels) for d in image.shape[:2]]
        result[coords[0], coords[1]] = 0

        return result

   

    @staticmethod
    def jpeg_compression(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        qualities = [80, 60, 40, 25, 10]
        quality = qualities[min(severity - 1, 4)]

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    

    @staticmethod
    def brightness_reduction(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        factors = [0.8, 0.6, 0.4, 0.25, 0.1]
        factor = factors[min(severity - 1, 4)]
        darkened = image.astype(np.float32) * factor
        return np.clip(darkened, 0, 255).astype(np.uint8)

    @staticmethod
    def low_contrast(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        factors = [0.9, 0.8, 0.65, 0.5, 0.3]
        f = factors[min(severity - 1, 4)]
        result = image.astype(np.float32) * f + 128 * (1 - f)
        return np.clip(result, 0, 255).astype(np.uint8)

   

    @staticmethod
    def occlusion(image: np.ndarray, severity: int = 1) -> np.ndarray:
        
        ratios = [0.05, 0.10, 0.20, 0.35, 0.50]
        ratio = ratios[min(severity - 1, 4)]

        result = image.copy()
        h, w = image.shape[:2]
        area = h * w * ratio

        
        n_rects = max(1, severity)
        rect_area = area / n_rects

        for _ in range(n_rects):
            rw = int(np.sqrt(rect_area * w / h))
            rh = int(rect_area / rw) if rw > 0 else 1
            rw = max(1, min(rw, w))
            rh = max(1, min(rh, h))

            x = np.random.randint(0, w - rw + 1)
            y = np.random.randint(0, h - rh + 1)
            result[y:y+rh, x:x+rw] = 0

        return result

  

    @classmethod
    def get_all_degradations(cls) -> list[DegradationConfig]:
        
        configs = []
        degradations = [
            ("gaussian_blur",       cls.gaussian_blur),
            ("motion_blur",         cls.motion_blur),
            ("gaussian_noise",      cls.gaussian_noise),
            ("salt_pepper_noise",   cls.salt_pepper_noise),
            ("jpeg_compression",    cls.jpeg_compression),
            ("brightness_reduction",cls.brightness_reduction),
            ("low_contrast",        cls.low_contrast),
            ("occlusion",           cls.occlusion),
        ]
        for name, fn in degradations:
            for severity in range(1, 6):
                configs.append(DegradationConfig(
                    name=f"{name}_s{severity}",
                    severity=severity,
                    transform=lambda img, f=fn, s=severity: f(img, s),
                ))
        return configs

    @classmethod
    def apply_combined(
        cls,
        image: np.ndarray,
        blur_severity: int = 0,
        noise_severity: int = 0,
        brightness_severity: int = 0,
        jpeg_severity: int = 0,
    ) -> np.ndarray:
        
        result = image.copy()
        if blur_severity > 0:
            result = cls.gaussian_blur(result, blur_severity)
        if noise_severity > 0:
            result = cls.gaussian_noise(result, noise_severity)
        if brightness_severity > 0:
            result = cls.brightness_reduction(result, brightness_severity)
        if jpeg_severity > 0:
            result = cls.jpeg_compression(result, jpeg_severity)
        return result
