import sys
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # ustaw CWD na root projektu


GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):  print(f"  {BLUE}→{RESET} {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 50)

errors   = []
warnings = []


header("1. Python")

version = sys.version_info
if version >= (3, 10):
    ok(f"Python {version.major}.{version.minor}.{version.micro}")
else:
    fail(f"Python {version.major}.{version.minor} — wymagane ≥ 3.10")
    errors.append("Python < 3.10")

info(f"Interpreter: {sys.executable}")
info(f"Root projektu: {ROOT}")


header("2. Akcelerator (GPU)")

try:
    import torch
    ok(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"CUDA GPU: {gpu} ({vram:.1f} GB VRAM)")
        info(f"CUDA wersja: {torch.version.cuda}")
        info("Rekomendacja: użyj device='cuda' w konfigach")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ok("Apple MPS (Metal) dostępne")
        info("Rekomendacja: użyj device='mps' w konfigach")
    else:
        warn("Brak GPU — trening będzie działał na CPU (bardzo wolno)")
        warn("Do eksperymentów użyj Google Colab (darmowy T4) lub Kaggle")
        warnings.append("Brak GPU")

except ImportError:
    fail("PyTorch NIE zainstalowany")
    errors.append("Brak PyTorch")



header("3. Biblioteki")

LIBRARIES = [
    ("torch",         "torch",          True),
    ("torchvision",   "torchvision",    True),
    ("timm",          "timm",           True),
    ("ultralytics",   "ultralytics",    True),
    ("cv2",           "opencv-python",  True),
    ("PIL",           "Pillow",         True),
    ("albumentations","albumentations", True),
    ("numpy",         "numpy",          True),
    ("pandas",        "pandas",         True),
    ("sklearn",       "scikit-learn",   True),
    ("yaml",          "pyyaml",         True),
    ("tqdm",          "tqdm",           True),
    ("matplotlib",    "matplotlib",     False),
    ("seaborn",       "seaborn",        False),
    ("wandb",         "wandb",          False),
]

missing_required = []
for import_name, pip_name, required in LIBRARIES:
    try:
        mod = __import__(import_name)
        version_str = getattr(mod, "__version__", "?")
        ok(f"{pip_name:<22} {version_str}")
    except ImportError:
        if required:
            fail(f"{pip_name:<22} NIE zainstalowane  →  pip install {pip_name}")
            missing_required.append(pip_name)
        else:
            warn(f"{pip_name:<22} opcjonalne, brak  →  pip install {pip_name}")

if missing_required:
    errors.append(f"Brak bibliotek: {', '.join(missing_required)}")
    print(f"\n  {YELLOW}Zainstaluj wszystko jedną komendą:{RESET}")
    print(f"  pip install -r requirements.txt")



header("4. Struktura katalogów")

REQUIRED_DIRS = [
    "data/raw", "data/processed", "data/augmented", "data/splits",
    "models/cnn", "models/vit", "models/shared",
    "modules/detection", "modules/ocr", "modules/classification", "modules/integration",
    "training/configs", "training/scripts",
    "evaluation/metrics", "evaluation/robustness", "evaluation/visualization",
    "utils", "notebooks", "scripts", "docs",
]

REQUIRED_FILES = [
    "models/shared/base_model.py",
    "models/cnn/resnet.py",
    "models/cnn/efficientnet.py",
    "models/vit/vit_base.py",
    "models/vit/swin.py",
    "modules/detection/base_detector.py",
    "modules/detection/vehicle_detector.py",
    "modules/detection/plate_detector.py",
    "modules/integration/pipeline.py",
    "evaluation/robustness/degradation.py",
    "training/configs/cnn_config.yaml",
    "training/configs/vit_config.yaml",
    "requirements.txt",
]

missing_dirs = []
for d in REQUIRED_DIRS:
    p = ROOT / d
    if p.exists():
        ok(d)
    else:
        fail(f"{d}  — BRAK")
        missing_dirs.append(d)

print()
missing_files = []
for f in REQUIRED_FILES:
    p = ROOT / f
    if p.exists():
        size = p.stat().st_size
        ok(f"{f}  ({size} B)")
    else:
        fail(f"{f}  — BRAK")
        missing_files.append(f)

if missing_dirs or missing_files:
    errors.append("Brakuje plików/katalogów projektu")


header("5. Importy modułów projektu")

own_imports = [
    ("models.shared.base_model",           "BaseModel"),
    ("models.cnn.resnet",                  "ResNetClassifier"),
    ("models.cnn.efficientnet",            "EfficientNetClassifier"),
    ("models.vit.vit_base",                "ViTClassifier"),
    ("models.vit.swin",                    "SwinClassifier"),
    ("modules.detection.base_detector",    "BaseDetector, Detection, DetectionResult"),
    ("modules.detection.vehicle_detector", "VehicleDetector"),
    ("modules.detection.plate_detector",   "PlateDetector"),
    ("modules.integration.pipeline",       "ALPRPipeline, ALPRResult"),
    ("evaluation.robustness.degradation",  "ImageDegradation"),
]

import_errors = []
for module_path, symbols in own_imports:
    try:
        __import__(module_path)
        ok(f"{module_path}")
    except ImportError as e:
        fail(f"{module_path}  →  {e}")
        import_errors.append(module_path)
    except Exception as e:
        warn(f"{module_path}  →  import OK, ale ostrzeżenie: {e}")

if import_errors:
    errors.append(f"Błędy importu: {import_errors}")


header("6. Modele — test forward pass (CPU)")

try:
    import torch
    import torch.nn as nn

    BATCH, C, H, W = 2, 3, 224, 224
    NUM_CLASSES = 10
    dummy_input = torch.randn(BATCH, C, H, W)

   
    try:
        from models.cnn.resnet import ResNetClassifier
        t0 = time.perf_counter()
        model = ResNetClassifier(num_classes=NUM_CLASSES, variant="resnet50", pretrained=False)
        out = model(dummy_input)
        elapsed = (time.perf_counter() - t0) * 1000
        assert out.shape == (BATCH, NUM_CLASSES), f"Zły kształt: {out.shape}"
        params = model.count_parameters()
        ok(f"ResNet-50  | output {tuple(out.shape)} | {params['total_M']} M param | {elapsed:.0f} ms")
    except Exception as e:
        fail(f"ResNet-50: {e}")
        errors.append("ResNet-50 forward pass")

   
    try:
        from models.cnn.efficientnet import EfficientNetClassifier
        t0 = time.perf_counter()
        model = EfficientNetClassifier(num_classes=NUM_CLASSES, variant="efficientnet_b4", pretrained=False)
        out = model(dummy_input)
        elapsed = (time.perf_counter() - t0) * 1000
        assert out.shape == (BATCH, NUM_CLASSES)
        params = model.count_parameters()
        ok(f"EfficientNet-B4 | output {tuple(out.shape)} | {params['total_M']} M param | {elapsed:.0f} ms")
    except Exception as e:
        fail(f"EfficientNet-B4: {e}")
        errors.append("EfficientNet forward pass")

    #
    try:
        import timm  
        from models.vit.vit_base import ViTClassifier
        t0 = time.perf_counter()
        model = ViTClassifier(num_classes=NUM_CLASSES, variant="vit_small_patch16_224", pretrained=False)
        out = model(dummy_input)
        elapsed = (time.perf_counter() - t0) * 1000
        assert out.shape == (BATCH, NUM_CLASSES)
        params = model.count_parameters()
        ok(f"ViT-Small/16  | output {tuple(out.shape)} | {params['total_M']} M param | {elapsed:.0f} ms")
    except ImportError:
        warn("ViT pominięty — brak timm (pip install timm)")
    except Exception as e:
        fail(f"ViT-Small: {e}")
        errors.append("ViT forward pass")


    try:
        import timm  
        from models.vit.swin import SwinClassifier
        t0 = time.perf_counter()
        model = SwinClassifier(num_classes=NUM_CLASSES, variant="swin_tiny_patch4_window7_224", pretrained=False)
        out = model(dummy_input)
        elapsed = (time.perf_counter() - t0) * 1000
        assert out.shape == (BATCH, NUM_CLASSES)
        params = model.count_parameters()
        ok(f"Swin-Tiny     | output {tuple(out.shape)} | {params['total_M']} M param | {elapsed:.0f} ms")
    except ImportError:
        warn("Swin pominięty — brak timm (pip install timm)")
    except Exception as e:
        fail(f"Swin-Tiny: {e}")
        errors.append("Swin forward pass")

   
    try:
        from models.cnn.resnet import ResNetClassifier
        m = ResNetClassifier(num_classes=5, pretrained=False)
        m.freeze_backbone()
        before = m.count_parameters()["trainable"]
        m.unfreeze_backbone(from_layer=6)
        after = m.count_parameters()["trainable"]
        assert after > before
        ok(f"freeze/unfreeze backbone  | {before:,} → {after:,} param trenow.")
    except Exception as e:
        fail(f"freeze/unfreeze: {e}")

except ImportError:
    warn("PyTorch niedostępny — pominięto testy modeli")


header("7. Moduł degradacji obrazu")

try:
    import numpy as np
    import cv2
    from evaluation.robustness.degradation import ImageDegradation

    img = np.random.randint(0, 255, (128, 320, 3), dtype=np.uint8)

    tests = [
        ("Gaussian blur s=3",       lambda i: ImageDegradation.gaussian_blur(i, 3)),
        ("Motion blur s=2",         lambda i: ImageDegradation.motion_blur(i, 2)),
        ("Gaussian noise s=3",      lambda i: ImageDegradation.gaussian_noise(i, 3)),
        ("Salt&Pepper noise s=2",   lambda i: ImageDegradation.salt_pepper_noise(i, 2)),
        ("JPEG compression s=4",    lambda i: ImageDegradation.jpeg_compression(i, 4)),
        ("Brightness reduction s=3",lambda i: ImageDegradation.brightness_reduction(i, 3)),
        ("Low contrast s=2",        lambda i: ImageDegradation.low_contrast(i, 2)),
        ("Occlusion s=3",           lambda i: ImageDegradation.occlusion(i, 3)),
        ("Combined degradations",   lambda i: ImageDegradation.apply_combined(i, blur_severity=2, noise_severity=2)),
    ]

    for name, fn in tests:
        try:
            result = fn(img)
            assert result.shape == img.shape, f"Zły kształt: {result.shape}"
            assert result.dtype == np.uint8
            ok(f"{name}")
        except Exception as e:
            fail(f"{name}: {e}")
            errors.append(f"Degradacja: {name}")

    all_configs = ImageDegradation.get_all_degradations()
    ok(f"get_all_degradations() → {len(all_configs)} konfiguracji (8 typów × 5 poziomów)")

except ImportError as e:
    warn(f"Pominięto test degradacji: {e}")




header("PODSUMOWANIE")

if not errors and not warnings:
    print(f"  {GREEN}{BOLD}✓ Wszystko działa poprawnie! Środowisko gotowe.{RESET}")
elif not errors:
    print(f"  {YELLOW}{BOLD} Ostrzeżenia ({len(warnings)}): środowisko działa, ale sprawdź poniższe:{RESET}")
    for w in warnings:
        warn(w)
else:
    print(f"  {RED}{BOLD}✗ Znaleziono {len(errors)} błąd(ów) — napraw przed przystąpieniem do pracy:{RESET}")
    for e in errors:
        fail(e)
    if warnings:
        print(f"\n  {YELLOW}Ostrzeżenia ({len(warnings)}):{RESET}")
        for w in warnings:
            warn(w)

print()
info("Następny krok: python scripts/prepare_data.py --dataset ccpd")
print()
