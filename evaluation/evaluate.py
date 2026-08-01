import argparse
import csv
import json
import sys
import time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_ocr_model(checkpoint_path: str, arch: str, device: torch.device):
    
    from models.shared.ocr_head import OCRModel
    from utils.dataset_ccpd import NUM_CHARS

    if arch.startswith("resnet"):
        from models.cnn.resnet import ResNetClassifier
        backbone = ResNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("efficientnet"):
        from models.cnn.efficientnet import EfficientNetClassifier
        backbone = EfficientNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("vit_"):
        from models.vit.vit_base import ViTClassifier
        backbone = ViTClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("swin_"):
        from models.vit.swin import SwinClassifier
        backbone = SwinClassifier(num_classes=1, variant=arch, pretrained=False)

    model = OCRModel(backbone=backbone, num_classes=NUM_CHARS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def load_attribute_model(checkpoint_path: str, arch: str, device: torch.device):
    
    from models.shared.attribute_head import VehicleAttributeModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_colors = ckpt.get("num_colors", 15)
    num_types  = ckpt.get("num_types",  12)
    num_makes  = ckpt.get("num_makes",  108)

    if arch.startswith("resnet"):
        from models.cnn.resnet import ResNetClassifier
        backbone = ResNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("efficientnet"):
        from models.cnn.efficientnet import EfficientNetClassifier
        backbone = EfficientNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("vit_"):
        from models.vit.vit_base import ViTClassifier
        backbone = ViTClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("swin_"):
        from models.vit.swin import SwinClassifier
        backbone = SwinClassifier(num_classes=1, variant=arch, pretrained=False)

    model = VehicleAttributeModel(
        backbone=backbone,
        num_colors=num_colors,
        num_types=num_types,
        num_makes=num_makes,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_ocr(model, loader: DataLoader, device: torch.device) -> dict:
    
    total_chars = correct_chars = 0
    total_plates = correct_plates = 0

    for images, targets in tqdm(loader, desc="  Eval OCR", leave=False):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits_list = model(images)
        B = images.size(0)

        preds = torch.stack([l.argmax(dim=1) for l in logits_list], dim=1) 
        correct_chars  += (preds == targets).sum().item()
        correct_plates += (preds == targets).all(dim=1).sum().item()
        total_chars    += B * 7
        total_plates   += B

    return {
        "char_acc":  correct_chars  / total_chars,
        "plate_acc": correct_plates / total_plates,
    }



@torch.no_grad()
def evaluate_attributes(model, loader: DataLoader, device: torch.device) -> dict:
    
    color_correct = color_total = 0
    type_correct  = type_total  = 0
    make_correct  = make_total  = 0

    for images, labels in tqdm(loader, desc="  Eval Attr", leave=False):
        images   = images.to(device, non_blocking=True)
        color_gt = labels["color"].to(device)
        type_gt  = labels["type"].to(device)
        make_gt  = labels["make"].to(device)

        color_logits, type_logits, make_logits = model(images)

        def acc(logits, gt):
            mask = gt >= 0
            if mask.sum() == 0:
                return 0, 0
            pred = logits[mask].argmax(dim=1)
            return (pred == gt[mask]).sum().item(), mask.sum().item()

        cc, ct = acc(color_logits, color_gt)
        tc, tt = acc(type_logits,  type_gt)
        mc, mt = acc(make_logits,  make_gt)

        color_correct += cc; color_total += ct
        type_correct  += tc; type_total  += tt
        make_correct  += mc; make_total  += mt

    color_acc = color_correct / max(color_total, 1)
    type_acc  = type_correct  / max(type_total,  1)
    make_acc  = make_correct  / max(make_total,  1)

    return {
        "color_acc": color_acc,
        "type_acc":  type_acc,
        "make_acc":  make_acc,
        "mean_acc":  (color_acc + type_acc + make_acc) / 3,
    }


def measure_inference_speed(
    model,
    input_size: tuple = (1, 3, 224, 224),
    device: torch.device = None,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy = torch.randn(*input_size, device=device)

    
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    import statistics
    mean_ms = statistics.mean(times)
    std_ms  = statistics.stdev(times)

    return {
        "mean_ms":  round(mean_ms, 3),
        "std_ms":   round(std_ms,  3),
        "fps":      round(1000.0 / mean_ms * input_size[0], 1),
        "device":   str(device),
    }



def run_e1_baseline(args, device):
    print(f"\n{'═'*60}")
    print(f"  E1 — Baseline")
    print(f"{'═'*60}")

    results = {}

    if args.task in ("ocr", "all"):
        print("\n  [OCR] Loading validation dataset...")
        from utils.dataset_ccpd import ProcessedCCPDOCRDataset
        val_dataset = ProcessedCCPDOCRDataset(root="data/processed/ccpd/val")
        val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                                 num_workers=args.workers, pin_memory=device.type=="cuda")

        for arch, ckpt_path in get_ocr_checkpoints(args):
            print(f"\n  OCR — {arch}")
            model  = load_ocr_model(ckpt_path, arch, device)
            metrics = evaluate_ocr(model, val_loader, device)
            speed   = measure_inference_speed(model, device=device)
            results[f"ocr_{arch}"] = {**metrics, **speed}
            print(f"  char_acc={metrics['char_acc']:.4f} | "
                  f"plate_acc={metrics['plate_acc']:.4f} | "
                  f"fps={speed['fps']}")

    if args.task in ("attributes", "all"):
        print("\n  [Atributes] Loading test dataset...")
        from utils.dataset_attributes import VehicleAttributeDataset
        val_dataset = VehicleAttributeDataset(
            compcars_root=args.compcars_root,
            color_root=args.color_root,
            split="test",
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type=="cuda")

        for arch, ckpt_path in get_attr_checkpoints(args):
            print(f"\n  Atributes — {arch}")
            model   = load_attribute_model(ckpt_path, arch, device)
            metrics = evaluate_attributes(model, val_loader, device)
            speed   = measure_inference_speed(model, device=device)
            results[f"attr_{arch}"] = {**metrics, **speed}
            print(f"  color={metrics['color_acc']:.4f} | "
                  f"type={metrics['type_acc']:.4f} | "
                  f"make={metrics['make_acc']:.4f} | "
                  f"mean={metrics['mean_acc']:.4f} | "
                  f"fps={speed['fps']}")

    save_results(results, "E1_baseline", args.output_dir)
    return results



def run_e2_robustness(args, device):
    print(f"\n{'═'*60}")
    print(f"  E2 — Resistance to degradation")
    print(f"{'═'*60}")

    SUBSETS = ["ccpd_blur", "ccpd_db", "ccpd_weather",
               "ccpd_tilt", "ccpd_fn", "ccpd_rotate", "ccpd_challenge"]

    results = {}

    if args.task in ("ocr", "all"):
        from utils.dataset_ccpd import ProcessedCCPDOCRDataset

        for arch, ckpt_path in get_ocr_checkpoints(args):
            print(f"\n  OCR — {arch}")
            model = load_ocr_model(ckpt_path, arch, device)
            results[f"ocr_{arch}"] = {}

            for subset in SUBSETS:
                subset_dir = Path("data/processed/ccpd") / subset
                if not subset_dir.exists():
                    print(f"   Missing subset: {subset}")
                    continue

                dataset = ProcessedCCPDOCRDataset(root=str(subset_dir))
                loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                                     num_workers=args.workers,
                                     pin_memory=device.type=="cuda")
                metrics = evaluate_ocr(model, loader, device)
                results[f"ocr_{arch}"][subset] = metrics
                print(f"  {subset:<20} char={metrics['char_acc']:.4f} | "
                      f"plate={metrics['plate_acc']:.4f}")


    if args.task in ("attributes", "all"):
        from utils.dataset_attributes import VehicleAttributeDataset
        from evaluation.robustness.degradation import ImageDegradation
        import cv2
        import numpy as np
        from torch.utils.data import Dataset

        class DegradedAttributeDataset(Dataset):
           
            def __init__(self, base_dataset, degradation_fn):
                self.base = base_dataset
                self.degradation_fn = degradation_fn

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                img_tensor, labels = self.base[idx]
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = img_tensor.numpy().transpose(1, 2, 0)
                img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
                img = self.degradation_fn(img)
                img = img.astype(np.float32) / 255.0
                img = (img - mean) / std
                return torch.from_numpy(img.transpose(2, 0, 1)), labels

        DEGRADATIONS = {
            "gaussian_blur_s3":    lambda i: ImageDegradation.gaussian_blur(i, 3),
            "gaussian_noise_s3":   lambda i: ImageDegradation.gaussian_noise(i, 3),
            "jpeg_compression_s3": lambda i: ImageDegradation.jpeg_compression(i, 3),
            "brightness_s3":       lambda i: ImageDegradation.brightness_reduction(i, 3),
            "low_contrast_s3":     lambda i: ImageDegradation.low_contrast(i, 3),
            "occlusion_s3":        lambda i: ImageDegradation.occlusion(i, 3),
        }

        base_val = VehicleAttributeDataset(
            compcars_root=args.compcars_root,
            color_root=args.color_root,
            split="test",
        )

        for arch, ckpt_path in get_attr_checkpoints(args):
            print(f"\n  Atributes E2 — {arch}")
            model = load_attribute_model(ckpt_path, arch, device)
            results[f"attr_{arch}"] = {}

            loader = DataLoader(base_val, batch_size=32, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type=="cuda")
            metrics = evaluate_attributes(model, loader, device)
            results[f"attr_{arch}"]["baseline"] = metrics
            print(f"  {'baseline':<25} mean={metrics['mean_acc']:.4f} | "
                  f"color={metrics['color_acc']:.4f} | "
                  f"type={metrics['type_acc']:.4f} | "
                  f"make={metrics['make_acc']:.4f}")

            
            for deg_name, deg_fn in DEGRADATIONS.items():
                deg_dataset = DegradedAttributeDataset(base_val, deg_fn)
                loader = DataLoader(deg_dataset, batch_size=32, shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=device.type=="cuda")
                metrics = evaluate_attributes(model, loader, device)
                results[f"attr_{arch}"][deg_name] = metrics
                print(f"  {deg_name:<25} "
                    f"mean={metrics['mean_acc']*100:5.1f}% | "
                    f"color={metrics['color_acc']*100:5.1f}% | "
                    f"type={metrics['type_acc']*100:5.1f}% | "
                    f"make={metrics['make_acc']*100:5.1f}%")
                
        
            print(f"\n  Summary {arch} — decline relative to the baseline:")
            baseline = results[f"attr_{arch}"].get("baseline", {})
            if baseline:
                print(f"  {'Degradation':<25} {'Δmean':>7} {'Δcolor':>7} "
                    f"{'Δtype':>7} {'Δmake':>7}")
                print(f"  {'-'*57}")
                for deg_name, m in results[f"attr_{arch}"].items():
                    if deg_name == "baseline":
                        continue
                    dm = (m['mean_acc']  - baseline['mean_acc'])  * 100
                    dc = (m['color_acc'] - baseline['color_acc']) * 100
                    dt = (m['type_acc']  - baseline['type_acc'])  * 100
                    dk = (m['make_acc']  - baseline['make_acc'])  * 100
                    print(f"  {deg_name:<25} "
                        f"{dm:+6.1f}% {dc:+6.1f}% "
                        f"{dt:+6.1f}% {dk:+6.1f}%")

    save_results(results, "E2_robustness", args.output_dir)
    return results


def run_e3_limited_data(args, device):

    print(f"\n{'═'*60}")
    print(f"  E3 — Limited data")
    print(f"{'═'*60}")

    
    FRACTIONS = {"10pct": 0.10, "20pct": 0.20, "30pct": 0.30, "50pct": 0.50, "80pct": 0.80, "100pct": 1.00}
    OCR_SAMPLES = {"10pct": 5000, "20pct": 10000, "30pct": 15000, "50pct": 25000, "80pct": 40000, "100pct": 50000}
    ATTR_SAMPLES = {"10pct": None, "20pct": None, "30pct": None,
                "50pct": None, "80pct": None, "100pct": None}

    results = {}

    if args.task in ("ocr", "all"):
        from utils.dataset_ccpd import ProcessedCCPDOCRDataset
        val_dataset = ProcessedCCPDOCRDataset(root="data/processed/ccpd/val")
        val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                                 num_workers=args.workers, pin_memory=device.type=="cuda")

        for arch, _ in get_ocr_checkpoints(args):
            results[f"ocr_{arch}"] = {}

            for fraction_name, n_samples in OCR_SAMPLES.items():
                if arch.startswith("vit_") or arch.startswith("swin_"):
                    ckpt_dir = Path(f"outputs/ocr_vit/E3/{arch}_{fraction_name}")
                else:
                    ckpt_dir = Path(f"outputs/ocr_cnn/E3/{arch}_{fraction_name}")

                ckpt_path = ckpt_dir / "checkpoint_best.pt"

                if not ckpt_path.exists():
                    script = "train_ocr_vit" if ("vit" in arch or "swin" in arch) else "train_ocr_cnn"
                    samples_arg = f"--max-samples {n_samples}" if n_samples else ""
                    print(f"   Missing checkpoint: {ckpt_path}")
                    print(f"    Run: python training/scripts/{script}.py "
                          f"--arch {arch} {samples_arg} --output-dir {ckpt_dir}")
                    continue

                print(f"\n  OCR — {arch} @ {fraction_name}")
                model   = load_ocr_model(str(ckpt_path), arch, device)
                metrics = evaluate_ocr(model, val_loader, device)
                results[f"ocr_{arch}"][fraction_name] = metrics
                print(f"  char={metrics['char_acc']:.4f} | plate={metrics['plate_acc']:.4f}")

    if args.task in ("attributes", "all"):
        from utils.dataset_attributes import VehicleAttributeDataset

        val_dataset = VehicleAttributeDataset(
            compcars_root=args.compcars_root,
            color_root=args.color_root,
            split="test",
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type=="cuda")

        for arch, _ in get_attr_checkpoints(args):
            results[f"attr_{arch}"] = {}

            for fraction_name, n_samples in ATTR_SAMPLES.items():
                ckpt_dir  = Path(f"outputs/attributes/E3/{arch}_{fraction_name}")
                ckpt_path = ckpt_dir / "checkpoint_best.pt"

                if not ckpt_path.exists():
                    samples_arg = f"--max-samples {n_samples}" if n_samples else ""
                    print(f"  Missing checkpoint: {ckpt_path}")
                    print(f"    Run: python training/scripts/train_attributes.py "
                          f"--arch {arch} {samples_arg} --output-dir {ckpt_dir}")
                    continue

                print(f"\n  Atributes — {arch} @ {fraction_name}")
                model   = load_attribute_model(str(ckpt_path), arch, device)
                metrics = evaluate_attributes(model, val_loader, device)
                results[f"attr_{arch}"][fraction_name] = metrics
                print(f"  color={metrics['color_acc']:.4f} | "
                      f"type={metrics['type_acc']:.4f} | "
                      f"make={metrics['make_acc']:.4f} | "
                      f"mean={metrics['mean_acc']:.4f}")

    save_results(results, "E3_limited_data", args.output_dir)
    return results


def run_e4_overfitting(args, device):
    
    print(f"\n{'═'*60}")
    print(f"  E4 — Overfitting analysis (learning curves)")
    print(f"{'═'*60}")

    results = {}

    def load_metrics_csv(csv_path: Path) -> list[dict]:
        if not csv_path.exists():
            return []
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) if k not in ("epoch",) else int(v)
                             for k, v in row.items()})
        return rows

    if args.task in ("ocr", "all"):
        for arch, _ in get_ocr_checkpoints(args):
            if arch.startswith("vit_") or arch.startswith("swin_"):
                csv_path = Path(f"outputs/ocr_vit/E1/{arch}/metrics.csv")
            else:
                csv_path = Path(f"outputs/ocr_cnn/E1/{arch}/metrics.csv")

            rows = load_metrics_csv(csv_path)
            if rows:
                results[f"ocr_{arch}"] = rows
                print(f"  OCR {arch}: {len(rows)} epok wczytanych z {csv_path}")
            else:
                print(f"  Missing file: {csv_path}")

    if args.task in ("attributes", "all"):
        for arch, _ in get_attr_checkpoints(args):
            csv_path = Path(f"outputs/attributes/E1/{arch}/metrics.csv")
            rows = load_metrics_csv(csv_path)
            if rows:
                results[f"attr_{arch}"] = rows
                print(f"  Attr {arch}: {len(rows)} epok | "
                    f"kolumny: {list(rows[0].keys())}")
            else:
                print(f"  Missing file: {csv_path}")

    save_results(results, "E4_overfitting", args.output_dir)
    return results



def run_e5_inference_speed(args, device):
    
    print(f"\n{'═'*60}")
    print(f"  E5 — Inference speed")
    print(f"{'═'*60}")

    BATCH_SIZES = [1, 8, 32, 64]
    IMG_SIZE = (3, 224, 224)
    results = {}

    def measure_model(model, arch, task_name):
        arch_results = {
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "batch_results": {},
        }
        print(f"\n  {task_name} — {arch}")
        print(f"  Parameters: {arch_results['params_M']:.2f}M")

        for bs in BATCH_SIZES:
            speed = measure_inference_speed(
                model,
                input_size=(bs, *IMG_SIZE),
                device=device,
                n_warmup=10,
                n_runs=50,
            )
            arch_results["batch_results"][str(bs)] = speed
            print(f"  batch={bs:>2} | "
                  f"mean={speed['mean_ms']:.2f}ms | "
                  f"fps={speed['fps']:.1f}")

        return arch_results

    if args.task in ("ocr", "all"):
        for arch, ckpt_path in get_ocr_checkpoints(args):
            model = load_ocr_model(ckpt_path, arch, device)
            results[f"ocr_{arch}"] = measure_model(model, arch, "OCR")

    if args.task in ("attributes", "all"):
        for arch, ckpt_path in get_attr_checkpoints(args):
            model = load_attribute_model(ckpt_path, arch, device)
            results[f"attr_{arch}"] = measure_model(model, arch, "Atributes")

    save_results(results, "E5_inference_speed", args.output_dir)
    return results


def get_ocr_checkpoints(args) -> list[tuple[str, str]]:
    
    checkpoints = []
    for arch in args.ocr_archs:
        if arch.startswith("vit_") or arch.startswith("swin_"):
            path = Path(f"outputs/ocr_vit/E1/{arch}/checkpoint_best.pt")
        else:
            path = Path(f"outputs/ocr_cnn/E1/{arch}/checkpoint_best.pt")

        if path.exists():
            checkpoints.append((arch, str(path)))
        else:
            print(f"  No OCR checkpoint for {arch}: {path}")
    return checkpoints


def get_attr_checkpoints(args) -> list[tuple[str, str]]:
    
    checkpoints = []
    for arch in args.attr_archs:
        path = Path(f"outputs/attributes/E1/{arch}/checkpoint_best.pt")
        if path.exists():
            checkpoints.append((arch, str(path)))
        else:
            print(f"  No atributes checkpoint for {arch}: {path}")
    return checkpoints


def save_results(results: dict, experiment: str, output_dir: str) -> None:
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Wyniki zapisane → {out_path}")

def measure_pipeline_speed(args, device):
    
    
    ccpd_dir  = Path("data/raw/ccpd/ccpd_base")
    comp_dir  = Path("data/raw/compcars/image")

    ccpd_images = list(ccpd_dir.glob("*.jpg"))[:50]
    comp_images = list(comp_dir.rglob("*.jpg"))[:50]

    if not ccpd_images:
        print("  No CCPD images for pipeline measurement")
        return {}

    results = {}
    N_WARMUP = 5
    N_RUNS   = 30

    
    for arch, ckpt_path in get_ocr_checkpoints(args):
        print(f"\n  Pipeline OCR — {arch}")

        
        ocr_model = load_ocr_model(ckpt_path, arch, device)

        
        try:
            from ultralytics import YOLO
            yolo = YOLO("models/yolov8n-lp.pt")
            use_yolo = True
        except Exception:
            use_yolo = False
            print("   No YOLO model — pipeline without detection")

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def run_ocr_pipeline(img_path):
           
            img = cv2.imread(str(img_path))
            if img is None:
                return

            
            if use_yolo:
                det_results = yolo(img, verbose=False, conf=0.25)[0]
                if len(det_results.boxes) > 0:
                    x1, y1, x2, y2 = map(int, det_results.boxes.xyxy[0].tolist())
                    crop = img[y1:y2, x1:x2]
                else:
                    crop = img
            else:
                crop = img

            
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (128, 64))
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - mean) / std
            tensor = torch.from_numpy(
                crop.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            
            with torch.no_grad():
                logits = ocr_model(tensor)

            
            from utils.dataset_ccpd import IDX_TO_CHAR
            chars = [l[0].argmax().item() for l in logits]
            text  = "".join(IDX_TO_CHAR.get(c, "?") for c in chars)
            return text

        
        for img_path in ccpd_images[:N_WARMUP]:
            run_ocr_pipeline(img_path)
        if device.type == "cuda":
            torch.cuda.synchronize()

       
        times = []
        images_cycle = ccpd_images * (N_RUNS // len(ccpd_images) + 1)
        for img_path in images_cycle[:N_RUNS]:
            t0 = time.perf_counter()
            run_ocr_pipeline(img_path)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        import statistics
        mean_ms = statistics.mean(times)
        std_ms  = statistics.stdev(times)
        fps     = 1000.0 / mean_ms

        results[f"ocr_pipeline_{arch}"] = {
            "mean_ms": round(mean_ms, 2),
            "std_ms":  round(std_ms, 2),
            "fps":     round(fps, 1),
            "stages":  "load + yolo + preprocess + ocr + decode",
        }
        print(f"  mean={mean_ms:.1f}ms | std={std_ms:.1f}ms | fps={fps:.1f}")

    
    if comp_images and args.task in ("attributes", "all"):
        for arch, ckpt_path in get_attr_checkpoints(args):
            print(f"\n  Atributes pipeline — {arch}")

            attr_model = load_attribute_model(ckpt_path, arch, device)

            try:
                from ultralytics import YOLO
                yolo_car = YOLO("yolov8n.pt")
                use_yolo_car = True
            except Exception:
                use_yolo_car = False

            VEHICLE_CLASSES = {2, 3, 5, 7}

            def run_attr_pipeline(img_path):
                
                img = cv2.imread(str(img_path))
                if img is None:
                    return

                
                if use_yolo_car:
                    det = yolo_car(img, verbose=False, conf=0.25)[0]
                    vehicle_boxes = [
                        b for b in det.boxes
                        if int(b.cls[0]) in VEHICLE_CLASSES
                    ]
                    if vehicle_boxes:
                        x1, y1, x2, y2 = map(int,
                            vehicle_boxes[0].xyxy[0].tolist())
                        crop = img[y1:y2, x1:x2]
                    else:
                        crop = img
                else:
                    crop = img

                
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, (224, 224))
                crop = crop.astype(np.float32) / 255.0
                crop = (crop - mean) / std
                tensor = torch.from_numpy(
                    crop.transpose(2, 0, 1)
                ).unsqueeze(0).to(device)

                
                with torch.no_grad():
                    color_l, type_l, make_l = attr_model(tensor)

                
                color = color_l[0].argmax().item()
                type_ = type_l[0].argmax().item()
                make  = make_l[0].argmax().item()
                return color, type_, make

            for img_path in comp_images[:N_WARMUP]:
                run_attr_pipeline(img_path)
            if device.type == "cuda":
                torch.cuda.synchronize()

        
            times = []
            images_cycle = comp_images * (N_RUNS // len(comp_images) + 1)
            for img_path in images_cycle[:N_RUNS]:
                t0 = time.perf_counter()
                run_attr_pipeline(img_path)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            mean_ms = statistics.mean(times)
            std_ms  = statistics.stdev(times)
            fps     = 1000.0 / mean_ms

            results[f"attr_pipeline_{arch}"] = {
                "mean_ms": round(mean_ms, 2),
                "std_ms":  round(std_ms, 2),
                "fps":     round(fps, 1),
                "stages":  "load + yolo + preprocess + attr + decode",
            }
            print(f"  mean={mean_ms:.1f}ms | fps={fps:.1f}")

    save_results(results, "E5_pipeline_speed", args.output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja CNN vs ViT")
    parser.add_argument("--task",       type=str, default="all",
                        choices=["ocr", "attributes", "all"])
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["E1", "E2", "E3", "E4", "E5", "all"])

    
    parser.add_argument("--ocr-archs",  type=str, nargs="+",
                        default=["resnet50", "vit_small_patch16_224"],
                        help="OCR architectures for evaluation")
    parser.add_argument("--attr-archs", type=str, nargs="+",
                        default=["resnet50", "vit_small_patch16_224"],
                        help="Atributes architectures for evaluation")

    
    parser.add_argument("--compcars-root", type=str, default="data/raw/compcars")
    parser.add_argument("--color-root",    type=str, default="data/raw/vehicle_color")
    parser.add_argument("--output-dir",    type=str, default="outputs/evaluation")
    parser.add_argument("--workers",       type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  Evaluation CNN vs ViT")
    print(f"{'═'*60}")
    print(f"  Device:  {device}")
    print(f"  Task:     {args.task}")
    print(f"  Experiment: {args.experiment}")
    print(f"  OCR archs:   {args.ocr_archs}")
    print(f"  Attr archs:  {args.attr_archs}")

    run_e1 = args.experiment in ("E1", "all")
    run_e2 = args.experiment in ("E2", "all")
    run_e3 = args.experiment in ("E3", "all")
    run_e4 = args.experiment in ("E4", "all")
    run_e5 = args.experiment in ("E5", "all")

    all_results = {}

    if run_e1:
        all_results["E1"] = run_e1_baseline(args, device)
    if run_e2:
        all_results["E2"] = run_e2_robustness(args, device)
    if run_e3:
        all_results["E3"] = run_e3_limited_data(args, device)
    if run_e4:
        all_results["E4"] = run_e4_overfitting(args, device)
    if run_e5:
        all_results["E5"] = run_e5_inference_speed(args, device)
        all_results["E5_pipeline"] = measure_pipeline_speed(args, device)

    
    save_results(all_results, "full_evaluation", args.output_dir)
    print(f"\n{'═'*60}")
    print(f"  Evaluation done!")
    print(f"  Results in: {args.output_dir}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
