import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.dataset_ccpd import (
    ProcessedCCPDOCRDataset, parse_ccpd_filename,
    NUM_CHARS, IDX_TO_CHAR,
)




class FullImageCCPDDataset(Dataset):
    

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        raw_dir: str,
        split_file: str,
        img_h: int = 64,
        img_w: int = 128,
        max_samples: int = None,
        seed: int = 42,
    ):
        self.raw_dir  = Path(raw_dir)
        self.img_h    = img_h
        self.img_w    = img_w

        
        lines = Path(split_file).read_text().strip().splitlines()

        import random
        rng = random.Random(seed)
        rng.shuffle(lines)
        if max_samples:
            lines = lines[:max_samples]

        
        self.samples = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            jpg_path = self.raw_dir / line
            if not jpg_path.exists():
                continue
            try:
                ann = parse_ccpd_filename(jpg_path.name)
                self.samples.append((jpg_path, ann["plate_chars"]))
            except Exception:
                continue

        print(f"  FullImageCCPDDataset: {len(self.samples):,} próbek")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_path, plate_chars = self.samples[idx]

        
        img = cv2.imread(str(jpg_path))
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_w, self.img_h))

        
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        chars_tensor = torch.tensor(plate_chars, dtype=torch.long)
        return img_tensor, chars_tensor, str(jpg_path)




def load_ocr_model(checkpoint_path: str, arch: str, device: torch.device):
    from models.shared.ocr_head import OCRModel

    if arch.startswith("resnet"):
        from models.cnn.resnet import ResNetClassifier
        backbone = ResNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("vit_"):
        from models.vit.vit_base import ViTClassifier
        backbone = ViTClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("swin_"):
        from models.vit.swin import SwinClassifier
        backbone = SwinClassifier(num_classes=1, variant=arch, pretrained=False)
    else:
        raise ValueError(f"Nieznana architektura: {arch}")

    model = OCRModel(backbone=backbone, num_classes=NUM_CHARS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model




@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    total_chars = correct_chars = 0
    total_plates = correct_plates = 0

    errors = []  

    for batch in tqdm(loader, desc="  Eval", leave=False):
        if len(batch) == 3:
            images, targets, paths = batch
        else:
            images, targets = batch
            paths = [""] * images.size(0)

        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits_list = model(images)
        B = images.size(0)

        preds = torch.stack([l.argmax(dim=1) for l in logits_list], dim=1)
        correct_mask = (preds == targets).all(dim=1)

        correct_chars  += (preds == targets).sum().item()
        correct_plates += correct_mask.sum().item()
        total_chars    += B * 7
        total_plates   += B

      
        for i in range(B):
            if not correct_mask[i] and len(errors) < 20:
                pred_text = "".join(IDX_TO_CHAR.get(p.item(), "?")
                                    for p in preds[i])
                true_text = "".join(IDX_TO_CHAR.get(t.item(), "?")
                                    for t in targets[i])
                errors.append((paths[i], pred_text, true_text))

    return {
        "char_acc":  correct_chars  / max(total_chars,  1),
        "plate_acc": correct_plates / max(total_plates, 1),
        "errors":    errors,
    }


def visualize_comparison(
    errors_crop: list,
    errors_nocrop: list,
    arch: str,
    output_path: str,
    n: int = 8,
    cols: int = 4,
):
 
   
    samples = errors_nocrop[:n]
    if not samples:
        print("  Brak błędów do wizualizacji")
        return

    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3.5, rows * 5))
    fig.patch.set_facecolor("white")

    if rows == 1:
        axes = [list(axes[0:cols]), list(axes[cols:2*cols])] if rows * 2 > 1 else [axes]

    fig.suptitle(
        f"Eksperyment: pełne zdjęcie vs wycięta tablica — {arch.upper()}\n"
        f"Górny rząd: pełne zdjęcie (nocrop) | Dolny rząd: wycięta tablica (crop)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    for i, (jpg_path, pred_nocrop, true_text) in enumerate(samples):
        col = i % cols
        row_full = (i // cols) * 2
        row_crop = row_full + 1

        if not jpg_path or not Path(jpg_path).exists():
            continue

        
        img_full = cv2.imread(jpg_path)
        if img_full is None:
            continue
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

      
        try:
            ann = parse_ccpd_filename(Path(jpg_path).name)
            x1, y1, x2, y2 = ann["bbox"]
            h, w = img_full.shape[:2]
            x1 = max(0, x1 - 4); y1 = max(0, y1 - 4)
            x2 = min(w, x2 + 4); y2 = min(h, y2 + 4)
            img_crop = img_full[y1:y2, x1:x2]

           
            img_full_marked = img_full.copy()
            cv2.rectangle(img_full_marked, (x1, y1), (x2, y2), (255, 100, 0), 3)
        except Exception:
            img_crop = img_full
            img_full_marked = img_full

       
        ax_full = axes[row_full][col] if rows > 1 else axes[0][col]
        ax_full.imshow(img_full_marked)
        ax_full.axis("off")
        ax_full.set_title(
            f"Pred: {pred_nocrop}",
            fontsize=9, color="#DC2626", fontweight="bold", pad=2,
        )
        ax_full.text(0.5, -0.08, f"GT: {true_text}",
                     fontsize=8, color="#555555", ha="center",
                     transform=ax_full.transAxes)

        
        ax_crop = axes[row_crop][col] if rows > 1 else axes[1][col]
        ax_crop.imshow(img_crop, aspect="auto")
        ax_crop.axis("off")
        ax_crop.set_title(
            f"GT: {true_text}",
            fontsize=9, color="#16A34A", fontweight="bold", pad=2,
        )
        ax_crop.text(0.5, -0.08, "wycięta tablica",
                     fontsize=8, color="#555555", ha="center",
                     transform=ax_crop.transAxes)

   
    for i in range(len(samples), rows * cols):
        col = i % cols
        for r in range(rows * 2):
            try:
                axes[r][col].axis("off")
            except Exception:
                pass

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wizualizacja → {output_path}")


def plot_summary(results: dict, output_path: str):
   

    models   = list(results.keys())
    crop_plate   = [results[m]["crop"]["plate_acc"]   * 100 for m in models]
    nocrop_plate = [results[m]["nocrop"]["plate_acc"] * 100 for m in models]
    crop_char    = [results[m]["crop"]["char_acc"]    * 100 for m in models]
    nocrop_char  = [results[m]["nocrop"]["char_acc"]  * 100 for m in models]

    x = np.arange(len(models))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Eksperyment: pełne zdjęcie vs wycięta tablica", fontweight="bold")

    for ax, crop_vals, nocrop_vals, ylabel, title in [
        (axes[0], crop_plate, nocrop_plate, "Plate Accuracy (%)", "Plate Accuracy"),
        (axes[1], crop_char,  nocrop_char,  "Char Accuracy (%)",  "Char Accuracy"),
    ]:
        bars1 = ax.bar(x - w/2, crop_vals,   w, label="Crop (wycięta tablica)",
                       color=["#1D4ED8", "#EA580C"][:len(models)], alpha=0.9)
        bars2 = ax.bar(x + w/2, nocrop_vals, w, label="No-crop (pełne zdjęcie)",
                       color=["#93C5FD", "#FDBA74"][:len(models)], alpha=0.9,
                       hatch="//")

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wykres podsumowania → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Eksperyment: crop vs no-crop OCR"
    )
    parser.add_argument("--arch", type=str, nargs="+",
                        default=["resnet50", "vit_small_patch16_224"],
                        help="Architektury do porównania")
    parser.add_argument("--n-samples",  type=int, default=500,
                        help="Liczba próbek do ewaluacji (domyślnie 500)")
    parser.add_argument("--batch",      type=int, default=64)
    parser.add_argument("--workers",    type=int, default=0)
    parser.add_argument("--ccpd-raw",   type=str,
                        default="data/raw/ccpd",
                        help="Katalog główny surowych danych CCPD")
    parser.add_argument("--ccpd-proc",  type=str,
                        default="data/processed/ccpd/val",
                        help="Katalog przetworzonych danych CCPD (crop)")
    parser.add_argument("--split-file", type=str,
                        default="data/raw/ccpd/splits/val.txt",
                        help="Plik podziału val CCPD")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/experiment_nocrop")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═' * 60}")
    print(f"  Eksperyment: pełne zdjęcie vs wycięta tablica (OCR)")
    print(f"{'═' * 60}")
    print(f"  Urządzenie:  {device}")
    print(f"  Próbki:      {args.n_samples}")
    print(f"  Architektury: {args.arch}")

    
    print("\n  Wczytywanie datasetu CROP (NPZ)...")
    crop_dataset = ProcessedCCPDOCRDataset(
        root=args.ccpd_proc,
        max_samples=args.n_samples,
    )
    crop_loader = DataLoader(
        crop_dataset, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    
    print("\n  Wczytywanie datasetu NO-CROP (pełne JPG)...")
    nocrop_dataset = FullImageCCPDDataset(
        raw_dir=args.ccpd_raw,
        split_file=args.split_file,
        max_samples=args.n_samples,
        seed=args.seed,
    )
    nocrop_loader = DataLoader(
        nocrop_dataset, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    all_results = {}

    for arch in args.arch:
        print(f"\n{'─' * 60}")
        print(f"  Architektura: {arch.upper()}")

        
        if arch.startswith("vit_") or arch.startswith("swin_"):
            ckpt_path = Path(f"outputs/ocr_vit/E1/{arch}/checkpoint_best.pt")
        else:
            ckpt_path = Path(f"outputs/ocr_cnn/E1/{arch}/checkpoint_best.pt")

        if not ckpt_path.exists():
            print(f"   Brak checkpointu: {ckpt_path}")
            continue

        print(f"  Checkpoint: {ckpt_path}")
        model = load_ocr_model(str(ckpt_path), arch, device)

       
        print("\n  [1/2] Ewaluacja CROP (wycięta tablica)...")
        metrics_crop = evaluate(model, crop_loader, device)

      
        print("  [2/2] Ewaluacja NO-CROP (pełne zdjęcie)...")
        metrics_nocrop = evaluate(model, nocrop_loader, device)

        all_results[arch] = {
            "crop":   metrics_crop,
            "nocrop": metrics_nocrop,
        }

       
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  {arch.upper():<43}│")
        print(f"  ├───────────────────┬──────────┬──────────────┤")
        print(f"  │ Podejście         │ plate_acc│ char_acc     │")
        print(f"  ├───────────────────┼──────────┼──────────────┤")
        print(f"  │ Crop (wycięcie)   │ "
              f"{metrics_crop['plate_acc']*100:7.2f}% │ "
              f"{metrics_crop['char_acc']*100:11.2f}% │")
        print(f"  │ No-crop (pełne)   │ "
              f"{metrics_nocrop['plate_acc']*100:7.2f}% │ "
              f"{metrics_nocrop['char_acc']*100:11.2f}% │")
        delta_plate = (metrics_crop['plate_acc'] - metrics_nocrop['plate_acc']) * 100
        delta_char  = (metrics_crop['char_acc']  - metrics_nocrop['char_acc'])  * 100
        print(f"  ├───────────────────┼──────────┼──────────────┤")
        print(f"  │ Różnica (↓ nocrop)│ "
              f"{delta_plate:+7.2f}% │ "
              f"{delta_char:+11.2f}% │")
        print(f"  └───────────────────┴──────────┴──────────────┘")

      
        viz_path = str(Path(args.output_dir) / f"errors_nocrop_{arch}.png")
        visualize_comparison(
            metrics_crop["errors"],
            metrics_nocrop["errors"],
            arch,
            viz_path,
        )

   
    if len(all_results) > 0:
        summary_path = str(Path(args.output_dir) / "summary_crop_vs_nocrop.png")
        plot_summary(all_results, summary_path)

    print(f"\n{'═' * 60}")
    print(f"  Eksperyment zakończony!")
    print(f"  Wyniki w: {args.output_dir}/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
