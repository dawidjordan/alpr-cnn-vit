import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from utils.dataset_ccpd import (
    ProcessedCCPDOCRDataset,
    NUM_CHARS, IDX_TO_CHAR
)
from models.shared.ocr_head import OCRModel



def load_model(model_path: str, arch: str, device: torch.device) -> OCRModel:
   

    if arch.startswith("resnet") or arch.startswith("efficientnet"):
        if arch.startswith("resnet"):
            from models.cnn.resnet import ResNetClassifier
            backbone = ResNetClassifier(num_classes=1, variant=arch, pretrained=False)
        else:
            from models.cnn.efficientnet import EfficientNetClassifier
            backbone = EfficientNetClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("vit_"):
        from models.vit.vit_base import ViTClassifier
        backbone = ViTClassifier(num_classes=1, variant=arch, pretrained=False)
    elif arch.startswith("swin_"):
        from models.vit.swin import SwinClassifier
        backbone = SwinClassifier(num_classes=1, variant=arch, pretrained=False)
    else:
        raise ValueError(f"Nieznana architektura: {arch}")

    model = OCRModel(backbone=backbone, num_classes=NUM_CHARS)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    saved_epoch = checkpoint.get("epoch", "?")
    saved_acc   = checkpoint.get("metrics", {}).get("val_plate_acc", "?")
    print(f"  Wczytano checkpoint: epoka {saved_epoch}, val_plate_acc={saved_acc}")

    return model




def decode_prediction(logits_list: list) -> tuple[str, list[float]]:
    
    chars = []
    confs = []
    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        chars.append(IDX_TO_CHAR.get(idx.item(), "?"))
        confs.append(conf.item())
    return "".join(chars), confs


def decode_target(chars_tensor: torch.Tensor) -> str:
    """Konwertuje tensor indeksów na tekst tablicy."""
    return "".join(IDX_TO_CHAR.get(i.item(), "?") for i in chars_tensor)




def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Odwraca normalizację ImageNet → obraz uint8 do wyświetlenia.
    tensor: (3, H, W) float32
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img




@torch.no_grad()
def visualize(
    model: OCRModel,
    dataset: ProcessedCCPDOCRDataset,
    device: torch.device,
    n: int = 20,
    only_errors: bool = False,
    output_path: str = "outputs/visualizations/predictions.png",
    cols: int = 4,
):
    
    model.eval()

    
    samples = []
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    print(f"  Szukam {'tylko błędnych' if only_errors else ''} przewidywań...")

    for idx in indices:
        if len(samples) >= n:
            break
        image_tensor, chars_tensor = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        
        logits_list = model(image_tensor)
        pred_text, confs = decode_prediction([l[0] for l in logits_list])
        true_text = decode_target(chars_tensor)
        print(f"idx={idx}  true={true_text}  pred={pred_text}")

        is_correct = (pred_text == true_text)

        if only_errors and is_correct:
            continue

        samples.append({
            "image":      image_tensor[0],
            "pred_text":  pred_text,
            "true_text":  true_text,
            "confs":      confs,
            "is_correct": is_correct,
            "min_conf":   min(confs),
        })

    if not samples:
        print("  Brak próbek do wyświetlenia (wszystkie poprawne?)")
        return

    print(f"  Zebrano {len(samples)} próbek")

    import matplotlib.font_manager as fm
    font_path = "assets/NotoSansCJKsc-Regular.otf"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = ["DejaVu Sans", prop.get_name()]
    
    
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.8))
    fig.patch.set_facecolor("white")

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, ax in enumerate(axes_flat):
        ax.set_facecolor("white")
        if i >= len(samples):
            ax.axis("off")
            continue

        s = samples[i]
        img = denormalize(s["image"])

        ax.imshow(img, aspect="auto")
        ax.axis("off")

        
        color = "#00ff88" if s["is_correct"] else "#ff4444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        
        pred_label = f"Pred: {s['pred_text']}"
        ax.set_title(pred_label, fontsize=13, color=color,
                     fontweight="bold", pad=3)

     
        conf_pct = s["min_conf"] * 100
        ax.text(
            0.5, -0.22,
            f"GT: {s['true_text']}  |  conf: {conf_pct:.0f}%",
            fontsize=11, color="#555555",
            ha="center", va="top",
            transform=ax.transAxes,
        )

        
        _draw_char_confidence_bar(ax, s["confs"], s["pred_text"], s["true_text"])

   
    correct_patch = mpatches.Patch(color="#00ff88", label="Poprawna tablica")
    error_patch   = mpatches.Patch(color="#ff4444", label="Błędna tablica")
    fig.legend(
        handles=[correct_patch, error_patch],
        loc="lower center", ncol=2,
        facecolor="white", edgecolor="#cccccc",
        labelcolor="black", fontsize=12,
        bbox_to_anchor=(0.5, 0.01),
    )

   
    n_correct = sum(1 for s in samples if s["is_correct"])
    plate_acc = n_correct / len(samples) * 100
    fig.suptitle(
        f"Przewidywania modelu OCR  |  plate_acc: {plate_acc:.1f}%  "
        f"({n_correct}/{len(samples)} poprawnych)",
        fontsize=15, color="black", fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.6)

    # Zapis
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Zapisano → {out_path}")


def _draw_char_confidence_bar(ax, confs: list, pred: str, true: str):
    
    n = len(confs)
    for j, (conf, pc, tc) in enumerate(zip(confs, pred, true)):
        is_char_correct = (pc == tc)
        base_color = np.array([0, 1, 0.5]) if is_char_correct else np.array([1, 0.2, 0.2])
        color = tuple(base_color * conf)
        rect = plt.Rectangle(
            (j / n, -0.18), 1 / n - 0.005, 0.15,
            transform=ax.transAxes,
            clip_on=False,
            color=color,
        )
        ax.add_patch(rect)



def main():
    parser = argparse.ArgumentParser(description="Wizualizacja przewidywań OCR")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ścieżka do checkpoint_best.pt")
    parser.add_argument("--arch", type=str, required=True,
                        help="Architektura: resnet50 | vit_small_patch16_224 | ...")
    parser.add_argument("--n", type=int, default=20,
                        help="Liczba próbek do wyświetlenia")
    parser.add_argument("--only-errors", action="store_true",
                        help="Pokaż tylko błędne przewidywania")
    parser.add_argument("--output", type=str, default=None,
                        help="Ścieżka wyjściowa PNG (domyślnie: outputs/visualizations/{arch}.png)")
    parser.add_argument("--cols", type=int, default=4,
                        help="Liczba kolumn w siatce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Który zbiór wizualizować")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output or f"outputs/visualizations/{args.arch}_{args.split}.png"

    print(f"\n{'═' * 55}")
    print(f"  Wizualizacja przewidywań OCR")
    print(f"{'═' * 55}")
    print(f"  Model:    {args.arch}")
    print(f"  Checkpoint: {args.model_path}")
    print(f"  Próbek:   {args.n}")

    
    print("\n  Wczytywanie modelu...")
    model = load_model(args.model_path, args.arch, device)

    
    print("  Wczytywanie datasetu...")
    if args.split == "train":
        viz_dataset = ProcessedCCPDOCRDataset(root="data/processed/ccpd/train")
    else:
        viz_dataset = ProcessedCCPDOCRDataset(root="data/processed/ccpd/val")

    print(f"  Wizualizuję zbiór {args.split} ({len(viz_dataset):,} próbek)")

    
    print(f"\n  Generowanie wizualizacji ({args.n} próbek)...")
    visualize(
        model=model,
        dataset=viz_dataset,
        device=device,
        n=args.n,
        only_errors=args.only_errors,
        output_path=output_path,
        cols=args.cols,
    )

    print(f"\n  Gotowe! Otwórz plik: {output_path}")


if __name__ == "__main__":
    main()
