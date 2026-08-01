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

from utils.dataset_attributes import VehicleAttributeDataset, get_valid_makes
from utils.dataset_vehicles import (
    COLOR_CLASSES, TYPE_IDX_TO_NAME, NUM_VEHICLE_TYPES, NUM_COLORS
)
from models.shared.attribute_head import VehicleAttributeModel



def build_make_names(compcars_root: str, make_to_idx: dict) -> dict:
    
    idx_to_name = {}
    try:
        import scipy.io
        mat = scipy.io.loadmat(
            str(Path(compcars_root) / "misc" / "make_model_name.mat")
        )
        make_names = mat["make_names"]
        for make_id, make_idx in make_to_idx.items():
            try:
                name = str(make_names[int(make_id) - 1][0][0])
                idx_to_name[make_idx] = name
            except Exception:
                idx_to_name[make_idx] = f"make_{make_id}"
    except Exception:
        
        for make_id, make_idx in make_to_idx.items():
            idx_to_name[make_idx] = f"make_{make_id}"
    return idx_to_name



def load_model(model_path: str, arch: str, device: torch.device,
               num_colors: int, num_types: int, num_makes: int) -> VehicleAttributeModel:

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
    else:
        raise ValueError(f"Nieznana architektura: {arch}")

    model = VehicleAttributeModel(
        backbone=backbone,
        num_colors=num_colors,
        num_types=num_types,
        num_makes=num_makes,
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    print(f"  Wczytano checkpoint: epoka {epoch}")
    print(f"  val_mean_acc={metrics.get('val_mean_acc', '?'):.4f} | "
          f"color={metrics.get('val_color_acc', '?'):.3f} | "
          f"type={metrics.get('val_type_acc', '?'):.3f} | "
          f"make={metrics.get('val_make_acc', '?'):.3f}")

    return model



def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)



@torch.no_grad()
def visualize(
    model: VehicleAttributeModel,
    dataset: VehicleAttributeDataset,
    device: torch.device,
    make_names: dict,
    n: int = 20,
    only_errors: bool = False,
    output_path: str = "outputs/visualizations/attributes.png",
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

        image_tensor, labels = dataset[idx]

       
        has_color = labels["color"].item() >= 0
        has_type  = labels["type"].item()  >= 0
        has_make  = labels["make"].item()  >= 0
        if not any([has_color, has_type, has_make]):
            continue

        img_input = image_tensor.unsqueeze(0).to(device)
        color_logits, type_logits, make_logits = model(img_input)

      
        pred_color = color_logits[0].argmax().item()
        pred_type  = type_logits[0].argmax().item()
        pred_make  = make_logits[0].argmax().item()

    
        conf_color = torch.softmax(color_logits[0], dim=-1).max().item()
        conf_type  = torch.softmax(type_logits[0],  dim=-1).max().item()
        conf_make  = torch.softmax(make_logits[0],  dim=-1).max().item()

        
        gt_color  = COLOR_CLASSES[labels["color"].item()] if has_color else "N/A"
        gt_type   = TYPE_IDX_TO_NAME.get(labels["type"].item(), "N/A") if has_type else "N/A"
        gt_make   = make_names.get(labels["make"].item(), "N/A") if has_make else "N/A"

        pr_color  = COLOR_CLASSES[pred_color]
        pr_type   = TYPE_IDX_TO_NAME.get(pred_type, "?")
        pr_make   = make_names.get(pred_make, "?")

     
        color_ok = (pred_color == labels["color"].item()) if has_color else None
        type_ok  = (pred_type  == labels["type"].item())  if has_type  else None
        make_ok  = (pred_make  == labels["make"].item())  if has_make  else None

        all_ok = all(r for r in [color_ok, type_ok, make_ok] if r is not None)

        if only_errors and all_ok:
            continue

        samples.append({
            "image":    image_tensor,
            "gt_color": gt_color,  "pr_color": pr_color,
            "gt_type":  gt_type,   "pr_type":  pr_type,
            "gt_make":  gt_make,   "pr_make":  pr_make,
            "color_ok": color_ok,  "type_ok":  type_ok,  "make_ok": make_ok,
            "conf_color": conf_color, "conf_type": conf_type, "conf_make": conf_make,
            "all_ok": all_ok,
            "has_color": has_color, "has_type": has_type, "has_make": has_make,
        })

    if not samples:
        print("  Brak próbek do wyświetlenia")
        return

    print(f"  Zebrano {len(samples)} próbek")

    
    import matplotlib.font_manager as fm
    font_path = "assets/NotoSansCJKsc-Regular.otf"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = ["DejaVu Sans", prop.get_name()]

    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 5.5))
    fig.patch.set_facecolor("white")

    if rows == 1:
        axes = [list(axes)] if cols > 1 else [[axes]]
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

        
        border_color = "#308f62" if s["all_ok"] else "#ff4444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

      
        def attr_color(ok):
            if ok is None:   return "#888888"
            return "#308f62" if ok else "#ff4444"

        lines_pred = []
        lines_gt   = []

        if s["has_color"]:
            lines_pred.append(f"Color: {s['pr_color']} ({s['conf_color']*100:.0f}%)")
            lines_gt.append(  f"Color: {s['gt_color']}")
        if s["has_type"]:
            lines_pred.append(f"Type:  {s['pr_type']} ({s['conf_type']*100:.0f}%)")
            lines_gt.append(  f"Type:  {s['gt_type']}")
        if s["has_make"]:
            lines_pred.append(f"Make:  {s['pr_make']} ({s['conf_make']*100:.0f}%)")
            lines_gt.append(  f"Make:  {s['gt_make']}")

        pred_text = "\n".join(lines_pred)
        gt_text   = "\n".join(lines_gt)

        ax.set_title(pred_text, fontsize=13, color=border_color,
                     fontweight="bold", pad=3, loc="left")
        ax.text(0.0, -0.05, "GT:\n" + gt_text, fontsize=13.5, color="#555555",
                ha="left", va="top", transform=ax.transAxes)

    
    correct_patch = mpatches.Patch(color="#308f62", label="Wszystkie atrybuty poprawne")
    error_patch   = mpatches.Patch(color="#ff4444", label="Co najmniej jeden błąd")
    fig.legend(handles=[correct_patch, error_patch], loc="lower center", ncol=2,
               facecolor="white", edgecolor="#cccccc", labelcolor="black",
               fontsize=13, bbox_to_anchor=(0.5, 0.01))

    n_correct = sum(1 for s in samples if s["all_ok"])
    mean_acc  = n_correct / len(samples) * 100
    fig.suptitle(
        f"Klasyfikacja atrybutów pojazdu  |  {mean_acc:.1f}% w pełni poprawnych "
        f"({n_correct}/{len(samples)})",
        fontsize=16, color="black", fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Zapisano → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Wizualizacja klasyfikacji atrybutów pojazdu")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--arch",       type=str, required=True)
    parser.add_argument("--n",          type=int, default=20)
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--cols",       type=int, default=4)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--compcars-root", type=str, default="data/raw/compcars")
    parser.add_argument("--color-root",    type=str, default="data/raw/vehicle_color")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output or f"outputs/visualizations/attributes_{args.arch}_{args.split}.png"

    print(f"\n{'═' * 55}")
    print(f"  Wizualizacja klasyfikacji atrybutów pojazdu")
    print(f"{'═' * 55}")
    print(f"  Model:      {args.arch}")
    print(f"  Checkpoint: {args.model_path}")
    print(f"  Zbiór:      {args.split}")
    print(f"  Próbek:     {args.n}")

    
    make_to_idx = get_valid_makes(args.compcars_root)
    make_names  = build_make_names(args.compcars_root, make_to_idx)
    num_makes   = len(make_to_idx)

    print(f"\n  Wczytywanie modelu...")
    model = load_model(
        model_path=args.model_path,
        arch=args.arch,
        device=device,
        num_colors=NUM_COLORS,
        num_types=NUM_VEHICLE_TYPES,
        num_makes=num_makes,
    )

    print(f"  Wczytywanie datasetu ({args.split})...")
    dataset = VehicleAttributeDataset(
        compcars_root=args.compcars_root,
        color_root=args.color_root,
        split=args.split,
    )
    print(f"  Załadowano {len(dataset):,} próbek")

  
    compcars_samples = [s for s in dataset.samples if s["type"] >= 0 or s["make"] >= 0]
    from torch.utils.data import Subset
    compcars_indices = [i for i, s in enumerate(dataset.samples)
                        if s["type"] >= 0 or s["make"] >= 0]
    color_indices    = [i for i, s in enumerate(dataset.samples)
                        if s["color"] >= 0]

    compcars_subset = Subset(dataset, compcars_indices)
    color_subset    = Subset(dataset, color_indices)

    out_make_type = output_path.replace(".png", "_type_make.png")
    out_color     = output_path.replace(".png", "_color.png")

    print(f"\n  Generowanie wizualizacji typ+marka ({args.n} próbek z CompCars)...")
    visualize(
        model=model,
        dataset=compcars_subset,
        device=device,
        make_names=make_names,
        n=args.n,
        only_errors=args.only_errors,
        output_path=out_make_type,
        cols=args.cols,
    )

    print(f"\n  Generowanie wizualizacji kolor ({args.n} próbek z VehicleColor)...")
    visualize(
        model=model,
        dataset=color_subset,
        device=device,
        make_names=make_names,
        n=args.n,
        only_errors=args.only_errors,
        output_path=out_color,
        cols=args.cols,
    )

    print(f"\n  Gotowe!")
    print(f"  Typ+Marka: {out_make_type}")
    print(f"  Kolor:     {out_color}")

if __name__ == "__main__":
    main()
