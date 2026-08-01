import argparse
import csv
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_attributes import VehicleAttributeDataset, make_attribute_dataloaders


def get_train_transform():
    try:
        import albumentations as A
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
            A.ImageCompression(compression_type="jpeg", quality_range=(70, 100), p=0.2),
        ])
    except ImportError:
        return None



def build_model(arch: str, pretrained: bool, num_colors: int, num_types: int, num_makes: int):
    from models.shared.attribute_head import VehicleAttributeModel

    if arch.startswith("resnet"):
        from models.cnn.resnet import ResNetClassifier
        backbone = ResNetClassifier(num_classes=1, variant=arch, pretrained=pretrained)
    elif arch.startswith("efficientnet"):
        from models.cnn.efficientnet import EfficientNetClassifier
        backbone = EfficientNetClassifier(num_classes=1, variant=arch, pretrained=pretrained)
    elif arch.startswith("vit_"):
        from models.vit.vit_base import ViTClassifier
        backbone = ViTClassifier(num_classes=1, variant=arch, pretrained=pretrained)
    elif arch.startswith("swin_"):
        from models.vit.swin import SwinClassifier
        backbone = SwinClassifier(num_classes=1, variant=arch, pretrained=pretrained)
    else:
        raise ValueError(f"Nieznana architektura: {arch}")

    model = VehicleAttributeModel(
        backbone=backbone,
        num_colors=num_colors,
        num_types=num_types,
        num_makes=num_makes,
    )

    params = model.count_parameters()
    print(f"  Model: {arch.upper()} + AttributeHeads")
    print(f"  Parametry: {params['total_M']} M total | {params['trainable_M']} M trenowalnych")

    return model



def train_one_epoch(model, loader, optimizer, device, scaler=None, grad_clip=None):
    model.train()

    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    total_loss = 0.0
    color_correct = color_total = 0
    type_correct  = type_total  = 0
    make_correct  = make_total  = 0

    progress = tqdm(loader, desc="  Train", leave=False, unit="batch")

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        color_gt = labels["color"].to(device)
        type_gt  = labels["type"].to(device)
        make_gt  = labels["make"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            color_logits, type_logits, make_logits = model(images)

            loss_color = criterion(color_logits, color_gt)
            loss_type  = criterion(type_logits,  type_gt)
            loss_make  = criterion(make_logits,  make_gt)
            losses = [l for l in [loss_color, loss_type, loss_make]
                    if not torch.isnan(l)]
            loss = sum(losses) if losses else torch.tensor(0.0, device=device)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

        
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

        progress.set_postfix(loss=f"{loss.item():.3f}")

    return {
        "train_loss":       total_loss / len(loader),
        "train_color_acc":  color_correct / max(color_total, 1),
        "train_type_acc":   type_correct  / max(type_total,  1),
        "train_make_acc":   make_correct  / max(make_total,  1),
    }


@torch.no_grad()
def evaluate(model, loader, device, prefix="val"):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    color_correct = color_total = 0
    type_correct  = type_total  = 0
    make_correct  = make_total  = 0

    for images, labels in tqdm(loader, desc=f"  {prefix.capitalize()}", leave=False):
        images   = images.to(device, non_blocking=True)
        color_gt = labels["color"].to(device)
        type_gt  = labels["type"].to(device)
        make_gt  = labels["make"].to(device)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            color_logits, type_logits, make_logits = model(images)
            loss_color = criterion(color_logits, color_gt)
            loss_type  = criterion(type_logits,  type_gt)
            loss_make  = criterion(make_logits,  make_gt)

           
            losses = [l for l in [loss_color, loss_type, loss_make]
                    if not torch.isnan(l)]
            loss = sum(losses) if losses else torch.tensor(0.0, device=device)

        total_loss += loss.item()

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
    mean_acc  = (color_acc + type_acc + make_acc) / 3

    return {
        f"{prefix}_loss":      total_loss / len(loader),
        f"{prefix}_color_acc": color_acc,
        f"{prefix}_type_acc":  type_acc,
        f"{prefix}_make_acc":  make_acc,
        f"{prefix}_mean_acc":  mean_acc,
    }



def main():
    parser = argparse.ArgumentParser(description="Trening klasyfikatora atrybutów pojazdu")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b4",
                                 "vit_small_patch16_224", "vit_base_patch16_224",
                                 "swin_tiny_patch4_window7_224"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--grad-clip",   type=float, default=None)
    parser.add_argument("--warmup-epochs",type=int,  default=0)
    parser.add_argument("--workers",     type=int,   default=0)
    parser.add_argument("--max-compcars", type=int, default=None,
                    help="Limit próbek CompCars (E3), np. 4772 = 30%")
    parser.add_argument("--max-color",    type=int, default=None,
                    help="Limit próbek VehicleColor (E3), np. 2180 = 30%")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir",  type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--compcars-root", type=str, default="data/raw/compcars")
    parser.add_argument("--color-root",    type=str, default="data/raw/vehicle_color")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

   
    is_vit = args.arch.startswith("vit_") or args.arch.startswith("swin_")
    if is_vit and args.grad_clip is None:
        args.grad_clip = 1.0
    if is_vit and args.warmup_epochs == 0:
        args.warmup_epochs = 5

    output_dir = args.output_dir or f"outputs/attributes/{args.arch}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═' * 60}")
    print(f"  Trening klasyfikatora atrybutów — {args.arch.upper()}")
    print(f"{'═' * 60}")
    print(f"  Urządzenie:  {device}")
    print(f"  Epoki:       {args.epochs}")
    print(f"  Batch size:  {args.batch}")
    print(f"  LR:          {args.lr}")

    # Dataset
    print("\n  Wczytywanie datasetów...")
    train_transform = get_train_transform()

    train_loader, val_loader = make_attribute_dataloaders(
        compcars_root=args.compcars_root,
        color_root=args.color_root,
        batch_size=args.batch,
        num_workers=args.workers,
        max_compcars=args.max_compcars,
        max_color=args.max_color,
        transform_train=train_transform,
        seed=args.seed,
    )

    num_makes  = train_loader.dataset.num_makes
    num_colors = train_loader.dataset.num_colors
    num_types  = train_loader.dataset.num_types

    print(f"\n  Budowanie modelu...")
    model = build_model(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        num_colors=num_colors,
        num_types=num_types,
        num_makes=num_makes,
    ).to(device)

   
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    if args.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 1e-4 + (1.0 - 1e-4) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

   
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.csv"
    best_mean_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 10

    print(f"\n{'═' * 60}")
    print(f"  Start trenowania → {output_dir}")
    print(f"{'═' * 60}\n")

    header_written = False

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_m = train_one_epoch(model, train_loader, optimizer, device,
                                  scaler, args.grad_clip)
        val_m   = evaluate(model, val_loader, device, "val")

        scheduler.step()
        epoch_time = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "lr": round(lr, 8),
            "epoch_time_s": round(epoch_time, 1),
            **{k: round(v, 6) for k, v in train_m.items()},
            **{k: round(v, 6) for k, v in val_m.items()},
        }

    
        mode = "a" if header_written else "w"
        with open(metrics_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

        print(
            f"  Epoka {epoch:>3}/{args.epochs} | "
            f"loss {train_m['train_loss']:.3f}→{val_m['val_loss']:.3f} | "
            f"color {val_m['val_color_acc']:.3f} | "
            f"type {val_m['val_type_acc']:.3f} | "
            f"make {val_m['val_make_acc']:.3f} | "
            f"mean {val_m['val_mean_acc']:.3f} | "
            f"{epoch_time:.0f}s"
        )

  
        if val_m["val_mean_acc"] > best_mean_acc:
            best_mean_acc = val_m["val_mean_acc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": val_m,
                "num_makes": num_makes,
                "num_colors": num_colors,
                "num_types": num_types,
            }, output_path / "checkpoint_best.pt")
            print(f"  ✓ Nowy najlepszy model (mean_acc={best_mean_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n  Early stopping po epoce {epoch}.")
                break

    print(f"\n  Trening zakończony. Najlepsza mean_acc: {best_mean_acc:.4f}")
    print(f"  Wyniki w: {output_dir}/")


if __name__ == "__main__":
    main()
