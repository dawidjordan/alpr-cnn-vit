import argparse
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from models.vit.vit_base import ViTClassifier
from models.vit.swin import SwinClassifier
from models.shared.ocr_head import OCRModel
from utils.dataset_ccpd import NUM_CHARS
from training.scripts.train_engine import Trainer, TrainerConfig



class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min_ratio: float = 1e-4,   # eta_min = base_lr * eta_min_ratio
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min_ratio = eta_min_ratio

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                
                return (epoch + 1) / warmup_epochs
            
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

        super().__init__(optimizer, lr_lambda)




def get_train_transform():

    try:
        import albumentations as A
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.GaussNoise(std_range=(0.02, 0.25), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ImageCompression(compression_type="jpeg", quality_range=(60, 100), p=0.3),
            A.Rotate(limit=8, p=0.4),
            A.Perspective(scale=(0.02, 0.07), p=0.3),
            A.CoarseDropout(                          # Random Erasing — symuluje okluzję
                num_holes_range=(1, 4),
                hole_height_range=(4, 8),
                hole_width_range=(8, 16),
                fill=0, p=0.3,
            ),
        ])
    except ImportError:
        print("  Albumentations niedostępne — trenowanie bez augmentacji")
        return None



def build_model(arch: str, pretrained: bool) -> OCRModel:
    
    if arch.startswith("vit_"):
        backbone = ViTClassifier(
            num_classes=1,        
            variant=arch,
            pretrained=pretrained,
            dropout=0.1,
        )
    elif arch.startswith("swin_"):
        backbone = SwinClassifier(
            num_classes=1,
            variant=arch,
            pretrained=pretrained,
            dropout=0.1,
        )
    else:
        raise ValueError(
            f"Nieznana architektura ViT: '{arch}'\n"
            f"Dostępne: vit_small_patch16_224, vit_base_patch16_224, "
            f"swin_tiny_patch4_window7_224, swin_small_patch4_window7_224"
        )

    model = OCRModel(
        backbone=backbone,
        num_classes=NUM_CHARS,
        hidden_dim=256,
        dropout=0.1,              
    )

    params = model.count_parameters()
    print(f"  Model: {arch.upper()} + OCRHead")
    print(f"  Parametry: {params['total_M']} M total | {params['trainable_M']} M trenowalnych")

    return model



def main():
    parser = argparse.ArgumentParser(description="Trening OCR ViT na CCPD")
    parser.add_argument("--arch", type=str, default="vit_small_patch16_224",
                        choices=[
                            "vit_small_patch16_224",
                            "vit_base_patch16_224",
                            "swin_tiny_patch4_window7_224",
                            "swin_small_patch4_window7_224",
                            "swin_base_patch4_window7_224",
                        ])
    parser.add_argument("--epochs",       type=int,   default=50,
                        help="ViT wymaga więcej epok niż CNN")
    parser.add_argument("--batch",        type=int,   default=32,
                        help="ViT wymaga więcej VRAM — mniejszy batch niż CNN")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="LR bazowy — 10x mniejszy niż CNN głowica!")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Wyższy weight decay dla Transformerów")
    parser.add_argument("--warmup-epochs",type=int,   default=5,
                        help="Epoki warmup — krytyczne dla stabilności ViT")
    parser.add_argument("--grad-clip",    type=float, default=1.0,
                        help="Gradient clipping — zapobiega eksplodującym gradientom")
    parser.add_argument("--output-dir",   type=str,   default=None)
    parser.add_argument("--max-samples",  type=int,   default=None,
                        help="Limit próbek (eksperyment E3)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--workers",      type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir or f"outputs/ocr_vit/{args.arch}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═' * 60}")
    print(f"  Trening OCR — ViT ({args.arch.upper()})")
    print(f"{'═' * 60}")
    print(f"  Urządzenie:    {device}")
    print(f"  Epoki:         {args.epochs} (warmup: {args.warmup_epochs})")
    print(f"  Batch size:    {args.batch}")
    print(f"  LR:            {args.lr} (weight_decay={args.weight_decay})")
    print(f"  Grad clip:     {args.grad_clip}")
    print(f"  Output:        {output_dir}")
    if args.max_samples:
        print(f"   Limit próbek: {args.max_samples} (eksperyment E3)")

    # Dataset
    print("\n  Wczytywanie datasetu CCPD...")
    from utils.dataset_ccpd import ProcessedCCPDOCRDataset
    from torch.utils.data import DataLoader

    train_dataset = ProcessedCCPDOCRDataset(
    root="data/processed/ccpd/train",
    max_samples=args.max_samples,
    )
    val_dataset = ProcessedCCPDOCRDataset(
        root="data/processed/ccpd/val",
        max_samples=None,
    )

    train_transform = get_train_transform()
    if train_transform is not None:
        train_dataset.transform = train_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                            shuffle=True,  num_workers=args.workers,
                            pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=torch.cuda.is_available())
    test_loader  = val_loader

    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    
    print("\n  Budowanie modelu...")
    model = build_model(args.arch, pretrained=not args.no_pretrained)

    
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.ocr_head.parameters(), "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)

  
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    config = TrainerConfig(
        output_dir=output_dir,
        epochs=args.epochs,
        grad_clip=args.grad_clip,          
        use_amp=True,
        save_best=True,
        early_stopping_patience=12,         
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    trainer.fit()
    trainer.test(test_loader)

    print(f"\n  Gotowe! Wyniki w: {output_dir}/")


if __name__ == "__main__":
    main()
