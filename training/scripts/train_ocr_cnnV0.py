import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from models.cnn.resnet import ResNetClassifier
from models.cnn.efficientnet import EfficientNetClassifier
from models.shared.ocr_head import OCRModel
from utils.dataset_ccpd import NUM_CHARS
from training.scripts.train_engine import Trainer, TrainerConfig



def get_train_transform():
   
    try:
        import albumentations as A
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.2), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ImageCompression(compression_type="jpeg", quality_range=(70, 100), p=0.2),
            A.Rotate(limit=5, p=0.3),           # małe obroty — nie więcej
            A.Perspective(scale=(0.02, 0.05), p=0.2),
        ])
    except ImportError:
        print("   Albumentations niedostępne — trenowanie bez augmentacji")
        return None


def build_model(arch: str, pretrained: bool) -> OCRModel:
   
    if arch.startswith("resnet"):
        backbone = ResNetClassifier(
            num_classes=1,          # zastępowane przez OCRHead
            variant=arch,
            pretrained=pretrained,
            dropout=0.3,
        )
    elif arch.startswith("efficientnet"):
        backbone = EfficientNetClassifier(
            num_classes=1,
            variant=arch,
            pretrained=pretrained,
            dropout=0.4,
        )
    else:
        raise ValueError(
            f"Nieznana architektura CNN: '{arch}'\n"
            f"Dostępne: resnet50, resnet101, efficientnet_b4, efficientnet_b7"
        )

    model = OCRModel(
        backbone=backbone,
        num_classes=NUM_CHARS,      
        hidden_dim=256,
        dropout=0.3,
    )

    params = model.count_parameters()
    print(f"  Model: {arch.upper()} + OCRHead")
    print(f"  Parametry: {params['total_M']} M total | {params['trainable_M']} M trenowalnych")

    return model



def build_optimizer_and_scheduler(model: OCRModel, args, steps_per_epoch: int):
    
   
    ocr_params = list(model.ocr_head.parameters())
    optimizer = torch.optim.AdamW(
        ocr_params,
        lr=args.lr_head,
        weight_decay=1e-4,
    )

  
    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
    )

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Trening OCR CNN na CCPD")
    parser.add_argument("--arch",        type=str,   default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b4", "efficientnet_b7"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr-head",     type=float, default=1e-3,
                        help="LR dla głowicy OCR (etap 1 — backbone zamrożony)")
    parser.add_argument("--lr-finetune", type=float, default=1e-4,
                        help="LR dla fine-tuningu całego modelu (etap 2)")
    parser.add_argument("--freeze-epochs", type=int, default=10,
                        help="Ile epok trenować tylko głowicę (etap 1)")
    parser.add_argument("--output-dir",  type=str,   default=None,
                        help="Katalog wyjściowy (domyślnie: outputs/ocr_cnn/{arch})")
    parser.add_argument("--max-samples", type=int,   default=None,
                        help="Limit próbek treningowych (eksperyment E3: np. 20000)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Trenuj od zera (bez wag ImageNet)")
    parser.add_argument("--workers",     type=int,   default=0,
                        help="Liczba workerów DataLoader (0 = bezpieczne na Windows)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir or f"outputs/ocr_cnn/{args.arch}"

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"  Trening OCR — CNN ({args.arch.upper()})")
    print(f"{'═' * 60}")
    print(f"  Urządzenie:  {device}")
    print(f"  Epoki:       {args.epochs} (zamrożony backbone: {args.freeze_epochs})")
    print(f"  Batch size:  {args.batch}")
    print(f"  Output:      {output_dir}")
    if args.max_samples:
        print(f"   Limit próbek: {args.max_samples} (eksperyment E3)")

   
    print("\n  Wczytywanie datasetu CCPD...")
    train_transform = get_train_transform()

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

    
    model.backbone.freeze_backbone()
    print(f"  Etap 1: backbone zamrożony ({args.freeze_epochs} epok)")

  
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args, steps_per_epoch=len(train_loader)
    )

    
    config_stage1 = TrainerConfig(
        output_dir=output_dir,
        epochs=args.freeze_epochs,
        grad_clip=None,           
        use_amp=True,
        save_best=True,
        early_stopping_patience=5,  
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_stage1,
        device=device,
    )
    trainer.fit()

    
    remaining_epochs = args.epochs - args.freeze_epochs
    if remaining_epochs > 0:
        print(f"\n  Etap 2: odmrażanie backbone ({remaining_epochs} epok, lr={args.lr_finetune})")
        model.backbone.unfreeze_backbone()

        
        optimizer_ft = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_finetune},
            {"params": model.ocr_head.parameters(), "lr": args.lr_finetune * 5},
        ], weight_decay=1e-4)

        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=remaining_epochs, eta_min=1e-6
        )

        config_stage2 = TrainerConfig(
            output_dir=output_dir,
            epochs=remaining_epochs,
            grad_clip=None,
            use_amp=True,
            save_best=True,
            early_stopping_patience=args.epochs // 5,
        )
        trainer_ft = Trainer(
            model=model,
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_stage2,
            device=device,
        )
        trainer_ft.fit()
        trainer_ft.test(test_loader)
    else:
        trainer.test(test_loader)

    print(f"\n  Gotowe! Wyniki w: {output_dir}/")


if __name__ == "__main__":
    main()
