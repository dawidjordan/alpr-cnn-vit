import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm



def ocr_accuracy(logits_list: list, targets: torch.Tensor) -> dict:
    
    batch_size = targets.shape[0]
    correct_chars = 0
    correct_plates = 0
    total_chars = batch_size * 7

    predictions = []
    for pos, logits in enumerate(logits_list):
        pred = logits.argmax(dim=1)      
        predictions.append(pred)
        correct_chars += (pred == targets[:, pos]).sum().item()

  
    pred_tensor = torch.stack(predictions, dim=1)  
    correct_plates = (pred_tensor == targets).all(dim=1).sum().item()

    return {
        "char_acc":  correct_chars / total_chars,
        "plate_acc": correct_plates / batch_size,
    }




class EarlyStopping:

    def __init__(
        self,
        patience: int = 8,
        mode: str = "max",       
        min_delta: float = 0.001, 
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int) -> bool:
        
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def status(self) -> str:
        return (
            f"EarlyStopping: best={self.best_value:.4f} "
            f"(epoch {self.best_epoch}) | patience {self.counter}/{self.patience}"
        )



class MetricsTracker:
   

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self._header_written = False

    def log(self, metrics: dict) -> None:
        
        self.rows.append(metrics)
        mode = "a" if self._header_written else "w"
        with open(self.output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)

    def best_epoch(self, metric: str = "val_plate_acc", mode: str = "max") -> dict:
       
        if not self.rows:
            return {}
        key = metric
        return max(self.rows, key=lambda r: r.get(key, 0)) if mode == "max" \
               else min(self.rows, key=lambda r: r.get(key, float("inf")))



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: Optional[float] = None,
) -> dict:
    
    model.train()
    total_loss = 0.0
    total_char_correct = 0
    total_plate_correct = 0
    total_chars = 0
    total_plates = 0

    progress = tqdm(loader, desc="  Train", leave=False, unit="batch")

    for images, targets in progress:
        images  = images.to(device, non_blocking=True)  
        targets = targets.to(device, non_blocking=True)  

        optimizer.zero_grad()

       
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits_list = model(images)   

          
            loss = sum(
                criterion(logits, targets[:, pos])
                for pos, logits in enumerate(logits_list)
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        
        batch_metrics = ocr_accuracy(logits_list, targets)
        b = images.size(0)
        total_loss         += loss.item()
        total_char_correct += batch_metrics["char_acc"] * b * 7
        total_plate_correct+= batch_metrics["plate_acc"] * b
        total_chars        += b * 7
        total_plates       += b

        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            char_acc=f"{batch_metrics['char_acc']:.3f}",
        )

    return {
        "train_loss":      total_loss / len(loader),
        "train_char_acc":  total_char_correct / total_chars,
        "train_plate_acc": total_plate_correct / total_plates,
    }



@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prefix: str = "val",
) -> dict:
   
    model.eval()
    total_loss = 0.0
    total_char_correct = 0
    total_plate_correct = 0
    total_chars = 0
    total_plates = 0

    progress = tqdm(loader, desc=f"  {prefix.capitalize()}", leave=False, unit="batch")

    for images, targets in progress:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits_list = model(images)
            loss = sum(
                criterion(logits, targets[:, pos])
                for pos, logits in enumerate(logits_list)
            )

        batch_metrics = ocr_accuracy(logits_list, targets)
        b = images.size(0)
        total_loss          += loss.item()
        total_char_correct  += batch_metrics["char_acc"] * b * 7
        total_plate_correct += batch_metrics["plate_acc"] * b
        total_chars         += b * 7
        total_plates        += b

    return {
        f"{prefix}_loss":      total_loss / len(loader),
        f"{prefix}_char_acc":  total_char_correct / total_chars,
        f"{prefix}_plate_acc": total_plate_correct / total_plates,
    }



@dataclass
class TrainerConfig:
    
    output_dir:      str   = "outputs/ocr_cnn"
    epochs:          int   = 30
    grad_clip:       Optional[float] = None
    use_amp:         bool  = True      # mixed precision (tylko GPU)
    save_best:       bool  = True
    early_stopping_patience: int = 8
    log_every_n_epochs:      int = 1


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler    = (
            torch.amp.GradScaler('cuda')
            if config.use_amp and device.type == "cuda"
            else None
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="max",
        )
        self.metrics = MetricsTracker(
            output_path=str(Path(config.output_dir) / "metrics.csv")
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_plate_acc = 0.0

    def fit(self) -> None:
       
        print(f"\n{'═' * 60}")
        print(f"  Start trenowania → {self.output_dir}")
        print(f"  Urządzenie: {self.device} | Epoki: {self.config.epochs}")
        print(f"  Mixed precision: {'TAK' if self.scaler else 'NIE'}")
        print(f"{'═' * 60}\n")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()

           
            train_metrics = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                scaler=self.scaler,
                grad_clip=self.config.grad_clip,
            )

          
            val_metrics = evaluate(
                model=self.model,
                loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                prefix="val",
            )

            epoch_time = time.perf_counter() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            
            row = {
                "epoch":           epoch,
                "lr":              round(lr, 8),
                "epoch_time_s":    round(epoch_time, 1),
                **{k: round(v, 6) for k, v in train_metrics.items()},
                **{k: round(v, 6) for k, v in val_metrics.items()},
            }
            self.metrics.log(row)

           
            print(
                f"  Epoka {epoch:>3}/{self.config.epochs} | "
                f"loss {train_metrics['train_loss']:.4f} → {val_metrics['val_loss']:.4f} | "
                f"char_acc {train_metrics['train_char_acc']:.4f} → {val_metrics['val_char_acc']:.4f} | "
                f"plate_acc {train_metrics['train_plate_acc']:.4f} → {val_metrics['val_plate_acc']:.4f} | "
                f"lr={lr:.2e} | {epoch_time:.0f}s"
            )

        
            if self.config.save_best:
                if val_metrics["val_plate_acc"] > self.best_val_plate_acc:
                    self.best_val_plate_acc = val_metrics["val_plate_acc"]
                    self._save_checkpoint(epoch, val_metrics, tag="best")
                    print(f"  ✓ Nowy najlepszy model zapisany (plate_acc={self.best_val_plate_acc:.4f})")

           
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_plate_acc"])
                else:
                    self.scheduler.step()

           
            if self.early_stopping(val_metrics["val_plate_acc"], epoch):
                print(f"\n  Early stopping po epoce {epoch}.")
                print(f"  {self.early_stopping.status}")
                break

        print(f"\n  Trening zakończony. Metryki zapisane → {self.metrics.output_path}")
        best = self.metrics.best_epoch("val_plate_acc", "max")
        print(
            f"  Najlepsza epoka: {best.get('epoch')} | "
            f"val_plate_acc={best.get('val_plate_acc'):.4f} | "
            f"val_char_acc={best.get('val_char_acc'):.4f}"
        )

    def test(self, test_loader: DataLoader) -> dict:
        
        best_path = self.output_dir / "checkpoint_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            print(f"\n  Wczytano najlepszy model z epoki {checkpoint.get('epoch', '?')}")

        test_metrics = evaluate(
            model=self.model,
            loader=test_loader,
            criterion=self.criterion,
            device=self.device,
            prefix="test",
        )

        print(f"\n{'═' * 60}")
        print("  WYNIKI NA ZBIORZE TESTOWYM:")
        for k, v in test_metrics.items():
            print(f"    {k:<25} {v:.4f}")
        print(f"{'═' * 60}")

       
        results_path = self.output_dir / "test_results.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
            writer.writeheader()
            writer.writerow({k: round(v, 6) for k, v in test_metrics.items()})
        print(f"  Wyniki zapisane → {results_path}")

        return test_metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = "best") -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "metrics":    metrics,
        }, path)
