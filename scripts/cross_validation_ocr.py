import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_split_lines(split_file: Path) -> list[str]:
    
    if not split_file.exists():
        raise FileNotFoundError(f"Brak pliku: {split_file}")
    lines = split_file.read_text().strip().splitlines()
    print(f"  Wczytano {len(lines):,} linii z {split_file.name}")
    return lines


def create_fold_files(
    all_lines: list[str],
    n_folds: int,
    output_dir: Path,
    seed: int = 42,
) -> list[tuple[Path, Path]]:
   
    rng = random.Random(seed)
    shuffled = all_lines.copy()
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled)
        folds.append(shuffled[start:end])

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_files = []

    for i in range(n_folds):
        val_lines   = folds[i]
        train_lines = []
        for j in range(n_folds):
            if j != i:
                train_lines.extend(folds[j])

        train_file = output_dir / f"fold{i+1}_train.txt"
        val_file   = output_dir / f"fold{i+1}_val.txt"

        train_file.write_text("\n".join(train_lines))
        val_file.write_text("\n".join(val_lines))

        print(f"  Fold {i+1}: train={len(train_lines):,} | val={len(val_lines):,}")
        fold_files.append((train_file, val_file))

    return fold_files


def run_training(
    arch: str,
    fold: int,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    epochs: int,
    batch: int,
    workers: int,
):
    
    fold_output = output_dir / f"fold{fold}"
    
 
    checkpoint = fold_output / "checkpoint_best.pt"
    if checkpoint.exists():
        print(f"  ✓ Fold {fold} już ukończony — pomijam")
        return True
    
    fold_output.mkdir(parents=True, exist_ok=True)

    if arch.startswith("vit_") or arch.startswith("swin_"):
        script = "training/scripts/train_ocr_vit.py"
    else:
        script = "training/scripts/train_ocr_cnn.py"

    cmd = [
        sys.executable, script,
        "--split-train", str(train_file),
        "--split-val",   str(val_file),
        "--epochs",      str(epochs),
        "--batch",       str(batch),
        "--workers",     str(workers),
        "--output-dir",  str(fold_output),
    ]

    print(f"\n  {'─'*55}")
    print(f"  Fold {fold} | {arch}")
    print(f"  Komenda: {' '.join(cmd)}")
    print(f"  {'─'*55}")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  Błąd treningu fold {fold} dla {arch}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="5-krotna walidacja krzyżowa OCR"
    )
    parser.add_argument(
        "--arch", nargs="+",
        default=["resnet50", "vit_small_patch16_224"],
        help="Architektury do trenowania"
    )
    parser.add_argument("--n-folds",   type=int, default=5)
    parser.add_argument("--epochs-cnn", type=int, default=30)
    parser.add_argument("--epochs-vit", type=int, default=50)
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--ccpd-splits", type=str,
        default="data/raw/ccpd/splits",
        help="Katalog z plikami train.txt i val.txt"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/cross_validation/ocr"
    )
    args = parser.parse_args()

    splits_dir  = ROOT / args.ccpd_splits
    output_dir  = ROOT / args.output_dir
    folds_dir   = output_dir / "folds"

    print(f"\n{'═'*60}")
    print(f"  5-krotna walidacja krzyżowa OCR")
    print(f"{'═'*60}")
    print(f"  Architektury: {args.arch}")
    print(f"  Foldów:       {args.n_folds}")
    print(f"  Seed:         {args.seed}")

    
    print(f"\n  Wczytywanie podziałów CCPD...")

    rng = random.Random(args.seed)

    all_train = load_split_lines(splits_dir / "train.txt")
    all_val   = load_split_lines(splits_dir / "val.txt")

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    train_lines = all_train[:50000]
    val_lines   = all_val[:50000]
    all_lines   = train_lines + val_lines
    print(f"  Używamy 50 000 z train + 50 000 z val = 100 000 łącznie")

   
    print(f"\n  Tworzenie {args.n_folds} foldów...")
    fold_files = create_fold_files(
        all_lines, args.n_folds, folds_dir, seed=args.seed
    )

    
    results = {}
    for arch in args.arch:
        arch_dir = output_dir / arch
        epochs = args.epochs_vit if (arch.startswith("vit_") or
                                      arch.startswith("swin_")) \
                 else args.epochs_cnn

        print(f"\n{'═'*60}")
        print(f"  Architektura: {arch.upper()} | Epoki: {epochs}")
        print(f"{'═'*60}")

        results[arch] = []
        for fold_idx, (train_file, val_file) in enumerate(fold_files, 1):
            success = run_training(
                arch=arch,
                fold=fold_idx,
                train_file=train_file,
                val_file=val_file,
                output_dir=arch_dir,
                epochs=epochs,
                batch=args.batch,
                workers=args.workers,
            )
            results[arch].append(success)

   
    print(f"\n{'═'*60}")
    print(f"  Podsumowanie")
    print(f"{'═'*60}")
    for arch, successes in results.items():
        n_ok = sum(successes)
        print(f"  {arch}: {n_ok}/{args.n_folds} foldów zakończonych pomyślnie")
        for i, ok in enumerate(successes, 1):
            status = "✓" if ok else "✗"
            print(f"    Fold {i}: {status}")

    print(f"\n  Wyniki w: {output_dir}/")
    print(f"  Następny krok: python scripts/wilcoxon_test.py")


if __name__ == "__main__":
    main()
