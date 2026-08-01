import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_training(
    arch: str,
    seed: int,
    run_idx: int,
    output_dir: Path,
    epochs: int,
    batch: int,
    workers: int,
):
    
    run_output = output_dir / f"seed{seed}"
    checkpoint = run_output / "checkpoint_best.pt"
    if checkpoint.exists():
        print(f"  ✓ seed={seed} już ukończony — pomijam")
        return True
    run_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "training/scripts/train_attributes.py",
        "--arch",       arch,
        "--epochs",     str(epochs),
        "--batch",      str(batch),
        "--workers",    str(workers),
        "--seed",       str(seed),
        "--output-dir", str(run_output),
    ]

    print(f"\n  {'─'*55}")
    print(f"  Przebieg {run_idx}/5 | {arch} | seed={seed}")
    print(f"  {'─'*55}")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"   Błąd treningu seed={seed} dla {arch}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Repeated hold-out dla klasyfikacji atrybutów"
    )
    parser.add_argument(
        "--arch", nargs="+",
        default=["resnet50", "vit_small_patch16_224"],
        help="Architektury do trenowania"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1, 2, 3, 4, 5],
        help="Ziarna losowości dla kolejnych przebiegów"
    )
    parser.add_argument("--epochs-cnn", type=int, default=30)
    parser.add_argument("--epochs-vit", type=int, default=50)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/cross_validation/attributes"
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir

    print(f"\n{'═'*60}")
    print(f"  Repeated Hold-Out — Klasyfikacja atrybutów")
    print(f"{'═'*60}")
    print(f"  Architektury: {args.arch}")
    print(f"  Seed:         {args.seeds}")
    print(f"  Uwaga: zbiór testowy jest zawsze ten sam (oficjalny podział)")

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
        for run_idx, seed in enumerate(args.seeds, 1):
            success = run_training(
                arch=arch,
                seed=seed,
                run_idx=run_idx,
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
        print(f"  {arch}: {n_ok}/{len(args.seeds)} przebiegów pomyślnych")
        for seed, ok in zip(args.seeds, successes):
            status = "✓" if ok else "✗"
            print(f"    seed={seed}: {status}")

    print(f"\n  Wyniki w: {output_dir}/")
    print(f"  Następny krok: python scripts/wilcoxon_test.py")


if __name__ == "__main__":
    main()
