import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import wilcoxon


def load_best_metric_from_csv(csv_path: Path, metric: str) -> float | None:
    
    if not csv_path.exists():
        return None
    try:
        import csv
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows or metric not in rows[0]:
            return None
        values = [float(r[metric]) for r in rows if r[metric]]
        return max(values) if values else None
    except Exception:
        return None


def load_best_metric_from_checkpoint(ckpt_path: Path, metric: str) -> float | None:
   
    if not ckpt_path.exists():
        return None
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        return metrics.get(metric, None)
    except Exception:
        return None


def collect_ocr_results(cv_dir: Path, arch: str, metric: str) -> list[float]:
   
    arch_dir = cv_dir / arch
    results = []

    for fold in range(1, 6):
        fold_dir = arch_dir / f"fold{fold}"

    
        val = load_best_metric_from_csv(
            fold_dir / "metrics.csv", f"val_{metric}"
        )
        
        if val is None:
            val = load_best_metric_from_checkpoint(
                fold_dir / "checkpoint_best.pt", f"val_{metric}"
            )

        if val is not None:
            results.append(val)
            print(f"    Fold {fold}: {metric}={val*100:.2f}%")
        else:
            print(f"    Fold {fold}: ⚠ Brak danych ({fold_dir})")

    return results


def collect_attr_results(
    holdout_dir: Path, arch: str, metric: str,
    seeds: list[int] = None,
) -> list[float]:
  
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    arch_dir = holdout_dir / arch
    results  = []

    for seed in seeds:
        seed_dir = arch_dir / f"seed{seed}"

        val = load_best_metric_from_csv(
            seed_dir / "metrics.csv", f"val_{metric}"
        )
        if val is None:
            val = load_best_metric_from_checkpoint(
                seed_dir / "checkpoint_best.pt", f"val_{metric}"
            )

        if val is not None:
            results.append(val)
            print(f"    seed={seed}: {metric}={val*100:.2f}%")
        else:
            print(f"    seed={seed}:  Brak danych ({seed_dir})")

    return results



def run_wilcoxon_test(
    values_cnn: list[float],
    values_vit: list[float],
    metric_name: str,
    alpha: float = 0.05,
) -> dict:
    
    n = min(len(values_cnn), len(values_vit))
    if n < 2:
        print(f"    ⚠ Za mało danych (CNN={len(values_cnn)}, ViT={len(values_vit)})")
        return {}

    cnn = np.array(values_cnn[:n])
    vit = np.array(values_vit[:n])

    mean_cnn = cnn.mean() * 100
    std_cnn  = cnn.std()  * 100
    mean_vit = vit.mean() * 100
    std_vit  = vit.std()  * 100

   
    try:
        stat, p_value = wilcoxon(cnn, vit, alternative="two-sided")
    except ValueError as e:
        
        print(f"     Test niemożliwy: {e}")
        stat, p_value = 0.0, 1.0

    is_significant = p_value < alpha
    winner = "ViT-Small" if mean_vit > mean_cnn else \
             "ResNet-50" if mean_cnn > mean_vit else "Remis"

    print(f"\n  {'─'*55}")
    print(f"  Metryka: {metric_name}")
    print(f"  {'─'*55}")
    print(f"  ResNet-50 (CNN): {mean_cnn:.2f}% ± {std_cnn:.2f}%")
    print(f"  ViT-Small (ViT): {mean_vit:.2f}% ± {std_vit:.2f}%")
    print(f"  Statystyka W:    {stat:.4f}")
    print(f"  p-wartość:       {p_value:.4f}")
    print(f"  Poziom alfa:     {alpha}")
    print(f"  Istotne stat.:   {'TAK ✓' if is_significant else 'NIE ✗'}")
    print(f"  Lepszy model:    {winner}")

    if is_significant:
        print(f"  → Różnica między CNN a ViT jest statystycznie istotna (p={p_value:.4f} < α={alpha})")
    else:
        print(f"  → Brak statystycznie istotnej różnicy (p={p_value:.4f} ≥ α={alpha})")

    return {
        "metric":         metric_name,
        "n_samples":      n,
        "mean_cnn":       round(mean_cnn, 4),
        "std_cnn":        round(std_cnn, 4),
        "mean_vit":       round(mean_vit, 4),
        "std_vit":        round(std_vit, 4),
        "W_statistic":    round(float(stat), 4),
        "p_value":        round(float(p_value), 4),
        "alpha":          alpha,
        "significant":    bool(is_significant),
        "winner":         winner,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Wilcoxona CNN vs ViT"
    )
    parser.add_argument("--alpha",       type=float, default=0.05)
    parser.add_argument("--seeds",  nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--cv-dir",
                        default="outputs/cross_validation/ocr")
    parser.add_argument("--holdout-dir",
                        default="outputs/cross_validation/attributes")
    parser.add_argument("--output",
                        default="outputs/cross_validation/wilcoxon_results.json")
    args = parser.parse_args()

    cv_dir      = ROOT / args.cv_dir
    holdout_dir = ROOT / args.holdout_dir
    output_path = ROOT / args.output

    archs_cnn = ["resnet50"]
    archs_vit = ["vit_small_patch16_224"]

    all_results = {}

    
    print(f"\n{'═'*60}")
    print(f"  TEST WILCOXONA — OCR (5-krotna walidacja krzyżowa)")
    print(f"{'═'*60}")

    ocr_metrics = ["plate_acc", "char_acc"]

    for metric in ocr_metrics:
        print(f"\n  Wczytywanie wyników OCR — {metric}")
        print(f"  ResNet-50:")
        cnn_vals = collect_ocr_results(cv_dir, "resnet50", metric)
        print(f"  ViT-Small:")
        vit_vals = collect_ocr_results(cv_dir, "vit_small_patch16_224", metric)

        if cnn_vals and vit_vals:
            result = run_wilcoxon_test(
                cnn_vals, vit_vals,
                f"OCR/{metric}",
                alpha=args.alpha,
            )
            all_results[f"ocr_{metric}"] = result

    
    print(f"\n{'═'*60}")
    print(f"  TEST WILCOXONA — Atrybuty (repeated hold-out)")
    print(f"{'═'*60}")

    attr_metrics = ["mean_acc", "color_acc", "type_acc", "make_acc"]

    for metric in attr_metrics:
        print(f"\n  Wczytywanie wyników atrybutów — {metric}")
        print(f"  ResNet-50:")
        cnn_vals = collect_attr_results(
            holdout_dir, "resnet50", metric, args.seeds
        )
        print(f"  ViT-Small:")
        vit_vals = collect_attr_results(
            holdout_dir, "vit_small_patch16_224", metric, args.seeds
        )

        if cnn_vals and vit_vals:
            result = run_wilcoxon_test(
                cnn_vals, vit_vals,
                f"Atrybuty/{metric}",
                alpha=args.alpha,
            )
            all_results[f"attr_{metric}"] = result

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    
    print(f"\n{'═'*60}")
    print(f"  PODSUMOWANIE TESTÓW WILCOXONA (α={args.alpha})")
    print(f"{'═'*60}")
    print(f"  {'Metryka':<30} {'CNN':>10} {'ViT':>10} {'p-wartość':>12} {'Istotna?':>10} {'Lepszy':>12}")
    print(f"  {'─'*84}")

    for key, r in all_results.items():
        if not r:
            continue
        sig = "TAK ✓" if r["significant"] else "NIE ✗"
        print(
            f"  {r['metric']:<30} "
            f"{r['mean_cnn']:>9.2f}% "
            f"{r['mean_vit']:>9.2f}% "
            f"{r['p_value']:>12.4f} "
            f"{sig:>10} "
            f"{r['winner']:>12}"
        )

    print(f"\n  Wyniki zapisane → {output_path}")


if __name__ == "__main__":
    main()
