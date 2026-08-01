import argparse
import json
import sys
from pathlib import Path
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS  = {"cnn": "#2196F3", "vit": "#FF5722"}
MARKERS = {"cnn": "o",       "vit": "s"}

def is_vit(arch):
    return arch.startswith("vit_") or arch.startswith("swin_")

def arch_label(arch):
    return {
        "resnet50":              "ResNet-50 (CNN)",
        "resnet101":             "ResNet-101 (CNN)",
        "vit_small_patch16_224": "ViT-Small (ViT)",
        "vit_base_patch16_224":  "ViT-Base (ViT)",
    }.get(arch, arch)

def arch_color(arch):
    return COLORS["vit"] if is_vit(arch) else COLORS["cnn"]

def arch_marker(arch):
    return MARKERS["vit"] if is_vit(arch) else MARKERS["cnn"]

def plot_e1_baseline(results: dict, output_dir: Path):
    
    ocr_data  = {k: v for k, v in results.items() if k.startswith("ocr_")}
    attr_data = {k: v for k, v in results.items() if k.startswith("attr_")}

    if ocr_data:
        fig, ax = plt.subplots(figsize=(7, 5))
        archs  = [k.replace("ocr_", "") for k in ocr_data]
        chars  = [ocr_data[f"ocr_{a}"]["char_acc"]  * 100 for a in archs]
        plates = [ocr_data[f"ocr_{a}"]["plate_acc"] * 100 for a in archs]
        colors = [arch_color(a) for a in archs]
        labels = [arch_label(a) for a in archs]

        x = np.arange(len(archs))
        w = 0.35
        bars1 = ax.bar(x - w/2, chars,  w, label="Character accuracy",
                       color=colors, alpha=0.85)
        bars2 = ax.bar(x + w/2, plates, w, label="Plates accuracy",
                       color=colors, alpha=0.5, hatch="//")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("E1 — OCR: license plate recognition")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 108)
       
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, framealpha=0.9)

        plt.tight_layout()
        out = output_dir / "E1_baseline_ocr.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")

    if attr_data:
        fig, ax = plt.subplots(figsize=(7, 5))
        archs      = [k.replace("attr_", "") for k in attr_data]
        colors_acc = [attr_data[f"attr_{a}"]["color_acc"] * 100 for a in archs]
        types_acc  = [attr_data[f"attr_{a}"]["type_acc"]  * 100 for a in archs]
        makes_acc  = [attr_data[f"attr_{a}"]["make_acc"]  * 100 for a in archs]
        colors_bar = [arch_color(a) for a in archs]
        labels     = [arch_label(a) for a in archs]

        x = np.arange(len(archs))
        w = 0.25
        ax.bar(x - w,   colors_acc, w, label="Kolor",  color=colors_bar, alpha=0.9)
        ax.bar(x,       types_acc,  w, label="Typ",    color=colors_bar, alpha=0.6, hatch="//")
        ax.bar(x + w,   makes_acc,  w, label="Marka",  color=colors_bar, alpha=0.3, hatch="xx")

        for bars, vals in [
            (ax.containers[0], colors_acc),
            (ax.containers[1], types_acc),
            (ax.containers[2], makes_acc),
        ]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("E1 — Klasyfikacja atrybutów pojazdu")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, framealpha=0.9)

        plt.tight_layout()
        out = output_dir / "E1_baseline_attr.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if ocr_data and attr_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("E1 — Wyniki bazowe CNN vs ViT", fontweight="bold")

        ax = axes[0]
        archs  = [k.replace("ocr_", "") for k in ocr_data]
        chars  = [ocr_data[f"ocr_{a}"]["char_acc"]  * 100 for a in archs]
        plates = [ocr_data[f"ocr_{a}"]["plate_acc"] * 100 for a in archs]
        colors = [arch_color(a) for a in archs]
        labels = [arch_label(a) for a in archs]
        x = np.arange(len(archs)); w = 0.35
        bars1 = ax.bar(x - w/2, chars,  w, label="Dokładność znaków",
                       color=colors, alpha=0.85)
        bars2 = ax.bar(x + w/2, plates, w, label="Dokładność tablic",
                       color=colors, alpha=0.5, hatch="//")
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Dokładność (%)")
        ax.set_title("OCR — Rozpoznawanie tablic rejestracyjnych")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, framealpha=0.9)

        ax = axes[1]
        archs      = [k.replace("attr_", "") for k in attr_data]
        colors_acc = [attr_data[f"attr_{a}"]["color_acc"] * 100 for a in archs]
        types_acc  = [attr_data[f"attr_{a}"]["type_acc"]  * 100 for a in archs]
        makes_acc  = [attr_data[f"attr_{a}"]["make_acc"]  * 100 for a in archs]
        colors_bar = [arch_color(a) for a in archs]
        labels     = [arch_label(a) for a in archs]
        x = np.arange(len(archs)); w = 0.25
        ax.bar(x - w, colors_acc, w, label="Kolor",  color=colors_bar, alpha=0.9)
        ax.bar(x,     types_acc,  w, label="Typ",    color=colors_bar, alpha=0.6, hatch="//")
        ax.bar(x + w, makes_acc,  w, label="Marka",  color=colors_bar, alpha=0.3, hatch="xx")
        ax.set_ylabel("Dokładność (%)")
        ax.set_title("Klasyfikacja atrybutów pojazdu")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, framealpha=0.9)

        plt.tight_layout()
        out = output_dir / "E1_baseline.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

def plot_e2_robustness(results: dict, output_dir: Path):
   
    SUBSET_LABELS = {
        "ccpd_blur":      "ccpd_blur",
        "ccpd_db":        "ccpd_db",
        "ccpd_weather":   "ccpd_weather",
        "ccpd_tilt":      "ccpd_tilt",
        "ccpd_fn":        "ccpd_fn",
        "ccpd_rotate":    "ccpd_rotate",
        "ccpd_challenge": "ccpd_challenge",
    }

    DEGRADATION_LABELS = {
        "baseline":            "Baseline",
        "gaussian_blur_s3":    "Rozmycie",
        "gaussian_noise_s3":   "Szum",
        "jpeg_compression_s3": "JPEG",
        "brightness_s3":       "Jasność",
        "low_contrast_s3":     "Kontrast",
        "occlusion_s3":        "Okluzja",
    }

    ocr_data  = {k: v for k, v in results.items() if k.startswith("ocr_")}
    attr_data = {k: v for k, v in results.items() if k.startswith("attr_")}

    def _plot_ocr_metric(metric, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, subset_results in ocr_data.items():
            arch = key.replace("ocr_", "")
            subsets = [s for s in SUBSET_LABELS if s in subset_results]
            if not subsets:
                continue
            values = [subset_results[s][metric] * 100 for s in subsets]
            labels = [SUBSET_LABELS[s] for s in subsets]
            ax.plot(labels, values, color=arch_color(arch),
                    marker=arch_marker(arch), label=arch_label(arch),
                    linewidth=2, markersize=7)
        ax.set_xlabel("Podzbior CCPD")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        out = output_dir / filename
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if ocr_data:
        _plot_ocr_metric(
            "plate_acc",
            "Dokładność tablic (%)",
            "E2 — Odporność OCR: dokładność tablic",
            "E2_ocr_plate_acc.png",
        )
       
    if attr_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, deg_results in attr_data.items():
            arch  = key.replace("attr_", "")
            degs  = [d for d in DEGRADATION_LABELS if d in deg_results]
            if not degs:
                continue
            values = [deg_results[d]["mean_acc"] * 100 for d in degs]
            labels = [DEGRADATION_LABELS[d] for d in degs]
            ax.plot(labels, values, color=arch_color(arch),
                    marker=arch_marker(arch), label=arch_label(arch),
                    linewidth=2, markersize=7)
        ax.set_xlabel("Typ degradacji")
        ax.set_ylabel("Średnia dokładność (%)")
        ax.set_title("E2 — Odporność klasyfikacji atrybutów na degradację obrazu")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        out = output_dir / "E2_attributes.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

def plot_e3_limited_data(results: dict, output_dir: Path):
    
    FRACTION_ORDER  = ["10pct", "20pct", "30pct", "50pct", "80pct", "100pct"]
    FRACTION_LABELS = {
        "10pct": "10%", "20pct": "20%", "30pct": "30%",
        "50pct": "50%", "80pct": "80%", "100pct": "100%",
    }

    ocr_data  = {k: v for k, v in results.items() if k.startswith("ocr_")}
    attr_data = {k: v for k, v in results.items() if k.startswith("attr_")}

    def _plot_single(data, metric, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, fraction_results in data.items():
            arch = key.replace("ocr_", "").replace("attr_", "")
            fractions = [f for f in FRACTION_ORDER if f in fraction_results]
            if not fractions:
                continue
            values = [fraction_results[f][metric] * 100 for f in fractions]
            labels = [FRACTION_LABELS[f] for f in fractions]
            ax.plot(labels, values, color=arch_color(arch),
                    marker=arch_marker(arch), label=arch_label(arch),
                    linewidth=2, markersize=8)
        ax.set_xlabel("Rozmiar zbioru treningowego")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        out = output_dir / filename
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if ocr_data:
        _plot_single(
            ocr_data, "plate_acc",
            "Dokładność tablic (%)",
            "E3 — OCR: generalizacja przy ograniczonych danych",
            "E3_ocr.png",
        )

    if attr_data:
        _plot_single(
            attr_data, "mean_acc",
            "Średnia dokładność (%)",
            "E3 — Atrybuty: generalizacja przy ograniczonych danych",
            "E3_attributes.png",
        )

    n_plots = (1 if ocr_data else 0) + (1 if attr_data else 0)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("E3 — Wpływ rozmiaru zbioru treningowego", fontweight="bold")

    idx = 0
    if ocr_data:
        ax = axes[idx]
        for key, fraction_results in ocr_data.items():
            arch = key.replace("ocr_", "")
            fractions = [f for f in FRACTION_ORDER if f in fraction_results]
            if not fractions:
                continue
            values = [fraction_results[f]["plate_acc"] * 100 for f in fractions]
            labels = [FRACTION_LABELS[f] for f in fractions]
            ax.plot(labels, values, color=arch_color(arch),
                    marker=arch_marker(arch), label=arch_label(arch),
                    linewidth=2, markersize=8)
        ax.set_xlabel("Rozmiar zbioru treningowego")
        ax.set_ylabel("Dokładność tablic (%)")
        ax.set_title("OCR — Generalizacja przy ograniczonych danych")
        ax.set_ylim(0, 100)
        ax.legend()
        idx += 1

    if attr_data:
        ax = axes[idx]
        for key, fraction_results in attr_data.items():
            arch = key.replace("attr_", "")
            fractions = [f for f in FRACTION_ORDER if f in fraction_results]
            if not fractions:
                continue
            values = [fraction_results[f]["mean_acc"] * 100 for f in fractions]
            labels = [FRACTION_LABELS[f] for f in fractions]
            ax.plot(labels, values, color=arch_color(arch),
                    marker=arch_marker(arch), label=arch_label(arch),
                    linewidth=2, markersize=8)
        ax.set_xlabel("Rozmiar zbioru treningowego")
        ax.set_ylabel("Średnia dokładność (%)")
        ax.set_title("Atrybuty — Generalizacja przy ograniczonych danych")
        ax.set_ylim(0, 100)
        ax.legend()

    plt.tight_layout()
    out = output_dir / "E3_limited_data.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano → {out}")

def plot_e4_learning_curves(results: dict, output_dir: Path):
    
    ocr_data  = {k: v for k, v in results.items() if k.startswith("ocr_")}
    attr_data = {k: v for k, v in results.items() if k.startswith("attr_")}

    for key, epochs_data in ocr_data.items():
        arch  = key.replace("ocr_", "")
        color = arch_color(arch)
        if not epochs_data:
            continue

        epochs     = [r["epoch"]          for r in epochs_data]
        train_loss = [r["train_loss"]      for r in epochs_data]
        val_loss   = [r.get("val_loss", 0) for r in epochs_data]
        train_acc  = [r.get("train_plate_acc", r.get("train_char_acc", 0)) * 100
                      for r in epochs_data]
        val_acc    = [r.get("val_plate_acc",   r.get("val_char_acc",   0)) * 100
                      for r in epochs_data]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"E4 — Krzywe uczenia OCR: {arch_label(arch)}", fontweight="bold")

        axes[0].plot(epochs, train_loss, color=color, label="Strata treningowa", linewidth=2)
        axes[0].plot(epochs, val_loss,   color=color, label="Strata walidacyjna",
                     linestyle="--", linewidth=2)
        axes[0].set_xlabel("Epoka")
        axes[0].set_ylabel("Strata")
        axes[0].set_title("Funkcja straty")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, color=color, label="Zbiór treningowy", linewidth=2)
        axes[1].plot(epochs, val_acc,   color=color, label="Zbiór walidacyjny",
                     linestyle="--", linewidth=2)
        axes[1].set_xlabel("Epoka")
        axes[1].set_ylabel("Dokładność tablic (%)")
        axes[1].set_title("Dokładność tablic rejestracyjnych")
        axes[1].set_ylim(0, 100)
        axes[1].legend()

        plt.tight_layout()
        safe = arch.replace("/", "_")
        out = output_dir / f"E4_ocr_{safe}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if ocr_data:
        n = len(ocr_data)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        fig.suptitle("E4 — Krzywe uczenia", fontweight="bold")
        if n == 1:
            axes = [axes]

        for row, (key, epochs_data) in enumerate(ocr_data.items()):
            arch  = key.replace("ocr_", "")
            color = arch_color(arch)
            if not epochs_data:
                continue
            epochs     = [r["epoch"]          for r in epochs_data]
            train_loss = [r["train_loss"]      for r in epochs_data]
            val_loss   = [r.get("val_loss", 0) for r in epochs_data]
            train_acc  = [r.get("train_plate_acc", r.get("train_char_acc", 0)) * 100
                          for r in epochs_data]
            val_acc    = [r.get("val_plate_acc",   r.get("val_char_acc",   0)) * 100
                          for r in epochs_data]

            axes[row][0].plot(epochs, train_loss, color=color,
                              label="Strata treningowa", linewidth=2)
            axes[row][0].plot(epochs, val_loss,   color=color,
                              label="Strata walidacyjna", linestyle="--", linewidth=2)
            axes[row][0].set_xlabel("Epoka")
            axes[row][0].set_ylabel("Strata")
            axes[row][0].set_title(f"{arch_label(arch)} — Funkcja straty")
            axes[row][0].legend()

            axes[row][1].plot(epochs, train_acc, color=color,
                              label="Zbiór treningowy", linewidth=2)
            axes[row][1].plot(epochs, val_acc,   color=color,
                              label="Zbiór walidacyjny", linestyle="--", linewidth=2)
            axes[row][1].set_xlabel("Epoka")
            axes[row][1].set_ylabel("Dokładność przewidywania tablic (%)")
            axes[row][1].set_title(f"{arch_label(arch)} — Dokładność tablic")
            axes[row][1].set_ylim(0, 100)
            axes[row][1].legend()

        plt.tight_layout()
        out = output_dir / "E4_learning_curves.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    for key, epochs_data in attr_data.items():
        arch  = key.replace("attr_", "")
        color = arch_color(arch)
        if not epochs_data:
            continue

        epochs     = [r["epoch"]               for r in epochs_data]
        train_loss = [r.get("train_loss", 0)   for r in epochs_data]
        val_loss   = [r.get("val_loss", 0)     for r in epochs_data]
        val_color  = [r.get("val_color_acc", 0) * 100 for r in epochs_data]
        val_type   = [r.get("val_type_acc",  0) * 100 for r in epochs_data]
        val_make   = [r.get("val_make_acc",  0) * 100 for r in epochs_data]
        val_mean   = [r.get("val_mean_acc",  0) * 100 for r in epochs_data]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"E4 — Krzywe uczenia atrybutów: {arch_label(arch)}", fontweight="bold")

        axes[0].plot(epochs, train_loss, color=color, label="Strata treningowa", linewidth=2)
        axes[0].plot(epochs, val_loss,   color=color, label="Strata walidacyjna",
                     linestyle="--", linewidth=2)
        axes[0].set_xlabel("Epoka")
        axes[0].set_ylabel("Strata")
        axes[0].set_title("Funkcja straty")
        axes[0].legend()

        axes[1].plot(epochs, val_color, color="#0D9488", label="Kolor",
                     linewidth=2, marker="o", markersize=3)
        axes[1].plot(epochs, val_type,  color="#1D4ED8", label="Typ",
                     linewidth=2, marker="s", markersize=3)
        axes[1].plot(epochs, val_make,  color="#EA580C", label="Marka",
                     linewidth=2, marker="^", markersize=3)
        axes[1].plot(epochs, val_mean,  color="#6B7280", label="Średnia",
                     linewidth=2.5, linestyle="--")
        axes[1].set_xlabel("Epoka")
        axes[1].set_ylabel("Dokładność walidacyjna (%)")
        axes[1].set_title("Dokładność na atrybut (walidacja)")
        axes[1].set_ylim(0, 100)
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        safe = arch.replace("/", "_")
        out = output_dir / f"E4_attr_{safe}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if attr_data:
        n = len(attr_data)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        fig.suptitle("E4 — Krzywe uczenia atrybutów pojazdu",
                     fontweight="bold")
        if n == 1:
            axes = [axes]

        for row, (key, epochs_data) in enumerate(attr_data.items()):
            arch  = key.replace("attr_", "")
            color = arch_color(arch)
            if not epochs_data:
                continue
            epochs     = [r["epoch"]               for r in epochs_data]
            train_loss = [r.get("train_loss", 0)   for r in epochs_data]
            val_loss   = [r.get("val_loss", 0)     for r in epochs_data]
            val_color  = [r.get("val_color_acc", 0) * 100 for r in epochs_data]
            val_type   = [r.get("val_type_acc",  0) * 100 for r in epochs_data]
            val_make   = [r.get("val_make_acc",  0) * 100 for r in epochs_data]
            val_mean   = [r.get("val_mean_acc",  0) * 100 for r in epochs_data]

            axes[row][0].plot(epochs, train_loss, color=color,
                              label="Strata treningowa", linewidth=2)
            axes[row][0].plot(epochs, val_loss,   color=color,
                              label="Strata walidacyjna", linestyle="--", linewidth=2)
            axes[row][0].set_xlabel("Epoka")
            axes[row][0].set_ylabel("Strata")
            axes[row][0].set_title(f"{arch_label(arch)} — Funkcja straty")
            axes[row][0].legend()

            axes[row][1].plot(epochs, val_color, color="#0D9488", label="Kolor",
                              linewidth=2, marker="o", markersize=3)
            axes[row][1].plot(epochs, val_type,  color="#1D4ED8", label="Typ",
                              linewidth=2, marker="s", markersize=3)
            axes[row][1].plot(epochs, val_make,  color="#EA580C", label="Marka",
                              linewidth=2, marker="^", markersize=3)
            axes[row][1].plot(epochs, val_mean,  color="#6B7280", label="Średnia",
                              linewidth=2.5, linestyle="--")
            axes[row][1].set_xlabel("Epoka")
            axes[row][1].set_ylabel("Dokładność walidacyjna (%)")
            axes[row][1].set_title(f"{arch_label(arch)} — Dokładność na atrybut")
            axes[row][1].set_ylim(0, 100)
            axes[row][1].legend(fontsize=9)

        plt.tight_layout()
        out = output_dir / "E4_learning_curves_attributes.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

def plot_e5_inference_speed(results: dict, output_dir: Path):
 
    BATCH_SIZES = [1, 8, 32, 64]

    ocr_data  = {k: v for k, v in results.items() if k.startswith("ocr_")}
    attr_data = {k: v for k, v in results.items() if k.startswith("attr_")}

    def _plot_metric(data, metric, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, d in data.items():
            arch = key.replace("ocr_", "").replace("attr_", "")
            batch_results = d.get("batch_results", {})
            values = [batch_results.get(str(bs), {}).get(metric, 0)
                      for bs in BATCH_SIZES]
            ax.plot([str(bs) for bs in BATCH_SIZES], values,
                    color=arch_color(arch), marker=arch_marker(arch),
                    label=arch_label(arch), linewidth=2, markersize=8)
        ax.set_xlabel("Rozmiar wsadu (batch size)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        out = output_dir / filename
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano → {out}")

    if ocr_data:
        _plot_metric(ocr_data, "fps",     "Przepustowość (FPS)",
                     "E5 — OCR: przepustowość inferencji",
                     "E5_ocr_fps.png")
        _plot_metric(ocr_data, "mean_ms", "Opóźnienie (ms)",
                     "E5 — OCR: opóźnienie inferencji",
                     "E5_ocr_latency.png")

    if attr_data:
        _plot_metric(attr_data, "fps",     "Przepustowość (FPS)",
                     "E5 — Atrybuty: przepustowość inferencji",
                     "E5_attr_fps.png")
        _plot_metric(attr_data, "mean_ms", "Opóźnienie (ms)",
                     "E5 — Atrybuty: opóźnienie inferencji",
                     "E5_attr_latency.png")

    n_cols = (1 if ocr_data else 0) + (1 if attr_data else 0)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10))
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle("E5 — Szybkość inferencji CNN vs ViT", fontweight="bold")

    for col, (data, task_title) in enumerate([
        (ocr_data,  "OCR — Rozpoznawanie tablic"),
        (attr_data, "Atrybuty — Klasyfikacja pojazdu"),
    ]):
        if not data:
            continue
        for row, (metric, ylabel, subtitle) in enumerate([
            ("fps",     "Przepustowość (FPS)",  "Przepustowość"),
            ("mean_ms", "Opóźnienie (ms)",       "Opóźnienie inferencji"),
        ]):
            ax = axes[row][col]
            for key, d in data.items():
                arch = key.replace("ocr_", "").replace("attr_", "")
                batch_results = d.get("batch_results", {})
                values = [batch_results.get(str(bs), {}).get(metric, 0)
                          for bs in BATCH_SIZES]
                ax.plot([str(bs) for bs in BATCH_SIZES], values,
                        color=arch_color(arch), marker=arch_marker(arch),
                        label=arch_label(arch), linewidth=2, markersize=8)
            ax.set_xlabel("Rozmiar wsadu (batch size)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{task_title}\n{subtitle}")
            ax.legend()

    param_lines = []
    for key, d in results.items():
        arch   = key.replace("ocr_", "").replace("attr_", "")
        params = d.get("params_M", 0)
        task   = "OCR" if key.startswith("ocr_") else "Atrybuty"
        param_lines.append(f"{arch_label(arch)} ({task}): {params:.1f}M param.")
    if param_lines:
        fig.text(0.5, -0.02, "  |  ".join(param_lines),
                 ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = output_dir / "E5_inference_speed.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano → {out}")

def plot_e5_pipeline_speed(results: dict, output_dir: Path):
    
    if not results:
        return

    ocr_data  = {k: v for k, v in results.items() if "ocr"  in k}
    attr_data = {k: v for k, v in results.items() if "attr" in k}

    labels_ocr  = [arch_label(k.replace("ocr_pipeline_",  "")) for k in ocr_data]
    labels_attr = [arch_label(k.replace("attr_pipeline_", "")) for k in attr_data]
    fps_ocr     = [v["fps"]     for v in ocr_data.values()]
    fps_attr    = [v["fps"]     for v in attr_data.values()]
    ms_ocr      = [v["mean_ms"] for v in ocr_data.values()]
    ms_attr     = [v["mean_ms"] for v in attr_data.values()]
    colors_ocr  = [arch_color(k.replace("ocr_pipeline_",  "")) for k in ocr_data]
    colors_attr = [arch_color(k.replace("attr_pipeline_", "")) for k in attr_data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("E5 — Szybkość pełnego przebiegu systemu",
                 fontweight="bold")

    ax = axes[0]
    x_ocr  = np.arange(len(labels_ocr))
    x_attr = x_ocr + len(labels_ocr) + 0.5
    w = 0.6

    bars_ocr  = ax.bar(x_ocr,  fps_ocr,  w, color=colors_ocr,  alpha=0.9,
                       label="_nolegend_")
    bars_attr = ax.bar(x_attr, fps_attr, w, color=colors_attr, alpha=0.6,
                       hatch="//", label="_nolegend_")

    for bar, val in zip(list(bars_ocr) + list(bars_attr),
                        fps_ocr + fps_attr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    all_labels = labels_ocr + labels_attr
    all_x      = list(x_ocr) + list(x_attr)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")
    ax.set_ylabel("Przepustowość (FPS)")
    ax.set_title("Przepustowość pełnego przebiegu", pad=17)
    ax.axvline(x=len(labels_ocr) - 0.25, color="gray",
               linestyle="--", alpha=0.5)
    ax.text(len(labels_ocr)/2 - 0.5,     ax.get_ylim()[1] * 1.01,
            "OCR",  ha="center", fontsize=10, color="gray")
    ax.text(len(labels_ocr) + len(labels_attr)/2 + 0.25,
            ax.get_ylim()[1] * 1.01,
            "Atrybuty", ha="center", fontsize=10, color="gray")
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars_ocr  = ax.bar(x_ocr,  ms_ocr,  w, color=colors_ocr,  alpha=0.9)
    bars_attr = ax.bar(x_attr, ms_attr, w, color=colors_attr, alpha=0.6,
                       hatch="//")

    for bar, val in zip(list(bars_ocr) + list(bars_attr),
                        ms_ocr + ms_attr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")
    ax.set_ylabel("Opóźnienie (ms)")
    ax.set_title("Opóźnienie pełnego przebiegu", pad=17)
    ax.axvline(x=len(labels_ocr) - 0.25, color="gray",
               linestyle="--", alpha=0.5)
    ax.text(len(labels_ocr)/2 - 0.5,     ax.get_ylim()[1] * 1.01,
            "OCR",  ha="center", fontsize=10, color="gray")
    ax.text(len(labels_ocr) + len(labels_attr)/2 + 0.25,
            ax.get_ylim()[1] * 1.01,
            "Atrybuty", ha="center", fontsize=10, color="gray")
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cnn_patch = mpatches.Patch(color=COLORS["cnn"], alpha=0.9,
                                label="ResNet-50 (CNN)")
    vit_patch = mpatches.Patch(color=COLORS["vit"], alpha=0.9,
                                label="ViT-Small (ViT)")
    fig.legend(handles=[cnn_patch, vit_patch], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9)

    plt.tight_layout()
    out = output_dir / "E5_pipeline_speed.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano → {out}")

def main():
    parser = argparse.ArgumentParser(description="Generowanie wykresów CNN vs ViT")
    parser.add_argument("--results-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--output-dir",  type=str, default="outputs/plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  Generowanie wykresów CNN vs ViT")
    print(f"{'═'*55}")

    e1_path = results_dir / "E1_baseline.json"
    if e1_path.exists():
        print("\n  E1 — Baseline...")
        with open(e1_path) as f:
            plot_e1_baseline(json.load(f), output_dir)

    e2_path = results_dir / "E2_robustness.json"
    if e2_path.exists():
        print("\n  E2 — Odporność...")
        with open(e2_path) as f:
            plot_e2_robustness(json.load(f), output_dir)

    e3_path = results_dir / "E3_limited_data.json"
    if e3_path.exists():
        print("\n  E3 — Ograniczone dane...")
        with open(e3_path) as f:
            plot_e3_limited_data(json.load(f), output_dir)

    e4_path = results_dir / "E4_overfitting.json"
    if e4_path.exists():
        print("\n  E4 — Krzywe uczenia...")
        with open(e4_path) as f:
            plot_e4_learning_curves(json.load(f), output_dir)

    e5_path = results_dir / "E5_inference_speed.json"
    if e5_path.exists():
        print("\n  E5 — Szybkość inferencji...")
        with open(e5_path) as f:
            plot_e5_inference_speed(json.load(f), output_dir)
    
    e5_pipeline_path = results_dir / "E5_pipeline_speed.json"
    if e5_pipeline_path.exists():
        print("\n  E5 — Szybkość pełnego przebiegu...")
        with open(e5_pipeline_path) as f:
            plot_e5_pipeline_speed(json.load(f), output_dir)

    print(f"\n  Wykresy zapisane w: {output_dir}/")
    print(f"{'═'*55}")

if __name__ == "__main__":
    main()
