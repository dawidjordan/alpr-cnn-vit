import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches



VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CLASS_COLORS = {
    "car":        "#2196F3",   # niebieski
    "truck":      "#FF5722",   # pomarańczowy
    "bus":        "#9C27B0",   # fioletowy
    "motorcycle": "#4CAF50",   # zielony
}


def load_model():
    
    from ultralytics import YOLO
    print("  Wczytywanie YOLOv8n (COCO)...")
    model = YOLO("yolov8n.pt")
    print("  Model gotowy")
    return model


def detect_vehicles(model, image_path: str, conf_threshold: float = 0.25):
   
    results = model(image_path, conf=conf_threshold, verbose=False)[0]
    image_bgr = cv2.imread(image_path)

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append({
            "class_name": VEHICLE_CLASSES[cls_id],
            "bbox":       (x1, y1, x2, y2),
            "confidence": conf,
        })

    return image_bgr, detections


def draw_detections(image_bgr, detections, image_path: str):
    
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls  = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cls, "#FF0000")

        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

       
        label = f"{cls.upper()} {conf:.0%}"
        ax.text(
            x1, y1 - 8, label,
            fontsize=12, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor="none"),
        )

    n = len(detections)
    status = f"Wykryto {n} pojazd{'ów' if n != 1 else ''}" if n > 0 else "Brak wykrytych pojazdów"
    ax.set_title(
        f"{status}  |  {Path(image_path).name}",
        fontsize=12, pad=8,
        color="#065F46" if n > 0 else "#991B1B",
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def process_single(model, image_path: str, output_path: str):
    
    print(f"  Obraz: {Path(image_path).name}")
    image_bgr, detections = detect_vehicles(model, image_path)

    if image_bgr is None:
        print(f"   Nie można wczytać obrazu: {image_path}")
        return

    for d in detections:
        print(f"    {d['class_name'].upper():12} conf={d['confidence']:.2f}  bbox={d['bbox']}")

    if not detections:
        print("    Brak wykrytych pojazdów")

    fig = draw_detections(image_bgr, detections, image_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Zapisano → {output_path}")


def process_grid(model, image_paths: list, output_path: str, cols: int = 3):
    
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    fig.patch.set_facecolor("white")

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        image_path = str(image_paths[i])
        image_bgr, detections = detect_vehicles(model, image_path)

        if image_bgr is None:
            ax.axis("off")
            continue

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls   = det["class_name"]
            conf  = det["confidence"]
            color = CLASS_COLORS.get(cls, "#FF0000")

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{cls} {conf:.0%}",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                              alpha=0.85, edgecolor="none"))

        n_det = len(detections)
        title_color = "#065F46" if n_det > 0 else "#991B1B"
        ax.set_title(
            f"{n_det} pojazd{'ów' if n_det != 1 else ''}  |  {Path(image_path).name[:30]}",
            fontsize=9, color=title_color, fontweight="bold", pad=4,
        )

    fig.suptitle(
        "Detekcja pojazdów — YOLOv8n (COCO)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Zapisano siatkę → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detekcja pojazdów — YOLOv8n")
    parser.add_argument("--image",      type=str, default=None,
                        help="Ścieżka do konkretnego zdjęcia")
    parser.add_argument("--compcars-root", type=str,
                        default="data/raw/compcars",
                        help="Katalog główny CompCars")
    parser.add_argument("--n",          type=int, default=6,
                        help="Liczba losowych zdjęć (gdy --image nie podane)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Próg pewności detekcji")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/visualizations/detection")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'═' * 55}")
    print(f"  Detekcja pojazdów — YOLOv8n (COCO)")
    print(f"{'═' * 55}")
    print(f"  Próg pewności: {args.conf}")

    model = load_model()

    if args.image:
        
        out = str(Path(args.output_dir) / "vehicle_detection.png")
        process_single(model, args.image, out)
    else:
        
        image_dir = Path(args.compcars_root) / "image"
        if not image_dir.exists():
            print(f"  Brak katalogu: {image_dir}")
            return

        all_images = list(image_dir.rglob("*.jpg"))
        if not all_images:
            print("  Brak zdjęć w CompCars")
            return

        selected = random.sample(all_images, min(args.n, len(all_images)))
        print(f"\n  Przetwarzam {len(selected)} losowych zdjęć z CompCars...")

        out = str(Path(args.output_dir) / "vehicle_detection_grid.png")
        process_grid(model, selected, out)

    print(f"\n  Gotowe!")


if __name__ == "__main__":
    main()
