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

from utils.dataset_ccpd import parse_ccpd_filename


def load_model():
    from ultralytics import YOLO
    from pathlib import Path
    import torch

    weights_path = Path("models/yolov8n-lp.pt")
    weights_path.parent.mkdir(exist_ok=True)

    
    if not weights_path.exists():
        print("  Trenowanie YOLOv8 na datasecie tablic (keremberke)...")
        try:
            from datasets import load_dataset
            import yaml, shutil

            ds = load_dataset("keremberke/license-plate-object-detection", "full")

            
            data_dir = Path("models/lp_dataset")
            for split, hf_split in [("train","train"), ("val","validation")]:
                img_dir = data_dir / split / "images"
                lbl_dir = data_dir / split / "labels"
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)

                for i, sample in enumerate(ds[hf_split]):
                    img = sample["image"]
                    w, h = img.size
                    img.save(str(img_dir / f"{i}.jpg"))

                    with open(lbl_dir / f"{i}.txt", "w") as f:
                        for obj in sample["objects"]:
                            bx, by, bw, bh = obj["bbox"]
                            cx = (bx + bw/2) / w
                            cy = (by + bh/2) / h
                            nw = bw / w
                            nh = bh / h
                            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            yaml_content = {
                "path": str(data_dir.resolve()),
                "train": "train/images",
                "val":   "val/images",
                "nc": 1,
                "names": ["license_plate"],
            }
            yaml_path = data_dir / "data.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            print("  Dataset przygotowany — trening YOLOv8n (10 epok)...")
            model = YOLO("yolov8n.pt")
            model.train(
                data=str(yaml_path),
                epochs=10,
                imgsz=640,
                batch=16,
                project="models",
                name="lp_train",
                verbose=False,
            )
            best = Path("models/lp_train/weights/best.pt")
            shutil.copy(best, weights_path)
            print(f"  Wagi zapisane → {weights_path}")

        except Exception as e:
            print(f"  Trening nieudany: {e}")
            model = YOLO("yolov8n.pt")
            return model, "coco"

    print(f"  Wczytywanie wag: {weights_path}")
    model = YOLO(str(weights_path))
    print("  Model License Plate gotowy")
    return model, "lp"


def detect_plates(model, model_type: str, image_path: str, conf_threshold: float = 0.25):
   
    results = model(image_path, conf=conf_threshold, verbose=False)[0]
    image_bgr = cv2.imread(image_path)

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])

    
        if model_type == "coco":
            coco_vehicle = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            if cls_id not in coco_vehicle:
                continue
            class_name = coco_vehicle[cls_id]
        else:
            class_name = "license_plate"

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append({
            "class_name": class_name,
            "bbox":       (x1, y1, x2, y2),
            "confidence": conf,
        })

    return image_bgr, detections


def draw_detections_with_gt(
    image_bgr,
    detections: list,
    gt_bbox: tuple,
    image_path: str,
    gt_text: str = "",
    compare_gt: bool = False,
):
  
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")

    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cls  = det["class_name"]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor="#1D4ED8", facecolor="none",
            label="YOLO",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 8, f"YOLO: {cls} {conf:.0%}",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1D4ED8",
                      alpha=0.85, edgecolor="none"),
        )

    
    if compare_gt and gt_bbox:
        gx1, gy1, gx2, gy2 = gt_bbox
        rect_gt = patches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=3, edgecolor="#16A34A", facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect_gt)
        label_gt = f"GT: {gt_text}" if gt_text else "GT bbox"
        ax.text(
            gx1, gy2 + 18, label_gt,
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#16A34A",
                      alpha=0.85, edgecolor="none"),
        )


    legend_elements = [
        patches.Patch(facecolor="#1D4ED8", label="YOLO — detekcja"),
    ]
    if compare_gt:
        legend_elements.append(
            patches.Patch(facecolor="#16A34A", label="GT — adnotacja CCPD")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    n = len(detections)
    status = f"Wykryto {n} obiekt{'ów' if n != 1 else ''}" if n > 0 else "Brak detekcji"
    ax.set_title(
        f"{status}  |  {Path(image_path).name[:50]}",
        fontsize=11, pad=8,
        color="#065F46" if n > 0 else "#991B1B",
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def process_grid(
    model, model_type: str,
    image_paths: list,
    output_path: str,
    compare_gt: bool = False,
    cols: int = 3,
    conf: float = 0.25,
):
    
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    fig.patch.set_facecolor("white")

    if rows == 1:
        axes = [list(axes)] if cols > 1 else [[axes]]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        image_path = str(image_paths[i])
        image_bgr, detections = detect_plates(model, model_type, image_path, conf)

        if image_bgr is None:
            ax.axis("off")
            continue

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")

      
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf_val = det["confidence"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor="#1D4ED8", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 5, 10), f"YOLO {conf_val:.0%}",
                    fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1D4ED8",
                              alpha=0.85, edgecolor="none"))

        
        if compare_gt:
            try:
                ann = parse_ccpd_filename(Path(image_path).name)
                gx1, gy1, gx2, gy2 = ann["bbox"]
                rect_gt = patches.Rectangle(
                    (gx1, gy1), gx2 - gx1, gy2 - gy1,
                    linewidth=2, edgecolor="#16A34A", facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect_gt)
            except Exception:
                pass

        n_det = len(detections)
        title_color = "#065F46" if n_det > 0 else "#991B1B"
        ax.set_title(
            f"{n_det} detekcj{'e' if n_det > 1 else 'a' if n_det == 1 else 'i'}  |  "
            f"{Path(image_path).name[:25]}",
            fontsize=8, color=title_color, fontweight="bold", pad=4,
        )

    
    legend_elements = [patches.Patch(facecolor="#1D4ED8", label="YOLO — detekcja")]
    if compare_gt:
        legend_elements.append(
            patches.Patch(facecolor="#16A34A", label="GT — adnotacja CCPD")
        )
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    model_name = "YOLOv8 License Plate" if model_type == "lp" else "YOLOv8n COCO (fallback)"
    fig.suptitle(
        f"Detekcja tablic rejestracyjnych — {model_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Zapisano siatkę → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detekcja tablic — YOLOv8 LP")
    parser.add_argument("--image",      type=str, default=None,
                        help="Ścieżka do konkretnego zdjęcia JPG z CCPD")
    parser.add_argument("--ccpd-root",  type=str,
                        default="data/raw/ccpd/ccpd_base",
                        help="Katalog z pełnymi zdjęciami CCPD")
    parser.add_argument("--n",          type=int, default=6,
                        help="Liczba losowych zdjęć (gdy --image nie podane)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Próg pewności detekcji")
    parser.add_argument("--compare-gt", action="store_true",
                        help="Pokaż też GT bbox z nazwy pliku CCPD (zielony)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/visualizations/detection")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'═' * 55}")
    print(f"  Detekcja tablic rejestracyjnych — YOLOv8 LP")
    print(f"{'═' * 55}")
    print(f"  Próg pewności: {args.conf}")
    print(f"  Porównanie z GT: {args.compare_gt}")

    model, model_type = load_model()

    if args.image:
        
        image_bgr, detections = detect_plates(model, model_type, args.image, args.conf)
        gt_bbox, gt_text = None, ""
        try:
            ann = parse_ccpd_filename(Path(args.image).name)
            gt_bbox = ann["bbox"]
            gt_text = ann["plate_text"]
        except Exception:
            pass

        print(f"  Obraz: {Path(args.image).name}")
        for d in detections:
            print(f"    {d['class_name']:15} conf={d['confidence']:.2f}  bbox={d['bbox']}")
        if not detections:
            print("    Brak detekcji")
        if gt_text:
            print(f"    GT tablica: {gt_text}  GT bbox: {gt_bbox}")

        fig = draw_detections_with_gt(
            image_bgr, detections, gt_bbox, args.image, gt_text,
            compare_gt=args.compare_gt,
        )
        out = str(Path(args.output_dir) / "plate_detection.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Zapisano → {out}")

    else:
       
        ccpd_dir = Path(args.ccpd_root)
        if not ccpd_dir.exists():
            print(f"  Brak katalogu: {ccpd_dir}")
            return

        all_images = list(ccpd_dir.glob("*.jpg"))
        if not all_images:
            print("  Brak zdjęć JPG w katalogu")
            return

        selected = random.sample(all_images, min(args.n, len(all_images)))
        print(f"\n  Przetwarzam {len(selected)} losowych zdjęć z CCPD...")

        suffix = "_gt" if args.compare_gt else ""
        out = str(Path(args.output_dir) / f"plate_detection_grid{suffix}.png")
        process_grid(
            model, model_type, selected, out,
            compare_gt=args.compare_gt,
            conf=args.conf,
        )

    print(f"\n  Gotowe!")


if __name__ == "__main__":
    main()
