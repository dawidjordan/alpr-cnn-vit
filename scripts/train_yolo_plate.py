import sys, yaml, shutil, random
from pathlib import Path
sys.path.insert(0, '.')

from ultralytics import YOLO
from utils.dataset_ccpd import parse_ccpd_filename

CCPD_DIR   = Path("data/raw/ccpd/ccpd_base")
OUT_DIR    = Path("models/yolo_plate_dataset")
WEIGHTS    = Path("models/yolov8n-lp.pt")
N_TRAIN    = 3000
N_VAL      = 500
SEED       = 42

def prepare_dataset():
    
    for split in ["train", "val"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    files = list(CCPD_DIR.glob("*.jpg"))
    random.seed(SEED)
    random.shuffle(files)
    train_files = files[:N_TRAIN]
    val_files   = files[N_TRAIN:N_TRAIN + N_VAL]

    for split, split_files in [("train", train_files), ("val", val_files)]:
        for src in split_files:
            try:
                ann = parse_ccpd_filename(src.name)
            except Exception:
                continue

            
            import cv2
            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]

            x1, y1, x2, y2 = ann["bbox"]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

           
            dst_img = OUT_DIR / split / "images" / src.name
            shutil.copy(src, dst_img)

            
            dst_lbl = OUT_DIR / split / "labels" / src.with_suffix(".txt").name
            with open(dst_lbl, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"  {split}: {len(split_files)} obrazów")

   
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(OUT_DIR.resolve()),
            "train": "train/images",
            "val":   "val/images",
            "nc":    1,
            "names": ["license_plate"],
        }, f)

    return yaml_path


if __name__ == "__main__":
    print("Przygotowanie datasetu YOLO z CCPD...")
    yaml_path = prepare_dataset()

    print("Trening YOLOv8n (10 epok)...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=10, imgsz=640, batch=16,
        project="models", name="lp_train",
    )

    best = Path("runs/detect/models/lp_train/weights/best.pt")
    shutil.copy(best, WEIGHTS)
    print(f"\nWagi zapisane → {WEIGHTS}")
    print("Teraz uruchom: python scripts/detect_plate.py --n 6 --compare-gt")