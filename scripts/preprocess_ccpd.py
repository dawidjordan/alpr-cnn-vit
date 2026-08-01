import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tqdm import tqdm
from utils.dataset_ccpd import parse_ccpd_filename

INPUT_ROOT  = Path("data/raw/ccpd")
OUTPUT_ROOT = Path("data/processed/ccpd")
SPLITS_ROOT = Path("data/raw/ccpd/splits")
IMG_H, IMG_W = 64, 128
PADDING = 4
MAX_SAMPLES = None   # limit próbek (None = wszystkie)


def load_split_file(split_file: Path, max_samples: int = None) -> list[Path]:
   
    lines = split_file.read_text().strip().splitlines()

    if max_samples is not None:
        
        import random
        random.seed(42)
        random.shuffle(lines)
        lines = lines[:max_samples]

    paths = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        full_path = INPUT_ROOT / line
        if full_path.exists():
            paths.append(full_path)

    return paths


def preprocess_split(split_name: str, max_samples: int = None):
   

    split_file = SPLITS_ROOT / f"{split_name}.txt"
    if not split_file.exists():
        print(f"Brak pliku podziału: {split_file}")
        return

    output_dir = OUTPUT_ROOT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    existing = len(list(output_dir.glob("*.npz"))) if output_dir.exists() else 0
    if existing > 0:
        print(f"\n{split_name}: już przetworzone ({existing:,} plików) — pomijam")
        return

    files = load_split_file(split_file, max_samples)
    print(f"\n{split_name}: {len(files):,} obrazów → {output_dir}")

    skipped = 0
    for filepath in tqdm(files, unit="img"):
        try:
            image = cv2.imread(str(filepath))
            if image is None:
                skipped += 1
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            ann = parse_ccpd_filename(filepath.name)
            x1, y1, x2, y2 = ann["bbox"]
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w, x2 + PADDING)
            y2 = min(h, y2 + PADDING)
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (IMG_W, IMG_H))

            chars = np.array(ann["plate_chars"], dtype=np.int32)
            out_path = output_dir / (filepath.stem + ".npz")
            np.savez_compressed(out_path, image=crop, chars=chars)

        except Exception as e:
            skipped += 1
            continue

    print(f"  Zapisano {len(files) - skipped:,} plików .npz → {output_dir}")
    if skipped:
        print(f"  Pominięto: {skipped}")

def generate_missing_split(subset_name: str):
    
    subset_dir = INPUT_ROOT / f"ccpd_{subset_name}"
    if not subset_dir.exists():
        print(f"Brak katalogu: {subset_dir}")
        return

    files = [f for f in sorted(subset_dir.glob("*.jpg"))
             if len(f.stem.split("-")) >= 7]

    split_file = SPLITS_ROOT / f"ccpd_{subset_name}.txt"
    with open(split_file, "w") as f:
        for filepath in files:
            f.write(f"ccpd_{subset_name}/{filepath.name}\n")

    print(f"Wygenerowano: {split_file} ({len(files):,} plików)")


if __name__ == "__main__":
    preprocess_split("train", max_samples=MAX_SAMPLES)
    preprocess_split("val",   max_samples=MAX_SAMPLES)

    generate_missing_split("weather")

    
    for subset in ["ccpd_blur", "ccpd_db", "ccpd_weather",
                   "ccpd_tilt", "ccpd_fn", "ccpd_rotate", "ccpd_challenge"]:
        preprocess_split(subset, max_samples=MAX_SAMPLES)

    print("\nPreprocessing zakończony!")
    print("Struktura:")
    for d in sorted(OUTPUT_ROOT.iterdir()):
        n = len(list(d.glob("*.npz")))
        print(f"  {d.name}/  {n:,} plików")

