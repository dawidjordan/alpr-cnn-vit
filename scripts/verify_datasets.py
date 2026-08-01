import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):   print(f"  {RED}✗{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):   print(f"  {BLUE}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 55)


def verify_ccpd(ccpd_root: str = "data/raw/ccpd"):
    header("CCPD 2019 — Tablice rejestracyjne")

    from utils.dataset_ccpd import (
        CCPDDetectionDataset, CCPDOCRDataset,
        CCPD_SUBSETS, parse_ccpd_filename, NUM_CHARS
    )

    root = Path(ccpd_root)
    if not root.exists():
        fail(f"Katalog nie istnieje: {root}")
        info("Pobierz CCPD: https://github.com/detectRecog/CCPD")
        return False

    info(f"Ścieżka: {root.resolve()}")
    info(f"Liczba klas znaków (OCR): {NUM_CHARS}")

    all_ok = True
    for subset_name, dir_name in CCPD_SUBSETS.items():
        subset_dir = root / dir_name
        if not subset_dir.exists():
            warn(f"Podzbór '{subset_name}' ({dir_name}) — BRAK katalogu")
            if subset_name == "base":
                fail("ccpd_base jest obowiązkowy!")
                all_ok = False
            continue

        jpg_files = list(subset_dir.glob("*.jpg"))
        valid = [f for f in jpg_files if len(f.stem.split("-")) >= 7]
        invalid = len(jpg_files) - len(valid)

        status = ok if len(valid) > 0 else fail
        status(f"{subset_name:<10} {len(valid):>7,} obrazów  "
               f"({'ccpd_base — obowiązkowy' if subset_name == 'base' else dir_name})"
               + (f"  ⚠ {invalid} nieprawidłowych" if invalid > 0 else ""))

    
    print()
    base_dir = root / "ccpd_base"
    if base_dir.exists():
        sample_files = list(base_dir.glob("*.jpg"))[:3]
        if sample_files:
            info("Test parsowania nazw plików (3 próbki):")
            parse_errors = 0
            for f in sample_files:
                try:
                    ann = parse_ccpd_filename(f.name)
                    ok(f"  {f.name[:40]}...  → '{ann['plate_text']}'  bbox={ann['bbox']}")
                except ValueError as e:
                    fail(f"  {f.name}: {e}")
                    parse_errors += 1
            if parse_errors == 0:
                ok("Parsowanie nazw plików działa poprawnie")

   
    print()
    if base_dir.exists():
        try:
            info("Test CCPDOCRDataset (5 próbek):")
            ds = CCPDOCRDataset(root=str(base_dir), max_samples=5)
            for i in range(min(3, len(ds))):
                img, chars = ds[i]
                ok(f"  [{i}] img={tuple(img.shape)}  chars={chars.tolist()}")
        except Exception as e:
            fail(f"CCPDOCRDataset: {e}")
            all_ok = False

        try:
            info("Test CCPDDetectionDataset (5 próbek):")
            ds = CCPDDetectionDataset(root=str(base_dir), max_samples=5)
            for i in range(min(3, len(ds))):
                img, target = ds[i]
                ok(f"  [{i}] img={tuple(img.shape)}  bbox_yolo={target['bbox_yolo'].tolist()[:2]}...  "
                   f"plate='{target['plate_text']}'")
        except Exception as e:
            fail(f"CCPDDetectionDataset: {e}")
            all_ok = False

    return all_ok


def verify_compcars(compcars_root: str = "data/raw/compcars"):
    header("CompCars — Typ i marka pojazdu")

    from utils.dataset_vehicles import CompCarsDataset, COMPCARS_TYPES, NUM_VEHICLE_TYPES

    root = Path(compcars_root)
    if not root.exists():
        warn(f"Katalog nie istnieje: {root}")
        info("Pobierz CompCars: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/")
        return False

    info(f"Ścieżka: {root.resolve()}")
    info(f"Liczba typów pojazdów: {NUM_VEHICLE_TYPES}")

    image_dir = root / "image"
    label_dir = root / "label"

    if not image_dir.exists():
        fail(f"Brak katalogu 'image' w {root}")
        return False

    
    makes = [d for d in image_dir.iterdir() if d.is_dir()]
    ok(f"Liczba marek (make): {len(makes)}")

    total_images = sum(1 for _ in image_dir.rglob("*.jpg"))
    ok(f"Łączna liczba obrazów: {total_images:,}")

    if label_dir.exists():
        ok(f"Katalog label/ istnieje")
        total_labels = sum(1 for _ in label_dir.rglob("*.txt"))
        info(f"Pliki etykiet: {total_labels:,}")
    else:
        warn("Brak katalogu label/ — klasyfikacja marki nadal możliwa, typ wymaga etykiet")

    
    print()
    try:
        info("Test CompCarsDataset task='type' (10 próbek):")
        ds = CompCarsDataset(root=str(root), task="type", max_samples=10)
        for i in range(min(3, len(ds))):
            img, label = ds[i]
            from utils.dataset_vehicles import TYPE_IDX_TO_NAME
            ok(f"  [{i}] img={tuple(img.shape)}  label={label.item()} ({TYPE_IDX_TO_NAME.get(label.item(), '?')})")
        ok(f"  Łącznie próbek: {len(ds)}  |  klas: {ds.num_classes}")
    except Exception as e:
        fail(f"CompCarsDataset (type): {e}")

    try:
        info("Test CompCarsDataset task='make' (10 próbek):")
        ds = CompCarsDataset(root=str(root), task="make", max_samples=10)
        ok(f"  Łącznie próbek: {len(ds)}  |  marek: {ds.num_classes}")
    except Exception as e:
        fail(f"CompCarsDataset (make): {e}")

    return True


def verify_color(color_root: str = "data/raw/vehicle_color"):
    header("Vehicle Color Recognition — Kolor pojazdu")

    from utils.dataset_vehicles import VehicleColorDataset, COLOR_CLASSES, NUM_COLORS

    root = Path(color_root)
    if not root.exists():
        warn(f"Katalog nie istnieje: {root}")
        info("Pobierz z Kaggle: vehicle-color-recognition-dataset")
        info("Wypakuj do: data/raw/vehicle_color/")
        return False

    info(f"Ścieżka: {root.resolve()}")
    info(f"Klasy kolorów ({NUM_COLORS}): {', '.join(COLOR_CLASSES)}")

    
    for color in COLOR_CLASSES:
        color_dir = root / "train" / color
        if color_dir.exists():
            n = len(list(color_dir.glob("*.jpg"))) + len(list(color_dir.glob("*.png")))
            ok(f"  {color:<10} {n:>5,} obrazów (train)")
        else:
            warn(f"  {color:<10} — brak katalogu")

    
    print()
    try:
        info("Test VehicleColorDataset (5 próbek):")
        ds = VehicleColorDataset(root=str(root), split="train", max_samples=5)
        for i in range(min(3, len(ds))):
            img, label = ds[i]
            from utils.dataset_vehicles import IDX_TO_COLOR
            ok(f"  [{i}] img={tuple(img.shape)}  label={label.item()} ({IDX_TO_COLOR[label.item()]})")
        ok(f"  Łącznie próbek: {len(ds)}")
    except Exception as e:
        fail(f"VehicleColorDataset: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Weryfikacja datasetów ALPR")
    parser.add_argument(
        "--dataset",
        choices=["ccpd", "compcars", "color", "all"],
        default="all",
        help="Który dataset zweryfikować (domyślnie: all)",
    )
    parser.add_argument("--ccpd-root",     default="data/raw/ccpd")
    parser.add_argument("--compcars-root", default="data/raw/compcars")
    parser.add_argument("--color-root",    default="data/raw/vehicle_color")
    args = parser.parse_args()

    print(f"\n{BOLD}Weryfikacja datasetów — ALPR Thesis{RESET}")

    results = {}
    if args.dataset in ("ccpd", "all"):
        results["ccpd"] = verify_ccpd(args.ccpd_root)
    if args.dataset in ("compcars", "all"):
        results["compcars"] = verify_compcars(args.compcars_root)
    if args.dataset in ("color", "all"):
        results["color"] = verify_color(args.color_root)

    
    header("PODSUMOWANIE")
    all_passed = all(v for v in results.values() if v is not None)
    for name, passed in results.items():
        if passed:
            ok(f"{name}")
        else:
            fail(f"{name} — wymaga uwagi")

    if all_passed:
        print(f"\n  {GREEN}{BOLD}✓ Wszystkie datasety gotowe do użycia!{RESET}")
        print(f"  Następny krok: python training/scripts/train_classifier.py")
    else:
        print(f"\n  {YELLOW}Pobierz brakujące datasety zgodnie z docs/dataset_guide.md{RESET}")
    print()


if __name__ == "__main__":
    main()
