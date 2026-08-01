import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.detection.plate_detector import PlateDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trening detektora tablicy rejestracyjnej YOLOv8"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/ccpd/dataset.yaml",
        help="Ścieżka do pliku dataset.yaml (format Ultralytics YOLO)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        help="Bazowy model YOLOv8 do fine-tuningu",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Liczba epok trenowania",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Rozmiar batcha",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Rozmiar obrazu wejściowego",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Urządzenie: auto | cuda | cpu",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Wznów trening z ostatniego checkpointu",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("BŁĄD: Zainstaluj ultralytics: pip install ultralytics")
        sys.exit(1)

    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"BŁĄD: Plik datasetu nie znaleziony: {dataset_path}")
        print("Uruchom najpierw: python scripts/prepare_data.py --dataset ccpd")
        sys.exit(1)

    print("=" * 60)
    print("  Trening detektora tablicy rejestracyjnej YOLOv8")
    print("=" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Model base: {args.base_model}")
    print(f"  Epoki:      {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    
    cfg = PlateDetector.training_config(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
    )

    
    cfg.update({
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "resume": args.resume,
    })

    
    model = YOLO(cfg.pop("model"))
    results = model.train(**cfg)

    print("\n" + "=" * 60)
    print("  Trening zakończony!")
    print(f"  Najlepszy model: outputs/plate_detector/yolov8m_ccpd/weights/best.pt")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("=" * 60)

    
    best_model_path = f"outputs/plate_detector/yolov8m_ccpd/weights/best.pt"
    if Path(best_model_path).exists():
        print("\nWeryfikacja modelu...")
        detector = PlateDetector(model_path=best_model_path)
        print(detector)
        print("\nModel gotowy do użycia w pipeline ALPR.")
    else:
        print(f"\nUWAGA: Nie znaleziono best.pt w oczekiwanej lokalizacji.")


if __name__ == "__main__":
    main()
