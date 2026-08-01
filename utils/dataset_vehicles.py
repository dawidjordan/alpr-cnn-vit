from pathlib import Path
from typing import Callable, Optional
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


COMPCARS_TYPES = {
    1:  "MPV",
    2:  "SUV",
    3:  "sedan",
    4:  "hatchback",
    5:  "minibus",
    6:  "fastback",
    7:  "estate",
    8:  "pickup",
    9:  "hardtop_convertible",
    10: "sports",
    11: "crossover",
    12: "convertible",
}

TYPE_IDX_TO_NAME = {i: name for i, (_, name) in enumerate(COMPCARS_TYPES.items())}
TYPE_NAME_TO_IDX = {name: i for i, name in TYPE_IDX_TO_NAME.items()}
NUM_VEHICLE_TYPES = len(COMPCARS_TYPES)   # 12


COLOR_CLASSES = ["beige", "black", "blue", "brown", "gold",
                 "green", "grey", "orange", "pink", "purple",
                 "red", "silver", "tan", "white", "yellow"]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_CLASSES)}
IDX_TO_COLOR = {i: c for c, i in COLOR_TO_IDX.items()}
NUM_COLORS = len(COLOR_CLASSES)   # 15



def _to_tensor(image: np.ndarray) -> torch.Tensor:
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = image.astype(np.float32) / 255.0
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1))




class CompCarsDataset(Dataset):
   
    def __init__(
        self,
        root: str,
        img_size: int = 224,
        task: str = "type",         # "type" lub "make" (marka)
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        min_samples_per_class: int = 10,
    ):
       
        self.root = Path(root)
        self.img_size = img_size
        self.task = task
        self.transform = transform

        image_dir = self.root / "image"
        label_dir = self.root / "label"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Katalog CompCars nie znaleziony: {image_dir}\n"
                f"Pobierz dataset: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/\n"
                f"Wypakuj do: data/raw/compcars/"
            )

        self.samples = []  # lista (ścieżka_obrazu, label_int)
        self._build_index(image_dir, label_dir, min_samples_per_class)

        if max_samples is not None:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"Brak próbek w CompCars ({task})")

    def _build_index(self, image_dir: Path, label_dir: Path, min_per_class: int):
        
        if self.task == "type":
            self._build_type_index(image_dir, label_dir, min_per_class)
        elif self.task == "make":
            self._build_make_index(image_dir, min_per_class)
        else:
            raise ValueError(f"Nieznane zadanie: '{self.task}'. Użyj 'type' lub 'make'")

    def _build_type_index(self, image_dir: Path, label_dir: Path, min_per_class: int):
        
        from collections import defaultdict
        class_samples = defaultdict(list)

        for label_file in label_dir.rglob("*.txt"):
            try:
                type_id = int(label_file.read_text().strip().split("\n")[0])
                if type_id not in COMPCARS_TYPES:
                    continue
                label_idx = type_id - 1  

                
                rel_path = label_file.relative_to(label_dir).with_suffix(".jpg")
                img_path = image_dir / rel_path
                if img_path.exists():
                    class_samples[label_idx].append((img_path, label_idx))
            except (ValueError, OSError):
                continue

        
        for label_idx, samples in class_samples.items():
            if len(samples) >= min_per_class:
                self.samples.extend(samples)

        
        self.classes = sorted(class_samples.keys())
        self.num_classes = len(self.classes)

    def _build_make_index(self, image_dir: Path, min_per_class: int):
       
        from collections import defaultdict
        make_to_idx = {}
        class_samples = defaultdict(list)

        for make_dir in sorted(image_dir.iterdir()):
            if not make_dir.is_dir():
                continue
            make_name = make_dir.name
            if make_name not in make_to_idx:
                make_to_idx[make_name] = len(make_to_idx)
            make_idx = make_to_idx[make_name]

            for img_path in make_dir.rglob("*.jpg"):
                class_samples[make_idx].append((img_path, make_idx))

        for make_idx, samples in class_samples.items():
            if len(samples) >= min_per_class:
                self.samples.extend(samples)

        self.make_to_idx = make_to_idx
        self.idx_to_make = {v: k for k, v in make_to_idx.items()}
        self.num_classes = len(make_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return _to_tensor(image), torch.tensor(label, dtype=torch.long)



class VehicleColorDataset(Dataset):
    

    def __init__(
        self,
        root: str,
        split: str = "train",        
        img_size: int = 224,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Katalog nie znaleziony: {split_dir}\n"
                f"Dostępne podziały: train / val / test"
            )

        self.samples = []
        self._build_index(split_dir)

        if max_samples is not None:
            import random
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"Brak próbek w: {split_dir}")

    def _build_index(self, split_dir: Path):
        
        found_colors = []
        for color_dir in sorted(split_dir.iterdir()):
            if not color_dir.is_dir():
                continue
            color_name = color_dir.name.lower()
            if color_name not in COLOR_TO_IDX:
                continue
            label = COLOR_TO_IDX[color_name]
            for img_path in color_dir.glob("*.jpg"):
                self.samples.append((img_path, label))
            for img_path in color_dir.glob("*.png"):
                self.samples.append((img_path, label))
            found_colors.append(color_name)

        missing = set(COLOR_CLASSES) - set(found_colors)
        if missing:
            print(f"[VehicleColorDataset] Brak podkatalogów dla: {missing}")

        self.num_classes = NUM_COLORS

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return _to_tensor(image), torch.tensor(label, dtype=torch.long)


def split_and_load(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    splits_dir: Optional[str] = None,
    split_name: str = "dataset",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    from utils.dataset_ccpd import split_dataset, make_dataloaders

    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=round(1.0 - train_ratio - val_ratio, 6),
        seed=seed,
        splits_dir=splits_dir,
        split_name=split_name,
    )

    return make_dataloaders(
        train_set, val_set, test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
