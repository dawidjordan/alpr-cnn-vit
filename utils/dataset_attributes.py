import json
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.dataset_vehicles import (
    CompCarsDataset, VehicleColorDataset,
    COMPCARS_TYPES, COLOR_CLASSES, COLOR_TO_IDX,
    TYPE_IDX_TO_NAME, NUM_VEHICLE_TYPES, NUM_COLORS,
)


MIN_MAKE_SAMPLES = 200


def get_valid_makes(compcars_root: str, min_samples: int = MIN_MAKE_SAMPLES) -> dict:
    
    image_dir = Path(compcars_root) / "image"
    make_counts = {}

    for make_dir in sorted(image_dir.iterdir()):
        if not make_dir.is_dir():
            continue
        count = sum(1 for _ in make_dir.rglob("*.jpg"))
        if count >= min_samples:
            make_counts[make_dir.name] = count

    
    sorted_makes = sorted(make_counts.keys(), key=lambda x: int(x))
    make_to_idx = {make_id: idx for idx, make_id in enumerate(sorted_makes)}

    return make_to_idx



class VehicleAttributeDataset(Dataset):
   

    def __init__(
        self,
        compcars_root: str,
        color_root: str,
        split: str = "train",         
        img_size: int = 224,
        transform: Optional[Callable] = None,
        max_compcars: Optional[int] = None,
        max_color: Optional[int] = None,
        min_make_samples: int = MIN_MAKE_SAMPLES,
        seed: int = 42,
    ):
        self.img_size  = img_size
        self.transform = transform

        
        self.make_to_idx = get_valid_makes(compcars_root, min_make_samples)
        self.num_makes   = len(self.make_to_idx)
        self.num_colors  = NUM_COLORS
        self.num_types   = NUM_VEHICLE_TYPES

        print(f"  Marki (make) po filtracji (min {min_make_samples} próbek): {self.num_makes}")
        print(f"  Typy pojazdów: {self.num_types}")
        print(f"  Kolory: {self.num_colors}")

       
        self.samples = []
        self._load_compcars(compcars_root, split, max_compcars, seed)
        self._load_colors(color_root, split if split != "test" else "val",
                        max_color, seed)

        print(f"  Łącznie próbek: {len(self.samples):,}")

    def _load_compcars(
        self,
        root: str,
        split: str,           
        max_samples: Optional[int],
        seed: int,
    ) -> None:
        
        root = Path(root)
        split_file = root / "train_test_split" / "classification" / f"{split}.txt"

        if not split_file.exists():
            print(f"   CompCars: brak pliku podziału {split_file}")
            return

        
        model_to_type = {}
        attrs_path = root / "misc" / "attributes.txt"
        if attrs_path.exists():
            for line in attrs_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        model_id = parts[0]
                        type_id  = int(parts[5])
                        if 1 <= type_id <= 12:
                            model_to_type[model_id] = type_id - 1  # 0-indexed
                    except ValueError:
                        continue

        
        lines = split_file.read_text().strip().splitlines()

        samples = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            img_path = root / "image" / line
            if not img_path.exists():
                continue

            
            parts = line.split("/")
            make_id  = parts[0]
            model_id = parts[1]

            
            if make_id not in self.make_to_idx:
                continue

            make_idx = self.make_to_idx[make_id]
            type_idx = model_to_type.get(model_id, -1)

            samples.append({
                "path":  img_path,
                "color": -1,
                "type":  type_idx,
                "make":  make_idx,
            })

        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples.extend(samples)
        print(f"  CompCars ({split}): {len(samples):,} próbek")

    def _load_colors(
        self,
        root: str,
        split: str,
        max_samples: Optional[int],
        seed: int,
    ) -> None:
        
        split_dir = Path(root) / split
        if not split_dir.exists():
            print(f"  Vehicle Color: brak katalogu {split_dir}")
            return

        samples = []
        for color_dir in sorted(split_dir.iterdir()):
            if not color_dir.is_dir():
                continue
            color_name = color_dir.name.lower()
            if color_name not in COLOR_TO_IDX:
                continue
            color_idx = COLOR_TO_IDX[color_name]

            for img_path in color_dir.glob("*.jpg"):
                samples.append({
                    "path":  img_path,
                    "color": color_idx,
                    "type":  -1,         
                    "make":  -1,         
                })
            for img_path in color_dir.glob("*.png"):
                samples.append({
                    "path":  img_path,
                    "color": color_idx,
                    "type":  -1,
                    "make":  -1,
                })

        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples.extend(samples)
        print(f"  Vehicle Color ({split}): {len(samples):,} próbek")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s = self.samples[idx]

        image = cv2.imread(str(s["path"]))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image_tensor = self._to_tensor(image)
        labels = {
            "color": torch.tensor(s["color"], dtype=torch.long),
            "type":  torch.tensor(s["type"],  dtype=torch.long),
            "make":  torch.tensor(s["make"],  dtype=torch.long),
        }
        return image_tensor, labels

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = image.astype(np.float32) / 255.0
        img  = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))



def make_attribute_dataloaders(
    compcars_root: str,
    color_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_compcars: Optional[int] = None,
    max_color: Optional[int] = None,
    transform_train: Optional[Callable] = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    
   
    train_dataset = VehicleAttributeDataset(
        compcars_root=compcars_root,
        color_root=color_root,
        split="train",
        transform=transform_train,
        max_compcars=max_compcars,
        max_color=max_color,
        seed=seed,
    )

    
    val_dataset = VehicleAttributeDataset(
        compcars_root=compcars_root,
        color_root=color_root,
        split="test",
        max_compcars=None,
        max_color=None,
        seed=seed,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader
