import json
import random
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import cv2 as _cv2
import random as _random


PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]

ALPHABETS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','O']

DIGITS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','O']


ALL_CHARS = PROVINCES + ALPHABETS + DIGITS
IDX_TO_CHAR = {i: c for i, c in enumerate(ALL_CHARS)}
CHAR_TO_IDX = {c: i for i, c in IDX_TO_CHAR.items()}
NUM_CHARS = len(ALL_CHARS)      


CCPD_SUBSETS = {
    "base":    "ccpd_base",
    "blur":    "ccpd_blur",
    "db":      "ccpd_db",
    "weather": "ccpd_weather",
    "tilt":    "ccpd_tilt",
    "fn":      "ccpd_fn",
    "rotate":    "ccpd_rotate",
    "challenge": "ccpd_challenge",
}



def parse_ccpd_filename(filename: str) -> dict:
   
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 7:
        raise ValueError(
            f"Nieprawidłowy format nazwy CCPD: '{filename}'\n"
            f"Oczekiwano 7 segmentów oddzielonych '-', otrzymano {len(parts)}"
        )

    
    try:
        bbox_parts = parts[2].split("_")
        pt1 = [int(x) for x in bbox_parts[0].split("&")]
        pt2 = [int(x) for x in bbox_parts[1].split("&")]
        bbox = [pt1[0], pt1[1], pt2[0], pt2[1]]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Błąd parsowania bbox w '{filename}': {e}")

    
    try:
        keypoints = [
            [int(x) for x in kp.split("&")]
            for kp in parts[3].split("_")
        ]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Błąd parsowania keypoints w '{filename}': {e}")

    
    try:
        char_indices = [int(x) for x in parts[4].split("_")]
        if len(char_indices) != 7:
            raise ValueError(f"Oczekiwano 7 znaków, otrzymano {len(char_indices)}")

        _P = len(PROVINCES)   
        _A = len(ALPHABETS)   
        plate_chars = [
            char_indices[0],                       
            _P + char_indices[1],                 
            _P + _A + char_indices[2],            
            _P + _A + char_indices[3],
            _P + _A + char_indices[4],
            _P + _A + char_indices[5],
            _P + _A + char_indices[6],
        ]
        plate_text = (
            PROVINCES[char_indices[0]]
            + ALPHABETS[char_indices[1]]
            + "".join(DIGITS[i] for i in char_indices[2:])
        )
    except (IndexError, ValueError) as e:
        raise ValueError(f"Błąd parsowania tablicy w '{filename}': {e}")

    
    try:
        brightness = int(parts[5])
        blurriness = int(parts[6])
    except (IndexError, ValueError):
        brightness, blurriness = -1, -1

    return {
        "bbox":       bbox,
        "keypoints":  keypoints,
        "plate_text": plate_text,
        "plate_chars": plate_chars,
        "brightness": brightness,
        "blurriness": blurriness,
    }




class CCPDDetectionDataset(Dataset):


    def __init__(
        self,
        root: str,
        img_size: int = 640,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Katalog CCPD nie znaleziony: {self.root}\n"
                f"Pobierz dataset: https://github.com/detectRecog/CCPD\n"
                f"Wypakuj do: data/raw/ccpd/"
            )

        
        all_files = sorted(self.root.glob("*.jpg"))
        valid_files = []
        for f in all_files:
            if len(f.stem.split("-")) >= 7:
                valid_files.append(f)

        if max_samples is not None:
            valid_files = valid_files[:max_samples]

        self.files = valid_files

        if len(self.files) == 0:
            raise RuntimeError(
                f"Brak plików JPG z poprawnymi adnotacjami w: {self.root}\n"
                f"Upewnij się że pobrano właściwy podzbór CCPD."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        filepath = self.files[idx]

   
        image = cv2.imread(str(filepath))
        if image is None:
            raise RuntimeError(f"Nie można wczytać obrazu: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       
        annotation = parse_ccpd_filename(filepath.name)
        h_orig, w_orig = image.shape[:2]

       
        image_resized = cv2.resize(image, (self.img_size, self.img_size))

        
        x1, y1, x2, y2 = annotation["bbox"]
        x1_n = x1 / w_orig
        y1_n = y1 / h_orig
        x2_n = x2 / w_orig
        y2_n = y2 / h_orig
        xc = (x1_n + x2_n) / 2
        yc = (y1_n + y2_n) / 2
        bw = x2_n - x1_n
        bh = y2_n - y1_n
        bbox_yolo = torch.tensor([xc, yc, bw, bh], dtype=torch.float32)

       
        if self.transform is not None:
            augmented = self.transform(image=image_resized)
            image_resized = augmented["image"]

       
        image_tensor = self._to_tensor(image_resized)

        target = {
            "bbox_yolo":  bbox_yolo,
            "plate_text": annotation["plate_text"],
            "filepath":   str(filepath),
        }

        return image_tensor, target

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """Konwertuje np.ndarray (H,W,3) uint8 → tensor (3,H,W) float32 znormalizowany."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = image.astype(np.float32) / 255.0
        img  = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))


class CCPDOCRDataset(Dataset):

    PLATE_HEIGHT = 64
    PLATE_WIDTH  = 128

    def __init__(
        self,
        root: str,
        img_size: tuple = (64, 128),
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        padding: int = 4,
    ):
       
        self.root = Path(root)
        self.img_h, self.img_w = img_size
        self.transform = transform
        self.padding = padding

        if not self.root.exists():
            raise FileNotFoundError(f"Katalog CCPD nie znaleziony: {self.root}")

        all_files = sorted(self.root.glob("*.jpg"))
        valid_files = [f for f in all_files if len(f.stem.split("-")) >= 7]

        if max_samples is not None:
            valid_files = valid_files[:max_samples]

        self.files = valid_files

        if len(self.files) == 0:
            raise RuntimeError(f"Brak plików w: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        filepath = self.files[idx]

        image = cv2.imread(str(filepath))
        if image is None:
            raise RuntimeError(f"Nie można wczytać: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = parse_ccpd_filename(filepath.name)
        h, w = image.shape[:2]

      
        x1, y1, x2, y2 = annotation["bbox"]
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)
        plate_crop = image[y1:y2, x1:x2]

       
        plate_crop = cv2.resize(plate_crop, (self.img_w, self.img_h))

      
        if self.transform is not None:
            augmented = self.transform(image=plate_crop)
            plate_crop = augmented["image"]

        image_tensor = CCPDDetectionDataset._to_tensor(plate_crop)
        chars_tensor = torch.tensor(annotation["plate_chars"], dtype=torch.long)

        return image_tensor, chars_tensor

    @property
    def num_classes(self) -> int:
        """Liczba możliwych klas znaków — używana przy budowie głowicy OCR."""
        return NUM_CHARS


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed: int = 42,
    splits_dir: Optional[str] = None,
    split_name: str = "ccpd",
) -> tuple[Subset, Subset, Subset]:
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Proporcje muszą sumować się do 1.0"

    n = len(dataset)
    splits_path = None

    
    if splits_dir is not None:
        n_samples = len(dataset)
        splits_path = Path(splits_dir) / f"{split_name}_{n_samples}_split.json"
        if splits_path.exists():
            with open(splits_path) as f:
                indices = json.load(f)
            print(f"Wczytano podział z: {splits_path}")
            return (
                Subset(dataset, indices["train"]),
                Subset(dataset, indices["val"]),
                Subset(dataset, indices["test"]),
            )

    
    rng = random.Random(seed)
    all_indices = list(range(n))
    rng.shuffle(all_indices)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train:n_train + n_val]
    test_idx  = all_indices[n_train + n_val:]

   
    if splits_path is not None:
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
        print(f"Zapisano podział do: {splits_path}")

    print(
        f"Podział datasetu (n={n}): "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def make_dataloaders(
    train_set: Subset,
    val_set:   Subset,
    test_set:  Subset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    _pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=True,    
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def get_subset_loader(
    root: str,
    subset: str,
    task: str = "ocr",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    
    if subset not in CCPD_SUBSETS:
        raise ValueError(
            f"Nieznany podzbór: '{subset}'. Dostępne: {list(CCPD_SUBSETS)}"
        )

    subset_dir = Path(root) / CCPD_SUBSETS[subset]

    if task == "ocr":
        dataset = CCPDOCRDataset(
            root=str(subset_dir),
            max_samples=max_samples,
        )
    elif task == "detection":
        dataset = CCPDDetectionDataset(
            root=str(subset_dir),
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Nieznane zadanie: '{task}'. Użyj 'ocr' lub 'detection'")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

class CachedCCPDOCRDataset(CCPDOCRDataset):
  

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"  Wczytywanie {len(self.files):,} obrazów do RAM...")
        self._cache = []
        import cv2
        from tqdm import tqdm
        for filepath in tqdm(self.files, desc="  Cache", unit="img", leave=False):
            image = cv2.imread(str(filepath))
            if image is None:
                image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                ann = parse_ccpd_filename(filepath.name)
                x1, y1, x2, y2 = ann["bbox"]
                x1 = max(0, x1 - self.padding)
                y1 = max(0, y1 - self.padding)
                x2 = min(w, x2 + self.padding)
                y2 = min(h, y2 + self.padding)
                image = image[y1:y2, x1:x2]
                image = cv2.resize(image, (self.img_w, self.img_h))
            self._cache.append(image)
        print(f"  Cache gotowy ({len(self._cache):,} obrazów w RAM)")

    def __getitem__(self, idx: int) -> tuple:
        image = self._cache[idx].copy()
        annotation = parse_ccpd_filename(self.files[idx].name)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image_tensor = CCPDDetectionDataset._to_tensor(image)
        chars_tensor = torch.tensor(annotation["plate_chars"], dtype=torch.long)
        return image_tensor, chars_tensor
    
class ProcessedCCPDOCRDataset(Dataset):
  

    def __init__(
        self,
        root: str,                      
        transform=None,
        max_samples=None,
    ):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Brak preprocessowanego datasetu: {self.root}\n"
                f"Uruchom: python scripts/preprocess_ccpd.py"
            )

        self.files = sorted(self.root.glob("*.npz"))
        if max_samples:
            self.files = self.files[:max_samples]

        if len(self.files) == 0:
            raise RuntimeError(f"Brak plików .npz w: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data  = np.load(str(self.files[idx]))
        image = data["image"]           
        chars = data["chars"]          

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image_tensor = CCPDDetectionDataset._to_tensor(image)
        chars_tensor = torch.tensor(chars, dtype=torch.long)
        return image_tensor, chars_tensor
    


class RawCCPDOCRDataset(torch.utils.data.Dataset):
   

    IMG_H = 64
    IMG_W = 128
    MEAN  = [0.485, 0.456, 0.406]
    STD   = [0.229, 0.224, 0.225]

    def __init__(
        self,
        raw_dir: str,
        split_file: str,
        max_samples: int = None,
        seed: int = 42,
        transform=None,
    ):
        self.raw_dir   = Path(raw_dir)
        self.transform = transform

        lines = Path(split_file).read_text().strip().splitlines()
        lines = [l.strip() for l in lines if l.strip()]

        if max_samples and max_samples < len(lines):
            rng = _random.Random(seed)
            rng.shuffle(lines)
            lines = lines[:max_samples]

        
        self.samples = []
        missing = 0
        for line in lines:
           
            p = self.raw_dir / line
            if not p.exists():
                
                p = self.raw_dir / Path(line).name
            if not p.exists():
                
                hits = list(self.raw_dir.rglob(Path(line).name))
                p = hits[0] if hits else None

            if p is None or not Path(p).exists():
                missing += 1
                continue

            try:
                ann = parse_ccpd_filename(Path(p).name)
                self.samples.append((Path(p), ann["plate_chars"]))
            except Exception:
                missing += 1

        print(f"  RawCCPDOCRDataset: {len(self.samples):,} próbek "
              f"(pominięto: {missing})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, plate_chars = self.samples[idx]

        
        img = _cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        else:
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)

           
            try:
                ann = parse_ccpd_filename(img_path.name)
                x1, y1, x2, y2 = ann["bbox"]
                h, w = img.shape[:2]
                x1 = max(0, x1 - 4); y1 = max(0, y1 - 4)
                x2 = min(w, x2 + 4); y2 = min(h, y2 + 4)
                img = img[y1:y2, x1:x2]
            except Exception:
                pass

            img = _cv2.resize(img, (self.IMG_W, self.IMG_H))

        
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

       
        img = img.astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std  = np.array(self.STD,  dtype=np.float32)
        img  = (img - mean) / std
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        chars_tensor = torch.tensor(plate_chars, dtype=torch.long)
        return img_tensor, chars_tensor