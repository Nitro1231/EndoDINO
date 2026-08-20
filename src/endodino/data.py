import csv
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from endodino.constants import (
    CLASS_TO_IDX,
    CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class GaussianNoise:
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return x + torch.randn_like(x) * self.std
        return x


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.3,
            ),
            transforms.ToTensor(),
            GaussianNoise(std=0.02, p=0.5),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "landmark"])
        writer.writerows(rows)


def prepare_splits(
    ugiad_splits: Path,
    images_root: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path]:
    """Train/val from on-disk images; official test.csv basenames are held out. Reuse if present."""
    paths = {name: out_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()):
        return paths

    test_names = set()
    for csv_path in ugiad_splits.rglob("test.csv"):
        for line in csv_path.read_text().splitlines():
            name = line.strip()
            if name:
                test_names.add(name)

    trainval, test_rows = [], []
    for image_path in sorted(images_root.rglob("*.png")):
        relpath = image_path.relative_to(images_root).as_posix()
        row = (relpath, image_path.parent.name)
        if image_path.name in test_names:
            test_rows.append(row)
        else:
            trainval.append(row)

    train_rows, val_rows = train_test_split(
        trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=[landmark for _, landmark in trainval],
    )
    _write_csv(paths["train"], train_rows)
    _write_csv(paths["val"], val_rows)
    _write_csv(paths["test"], test_rows)
    return paths


class LandmarkDataset(Dataset):
    def __init__(self, csv_path: Path, images_root: Path, transform):
        with csv_path.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.images_root / row["relpath"]).convert("RGB")
        return self.transform(image), CLASS_TO_IDX[row["landmark"]]

    def label_ids(self) -> list[int]:
        return [CLASS_TO_IDX[row["landmark"]] for row in self.rows]


def class_weights(label_ids: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(label_ids), minlength=len(CLASSES)).float()
    return counts.sum() / (len(CLASSES) * counts.clamp_min(1.0))
