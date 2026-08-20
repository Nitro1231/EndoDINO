import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from endodino.constants import (
    CLASS_TO_IDX,
    CLASSES,
    DEFAULT_LABEL_COLUMN,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OTHERCLASS,
)

SET_TYPE_TO_SPLIT = {"Train": "train", "Validation": "val", "Test": "test"}


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


def _normalize_label(raw: str) -> str:
    return "NA" if raw == OTHERCLASS else raw


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "landmark", "patient"])
        writer.writerows(rows)


def prepare_splits(
    labels_csv: Path,
    out_dir: Path,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> dict[str, Path]:
    """Build train/val/test CSVs from GastroHUN official splits. Reuse if manifest matches."""
    paths = {name: out_dir / f"{name}.csv" for name in ("train", "val", "test")}
    manifest_path = out_dir / "manifest.json"

    with labels_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    buckets = {name: [] for name in paths}
    for row in rows:
        label = row[label_column].strip()
        if not label:
            continue
        landmark = _normalize_label(label)
        split = SET_TYPE_TO_SPLIT[row["set_type"]]
        patient = row["num patient"]
        relpath = f"{patient}/{row['filename']}"
        buckets[split].append((relpath, landmark, patient))

    counts = {name: len(buckets[name]) for name in paths}
    expected = {"label_column": label_column, "counts": counts}
    if manifest_path.exists() and all(p.exists() for p in paths.values()):
        if json.loads(manifest_path.read_text()) == expected:
            return paths

    for name, path in paths.items():
        _write_csv(path, buckets[name])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
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
