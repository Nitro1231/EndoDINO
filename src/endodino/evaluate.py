import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from endodino.constants import CLASSES, DEFAULT_IMAGES, DEFAULT_OUTPUTS, NUM_CLASSES
from endodino.data import LandmarkDataset, eval_transform
from endodino.model import LandmarkClassifier

log = logging.getLogger("endodino")


def metrics_from_preds(y_true, y_pred) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
        ),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, criterion=None, desc="val"):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = metrics_from_preds(y_true, y_pred)
    if criterion is not None:
        metrics["loss"] = total_loss / n
    return metrics, y_true, y_pred


def plot_confusion_matrix(cm, path=None):
    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def load_trained_model(checkpoint: Path, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LandmarkClassifier()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Evaluate 23-class SSS landmark classifier on a labeled split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_OUTPUTS / "splits")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS / "eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  split=%s  checkpoint=%s", device, args.split, args.checkpoint)
    model, _ = load_trained_model(args.checkpoint, device)
    csv_path = args.splits_dir / f"{args.split}.csv"
    dataset = LandmarkDataset(csv_path, args.images, eval_transform())
    log.info("%s images: %d", args.split, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, y_true, y_pred = eval_epoch(model, loader, device, desc=args.split)
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.split}_report.txt").write_text(report)
    np.savetxt(args.out_dir / f"{args.split}_confusion.csv", cm, fmt="%d", delimiter=",")
    fig = plot_confusion_matrix(cm, args.out_dir / f"{args.split}_confusion.png")
    plt.close(fig)

    log.info(
        "acc=%.4f  balanced_acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
        metrics["acc"],
        metrics["balanced_acc"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )
    log.info("wrote %s", args.out_dir)
    print(report)


if __name__ == "__main__":
    main()
