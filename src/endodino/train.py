import argparse
import logging
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from endodino.constants import (
    CLASSES,
    DEFAULT_IMAGES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_LABELS,
    DEFAULT_OUTPUTS,
    DEFAULT_WEIGHTS,
    LABEL_COLUMNS,
    NUM_CLASSES,
)
from endodino.data import (
    LandmarkDataset,
    class_weights,
    eval_transform,
    prepare_splits,
    train_transform,
)
from endodino.evaluate import eval_epoch, plot_confusion_matrix
from endodino.model import LandmarkClassifier

log = logging.getLogger("endodino")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GastroNet DINOv2 for 23-class SSS landmark classification."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, choices=LABEL_COLUMNS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="endodino")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  freeze_backbone=%s  wandb=%s", device, args.freeze_backbone, args.wandb)
    splits_dir = args.output_dir / "splits"
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_paths = prepare_splits(args.labels, splits_dir, args.label_column)

    train_ds = LandmarkDataset(split_paths["train"], args.images, train_transform())
    val_ds = LandmarkDataset(split_paths["val"], args.images, eval_transform())
    log.info(
        "splits  label=%s  train=%d  val=%d  test=%d",
        args.label_column,
        len(train_ds),
        len(val_ds),
        len(split_paths["test"].read_text().splitlines()) - 1,
    )
    log.info("loading backbone %s", args.weights)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = LandmarkClassifier(args.weights, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds.label_ids()).to(device))
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                {"params": model.head.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=args.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run = None
    if args.wandb:
        mode = "probe" if args.freeze_backbone else "ft"
        run = wandb.init(
            project=args.wandb_project,
            name=f"s{args.seed}_{mode}",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )

    ranking = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        running_loss, n = 0.0, 0
        batches = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            batches.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss = running_loss / n

        val_metrics, y_true, y_pred = eval_epoch(
            model, val_loader, device, criterion, desc=f"val {epoch}/{args.epochs}"
        )
        log.info(
            "epoch %02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_macro_f1=%.4f  lr_head=%.2e",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["macro_f1"],
            optimizer.param_groups[-1]["lr"],
        )

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": {
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in val_metrics.items()
            },
            "classes": CLASSES,
            "args": vars(args),
        }
        last_path = ckpt_dir / "last.pt"
        torch.save(payload, last_path)
        score = val_metrics["macro_f1"]
        if len(ranking) < 3 or score > ranking[-1][0]:
            epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            shutil.copyfile(last_path, epoch_path)
            ranking.append((score, epoch_path))
            ranking.sort(key=lambda item: item[0], reverse=True)
            for _, path in ranking[3:]:
                path.unlink()
            ranking = ranking[:3]
            for i, (_, path) in enumerate(ranking, 1):
                shutil.copyfile(path, ckpt_dir / f"top{i}.pt")
            log.info(
                "saved top-%d  best_macro_f1=%.4f  (%s)",
                len(ranking),
                ranking[0][0],
                ", ".join(p.stem for _, p in ranking),
            )

        if run is not None:
            wandb_log = {
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "val/balanced_acc": val_metrics["balanced_acc"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/weighted_f1": val_metrics["weighted_f1"],
                "lr/head": optimizer.param_groups[-1]["lr"],
                "epoch": epoch,
            }
            if len(optimizer.param_groups) > 1:
                wandb_log["lr/backbone"] = optimizer.param_groups[0]["lr"]
            for name, f1 in zip(CLASSES, val_metrics["per_class_f1"]):
                wandb_log[f"val/f1_{name}"] = float(f1)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
            fig = plot_confusion_matrix(cm)
            wandb_log["val/confusion_matrix"] = wandb.Image(fig)
            run.log(wandb_log, step=epoch)
            plt.close(fig)

    log.info("done  checkpoints in %s", ckpt_dir)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
