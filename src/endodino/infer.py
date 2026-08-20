import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from endodino.constants import (
    CLASSES,
    CLASS_LABELS_EN,
    CLASS_LABELS_KR,
    DEFAULT_OUTPUTS,
    DEFAULT_TEST_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from endodino.data import eval_transform
from endodino.evaluate import load_trained_model

log = logging.getLogger("endodino")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _configure_korean_font() -> None:
    """Prefer a Hangul-capable font so Korean labels render without glyph warnings."""
    preferred = (
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


def to_vis_image(x):
    mean = x.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(IMAGENET_STD)[:, None, None]
    return to_pil_image((x * std + mean).clamp(0, 1).cpu())


def annotate(image, text):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw.rectangle((0, 0, vis.width, 32), fill="black")
    draw.text((8, 8), text, fill="white")
    return vis


def save_detailed(path, original, processed, probs, pred, name):
    values = [float(p) for p in probs]
    order = sorted(range(len(CLASSES)), key=lambda i: values[i], reverse=True)
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_proc = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    ax_orig.imshow(original)
    ax_orig.set_title("Original")
    ax_orig.axis("off")
    ax_proc.imshow(processed)
    ax_proc.set_title("Processed")
    ax_proc.axis("off")

    y = range(len(order))
    heights = [values[i] for i in order]
    colors = ["#1f6feb" if i == pred else "#c5ccd6" for i in order]
    ax_bar.barh(list(y), heights, color=colors)
    ax_bar.set_yticks(
        list(y),
        [f"{CLASS_LABELS_EN[CLASSES[i]]}\n{CLASS_LABELS_KR[CLASSES[i]]}" for i in order],
        fontsize=8,
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Landmark probabilities")
    for yi, value in zip(y, heights):
        ax_bar.text(value + 0.02, yi, f"{value:.2f}", va="center", fontsize=9)

    pred_name = CLASSES[pred]
    fig.suptitle(
        f"{name}  |  {CLASS_LABELS_EN[pred_name]} / {CLASS_LABELS_KR[pred_name]}  ({values[pred]:.2f})"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run landmark classifier on unlabeled images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "predictions.csv")
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_OUTPUTS / "predictions")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Save a figure with original, processed crop, and class-probability bars.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "device=%s  input=%s  checkpoint=%s  detailed=%s",
        device,
        args.input,
        args.checkpoint,
        args.detailed,
    )
    model, _ = load_trained_model(args.checkpoint, device)
    model.eval()
    transform = eval_transform()
    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    log.info("images: %d", len(paths))
    if args.detailed:
        _configure_korean_font()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pred", "confidence", "vis_path", *[f"prob_{c}" for c in CLASSES]])
        with torch.no_grad():
            for path in tqdm(paths, desc="infer", dynamic_ncols=True):
                original = Image.open(path).convert("RGB")
                x = transform(original).unsqueeze(0).to(device)
                prob = torch.softmax(model(x), dim=1)[0]
                pred = int(prob.argmax())
                processed = to_vis_image(x[0])
                vis_path = args.vis_dir / f"{path.stem}_{CLASSES[pred]}.jpg"
                if args.detailed:
                    save_detailed(vis_path, original, processed, prob, pred, path.name)
                else:
                    annotate(processed, f"{CLASSES[pred]}  {float(prob[pred]):.2f}").save(
                        vis_path, quality=95
                    )
                writer.writerow(
                    [
                        path,
                        CLASSES[pred],
                        f"{prob[pred]:.6f}",
                        vis_path,
                        *[f"{p:.6f}" for p in prob],
                    ]
                )
                log.info(
                    "%s  %s  (%.3f)  %s",
                    path.name,
                    CLASSES[pred],
                    float(prob[pred]),
                    vis_path.name,
                )
    log.info("wrote %s  and %d images in %s", args.output, len(paths), args.vis_dir)


if __name__ == "__main__":
    main()
