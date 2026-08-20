# EndoDINO

GastroNet-5M DINOv2 ViT-B fine-tuned on UGIAD for 9-class EGD anatomical landmark classification.

## Setup

```bash
pip install -e .
wandb login
```

Place files as:

```
weight/dinov2.pth              # GastroNet DINOv2 ViT-B
data/UGIAD-dataset/            # images/ and splits/
data/test/                     # unlabeled frames for inference
```

- Model weights: [GastroNet-5M DINOv2 ViT-B](https://cortex.thetavision.nl/dataset-provider/listing/2/)
- Landmark labels: [FastUGI-Net](https://github.com/Nitro1231/FastUGI-Net)
- Dataset: [UGIAD](http://drive.google.com/file/u/1/d/1mrJiWXsGEDMog2uoM5EmBtjEtYbx8t2F/view?usp=drive_link)

## Train

```bash
python -m endodino.train
```

Logs to [Weights & Biases](https://wandb.ai) (`endodino` project). Use `--no-wandb` to skip. Checkpoints are ranked by validation macro-F1:

```
outputs/checkpoints/top1.pt
outputs/checkpoints/top2.pt
outputs/checkpoints/top3.pt
```

Linear probe instead of full fine-tune:

```bash
python -m endodino.train --freeze-backbone
```

## Evaluate

```bash
python -m endodino.evaluate --split test --checkpoint outputs/checkpoints/top1.pt
python -m endodino.evaluate --split val --checkpoint outputs/checkpoints/top1.pt
```

Writes a classification report and confusion matrix to `outputs/eval/`.

## Infer

```bash
python -m endodino.infer --input data/test --checkpoint outputs/checkpoints/top1.pt
python -m endodino.infer --input data/test --checkpoint outputs/checkpoints/top1.pt --detailed
```

`--detailed` saves original, processed crop, and bilingual probability bars. Outputs:

```
outputs/predictions.csv
outputs/predictions/*.jpg
```
