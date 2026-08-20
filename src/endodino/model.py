import timm
import torch
from torch import nn

from endodino.constants import IMAGE_SIZE, NUM_CLASSES


class LandmarkClassifier(nn.Module):
    def __init__(self, weights_path=None, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=IMAGE_SIZE,
        )
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            clean = {k.replace("backbone.", ""): v for k, v in state["teacher"].items()}
            self.backbone.load_state_dict(clean, strict=False)
        self.head = nn.Linear(self.backbone.num_features, NUM_CLASSES)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.head(self.backbone(x))
