from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSES = [
    "esophagus",
    "squamocolumnar_junction",
    "fundus",
    "body_antegrade",
    "body_retroflex",
    "angulus",
    "antrum",
    "duodenal_bulb",
    "descending_part_of_duodenum",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

CLASS_LABELS_EN = {
    "esophagus": "Esophagus",
    "squamocolumnar_junction": "Squamocolumnar Junction (SCJ)",
    "fundus": "Gastric Fundus",
    "body_antegrade": "Gastric Body, Antegrade",
    "body_retroflex": "Gastric Body, Retroflex",
    "angulus": "Gastric Angulus",
    "antrum": "Gastric Antrum",
    "duodenal_bulb": "Duodenal Bulb",
    "descending_part_of_duodenum": "Descending Part of the Duodenum",
}
CLASS_LABELS_KR = {
    "esophagus": "식도",
    "squamocolumnar_junction": "편평원주상피접합부",
    "fundus": "위저부",
    "body_antegrade": "위체부, 정방향",
    "body_retroflex": "위체부, 반전",
    "angulus": "위각절흔(위각)",
    "antrum": "위전정부",
    "duodenal_bulb": "십이지장 구부",
    "descending_part_of_duodenum": "십이지장 하행부",
}

IMAGE_SIZE = 336
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGES = REPO_ROOT / "data" / "UGIAD-dataset" / "images"
DEFAULT_UGIAD_SPLITS = REPO_ROOT / "data" / "UGIAD-dataset" / "splits"
DEFAULT_WEIGHTS = REPO_ROOT / "weight" / "dinov2.pth"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_TEST_DIR = REPO_ROOT / "data" / "test"
