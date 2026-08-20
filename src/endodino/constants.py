from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSES = [
    "A1", "L1", "P1", "G1",
    "A2", "L2", "P2", "G2",
    "A3", "L3", "P3", "G3",
    "A4", "L4", "P4", "G4",
    "A5", "L5", "P5",
    "A6", "L6", "P6",
    "NA",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

CLASS_LABELS_EN = {
    "A1": "Antrum, Anterior wall (A1)",
    "L1": "Antrum, Lesser curvature (L1)",
    "P1": "Antrum, Posterior wall (P1)",
    "G1": "Antrum, Greater curvature (G1)",
    "A2": "Lower body, Anterior wall (A2)",
    "L2": "Lower body, Lesser curvature (L2)",
    "P2": "Lower body, Posterior wall (P2)",
    "G2": "Lower body, Greater curvature (G2)",
    "A3": "Upper-middle body, Anterior wall (A3)",
    "L3": "Upper-middle body, Lesser curvature (L3)",
    "P3": "Upper-middle body, Posterior wall (P3)",
    "G3": "Upper-middle body, Greater curvature (G3)",
    "A4": "Fundus/cardia, Anterior wall (A4)",
    "L4": "Fundus/cardia, Lesser curvature (L4)",
    "P4": "Fundus/cardia, Posterior wall (P4)",
    "G4": "Fundus/cardia, Greater curvature (G4)",
    "A5": "Upper-middle body retroflex, Anterior wall (A5)",
    "L5": "Upper-middle body retroflex, Lesser curvature (L5)",
    "P5": "Upper-middle body retroflex, Posterior wall (P5)",
    "A6": "Incisura, Anterior wall (A6)",
    "L6": "Incisura, Lesser curvature (L6)",
    "P6": "Incisura, Posterior wall (P6)",
    "NA": "Unqualified / not applicable (NA)",
}
CLASS_LABELS_KR = {
    "A1": "전정부, 전벽 (A1)",
    "L1": "전정부, 소만 (L1)",
    "P1": "전정부, 후벽 (P1)",
    "G1": "전정부, 대만 (G1)",
    "A2": "위체하부, 전벽 (A2)",
    "L2": "위체하부, 소만 (L2)",
    "P2": "위체하부, 후벽 (P2)",
    "G2": "위체하부, 대만 (G2)",
    "A3": "위체중상부, 전벽 (A3)",
    "L3": "위체중상부, 소만 (L3)",
    "P3": "위체중상부, 후벽 (P3)",
    "G3": "위체중상부, 대만 (G3)",
    "A4": "위저부/분문부, 전벽 (A4)",
    "L4": "위저부/분문부, 소만 (L4)",
    "P4": "위저부/분문부, 후벽 (P4)",
    "G4": "위저부/분문부, 대만 (G4)",
    "A5": "위체중상부 반전, 전벽 (A5)",
    "L5": "위체중상부 반전, 소만 (L5)",
    "P5": "위체중상부 반전, 후벽 (P5)",
    "A6": "위각부, 전벽 (A6)",
    "L6": "위각부, 소만 (L6)",
    "P6": "위각부, 후벽 (P6)",
    "NA": "부적합 / 해당 없음 (NA)",
}

LABEL_COLUMNS = (
    "Complete agreement",
    "Triple agreement",
    "FG agreement",
    "G agreement",
    "FG1 (Team A)",
    "FG2 (Team A)",
    "G1 (Team B)",
    "G2 (Team B)",
    "FG1-G1 agreement",
    "FG1-G2 agreement",
    "FG2-G1 agreement",
    "FG2-G2 agreement",
)
DEFAULT_LABEL_COLUMN = "Complete agreement"
OTHERCLASS = "OTHERCLASS"

IMAGE_SIZE = 336
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGES = REPO_ROOT / "data" / "GastroHUN"
DEFAULT_LABELS = DEFAULT_IMAGES / "official_splits" / "image_classification.csv"
DEFAULT_WEIGHTS = REPO_ROOT / "weight" / "dinov2.pth"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_TEST_DIR = REPO_ROOT / "data" / "test"
