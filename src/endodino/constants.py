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
    "A1": "Antrum, Anterior (A1)",
    "L1": "Antrum, Lesser Curvature (L1)",
    "P1": "Antrum, Posterior (P1)",
    "G1": "Antrum, Greater Curvature (G1)",
    "A2": "Lower Body, Anterior (A2)",
    "L2": "Lower Body, Lesser Curvature (L2)",
    "P2": "Lower Body, Posterior (P2)",
    "G2": "Lower Body, Greater Curvature (G2)",
    "A3": "Upper-Middle Body, Anterior (A3)",
    "L3": "Upper-Middle Body, Lesser Curvature (L3)",
    "P3": "Upper-Middle Body, Posterior (P3)",
    "G3": "Upper-Middle Body, Greater Curvature (G3)",
    "A4": "Fundus/Cardia, Anterior (A4)",
    "L4": "Fundus/Cardia, Lesser Curvature (L4)",
    "P4": "Fundus/Cardia, Posterior (P4)",
    "G4": "Fundus/Cardia, Greater Curvature (G4)",
    "A5": "Upper-Middle Body Retroflex, Anterior (A5)",
    "L5": "Upper-Middle Body Retroflex, Lesser Curvature (L5)",
    "P5": "Upper-Middle Body Retroflex, Posterior (P5)",
    "A6": "Incisura, Anterior (A6)",
    "L6": "Incisura, Lesser Curvature (L6)",
    "P6": "Incisura, Posterior (P6)",
    "NA": "Unqualified / Other (NA)",
}
CLASS_LABELS_KR = {
    "A1": "전정부, 전벽 (A1)",
    "L1": "전정부, 소만 (L1)",
    "P1": "전정부, 후벽 (P1)",
    "G1": "전정부, 대만 (G1)",
    "A2": "위체부 하부, 전벽 (A2)",
    "L2": "위체부 하부, 소만 (L2)",
    "P2": "위체부 하부, 후벽 (P2)",
    "G2": "위체부 하부, 대만 (G2)",
    "A3": "위체부 중상부, 전벽 (A3)",
    "L3": "위체부 중상부, 소만 (L3)",
    "P3": "위체부 중상부, 후벽 (P3)",
    "G3": "위체부 중상부, 대만 (G3)",
    "A4": "위저부/분문부, 전벽 (A4)",
    "L4": "위저부/분문부, 소만 (L4)",
    "P4": "위저부/분문부, 후벽 (P4)",
    "G4": "위저부/분문부, 대만 (G4)",
    "A5": "위체부 중상부 반전, 전벽 (A5)",
    "L5": "위체부 중상부 반전, 소만 (L5)",
    "P5": "위체부 중상부 반전, 후벽 (P5)",
    "A6": "위각, 전벽 (A6)",
    "L6": "위각, 소만 (L6)",
    "P6": "위각, 후벽 (P6)",
    "NA": "부적합 / 기타 (NA)",
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
