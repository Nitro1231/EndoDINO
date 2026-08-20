# EndoDINO

[English](README.md) | 한국어

GastroNet-5M DINOv2 ViT-B를 GastroHUN에서 23클래스 SSS(위 체계적 선별 프로토콜, Systematic Screening Protocol for the Stomach) 랜드마크 분류로 파인튜닝한 모델입니다. Kenshi Yao 관찰점 22개와 NA를 포함합니다.

## 클래스

각 관찰점 코드는 **벽** + **부위**입니다 (예: `G3` = 위체부 중상부의 대만). 벽: **A** 전벽 (anterior), **L** 소만 (lesser curvature), **P** 후벽 (posterior), **G** 대만 (greater curvature). 반전(5)과 위각(6)에는 대만 클래스가 없습니다.

![SSS 클래스 참고도](assets/sss_class_reference.jpg)

| 코드 | 영어 | 한국어 |
|------|------|--------|
| A1 | Antrum, Anterior | 전정부, 전벽 |
| L1 | Antrum, Lesser Curvature | 전정부, 소만 |
| P1 | Antrum, Posterior | 전정부, 후벽 |
| G1 | Antrum, Greater Curvature | 전정부, 대만 |
| A2 | Lower Body, Anterior | 위체부 하부, 전벽 |
| L2 | Lower Body, Lesser Curvature | 위체부 하부, 소만 |
| P2 | Lower Body, Posterior | 위체부 하부, 후벽 |
| G2 | Lower Body, Greater Curvature | 위체부 하부, 대만 |
| A3 | Upper-Middle Body, Anterior | 위체부 중상부, 전벽 |
| L3 | Upper-Middle Body, Lesser Curvature | 위체부 중상부, 소만 |
| P3 | Upper-Middle Body, Posterior | 위체부 중상부, 후벽 |
| G3 | Upper-Middle Body, Greater Curvature | 위체부 중상부, 대만 |
| A4 | Fundus/Cardia, Anterior | 위저부/분문부, 전벽 |
| L4 | Fundus/Cardia, Lesser Curvature | 위저부/분문부, 소만 |
| P4 | Fundus/Cardia, Posterior | 위저부/분문부, 후벽 |
| G4 | Fundus/Cardia, Greater Curvature | 위저부/분문부, 대만 |
| A5 | Upper-Middle Body Retroflex, Anterior | 위체부 중상부 반전, 전벽 |
| L5 | Upper-Middle Body Retroflex, Lesser Curvature | 위체부 중상부 반전, 소만 |
| P5 | Upper-Middle Body Retroflex, Posterior | 위체부 중상부 반전, 후벽 |
| A6 | Incisura, Anterior | 위각, 전벽 |
| L6 | Incisura, Lesser Curvature | 위각, 소만 |
| P6 | Incisura, Posterior | 위각, 후벽 |
| NA | Unqualified / Other | 부적합 / 기타 |

## 설정

```bash
pip install -e .
wandb login
```

파일을 다음과 같이 배치하세요:

```
weight/dinov2.pth              # GastroNet DINOv2 ViT-B
data/GastroHUN/                # 환자 폴더, metadata/, official_splits/
data/test/                     # 추론용 미라벨 프레임
```

- 모델 가중치: [GastroNet-5M DINOv2 ViT-B](https://cortex.thetavision.nl/dataset-provider/listing/2/)
- 데이터셋: [GastroHUN](https://www.nature.com/articles/s41597-025-04401-5)

공식 환자 단위 분할은 `data/GastroHUN/official_splits/image_classification.csv`에서 가져옵니다. 학습 기본값은 **4명 평가자 완전 일치**(논문 Scenario A)입니다: 3,722 / 793 / 803장. `OTHERCLASS`는 `NA`로 매핑됩니다.

## 학습

```bash
python -m endodino.train
```

[Weights & Biases](https://wandb.ai)에 기록합니다 (`endodino` 프로젝트). 건너뛰려면 `--no-wandb`를 사용하세요. 체크포인트는 검증 macro-F1 기준으로 순위가 매겨집니다:

```
outputs/checkpoints/top1.pt
outputs/checkpoints/top2.pt
outputs/checkpoints/top3.pt
```

전체 파인튜닝 대신 선형 프로브:

```bash
python -m endodino.train --freeze-backbone
```

다른 GastroHUN 라벨 열(평가자 또는 합의)을 사용하려면:

```bash
python -m endodino.train --label-column "Triple agreement"
```

## 평가

```bash
python -m endodino.evaluate --split test --checkpoint outputs/checkpoints/top1.pt
python -m endodino.evaluate --split val --checkpoint outputs/checkpoints/top1.pt
```

분류 리포트와 혼동 행렬을 `outputs/eval/`에 저장합니다.

## 추론

```bash
python -m endodino.infer --input data/test --checkpoint outputs/checkpoints/top1.pt
python -m endodino.infer --input data/test --checkpoint outputs/checkpoints/top1.pt --detailed
```

`--detailed`는 원본, 전처리된 크롭, 한·영 확률 막대 그래프를 저장합니다. 출력:

```
outputs/predictions.csv
outputs/predictions/*.jpg
```
