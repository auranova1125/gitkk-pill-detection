# SSDLite320 · 탐지기 v2 · 기준선과 증강 A1 결과

이 문서는 2026-11-04~05에 Colab에서 수행한 SSDLite320 탐지기 실험의 결과 기록이다.
두 실행은 **증강과 early stopping patience만** 다르고 나머지 조건은 같다.
PR 정리 과정에서 다시 학습하지 않았다.

## 데이터·학습

| 항목 | 기준선 | A1 |
| --- | --- | --- |
| run | `20261104-ssdlite320-det-v2-baseline-01` | `20261104-ssdlite320-det-v2-aug-flipcolor-01` |
| 노트북 | [`train_detector_ssdlite320_v2_baseline_colab.ipynb`](../../notebooks/experiments/train_detector_ssdlite320_v2_baseline_colab.ipynb) | [`train_detector_ssdlite320_v2_aug_a1_colab.ipynb`](../../notebooks/experiments/train_detector_ssdlite320_v2_aug_a1_colab.ipynb) |
| 좌우 / 상하 뒤집기 | 0 / 0 | **0.5 / 0.5** |
| 밝기 / 대비 | 0 / 0 | **0.15 / 0.2** |
| early stopping patience | 7 | **10** |
| 최대 / 실제 epoch | 50 / 23 (early stopping) | 50 / 50 (최대 epoch 도달) |
| Best epoch | 16 | 44 |

공통 조건은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 모델 | SSDLite320 MobileNetV3-Large (`ssdlite320_mobilenet_v3_large`), backbone `IMAGENET1K_V1`, head 무작위 초기화 |
| Train / val | 8,143장 / 2,026장·7,771개 (탐지기 v2, PR #34) |
| split_hash | `ac6618cdf2fd69352b0d3848b69064eadbc6527d6153795c3cc2de1c2dfa7d32` |
| Optimizer | SGD · lr 0.005 · momentum 0.9 · weight decay 0.0005 |
| Batch / seed | 8 / 42 |
| Score / NMS threshold | 0.25 / 0.45 |
| Best 선택 기준 | val mAP@[0.75:0.95] |
| 장치 | Colab NVIDIA L4 |

epoch는 이 문서에서 1부터 센다. `metrics.csv`의 `epoch` 열은 0부터 센다
(기준선 best `epoch=15`, A1 best `epoch=43`).

### A1 증강을 고른 근거 (학습 전 기록)

- 뒤집기: flip-TTA에서 박스 중심이 1.2% 밀려, 위치 편향을 교정하려고 넣었다. 박스도 정확히 같이 변환된다.
- 밝기: test 밝기 중앙값 116, train 125(약 0.93배)의 차이를 덮는 범위로 정했다.
- 대비: test 대비(표준편차) 11.6~32.3이 train 범위와 비슷해서 약하게 넣었다.
- patience: 증강을 넣으면 수렴이 느려지는데, 기준선은 오르는 중에 멈췄다.
- 흐림·회전·크기/위치 변화는 넣지 않았다. train과 test 분포 차이가 작고, AP@0.95를 해칠 수 있기 때문이다.

### 채택 기준 (학습 전에 정함)

best mAP@[0.75:0.95]가 기준선 0.9402 + 0.01 = **0.9502 이상**이면 채택한다.
epoch 사이 흔들림이 ±0.01~0.02라서 그보다 작은 차이는 우연과 구분하기 어렵다.

## 결과

### 학습 기록 (val mAP@[0.75:0.95])

| 지표 | 기준선 | A1 | 차이 |
| --- | ---: | ---: | ---: |
| Best | 0.9402 | 0.9498 | +0.0096 |
| 상위 3 epoch 평균 | 0.9382 | 0.9489 | +0.0107 |

### Best epoch의 전체 지표 (`metrics.csv`)

| 지표 | 기준선 | A1 |
| --- | ---: | ---: |
| COCO mAP@[0.50:0.95] | 0.9691 | 0.9739 |
| AP@0.50 | 1.0000 | 1.0000 |
| AP@0.75 | 0.9894 | 0.9895 |
| mAP@[0.75:0.95] | 0.9402 | 0.9498 |
| Mean average recall | 0.9837 | 0.9870 |
| score 0.25 후 recall (IoU 0.5) | 0.9995 | 0.9999 |
| score 0.25 후 예측 수 / GT 7,771 | 7,767 | 7,770 |

### IoU 기준별 AP와 출처별 AP@0.95 (best.pt 재평가)

두 `best.pt`로 v2 val을 다시 예측해 pycocotools로 계산했다(score 0.25 적용 후, 단일 pill 클래스).
codeit은 대회 제공 사진, aihub는 AI Hub 사진이다.

| 모델 | 출처 | 이미지 | mAP@[.75:.95] | AP@0.75 | AP@0.80 | AP@0.85 | AP@0.90 | AP@0.95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기준선 | 전체 | 2,026 | 0.9404 | 0.9894 | 0.9892 | 0.9890 | 0.9846 | 0.7495 |
| 기준선 | codeit | 56 | 0.9308 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6539 |
| 기준선 | aihub | 1,970 | 0.9404 | 0.9894 | 0.9892 | 0.9890 | 0.9846 | 0.7499 |
| A1 | 전체 | 2,026 | 0.9476 | 0.9895 | 0.9891 | 0.9888 | 0.9863 | **0.7844** |
| A1 | codeit | 56 | 0.9444 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.7220** |
| A1 | aihub | 1,970 | 0.9477 | 0.9895 | 0.9891 | 0.9888 | 0.9863 | 0.7847 |

AP@0.75~0.90은 두 모델이 거의 같고, 차이는 대부분 **AP@0.95**에서 난다
(전체 0.7495 → 0.7844, codeit 0.6539 → 0.7220).

## 판단

- 사전 채택 기준(best +0.01)에는 **+0.0096으로 미달**했다.
- 상위 3 epoch 평균(+0.0107)과 AP@0.95(전체, codeit 모두 상승)는 A1이 낫다.
  뒤집기의 목적이던 박스 정밀도 개선과 방향이 맞는다.
- A1은 최대 50 epoch에서 끝났고 best가 44 epoch였다. early stopping으로 멈춘 것이 아니므로
  epoch를 늘리면 더 오를 여지가 있다.
- 채택 여부는 PR 리뷰에서 결정한다.

## Kaggle 제출

| 실행 | Public score |
| --- | ---: |
| 기준선 SSDLite320 v2 (`sub03_det_v2`) | 0.58765 |
| A1 | 제출하지 않음 (채택 기준 미달) |

기준선 점수는 A1 노트북 10절에 기록한 값이다.

## 해석 시 주의

- Val은 best epoch 선택에 사용했으므로 독립 test 성능이 아니다.
- codeit val은 56장뿐이라 AP@0.95 차이(+0.0681)의 신뢰 구간이 넓다.
- [YOLO11s v2 결과](detector_yolo11s_v2.md)의 고 IoU mAP(0.986151)와 숫자를 직접 비교하지 않는다.
  학습 중 평가 구현과 재평가 조건이 같은지 아직 확인하지 않았다.
- A1 노트북 7절은 Drive의 `metrics.csv`를 읽어 "51에폭 학습"으로 출력했다. 하지만
  학습 출력(`completed_epoch: 49`)과 로컬로 받은 `metrics.csv`(50행, epoch 0~49)는 50 epoch다.
  재개 과정에서 행이 하나 더 기록된 것으로 보이며, 원인은 확인하지 않았다.
  best와 상위 3 epoch 평균은 두 파일에서 같다.

## 재현과 보존 경로

1. 노트북을 Colab GPU 런타임에서 위에서부터 실행한다. 두 노트북 모두 main의 레포 코드(`uv run`, Python 3.12)로 학습한다.
2. 기준선은 공식 노트북 `notebooks/baseline/train_detector.ipynb`의 설정 셀만 바꿔 실행한다.
   A1은 `configs/detector.yaml`에서 증강·patience·경로만 바꾼 설정으로 `main.py train-detector`를 실행한다.
3. 결과는 개인 Drive `MyDrive/gitkk_runs/detector/<run>/`에 저장했다
   (`best.pt`, `last.pt`, `label_map.json`, `metrics.csv`, A1의 `ap_by_iou.csv`, `compare_baseline.png`).

이 저장소에는 가중치와 실행 출력을 올리지 않는다. 노트북 출력은 커밋할 때 nbstripout이 제거한다.

## 후속 확인

- A1 채택 여부 결정. 채택하면 Kaggle에 제출해 기준선 0.58765와 비교한다.
- A1의 최대 epoch를 늘린 재실행 검토.
- YOLO11s와 같은 공통 분류기·평가 조건으로 SSDLite320 e2e 비교.
- 결과 폴더를 팀 공용 `gitkk_archive/outputs/detector/`로 옮길지 결정.
