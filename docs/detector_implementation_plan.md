# 단일 클래스 알약 탐지기 구현 계획서

> **문서 상태:** 초기 구현 계획과 설계 결정 기록이다. 현재 실행 방법과 경로는 루트 [`README.md`](../README.md), 데이터 전달 계약은 [`detector_input_spec.md`](detector_input_spec.md)를 기준으로 한다.

## 1. 목표

조합 이미지에서 알약의 위치만 찾는 단일 클래스 object detector를 구현한다. 탐지기는 세부 품목을 구분하지 않고 모든 알약을 `pill`로 예측한다. 예측 bbox는 기존 ResNet-50 분류기의 crop 입력으로 전달할 수 있게 한다.

```text
원본 이미지
  → SSDLite 단일 클래스 탐지
  → pill bbox와 detector score
  → bbox crop 및 분류기 입력 전처리
  → ResNet-50 세부 품목 분류
  → detector score × classifier score
```

현재 데이터에서는 한 이미지에 최대 4개의 알약이 있는 것으로 가정한다. 이 값은 데이터 특성에 관한 현재 기준이며 모델 구조의 고정값으로 사용하지 않는다.

## 2. 구현 범위

### 포함

- 외부에서 전달받은 `train_loader`, `val_loader`, label map과 데이터 요약 검증
- torchvision `SSDLite320 MobileNetV3-Large` 베이스라인
- 모델 교체가 가능한 builder registry
- 학습, validation, checkpoint 저장과 재개
- bbox mAP와 recall 기반 평가
- 이미지별 예측 bbox, score와 오류 분석 결과 저장
- 예측 bbox를 기존 분류기 입력으로 변환하는 연결 어댑터
- Notebook 기반 전체 실행 흐름
- 합성 데이터 단위 테스트와 MPS/CUDA/CPU 실행 경로
- 실험 결과 보고서 틀

### 제외

- 원본 annotation JSON 탐색과 DataLoader 구현
- train/validation 분할 생성
- 탐지기가 세부 약품명을 직접 예측하는 다중 클래스 학습
- detector와 classifier의 joint training
- 제출 파일 생성과 모델 ensemble
- 전체 데이터 최종 재학습 여부 결정

데이터 구성 코드는 별도 파트에서 제공하며, 분류기와 마찬가지로 탐지기 코드가 요구하는 입력 형식은 [`detector_input_spec.md`](detector_input_spec.md)에 명시한다.

## 3. 확인된 원본 데이터 구조

제공된 JSON은 필드 구조를 확인하기 위한 예시이며, 실제 전달 annotation은 이미지 안의 모든 알약 bbox를 포함하도록 수정된 COCO 계열 구조다.

```text
images[]
annotations[]
categories[]
```

- `annotations.bbox`: `[x, y, width, height]`
- `annotations.image_id`: `images.id` 참조
- `annotations.category_id`: 원래 약품 category
- `categories.supercategory`: `pill`
- 이미지 크기: `images.width`, `images.height`

탐지 학습에서는 원래 약품 `category_id`를 탐지 label로 사용하지 않는다. 모든 유효한 알약 annotation을 foreground label `1`로 변환한다. label `0`은 torchvision detector가 사용하는 background다.

실제 전달 데이터에서는 이미지 하나의 `annotations`에 그 이미지에서 보이는 모든 알약 bbox가 들어 있어야 한다. 탐지기 입력에서는 해당 annotation 목록 전체를 한 이미지의 target으로 변환한다. 탐지기 코드가 여러 품목별 JSON을 찾아 합치는 작업은 수행하지 않는다.

## 4. 베이스라인 모델

### 4.1 선택 모델

첫 베이스라인은 다음 모델로 한다.

```text
SSDLite320 MobileNetV3-Large
```

선택 이유는 다음과 같다.

- SSD 계열의 single-stage 구조라 팀원이 학습·추론 흐름을 이해하기 쉽다.
- torchvision에 모델 builder와 기본 후처리가 포함되어 있다.
- MobileNetV3 backbone이라 로컬 MPS와 Colab CUDA에서 빠르게 반복 실험하기 적합하다.
- 학습 loop가 torchvision의 다른 detector와 유사해 이후 모델 교체가 쉽다.

torchvision 모델의 `num_classes`는 background를 포함한다. foreground가 `pill` 하나이므로 모델에는 다음 값을 전달한다.

```python
num_classes = 2  # background 0 + pill 1
```

### 4.2 초기 가중치

기준 구현은 ImageNet 사전학습 MobileNetV3 backbone과 새 detection head를 사용한다.

```python
ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    num_classes=2,
)
```

COCO 전체 detector 가중치는 COCO 클래스 수에 맞는 head를 포함하므로 기준 구현에서 그대로 사용하지 않는다. COCO detector head 교체 실험은 baseline 이후 선택 실험으로 남긴다.

### 4.3 비교 모델

기준 모델의 위치 정밀도가 부족할 때 `SSD300 VGG16`을 동일한 입력·학습·평가 코드로 비교한다. 비교는 다음 항목으로 판단한다.

- 높은 IoU 구간 mAP
- pill recall
- 이미지당 추론 시간
- peak memory
- checkpoint 크기

SSD300 역시 높은 IoU 성능이 충분하지 않다면 SSD 계열 안에서 무리하게 조정하기보다 FPN 기반 detector를 후속 후보로 검토한다. 이 결정은 baseline 결과를 확인한 뒤 별도 승인받는다.

## 5. 공통 모델 인터페이스

학습과 평가 코드는 특정 모델 이름을 직접 참조하지 않는다. `detector/model.py`의 builder만 모델별 차이를 알고, 나머지는 torchvision detection 인터페이스를 사용한다.

### 학습 모드

```python
losses = model(images, targets)
total_loss = sum(losses.values())
```

- `images`: `list[Tensor[C, H, W]]`
- `targets`: `list[dict[str, Tensor]]`
- 반환값: 이름별 loss tensor mapping

### 평가·추론 모드

```python
predictions = model(images)
```

이미지마다 다음 형식의 dict를 반환한다고 가정한다.

```python
{
    "boxes": FloatTensor[K, 4],
    "labels": LongTensor[K],
    "scores": FloatTensor[K],
}
```

`boxes`는 원본 이미지 좌표계의 `[x1, y1, x2, y2]`다.

## 6. 모델 교체 구조

`DetectorConfig.model_name`과 builder registry를 사용한다.

```python
MODEL_BUILDERS = {
    "ssdlite320_mobilenet_v3_large": build_ssdlite320,
    "ssd300_vgg16": build_ssd300,
}
```

새 torchvision detector를 추가할 때는 다음 작업만 수행하도록 설계한다.

1. 공통 학습·추론 인터페이스를 만족하는 builder 작성
2. registry에 모델 이름 등록
3. 모델별 기본 입력 크기와 초기 가중치 설정
4. 합성 batch forward/backward 테스트 추가
5. config의 `model_name` 변경

Ultralytics처럼 호출 방식이 다른 모델을 나중에 추가할 경우에는 학습 loop를 분기하지 않고 공통 `DetectorBackend` 인터페이스를 구현하는 adapter를 추가한다.

## 7. 현재 파일 구조

```text
configs/
├── detector.py
├── detector.yaml
└── yaml_loading.py

detector/
├── __init__.py
├── input_validation.py
├── model.py
├── train.py
├── evaluate.py
├── checkpoint.py
└── inference.py

pipeline/
└── pill_recognition.py

notebooks/
└── train_detector.ipynb

docs/
├── detector_implementation_plan.md
└── detector_input_spec.md

reports/
└── detector_report.md

tests/
├── test_detector_input.py
├── test_detector_model.py
├── test_detector_training.py
├── test_detector_checkpoint.py
├── test_detector_evaluation.py
└── test_detector_inference.py
```

### 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `configs/detector.py` | 모델 이름, 학습률, epoch, threshold, 실행 경로와 YAML 로더 |
| `input_validation.py` | batch, label map, 데이터 요약 검증 |
| `model.py` | model registry와 SSDLite/SSD builder |
| `train.py` | 장치 선택, optimizer, epoch loop와 전체 학습 실행 |
| `evaluate.py` | mAP, recall, 이미지별 예측과 오류 분석 |
| `checkpoint.py` | last/best 저장, 호환성 확인과 재개 |
| `inference.py` | score filtering, 예측 직렬화와 시각화용 결과 생성 |
| `pipeline/pill_recognition.py` | 예측 bbox crop을 분류기 batch로 변환하고 두 모델의 결과를 결합 |

## 8. 설정 설계

`DetectorConfig`에는 최소한 다음 항목을 둔다.

```python
model_name: str = "ssdlite320_mobilenet_v3_large"
run_name: str
output_dir: Path
label_map_path: Path
data_summary_path: Path
seed: int = 42
batch_size: int = 8
max_epochs: int = 50
learning_rate: float
momentum: float = 0.9
weight_decay: float
score_threshold: float
nms_threshold: float
early_stopping_patience: int
resume_from: Path | None = None
```

학습률과 batch size는 데이터 크기와 장치 메모리를 확인한 뒤 pilot run에서 확정한다. torchvision의 COCO용 SSDLite 학습 recipe는 대규모 다중 GPU 조건이므로 그 값을 그대로 복사하지 않고 기준점으로만 사용한다.

모델별 기본값은 builder 또는 별도 profile에 두고 공통 config 구조는 유지한다.

## 9. 입력과 전처리 원칙

정확한 형식은 입력 명세서에서 정의한다. 핵심 원칙은 다음과 같다.

- DataLoader는 `images`, `targets`, `meta`를 반환한다.
- 이미지는 batch tensor가 아니라 원본 크기를 유지한 tensor list다.
- 이미지 tensor는 RGB, `float32`, `[3, H, W]`, 값 범위 `[0, 1]`이다.
- ImageNet normalization과 `320×320` resize는 SSDLite 내부 transform에 맡긴다.
- source bbox `[x, y, width, height]`는 DataLoader에서 `[x1, y1, x2, y2]`로 변환한다.
- 모든 bbox는 이미지 경계로 clip하고 면적이 0인 박스를 제거하거나 명시적으로 오류 처리한다.
- 모든 foreground label은 `1`이다.
- train과 validation은 촬영 그룹 단위로 분리하고 overlap은 0이어야 한다.
- 이미지별 source annotation 수와 target box 수가 정확히 같아야 한다.
- annotation ID, box, label, area와 iscrowd의 개수가 모두 일치해야 한다.

탐지 augmentation은 bbox도 함께 변환하는 라이브러리를 사용해야 한다. 첫 baseline은 과도한 합성보다 검증하기 쉬운 변환부터 시작한다.

### 학습 권장 변환

- dtype 변환과 `[0, 1]` scaling
- horizontal flip
- vertical flip
- 제한적인 photometric distortion

분류기와 달리 detector는 품목 각인을 판별하지 않으므로 flip으로 인한 거울상 자체는 문제되지 않는다. 다만 validation에는 random transform을 적용하지 않는다.

회전, random crop, scale jitter는 bbox clipping과 작은 객체 손실을 함께 검토해야 하므로 baseline 이후 ablation으로 추가한다.

## 10. 학습 방식

### 10.1 장치

기존 분류기와 같은 우선순위를 사용한다.

```text
CUDA → MPS → CPU
```

- CUDA: AMP와 GradScaler 사용
- MPS: float32 기준
- CPU: 테스트와 fallback

torchvision detection 모델은 장치별 연산 지원 차이가 있을 수 있으므로 구현 초기에 SSDLite forward/backward와 NMS smoke test를 실행한다. MPS에서 지원되지 않는 연산이 확인되면 학습은 CUDA를 기준으로 하고 로컬에서는 CPU smoke test를 제공한다.

### 10.2 optimizer와 scheduler

초기 baseline은 SGD 계열을 우선 검토한다.

```text
optimizer: SGD(momentum=0.9)
scheduler: CosineAnnealingLR 또는 MultiStepLR
```

실제 learning rate와 weight decay는 다음 순서로 확정한다.

1. 한 batch overfit test
2. 소규모 subset pilot
3. 전체 train baseline
4. validation 곡선을 보고 scheduler와 epoch 조정

### 10.3 한 epoch 학습

1. images와 target tensor를 장치로 이동
2. `model(images, targets)` 호출
3. 개별 loss가 모두 유한한지 확인
4. 전체 loss 합산
5. backward와 optimizer step
6. loss component별 평균 기록

gradient clipping은 baseline 결과에서 폭주가 확인될 때만 추가한다.

## 11. 평가 기준

### 11.1 Detector 단독 평가

다음 지표를 저장한다.

- COCO-style mAP@[0.50:0.95]
- AP@0.50
- AP@0.75
- 대회 지표에 맞춘 mAP@[0.75:0.95]
- mean average recall
- 이미지당 GT 개수와 예측 개수 차이
- score threshold 적용 전후 recall

`torchmetrics.detection.MeanAveragePrecision` 사용을 기준으로 하며 필요한 의존성 동작을 구현 단계에서 확인한다.

Best checkpoint의 1차 기준은 높은 IoU 구간 `mAP@[0.75:0.95]`로 한다. 동률이면 recall을 비교한다. 탐지기 loss는 학습 상태 확인용이며 best 선택 기준으로 사용하지 않는다.

### 11.2 연결 평가

탐지기와 분류기를 연결한 뒤 다음을 분리해서 기록한다.

- detector bbox recall
- GT bbox를 사용한 분류기 성능
- 예측 bbox를 사용한 분류기 성능
- 최종 `category_id` 기준 end-to-end mAP@[0.75:0.95]

이 구분을 통해 오류가 bbox 누락·위치 부정확인지, crop 분류 오류인지 판단한다.

### 11.3 threshold 선택

`score_threshold`와 `nms_threshold`는 validation에서 결정하고 보고서에 기록한다. 이미지당 최대 4개라는 이유만으로 예측을 처음부터 4개로 강제하지 않는다. 누락과 중복 예측을 먼저 관찰한 뒤 필요하면 후처리에서 최대 개수 제한을 비교한다.

## 12. Checkpoint와 실행 결과

```text
outputs/{run_name}/
├── resolved_config.json
├── label_map.json
├── data_summary.json
├── last.pt
├── best.pt
├── metrics.csv
├── val_predictions.json
├── error_samples.json
└── visualizations/
```

Checkpoint에는 다음 상태를 포함한다.

- checkpoint version
- model name과 모델 생성 설정
- 완료 epoch
- model state
- optimizer와 scheduler state
- GradScaler state
- best metric과 early stopping 상태
- RNG state
- label map hash
- data split hash
- torchvision version

모델 이름, label map 또는 split hash가 현재 실행과 다르면 자동 재개를 거부한다.

## 13. 기존 분류기 연결 범위

`pipeline/pill_recognition.py`가 두 모델 사이의 경계를 소유하며, `detector/`는 분류기 입력 형식에 의존하지 않는다.

입력:

- 원본 RGB 이미지
- detector의 `boxes`, `scores`, `labels`
- `image_id`, `detection_id`, detector 이름

처리:

1. `pill` label만 선택
2. score threshold 적용
3. bbox clip
4. 원본 이미지 crop
5. 분류기 기준의 정사각형 padding
6. `224×224` resize와 ImageNet normalization
7. 기존 `predict_detections()` 입력 batch 생성

출력은 기존 분류기 연결 규격을 그대로 사용한다.

```text
category_id
item_seq
detector_score
classifier_score
final_score
```

```text
final_score = detector_score × classifier_score
```

분류기 checkpoint 경로와 class map은 bridge가 소유하지 않고 호출자가 전달한다. 이렇게 하면 detector 단독 실험과 분류기 연결 실험을 독립적으로 실행할 수 있다.

## 14. 학습 Notebook

`notebooks/train_detector.ipynb`는 orchestration만 담당한다.

1. 프로젝트 import와 경로 설정
2. `DetectorConfig` 생성
3. `data_factory_path`를 통해 loader, label map과 data summary 생성
4. 입력 검증
5. 장치와 모델 확인
6. 신규 학습 또는 `last.pt` 재개
7. loss와 mAP 곡선 표시
8. `best.pt` validation 평가
9. 예측 bbox와 대표 오류 시각화
10. 선택적으로 classifier bridge smoke test

Dataset, augmentation, 모델 클래스, epoch loop와 metric 구현을 Notebook 안에 중복 작성하지 않는다.

## 15. 단계별 구현과 검토 지점

각 단계는 테스트 통과 후 사용자 검토를 받고 별도 커밋한다. Push는 전체 작업이 끝난 뒤 별도로 수행한다.

### 0단계. 계획과 입력 명세

- 구현 계획서 작성
- DataLoader, label map과 data summary 명세 작성
- 이미지별 annotation 완전성 요구사항 문서화

제안 커밋:

```text
chore: 탐지기 구현 계획 및 입력 명세 추가
```

### 1단계. 설정과 입력 검증

- `DetectorConfig`
- label map 검증
- data summary 검증
- 학습·validation batch 검증
- source annotation 수와 변환된 target box 수 일치 검증
- annotation ID 중복 및 누락 검증
- 합성 batch 테스트

제안 커밋:

```text
feat: 탐지기 설정 및 입력 검증 추가
```

### 2단계. 모델 registry와 SSDLite

- 공통 builder registry
- `SSDLite320 MobileNetV3-Large` 생성
- background 포함 클래스 수 처리
- train/eval forward smoke test
- MPS/CUDA/CPU 장치 경로 확인

제안 커밋:

```text
feat: SSDLite 탐지 모델 구성 추가
```

### 3단계. 학습과 평가

- optimizer와 scheduler
- 한 epoch 학습
- mAP와 recall 평가
- threshold와 이미지별 개수 분석
- empty loader와 non-finite loss 방어

제안 커밋:

```text
feat: 탐지기 학습 및 평가 루프 추가
```

### 4단계. Checkpoint와 학습 재개

- `last.pt`, `best.pt`
- RNG와 optimizer 상태 복원
- 모델 이름, label map, split 호환성 확인
- early stopping

제안 커밋:

```text
feat: 탐지기 체크포인트 및 학습 재개 추가
```

### 5단계. 추론과 분류기 연결

- detector 예측 직렬화
- bbox filtering과 오류 기록
- classifier batch 변환
- detector/classifier score 결합 확인

제안 커밋:

```text
feat: 탐지 결과 분류기 연결 추가
```

### 6단계. Notebook과 결과 문서

- `train_detector.ipynb`
- 학습 곡선과 bbox 시각화
- 결과 보고서 틀
- README 실행 안내
- 실제 SSDLite 소형 smoke test

제안 커밋:

```text
feat: 탐지기 학습 노트북 및 결과 저장 추가
```

### 선택 단계. SSD300 비교

SSDLite baseline 결과 검토 후 승인받아 진행한다.

- SSD300 builder 추가
- 같은 split과 metric으로 비교
- 속도, 메모리와 높은 IoU 성능 비교

제안 커밋:

```text
feat: SSD300 비교 실험 구성 추가
```

## 16. 테스트 계획

실제 DataLoader가 준비되기 전에는 합성 이미지와 bbox를 사용한다.

- 올바른 batch 검증 통과
- 잘못된 dtype, shape, bbox, label 거부
- 한 이미지의 여러 bbox 처리
- source annotation 수와 target bbox 수 불일치 거부
- 중복 annotation ID와 target 필드 길이 불일치 거부
- SSDLite 생성과 출력 head 클래스 수 확인
- CPU forward와 backward
- 장치 선택
- loss 집계와 non-finite 방어
- metric 집계
- checkpoint 저장과 재개
- 모델 이름 및 split 불일치 거부
- bbox crop과 분류기 batch 변환
- 빈 예측과 mapping 오류 처리

실제 loader 연결 후에는 다음 smoke test를 추가한다.

- train/validation 각 첫 batch 입력 검증
- 한 batch overfit
- MPS 또는 CUDA에서 한 epoch
- checkpoint 재개 후 한 epoch
- 예측 bbox 시각화

## 17. 주요 위험과 대응

### 이미지별 annotation 누락

실제 전달 annotation에는 이미지에서 보이는 모든 pill bbox가 있어야 한다. 일부 알약의 bbox가 빠지면 해당 알약이 background로 학습된다. 입력 검증은 source annotation 수, annotation ID 수와 target bbox 수를 비교해 변환 과정의 누락을 탐지한다. 다만 이미지 픽셀만 보고 원본 annotation 자체에 빠진 알약이 있는지 판단할 수는 없으므로 첫 batch와 무작위 샘플을 시각화해 원본 완전성도 확인한다.

### 높은 IoU 구간의 위치 정밀도

SSDLite320은 속도 중심 모델이라 mAP@[0.75:0.95]에서 한계가 있을 수 있다. AP@0.50만으로 모델을 판단하지 않고 AP@0.75와 높은 IoU 평균을 best 기준으로 사용한다.

### bbox 형식 혼동

원본은 XYWH이고 torchvision model target은 XYXY다. 입력 검증에서 `x2 > x1`, `y2 > y1`, 이미지 경계와 area 일치를 확인한다.

### score threshold로 인한 recall 감소

분류기에 전달하기 전에 너무 높은 threshold를 적용하면 복구할 수 없는 누락이 생긴다. threshold 미적용 예측을 저장하고 validation에서 recall과 중복을 함께 비교한다.

### 장치별 연산 지원

torchvision detection API는 beta 상태이며 장치별 지원 차이가 있을 수 있다. 모델 구현 직후 MPS forward/backward/NMS를 실제 실행하고, 실패 시 CUDA 학습과 CPU 로컬 테스트 경로를 명시한다.

## 18. 완료 기준

- 명세에 맞는 외부 DataLoader를 검증할 수 있다.
- SSDLite가 `pill=1`, background `0`으로 학습된다.
- CUDA, MPS, CPU 중 가능한 장치가 선택된다.
- epoch별 loss와 validation metric이 기록된다.
- `last.pt`와 `best.pt`에서 학습을 재개할 수 있다.
- validation 예측과 대표 오류를 저장하고 시각화할 수 있다.
- 예측 bbox를 기존 분류기 입력으로 변환할 수 있다.
- config의 `model_name` 변경으로 등록된 detector를 교체할 수 있다.
- 단위 테스트, lint와 format 검사가 통과한다.

## 19. 참고 자료

- [torchvision SSDLite320 MobileNetV3-Large 문서](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
- [TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [torchvision detection reference training recipes](https://github.com/pytorch/vision/tree/main/references/detection)
