# 경구약제 118종 분류기 구현 계획서

> **문서 상태:** 초기 구현 계획과 설계 결정 기록이다. 현재 실행 방법과 경로는 루트 [`README.md`](../../README.md), 데이터 전달 계약은 [분류기 입력 명세](../specs/classifier_input_spec.md)를 기준으로 한다.

## 1. 목표와 범위

조합 이미지에서 정답 또는 탐지 bbox로 잘라낸 알약을 118개 품목 중 하나로 분류하는 ResNet-50 모델을 구현한다.

이번 구현은 데이터 구성이 완료되어 `train_loader`, `val_loader`와 클래스 매핑이 제공되는 시점부터 시작한다. 금지 세트 제외, train/validation 분할, bbox crop과 입력 tensor 생성은 데이터 구성 파트에서 담당한다.

현재 후보 집합이 118종이므로 분류기는 처음부터 **118-way 단일 softmax 분류**로 학습한다. Seen/Unseen gate는 사용하지 않는다. 118종 밖의 품목을 예측해야 한다는 사실이 추후 확인되면 해당 품목의 데이터와 출력 클래스를 먼저 추가한 뒤 다시 판단한다.

### 산출물

| 산출물 | 내용 |
|---|---|
| Python 코드 | 모델, 학습, 평가, checkpoint, 추론 코드 |
| 학습 노트북 | 학습 시작·재개, best 모델 평가, 결과 시각화 |
| 모델 파일 | `last.pt`, `best.pt`, 필요 시 전체 데이터로 재학습한 `final.pt` |
| 실험 결과 | 설정, 학습 이력, validation 예측, 클래스별 평가와 오류 사례 |
| 보고서 | 데이터 요약, 모델 구성, 학습 조건, 결과와 한계 |

---

## 2. 전체 처리 흐름

```text
data team
  └─ train_loader / val_loader / class_map / data_summary
                            ↓
                118-way ResNet-50 학습
                            ↓
                 best checkpoint 선택
                            ↓
          Top-1 / Top-5 / macro F1 / 클래스별 분석
                            ↓
      탐지 bbox crop 입력 → category_id와 최종 score 생성
```

GT bbox로 측정한 결과는 분류기 자체의 성능을 평가하는 데 사용한다. 최종 모델의 유효성은 탐지기가 예측한 bbox를 입력했을 때 대회 end-to-end mAP@[0.75:0.95]에서도 확인한다.

---

## 3. 데이터 구성 파트에 요청할 입력

분류기 구현을 시작하기 전에 데이터 구성 파트로부터 다음 네 가지를 전달받는다.

1. `train_loader`
2. `val_loader`
3. `class_map.json`
4. `data_summary.json`

TS2 또는 test 데이터는 클래스 목록, 분할, 전처리 결정, 하이퍼파라미터 선택에 사용하지 않는다.

### 원본 JSON에서 필요한 값

공유된 annotation JSON은 `images`, `annotations`, `categories` 배열로 구성되어 있다. 데이터 구성 파트는 `annotations.image_id`와 `images.id`를 연결하여 알약 하나당 한 샘플을 만든다.

| 분류 입력 | 원본 JSON |
|---|---|
| 이미지 파일 | `images.file_name` |
| 이미지 크기 | `images.width`, `images.height` |
| `item_seq` | `images.item_seq`를 문자열로 변환 |
| 품목명 | `images.dl_name` 또는 `categories.name` |
| `category_id` | `annotations.category_id` |
| bbox | `annotations.bbox`의 `[x, y, width, height]` |
| 앞·뒷면 | `images.drug_dir` |
| 배경·조명 | `images.back_color`, `images.light_color` |
| 촬영각 | `images.camera_la`, `images.camera_lo` |

같은 조합 이미지가 품목별 annotation JSON에 반복될 수 있으므로 `file_name`이나 `image_id`만으로 중복 제거하지 않는다. annotation 하나를 알약 샘플 하나로 유지한다.

촬영 그룹은 파일명에서 첫 번째 `_` 앞부분으로 만든다. 예를 들어 아래 세부 촬영 조건이 달라도 같은 조합이면 동일한 `group_id`를 사용한다.

```text
file_name: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
group_id:  K-001900-016548-019607-029451
```

### 3.1 DataLoader batch 형식

`train_loader`와 `val_loader`는 같은 형태의 dict를 반환한다.

```python
batch = {
    "images": FloatTensor,      # [B, 3, 224, 224]
    "targets": LongTensor,     # [B], 값의 범위는 0..117
    "meta": list[dict],        # 길이 B
}
```

#### `images`

- RGB 이미지
- shape: `[batch_size, 3, 224, 224]`
- dtype: `torch.float32`
- ImageNet mean/std로 정규화된 tensor
- NaN 또는 Inf가 없어야 함

#### `targets`

- shape: `[batch_size]`
- dtype: `torch.int64`
- 값의 범위: `0..117`
- 값은 `class_map.json`의 `class_index`와 정확히 대응

#### `meta`

각 원소는 최소한 다음 필드를 포함한다.

```python
{
    "sample_id": str,
    "image_id": str,
    "image_path": str,
    "category_id": int,
    "item_seq": str,
    "bbox_xywh": [float, float, float, float],
    "group_id": str,
    "drug_dir": str | None,
}
```

`bbox_xywh`의 순서는 `[x, y, width, height]`이다. `item_seq`는 숫자로 변환하지 않고 문자열로 전달한다.

### 3.2 `class_map.json`

모델의 출력 index를 실제 품목으로 되돌리기 위한 고정 매핑이다.

```json
{
  "num_classes": 118,
  "classes": [
    {
      "class_index": 0,
      "category_id": 1900,
      "item_seq": "198700706",
      "name": "보령부스파정 5mg"
    }
  ]
}
```

다음 조건을 만족해야 한다.

- `num_classes`는 118이다.
- `class_index`는 중복 없이 `0..117`을 모두 포함한다.
- `category_id`와 `item_seq`는 각각 중복 없이 일대일 대응한다.
- 같은 매핑 파일을 학습, 평가, 추론에서 사용한다.

### 3.3 `data_summary.json`

분류기 코드가 입력 데이터의 상태를 확인할 수 있도록 다음 정보를 포함한다.

```json
{
  "num_classes": 118,
  "num_train_samples": 0,
  "num_val_samples": 0,
  "train_class_counts": {"0": 0},
  "val_class_counts": {"0": 0},
  "train_group_count": 0,
  "val_group_count": 0,
  "group_overlap_count": 0,
  "validation_class_indices": [],
  "excluded_forbidden_count": 0,
  "split_hash": "고정 split의 hash"
}
```

분류기 실행 전 다음을 검사한다.

- `num_classes == 118`
- 모든 클래스가 train에 최소 한 장 이상 존재
- `group_overlap_count == 0`
- train과 validation에 금지 데이터가 없음
- `class_map.json`과 DataLoader target이 일치

Validation 샘플이 없는 클래스가 있다면 해당 목록을 명시적으로 전달한다. 해당 클래스는 macro F1 계산에서 제외하고 validation coverage에 반영한다.

### 3.4 crop과 전처리 요구사항

공유된 annotation JSON에서 bbox는 `[x, y, width, height]` 형식이다. 데이터 구성 파트는 아래 순서로 입력을 만든다.

#### 학습 데이터

1. bbox를 이미지 경계로 clip한다.
2. 기준 데이터에서는 bbox 여백을 추가하지 않는다.
3. RGB 알약 crop에 `RandomRotation(-180°~180°, expand=True, bilinear)`을 적용한다.
4. 회전 후 긴 변을 기준으로 정사각형 중앙 패딩한다.
5. 회전·패딩 영역은 RGB `(124, 116, 104)`로 채운다.
6. `224×224`로 resize한다.
7. ImageNet mean/std로 normalize한다.

각인이 거울상으로 바뀌는 것을 피하기 위해 기준 데이터에는 horizontal/vertical flip을 적용하지 않는다. 색상이 품목 구분에 중요하므로 color jitter, grayscale, 강한 blur와 random erasing도 기준 데이터에는 적용하지 않는다.

#### Validation 데이터

1. bbox clip과 crop
2. RGB `(124, 116, 104)` 중앙 정사각형 패딩
3. `224×224` resize
4. ImageNet normalization

Validation에는 random transform을 사용하지 않는다.

### 3.5 클래스 불균형

클래스별 이미지 수가 약 4~50장으로 예상되므로 `train_loader`는 `WeightedRandomSampler`를 사용한다.

```text
각 샘플의 weight = 1 / 해당 클래스의 train 샘플 수
num_samples       = train dataset 길이
replacement       = True
shuffle           = False
drop_last         = False
```

`val_loader`는 sampler와 shuffle을 사용하지 않고 전체 validation 샘플을 한 번씩 반환한다.

---

## 4. 모델

### 구조

- `torchvision.models.resnet50`
- `ResNet50_Weights.IMAGENET1K_V2` 사전학습 weight
- 마지막 fully connected layer를 `Linear(2048, num_classes)`로 교체
- `num_classes`는 `class_map.json`에서 읽고 118인지 검사
- 모델 출력 shape: `[batch_size, 118]`

모델 생성 함수는 소형 테스트에서도 사용할 수 있도록 클래스 수를 인자로 받는다.

```python
build_model(num_classes: int, pretrained: bool = True) -> nn.Module
```

실제 학습에서는 `num_classes=118`, smoke test에서는 더 작은 값을 사용할 수 있다.

### 학습 단계

#### 1단계: 분류 head 학습

- epoch 0~2, 총 3 epoch
- ResNet-50 backbone은 `requires_grad=False`
- frozen BatchNorm layer는 eval mode로 유지
- FC layer만 학습
- AdamW, learning rate `1e-3`, weight decay `1e-4`

#### 2단계: `layer4`와 FC fine-tuning

- epoch 3부터 최대 epoch 29까지
- `layer4`와 FC만 `requires_grad=True`
- 그 이전 layer와 BatchNorm은 frozen 상태 유지
- AdamW parameter group
  - `layer4`: learning rate `1e-4`
  - FC: learning rate `5e-4`
- weight decay `1e-4`
- `CosineAnnealingLR(T_max=27)`
- early stopping은 2단계가 시작된 뒤부터 계산

1단계에서 2단계로 넘어갈 때 optimizer와 scheduler를 새로 생성한다. Checkpoint에는 현재 `training_stage`를 저장하여 재개할 때 동일한 단계와 optimizer 상태를 복원한다.

### Loss

`CrossEntropyLoss`를 사용한다. DataLoader의 sampler가 클래스 균형을 조정하므로 loss에 class weight를 중복 적용하지 않는다.

---

## 5. 초기 학습 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| input size | 224×224 |
| batch size | 32 |
| 최대 epoch | 30 |
| head-only epoch | 3 |
| optimizer | AdamW |
| loss | CrossEntropyLoss |
| best 기준 | validation macro F1 |
| 동률 처리 | Top-1이 높은 epoch 선택 |
| early stopping | 2단계에서 macro F1이 7 epoch 동안 개선되지 않을 때 |

`batch_size=32`에서 메모리 부족이 발생하면 16으로 낮춘다. 첫 기준 모델이 완료되기 전에는 다른 학습값을 임의로 변경하지 않는다.

### Colab과 MPS

장치는 다음 순서로 선택한다.

```text
CUDA → MPS → CPU
```

- CUDA에서는 automatic mixed precision과 GradScaler를 사용한다.
- MPS와 CPU에서는 float32로 실행한다.
- CUDA DataLoader는 `num_workers=2`, `pin_memory=True`를 기본값으로 한다.
- MPS DataLoader는 `num_workers=0`, `pin_memory=False`를 기본값으로 한다.
- checkpoint를 불러올 때 현재 장치에 맞춰 `map_location`을 지정한다.
- Colab의 `output_dir`는 Google Drive 아래의 영구 경로로 설정할 수 있어야 한다.

---

## 6. 코드 구성

```text
project/
├── configs/
│   ├── classifier.py
│   ├── classifier.yaml
│   └── yaml_loading.py
├── data_loading/
│   ├── classifier_dataset.py
│   ├── transforms.py
│   └── validation.py
├── classifier/
│   ├── input_validation.py
│   ├── training_data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── checkpoint.py
│   └── inference.py
├── utils/
│   ├── io.py
│   └── runtime.py
├── pipeline/
│   └── pill_recognition.py
├── notebooks/
│   └── train_classifier.ipynb
├── outputs/
├── reports/
│   └── classifier_report.md
└── tests/
    └── test_classifier.py
```

| 파일 | 역할 |
|---|---|
| `configs/classifier.py` | 분류기 학습·실행 설정과 YAML 로더 |
| `input_validation.py` | loader batch, class map, data summary 검사 |
| `training_data.py` | 데이터 factory 호출, 학습 입력 검증과 장치 선택 |
| `model.py` | ResNet-50 생성, freeze와 unfreeze |
| `train.py` | 2단계 학습, validation, early stopping |
| `evaluate.py` | 지표 계산, 예측과 오류 사례 저장 |
| `checkpoint.py` | 모델 저장과 중단 지점 재개 |
| `inference.py` | 품목 예측, class map 변환, 탐지 점수 결합 |
| `utils/runtime.py` | 장치 선택, seed 및 RNG 상태 설정·복원 |

`ClassifierConfig`에는 최소한 다음 필드를 둔다.

```python
run_name: str
output_dir: Path
class_map_path: Path
data_summary_path: Path
seed: int = 42
batch_size: int = 32
max_epochs: int = 30
head_only_epochs: int = 3
lr_head_only: float = 1e-3
lr_layer4: float = 1e-4
lr_fc: float = 5e-4
weight_decay: float = 1e-4
early_stopping_patience: int = 7
resume_from: Path | None = None
```

`run_training`은 DataLoader를 외부에서 전달받는다.

```python
run_training(
    config: ClassifierConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_map: dict,
    data_summary: dict,
) -> TrainingResult
```

이 구조를 사용하면 데이터 구성 코드의 위치와 내부 구현에 관계없이 동일한 분류기 코드를 사용할 수 있다.

---

## 7. 학습 노트북

`notebooks/baseline/train_classifier.ipynb`는 다음 순서로 실행한다.

1. 프로젝트 모듈 import
2. `ClassifierConfig` 생성
3. 설정된 batch size와 실행 장치에 맞춰 데이터 구성 파트의 함수로 `train_loader`, `val_loader`, `class_map`, `data_summary` 생성
4. 입력 계약 검사
5. 새 학습 또는 `resume_from` checkpoint 재개
6. `best.pt`를 불러와 validation 평가
7. loss, Top-1, Top-5, macro F1 곡선 표시
8. 클래스별 성능과 대표 오분류 확인

Dataset, transform, 모델 클래스, epoch loop, metric과 checkpoint 함수는 노트북 안에서 정의하지 않는다.

---

## 8. Checkpoint와 결과 저장

각 epoch 종료 시 `last.pt`를 저장하고, best 기준을 갱신하면 `best.pt`를 저장한다.

```text
outputs/{run_name}/
├── resolved_config.json
├── class_map.json
├── data_summary.json
├── last.pt
├── best.pt
├── metrics.csv
├── val_predictions.json
└── error_samples.json
```

Checkpoint에는 다음 상태를 포함한다.

- 완료한 epoch
- `training_stage`
- model state
- optimizer state
- scheduler state 또는 `None`
- CUDA GradScaler state 또는 `None`
- best macro F1과 Top-1
- random, NumPy, PyTorch RNG state
- class map hash
- data summary 또는 split hash

`resume_from`이 지정되면 저장된 다음 epoch부터 학습한다. 현재 class map이나 split hash가 checkpoint와 다르면 재개하지 않고 오류를 출력한다.

Colab에서 재개할 실험은 `output_dir`를 Google Drive 경로로 지정한다.

---

## 9. 평가

다음 지표를 validation 전체와 클래스별로 저장한다.

- validation loss
- Top-1 accuracy
- Top-5 accuracy
- macro F1
- 클래스별 precision, recall, F1, support
- validation class coverage
- 가장 자주 혼동되는 품목 쌍

Macro F1은 validation support가 한 장 이상인 클래스만 대상으로 계산한다.

```text
validation class coverage
= validation에 한 장 이상 존재하는 클래스 수 / 118
```

Validation에 존재하지 않는 클래스의 F1을 임의로 0으로 넣지 않고 `null`로 저장한다. 보고서에는 macro F1과 coverage를 함께 기재한다.

클래스별 데이터 수가 다르므로 전체 Top-1만으로 모델을 판단하지 않는다. Macro F1과 소수 클래스 recall을 함께 확인한다.

---

## 10. 최종 모델

Holdout validation으로 학습 방식과 사용할 epoch를 확정한 뒤 필요하면 train과 validation을 합친 전체 허용 데이터로 다시 학습한다.

- 재학습 epoch 수는 `best.pt`가 선택된 epoch 수를 사용한다.
- 모델과 optimizer 설정은 validation 실험에서 확정한 값을 그대로 사용한다.
- 전체 데이터 재학습 모델은 `final.pt`로 저장한다.
- 보고서의 성능 수치는 holdout `best.pt` 결과를 사용한다.
- 대회 추론에는 `final.pt`를 사용할 수 있다.

전체 데이터 재학습 여부는 detector와 연결한 validation 성능 및 제출 일정에 따라 결정한다.

---

## 11. 탐지기 연결 규격

분류기는 탐지기 종류와 관계없이 아래 정보를 전달받는다.

```python
{
    "images": FloatTensor,       # [B, 3, 224, 224]
    "meta": [
        {
            "image_id": str,
            "detection_id": str,
            "bbox_xywh": [float, float, float, float],
            "detector_score": float,
            "detector_name": str,
        }
    ]
}
```

분류 결과에는 다음 값을 추가한다.

```python
{
    "class_index": int,
    "category_id": int,
    "item_seq": str,
    "classifier_score": float,
    "final_score": float,
}
```

```text
final_score = detector_score × classifier_score
```

GT bbox 평가에서는 `detector_score=1.0`으로 둔다. `class_index`를 `category_id`로 변환하지 못하는 경우 해당 예측을 제거하지 않고 오류로 기록한다.

탐지 모델별 학습 loop와 NMS는 이번 구현 범위가 아니다. 세 탐지기는 동일한 crop DataLoader 형식으로 결과를 넘기면 된다.

---

## 12. 구현 순서

### 1단계. 입력 계약 검사

- 가짜 8-class DataLoader로 batch 검사 코드 작성
- 실제 loader 연결 후 shape, dtype, target 범위 검사
- class map과 data summary 검사
- group overlap과 validation coverage 확인

### 2단계. 모델과 학습 loop

- `build_model(num_classes)` 구현
- head-only 학습 구현
- `layer4`와 FC fine-tuning 구현
- CUDA와 MPS 장치 처리 구현

### 3단계. Checkpoint 재개

- `last.pt`, `best.pt` 저장
- 1단계 학습 중 재개 확인
- 2단계 학습 중 재개 확인
- stage 전환 후 optimizer와 scheduler 복원 확인

### 4단계. 118-way 기준 모델 학습

- 고정 train/validation loader 사용
- best checkpoint 선택
- validation 예측과 클래스별 오류 저장

### 5단계. 탐지 결과 연결

- 탐지 crop loader 입력
- class map을 이용한 `category_id` 변환
- 탐지 점수와 분류 점수 결합
- end-to-end evaluator에 전달할 결과 생성

### 6단계. 보고서 작성

- 데이터 구성과 클래스 불균형
- 모델과 학습 조건
- 학습 곡선과 validation 결과
- 클래스별 성능 및 대표 오류
- 탐지 bbox 연결 결과
- 한계와 필요한 후속 실험

---

## 13. 완료 조건

- 전달받은 DataLoader와 class map을 입력 계약에 따라 검증할 수 있다.
- 118-way ResNet-50의 forward와 backward가 Colab CUDA와 MPS에서 실행된다.
- Head-only 단계와 `layer4` fine-tuning 단계가 순서대로 실행된다.
- 중단된 학습을 `last.pt`에서 다음 epoch부터 재개할 수 있다.
- `best.pt`, 설정, 지표, validation 예측과 오류 사례가 저장된다.
- Top-1, Top-5, macro F1, validation coverage와 클래스별 recall이 보고된다.
- 탐지 crop DataLoader를 받아 `category_id`, `classifier_score`, `final_score`를 생성할 수 있다.
- `reports/classifier_report.md`에 실험 방법과 결과가 정리된다.
