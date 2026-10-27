# 분류기 데이터 및 함수 입력 명세서

## 1. 목적

이 문서는 데이터 구성 코드가 경구약제 분류기에 전달해야 하는 함수와 데이터 형식을 정의한다. 아래 명세를 만족하면 `classifier.input_validation`의 입력 검증과 `classifier.train.run_training()`을 통과할 수 있다.

이 문서에서 `N`은 현재 실행이 기대하는 클래스 수를 뜻한다. 구현의 기본값과 현재 데이터 예시는 118이지만 고정된 절대값은 아니다. 클래스 수가 바뀌면 `expected_num_classes=N`을 전달하고, 같은 `N`을 `class_map`, `data_summary`, target과 모델 출력 전체에 일관되게 적용한다.

데이터 구성 파트가 최종적으로 제공해야 하는 값은 다음 네 가지다.

```python
train_loader, val_loader, class_map, data_summary
```

분류기 코드는 데이터 탐색, annotation 파싱, bbox crop, train/validation 분할을 수행하지 않는다. 이 작업은 모두 데이터 구성 파트의 책임이다.

## 2. 데이터 구성 함수

### 2.1 함수 경로

학습 노트북의 `data_factory_path`에는 다음 형식으로 함수 위치를 지정한다.

```python
data_factory_path = "data.classifier_data:build_classifier_inputs"
```

- `data.classifier_data`: Python에서 import 가능한 모듈 경로
- `build_classifier_inputs`: 해당 모듈 안의 함수 이름

### 2.2 함수 시그니처

데이터 구성 함수는 keyword argument로 `config`와 `device`를 받고 정확히 네 값을 반환해야 한다.

```python
from typing import Any

import torch
from torch.utils.data import DataLoader

from classifier import TrainConfig


def build_classifier_inputs(
    *,
    config: TrainConfig,
    device: torch.device,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    ...
    return train_loader, val_loader, class_map, data_summary
```

함수는 여러 번 호출해도 같은 분할과 같은 `class_map`을 반환해야 한다. 학습 노트북은 첫 batch를 검증한 후 sampler 상태에 영향을 주지 않도록 이 함수를 다시 호출한다.

### 2.3 `config` 사용

데이터 구성 함수는 최소한 다음 설정을 사용해야 한다.

| 필드 | 용도 |
| --- | --- |
| `config.batch_size` | 두 DataLoader의 batch 크기 |
| `config.seed` | 분할, sampler와 worker의 재현 가능한 seed |
| `config.class_map_path` | 원본 `class_map.json` 위치 |
| `config.data_summary_path` | 원본 `data_summary.json` 위치 |

`run_training()`은 `class_map_path`와 `data_summary_path`를 직접 읽지 않는다. 데이터 구성 함수가 파일을 읽어서 Python dict로 반환해야 한다. 학습 코드는 전달받은 dict를 실행 디렉터리에 다시 저장한다.

`device`는 DataLoader 옵션을 정할 때 사용한다. Dataset에서 반환하는 tensor는 CPU tensor여도 되며, 학습 코드가 선택된 장치로 이동한다.

권장 DataLoader 설정은 다음과 같다.

| 실행 장치 | `num_workers` | `pin_memory` |
| --- | ---: | --- |
| CUDA | 2 | `True` |
| MPS | 0 | `False` |
| CPU | 0 | `False` |

### 2.4 원본 annotation에서 샘플 만들기

공유 annotation JSON이 `images`, `annotations`, `categories` 배열로 구성된 경우 `annotations.image_id`와 `images.id`를 연결하고, annotation 하나를 알약 샘플 하나로 만든다.

| 분류기 데이터 | 원본 JSON |
| --- | --- |
| 이미지 파일 | `images.file_name` |
| 이미지 크기 | `images.width`, `images.height` |
| `item_seq` | `images.item_seq`를 문자열로 변환 |
| 품목명 | `images.dl_name` 또는 `categories.name` |
| `category_id` | `annotations.category_id` |
| `bbox_xywh` | `annotations.bbox` |
| `drug_dir` | `images.drug_dir` |

같은 조합 이미지가 품목별 annotation 파일에 반복될 수 있으므로 `file_name`이나 `image_id`만으로 알약 샘플을 중복 제거하지 않는다.

`group_id`는 파일명에서 첫 번째 `_` 앞부분을 사용하는 것을 기준으로 한다.

```text
file_name: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
group_id:  K-001900-016548-019607-029451
```

같은 `group_id`에 속한 샘플은 모두 train 또는 모두 validation 중 한쪽에만 배치한다.

## 3. 학습 및 Validation DataLoader

`train_loader`와 `val_loader`는 동일한 batch 구조를 반환해야 한다.

```python
batch = {
    "images": images,
    "targets": targets,
    "meta": meta,
}
```

추가 필드를 넣는 것은 허용되지만 위 세 필드는 반드시 있어야 한다.

### 3.1 `images`

| 항목 | 조건 |
| --- | --- |
| Python 타입 | `torch.Tensor` |
| dtype | `torch.float32` |
| shape | `[B, 3, 224, 224]` |
| 채널 순서 | RGB |
| 값 | ImageNet mean/std로 정규화된 유한한 값 |
| batch 크기 | `B >= 1` |

NaN 또는 Inf가 하나라도 포함되면 검증에 실패한다.

### 3.2 `targets`

| 항목 | 조건 |
| --- | --- |
| Python 타입 | `torch.Tensor` |
| dtype | `torch.int64` |
| shape | `[B]` |
| 값 범위 | `0..N-1` |

각 값은 `class_map["classes"]`의 `class_index`와 정확히 대응해야 한다.

### 3.3 `meta`

`meta`는 길이가 `B`인 `list[dict]`여야 한다. PyTorch 기본 `collate_fn`이 만드는 dict of lists 형식은 허용되지 않는다.

샘플마다 다음 필드가 필요하다.

```python
sample_meta = {
    "sample_id": "sample-000001",
    "image_id": "image-000001",
    "image_path": "/data/images/example.png",
    "category_id": 1900,
    "item_seq": "198700706",
    "bbox_xywh": [120.0, 80.0, 64.0, 48.0],
    "group_id": "K-001900-016548-019607-029451",
    "drug_dir": "앞면",
}
```

| 필드 | 타입 | 필수 조건과 의미 |
| --- | --- | --- |
| `sample_id` | `str` | 비어 있지 않은 샘플 식별자. annotation 단위로 고유하게 만드는 것을 권장 |
| `image_id` | `str` | 비어 있지 않은 원본 이미지 식별자 |
| `image_path` | `str` | 비어 있지 않은 원본 이미지 경로 |
| `category_id` | `int` | target 클래스의 `class_map.category_id`와 같아야 함 |
| `item_seq` | `str` | target 클래스의 `class_map.item_seq`와 같아야 하며 숫자로 변환하지 않음 |
| `bbox_xywh` | 숫자 4개 | 원본 이미지 좌표의 `[x, y, width, height]`; 모든 값은 유한하고 width와 height는 양수 |
| `group_id` | `str` | 비어 있지 않은 촬영 그룹 식별자 |
| `drug_dir` | `str \| None` | 앞면·뒷면 등의 방향 정보. 정보가 없으면 `None` |

`category_id`와 `item_seq`는 같은 샘플의 `target`이 가리키는 클래스 정보와 모두 일치해야 한다.

### 3.4 Dataset item과 `collate_fn` 예시

Dataset의 샘플 반환 형식은 자유롭게 정할 수 있지만, DataLoader의 최종 batch는 위 형식이어야 한다. 다음 형태를 권장한다.

```python
from typing import Any

import torch


def collate_classifier_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "targets": torch.tensor(
            [sample["target"] for sample in samples],
            dtype=torch.int64,
        ),
        "meta": [sample["meta"] for sample in samples],
    }
```

Dataset의 `__getitem__()`은 위 `collate_fn`을 기준으로 다음 형태를 반환한다.

```python
{
    "image": image_tensor,  # float32, [3, 224, 224]
    "target": class_index,  # Python int
    "meta": sample_meta,
}
```

## 4. `class_map`

모델 출력 index를 실제 품목 정보로 변환하는 고정 매핑이다. JSON 파일을 읽은 Python dict 형태로 전달한다.

아래 JSON은 현재 118종 데이터를 가정한 일부 예시다. 다른 클래스 수를 사용할 때는 `num_classes`와 클래스 항목 수를 함께 변경한다.

```json
{
  "num_classes": 118,
  "classes": [
    {
      "class_index": 0,
      "category_id": 1900,
      "item_seq": "198700706",
      "name": "보령부스파정 5mg"
    },
    {
      "class_index": 1,
      "category_id": 1901,
      "item_seq": "200000001",
      "name": "예시 품목"
    }
  ]
}
```

필수 조건은 다음과 같다.

- `num_classes`는 현재 실행이 기대하는 양의 정수 `N`이다.
- `classes`는 길이가 `N`인 list다.
- 각 항목에는 `class_index`, `category_id`, `item_seq`, `name`이 모두 있어야 한다.
- `class_index`는 정수이며 `0..N-1`을 각각 정확히 한 번 포함한다.
- `category_id`는 정수이며 전체 클래스에서 중복되지 않는다.
- `item_seq`는 비어 있지 않은 문자열이며 전체 클래스에서 중복되지 않는다.
- `name`은 비어 있지 않은 문자열이다.

`classes`는 `class_index` 오름차순으로 저장하는 것을 권장한다. 검증 코드가 순서를 강제하지는 않지만 사람이 확인하기 쉽고 파일 변경을 추적하기 편하다.

학습, 재개, 평가, 추론에는 내용이 동일한 `class_map`을 사용해야 한다. 키 순서는 hash에 영향을 주지 않지만 list 순서와 추가 필드를 포함한 실제 내용이 바뀌면 checkpoint 호환성 검증에 실패한다. 학습을 시작한 후에는 매핑 파일을 수정하지 않는 것이 원칙이다.

## 5. `data_summary`

데이터 분할과 클래스 coverage를 요약한 Python dict다.

아래 JSON도 현재 118종 데이터를 가정한 예시이며 클래스별 count는 일부만 표시한다.

```json
{
  "num_classes": 118,
  "num_train_samples": 4720,
  "num_val_samples": 1180,
  "train_class_counts": {
    "0": 40,
    "1": 40
  },
  "val_class_counts": {
    "0": 10,
    "1": 10
  },
  "train_group_count": 1200,
  "val_group_count": 300,
  "group_overlap_count": 0,
  "validation_class_indices": [0, 1],
  "split_hash": "분할 결과로 계산한 고정 hash"
}
```

실제 `train_class_counts`와 `val_class_counts`에는 문자열 키 `"0"`부터 `"N-1"`까지 모두 있어야 한다.

| 필드 | 타입 | 필수 조건 |
| --- | --- | --- |
| `num_classes` | `int` | `class_map.num_classes` 및 현재 실행의 `N`과 같음 |
| `num_train_samples` | `int` | 0 이상이며 `train_class_counts` 합과 같음 |
| `num_val_samples` | `int` | 0 이상이며 `val_class_counts` 합과 같음 |
| `train_class_counts` | `dict[str, int]` | `"0".."N-1"`을 모두 포함하고 모든 count가 1 이상 |
| `val_class_counts` | `dict[str, int]` | `"0".."N-1"`을 모두 포함하고 count가 0 이상 |
| `train_group_count` | `int` | 0 이상 |
| `val_group_count` | `int` | 0 이상 |
| `group_overlap_count` | `int` | 반드시 `0` |
| `validation_class_indices` | `list[int]` | validation count가 1 이상인 클래스 index와 정확히 일치 |
| `split_hash` | `str` | 비어 있지 않은 고정 분할 식별자 |

Validation에 샘플이 없는 클래스는 허용된다. 해당 클래스의 count를 `0`으로 기록하고 `validation_class_indices`에서는 제외한다. 평가 시 이 클래스의 F1은 `None`이 되며 Macro F1 계산에서는 제외된다.

입력 검증 자체는 `num_val_samples=0`도 허용하지만, 실제 학습의 validation 단계는 비어 있는 loader를 허용하지 않는다. 학습을 실행하려면 validation에 최소 한 개 이상의 샘플이 있어야 한다.

`split_hash`는 같은 train/validation 분할이면 항상 같은 값이어야 한다. 정렬된 train/validation 샘플 ID 또는 group ID와 분할 방식 버전을 직렬화한 뒤 SHA-256으로 계산하는 방식을 권장한다.

금지 데이터 제외 여부는 분류기 입력 검증에서 확인하지 않는다. 필요하면 `excluded_forbidden_count` 같은 추가 필드를 넣을 수 있지만, 제외 작업과 확인은 데이터 구성 파트에서 담당한다.

## 6. 전처리 요구사항

이 절의 조건 중 shape, dtype, NaN/Inf는 코드가 검사한다. crop 방식, 보간법, padding 색상, augmentation 구성은 최종 tensor만으로 판별할 수 없으므로 데이터 구성 코드와 리뷰에서 확인해야 한다.

### 6.1 공통 처리

1. annotation bbox `[x, y, width, height]`를 이미지 경계로 clip한다.
2. RGB 이미지로 crop한다.
3. 기준 데이터에서는 bbox 여백을 추가하지 않는다.
4. 정사각형 중앙 padding을 적용한다.
5. padding 영역은 RGB `(124, 116, 104)`로 채운다.
6. `224×224`로 resize한다.
7. 다음 ImageNet 통계로 normalize한다.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 6.2 학습 데이터

crop 후 정사각형 padding 전에 다음 회전을 적용한다.

```text
RandomRotation(-180°~180°, expand=True, bilinear)
```

회전으로 새로 생긴 영역도 RGB `(124, 116, 104)`로 채운다.

기준 학습에서는 다음 augmentation을 사용하지 않는다.

- horizontal flip
- vertical flip
- color jitter
- grayscale
- 강한 blur
- random erasing

특히 flip은 알약 각인을 거울상으로 바꿀 수 있으므로 제외한다.

### 6.3 Validation 데이터

Validation에는 random transform을 적용하지 않는다. 같은 샘플은 실행할 때마다 같은 tensor를 만들어야 한다.

```text
bbox clip → RGB crop → 중앙 정사각형 padding → resize → normalize
```

## 7. 학습 sampler

클래스 불균형을 완화하기 위해 `train_loader`에는 `WeightedRandomSampler` 사용을 권장한다.

```python
sample_weight = 1.0 / train_class_counts[str(class_index)]
```

설정은 다음과 같다.

```python
WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)
```

sampler를 사용하면 DataLoader의 `shuffle`은 반드시 `False`로 설정한다. `val_loader`에는 weighted sampler와 shuffle을 사용하지 않는다.

분류기는 sampler가 클래스 균형을 조정한다고 가정하므로 `CrossEntropyLoss`에 class weight를 추가로 적용하지 않는다.

## 8. 탐지 결과 분류용 DataLoader

Detector가 만든 bbox crop을 `predict_detections()`에 전달할 때는 target이 없는 별도 batch 형식을 사용한다.

```python
batch = {
    "images": images,
    "meta": [
        {
            "image_id": "image-000001",
            "detection_id": "det-000001",
            "bbox_xywh": [120.0, 80.0, 64.0, 48.0],
            "detector_score": 0.91,
            "detector_name": "detector-a",
        }
    ],
}
```

### 8.1 `images`

학습 batch와 동일하게 `torch.float32`, `[B, 3, 224, 224]`, ImageNet normalization 형식이어야 한다. 각 batch는 한 개 이상의 샘플을 포함하고 NaN 또는 Inf가 없어야 한다.

### 8.2 `meta`

| 필드 | 타입 | 필수 조건 |
| --- | --- | --- |
| `image_id` | `str` | 비어 있지 않음 |
| `detection_id` | `str` | 비어 있지 않음 |
| `bbox_xywh` | 숫자 4개 | 유한한 `[x, y, width, height]`, width와 height는 양수 |
| `detector_score` | `int \| float` | 유한한 `0.0..1.0` |
| `detector_name` | `str` | 비어 있지 않음 |

GT bbox로 detector 연결 형식을 평가할 때는 `detector_score=1.0`으로 둔다.

분류기는 다음 값을 생성한다.

```python
{
    "class_index": 0,
    "category_id": 1900,
    "item_seq": "198700706",
    "classifier_score": 0.88,
    "final_score": 0.8008,
}
```

```text
final_score = detector_score × classifier_score
```

## 9. 실행 전 입력 검증

데이터 구성 함수를 구현한 뒤 다음 코드가 오류 없이 실행되어야 한다.

```python
from classifier import (
    TrainConfig,
    select_device,
    validate_batch,
    validate_class_map,
    validate_data_summary,
)


device = select_device()
config = TrainConfig(...)
expected_num_classes = 118  # 현재 데이터 예시이며 필요하면 다른 N으로 변경
train_loader, val_loader, class_map, data_summary = build_classifier_inputs(
    config=config,
    device=device,
)

validate_class_map(
    class_map,
    expected_num_classes=expected_num_classes,
)
validate_data_summary(
    data_summary,
    class_map,
    expected_num_classes=expected_num_classes,
)
validate_batch(
    next(iter(train_loader)),
    class_map,
    expected_num_classes=expected_num_classes,
)
validate_batch(
    next(iter(val_loader)),
    class_map,
    expected_num_classes=expected_num_classes,
)
print("입력 검증 완료")
```

현재 118종 데이터에서는 `expected_num_classes`를 생략해도 기본값 118이 사용된다. 다른 클래스 수를 사용할 때는 검증뿐 아니라 학습·상세 평가·checkpoint 로딩에도 같은 값을 전달한다.

```python
training_result = run_training(
    config,
    train_loader,
    val_loader,
    class_map,
    data_summary,
    expected_num_classes=expected_num_classes,
)
```

첫 batch 검증으로 sampler 또는 random transform의 난수 상태가 바뀔 수 있다. 실제 학습 직전에는 데이터 구성 함수를 다시 호출해 loader를 새로 만드는 것을 권장한다. 학습 노트북은 이 과정을 이미 수행한다.

탐지 결과 batch는 다음처럼 검사한다.

```python
from classifier import validate_detection_batch


validate_detection_batch(next(iter(detection_loader)))
print("탐지 입력 검증 완료")
```

## 10. 데이터 구성 완료 체크리스트

- [ ] 데이터 구성 함수가 `config`, `device` keyword argument를 받는다.
- [ ] 함수가 `train_loader`, `val_loader`, `class_map`, `data_summary` 순서로 반환한다.
- [ ] 같은 seed로 호출하면 같은 split과 class map이 만들어진다.
- [ ] `class_map`에 `N`개 클래스가 있고 index가 `0..N-1`이다.
- [ ] 모든 클래스가 train에 한 장 이상 존재한다.
- [ ] train과 validation의 `group_id`가 겹치지 않는다.
- [ ] 클래스별 count 합이 전체 샘플 수와 같다.
- [ ] batch image가 `float32 [B, 3, 224, 224]`다.
- [ ] batch target이 `int64 [B]`다.
- [ ] `meta`가 dict of lists가 아니라 `list[dict]`다.
- [ ] target과 metadata의 `category_id`, `item_seq`가 일치한다.
- [ ] validation transform에 무작위 처리가 없다.
- [ ] `WeightedRandomSampler` 사용 시 `shuffle=False`다.
- [ ] 네 번의 입력 검증 호출이 모두 통과한다.

## 11. 자주 발생할 수 있는 실패

| 증상 | 확인할 내용 |
| --- | --- |
| `batch.meta must be a list` | 기본 `collate_fn` 대신 metadata list를 유지하는 함수 사용 |
| `batch.images must have shape [B, 3, 224, 224]` | resize, 채널 순서와 batch 차원 확인 |
| `batch.images must use torch.float32` | `float64`, `uint8` tensor가 아닌지 확인 |
| `batch.targets must use torch.int64` | `torch.long`으로 생성했는지 확인 |
| `category_id does not match target class` | Dataset의 target 생성과 class map 조회가 같은 매핑을 쓰는지 확인 |
| `item_seq does not match target class` | `item_seq`를 정수로 바꾸지 않았는지 확인 |
| `train_class_counts must ... greater than zero` | train에서 빠진 클래스가 있는지 확인 |
| `validation_class_indices must match ...` | val count가 1 이상인 index만 목록에 넣었는지 확인 |
| `group_overlap_count must be zero` | 이미지 단위가 아니라 촬영 조합 `group_id` 단위로 분할했는지 확인 |
| checkpoint class map hash 불일치 | 학습 후 class map 내용 또는 클래스 순서를 변경했는지 확인 |
| checkpoint split hash 불일치 | 학습 재개 시 다른 train/validation 분할을 전달했는지 확인 |
