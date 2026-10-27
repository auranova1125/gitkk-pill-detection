# 단일 클래스 탐지기 데이터 및 함수 입력 명세서

## 1. 목적과 범위

이 문서는 데이터 구성 코드가 단일 클래스 알약 탐지기에 전달해야 하는 함수와 데이터 형식을 정의한다.

탐지기는 모든 알약을 하나의 foreground 클래스 `pill`로 처리한다.

```text
background label = 0
pill label       = 1
model num_classes = 2
```

`num_classes=2`는 torchvision detection 모델이 background를 출력 클래스 수에 포함하기 때문이다. 원본 annotation의 약품별 `category_id`는 탐지 target label로 사용하지 않는다.

원본 JSON 탐색, 그룹 분할과 augmentation은 데이터 구성 파트가 담당한다. 탐지기 구현은 구성 완료된 DataLoader부터 시작한다.

## 2. 데이터 구성 함수

### 2.1 함수 경로

학습 Notebook은 다음 형식의 문자열로 데이터 구성 함수를 불러온다.

```python
data_factory_path = "data.detector_data:build_detector_inputs"
```

### 2.2 함수 시그니처

```python
from typing import Any

import torch
from torch.utils.data import DataLoader

from detector import DetectorConfig


def build_detector_inputs(
    *,
    config: DetectorConfig,
    device: torch.device,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    ...
    return train_loader, val_loader, label_map, data_summary
```

반환 순서는 다음과 같다.

1. `train_loader`
2. `val_loader`
3. `label_map`
4. `data_summary`

같은 seed와 데이터 버전으로 다시 호출하면 같은 split, label map과 summary가 만들어져야 한다. Notebook은 첫 batch를 검증한 뒤 실제 학습 전에 loader를 다시 생성할 수 있다.

### 2.3 권장 DataLoader 설정

| 장치 | `num_workers` | `pin_memory` |
| --- | ---: | --- |
| CUDA | 2 | `True` |
| MPS | 0 | `False` |
| CPU | 0 | `False` |

Dataset에서 반환하는 tensor는 CPU에 두고 학습 loop에서 선택된 장치로 이동한다.

## 3. 원본 JSON을 이미지 샘플로 구성하는 규칙

확인된 예시 JSON에는 `images`, `annotations`, `categories` 배열이 있다. 실제 전달 JSON은 같은 구조를 바탕으로 이미지 안의 모든 알약 bbox를 `annotations`에 포함하도록 수정된다.

| 탐지 데이터 | 원본 JSON |
| --- | --- |
| 이미지 파일 | `images.file_name` 또는 `images.imgfile` |
| 이미지 크기 | `images.width`, `images.height` |
| source image ID | `images.id` |
| bbox | `annotations.bbox` |
| annotation ID | `annotations.id` |
| source category | `annotations.category_id` |
| foreground 의미 | `categories.supercategory == "pill"` |

원본 bbox는 `[x, y, width, height]`다. Dataset은 이를 torchvision target에 필요한 `[x1, y1, x2, y2]`로 변환한다.

```python
x1 = x
y1 = y
x2 = x + width
y2 = y + height
```

### 3.1 이미지 하나에 완전한 annotation 목록 사용

실제 전달 annotation은 이미지 하나와 그 이미지에서 보이는 모든 알약 bbox를 함께 제공한다. Dataset은 JSON의 annotation 목록 전체를 한 이미지 target으로 변환한다.

```text
한 실제 이미지
  ├── pill bbox 1
  ├── pill bbox 2
  ├── pill bbox 3
  └── pill bbox 4
```

탐지기 또는 Dataset이 여러 품목별 JSON을 추가로 탐색해 bbox를 병합하지 않는다. 따라서 전달 시점에 이미지 속 일부 알약 bbox가 누락되지 않았는지 데이터 구성 단계에서 확인해야 한다. 빠진 알약은 학습 과정에서 background로 취급될 수 있다.

평가용 `image_id`는 전체 Dataset에서 유일한 정수여야 한다. 원본 값은 `source_image_id`로 보존한다.

### 3.2 그룹 분할

촬영 그룹은 파일명의 첫 번째 `_` 앞부분으로 만든다.

```text
file_name: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
group_id:  K-001900-016548-019607-029451
```

같은 `group_id`의 이미지는 모두 train 또는 모두 validation에 들어가야 한다. 두 split의 그룹 중복은 0이어야 한다.

## 4. DataLoader batch 형식

`train_loader`와 `val_loader`는 다음 dict를 반환한다.

```python
batch = {
    "images": images,
    "targets": targets,
    "meta": meta,
}
```

- `images`: 길이 `B`인 이미지 tensor list
- `targets`: 길이 `B`인 target dict list
- `meta`: 길이 `B`인 metadata dict list

세 list의 같은 위치는 반드시 같은 이미지를 나타내야 한다. 추가 필드는 허용하지만 위 세 필드는 필수다.

## 5. `images`

```python
images: list[torch.Tensor]
```

각 이미지 tensor 조건은 다음과 같다.

| 항목 | 조건 |
| --- | --- |
| dtype | `torch.float32` |
| shape | `[3, H, W]` |
| 채널 | RGB |
| 크기 | `H > 0`, `W > 0` |
| 값 범위 | `0.0..1.0` |
| 값 상태 | NaN과 Inf 없음 |

이미지는 서로 다른 높이와 너비를 가질 수 있으므로 `[B, 3, H, W]`로 강제 stack하지 않는다.

DataLoader는 ImageNet mean/std normalization을 적용하지 않는다. SSDLite 내부 transform이 normalization과 고정 입력 크기 변환을 수행한다. 외부에서 미리 `320×320`로 resize할 필요도 없다.

geometry augmentation을 적용했다면 bbox에도 동일한 변환이 반영되어야 한다.

## 6. `targets`

```python
targets: list[dict[str, Any]]
```

각 target은 다음 필드를 포함한다.

```python
target = {
    "boxes": boxes,
    "labels": labels,
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

현재 데이터에서는 이미지당 1~4개 알약이 예상되지만, tensor 형식은 일반적인 `K`개 객체를 지원해야 한다.

### 6.1 `boxes`

| 항목 | 조건 |
| --- | --- |
| dtype | `torch.float32` |
| shape | `[K, 4]` |
| 좌표 형식 | `[x1, y1, x2, y2]` |
| x 범위 | `0 <= x1 < x2 <= W` |
| y 범위 | `0 <= y1 < y2 <= H` |

모든 bbox는 이미지 경계로 clip한 뒤 양의 너비와 높이를 가져야 한다. clip 결과 면적이 0인 박스는 조용히 통과시키지 말고 데이터 오류로 기록하는 것을 권장한다.

### 6.2 `labels`

| 항목 | 조건 |
| --- | --- |
| dtype | `torch.int64` |
| shape | `[K]` |
| 값 | 모든 원소가 `1` |

`0`은 background이므로 Dataset target에 넣지 않는다. 원본 약품별 `category_id`도 target label로 넣지 않는다.

### 6.3 `image_id`

```python
image_id: int
```

전체 Dataset에서 유일한 0 이상의 정수다. train과 validation 각각의 평가 범위에서도 중복되면 안 된다.

### 6.4 `area`

| 항목 | 조건 |
| --- | --- |
| dtype | `torch.float32` |
| shape | `[K]` |
| 값 | `(x2 - x1) * (y2 - y1)` |

COCO-style metric에서 객체 크기별 평가에 사용한다. source JSON의 `area`를 그대로 신뢰하기보다 clip과 augmentation이 끝난 최종 bbox에서 다시 계산한다.

### 6.5 `iscrowd`

| 항목 | 조건 |
| --- | --- |
| dtype | `torch.int64` |
| shape | `[K]` |
| 값 | 일반 알약은 `0` |

source annotation의 `iscrowd`가 있다면 의미를 유지하되, 현재 정상 데이터에서는 모두 0을 예상한다.

## 7. `meta`

`meta`는 모델 학습 target과 분리한 추적·시각화·분류기 연결 정보다.

```python
sample_meta = {
    "image_id": 0,
    "source_image_id": "34",
    "file_name": "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png",
    "image_path": "/data/images/example.png",
    "group_id": "K-001900-016548-019607-029451",
    "original_size": [1280, 976],
    "source_annotation_count": 1,
    "annotation_ids": [133],
    "bbox_xywh": [[644.0, 845.0, 189.0, 190.0]],
}
```

| 필드 | 타입 | 조건과 의미 |
| --- | --- | --- |
| `image_id` | `int` | target의 `image_id`와 같음 |
| `source_image_id` | `str` | 원본 JSON ID를 문자열로 보존 |
| `file_name` | `str` | 비어 있지 않은 원본 파일명 |
| `image_path` | `str` | 비어 있지 않은 실제 이미지 경로 |
| `group_id` | `str` | 비어 있지 않은 촬영 그룹 ID |
| `original_size` | 정수 2개 | `[height, width]` |
| `source_annotation_count` | `int` | source JSON의 해당 이미지 annotation 수, 1 이상 |
| `annotation_ids` | `list[int]` | 이미지에 포함된 annotation ID 목록, 길이 `K` |
| `bbox_xywh` | 숫자 4개 list의 list | 원본 추적용 bbox 목록, 길이 `K` |

augmentation 이후 box 분석이 필요하면 `transformed_boxes_xyxy` 같은 추가 필드를 넣을 수 있다.

입력 검증은 다음 값이 모두 같은지 확인한다.

```text
source_annotation_count
= len(annotation_ids)
= len(bbox_xywh)
= len(target["boxes"])
= len(target["labels"])
= len(target["area"])
= len(target["iscrowd"])
```

`annotation_ids`는 이미지 안에서 중복되면 안 된다. 이 검증은 source annotation이 DataLoader target으로 변환되는 과정에서 사라지거나 중복된 항목을 찾는다.

이미지 픽셀을 분석해 실제 알약 수를 새로 추론하는 검증은 수행하지 않는다. 따라서 source JSON 자체가 모든 알약을 표시했는지는 데이터 생성 단계와 bbox 시각화 점검에서 확인해야 한다.

## 8. `collate_fn`

Detection 입력은 이미지마다 크기와 객체 수가 다르므로 기본 stack 방식 대신 list를 유지한다.

```python
from typing import Any


def collate_detector_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in samples],
        "targets": [sample["target"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }
```

Dataset의 `__getitem__()`은 다음 형태를 권장한다.

```python
{
    "image": image_tensor,
    "target": target,
    "meta": sample_meta,
}
```

## 9. `label_map`

```json
{
  "num_foreground_classes": 1,
  "num_model_classes": 2,
  "background_label": 0,
  "classes": [
    {
      "label": 1,
      "name": "pill"
    }
  ]
}
```

필수 조건은 다음과 같다.

- foreground 클래스는 현재 `pill` 하나다.
- background label은 `0`이다.
- pill label은 `1`이다.
- `num_model_classes`는 background를 포함해 `2`다.
- 학습, 평가와 추론에서 내용이 같은 파일을 사용한다.

구조는 이후 foreground 클래스가 추가되더라도 확장할 수 있게 list를 사용하지만, 현재 baseline은 단일 클래스만 허용한다.

## 10. `data_summary`

```json
{
  "num_train_images": 1000,
  "num_val_images": 200,
  "num_train_source_annotations": 2500,
  "num_val_source_annotations": 500,
  "num_train_instances": 2500,
  "num_val_instances": 500,
  "train_group_count": 250,
  "val_group_count": 50,
  "group_overlap_count": 0,
  "train_empty_image_count": 0,
  "val_empty_image_count": 0,
  "train_max_instances_per_image": 4,
  "val_max_instances_per_image": 4,
  "train_label_counts": {"1": 2500},
  "val_label_counts": {"1": 500},
  "split_hash": "고정 분할 hash"
}
```

숫자는 형식 설명을 위한 예시다. 현재 데이터의 실제 집계값을 넣어야 한다.

필수 검증 조건은 다음과 같다.

- 이미지 수와 instance 수는 0 이상의 정수다.
- 실제 학습 실행 시 train과 validation 이미지가 각각 한 장 이상이다.
- `num_train_source_annotations == num_train_instances`다.
- `num_val_source_annotations == num_val_instances`다.
- `num_train_instances == sum(train_label_counts.values())`다.
- `num_val_instances == sum(val_label_counts.values())`다.
- label count key는 pill label `"1"`이다.
- `group_overlap_count == 0`이다.
- empty image를 허용하지 않는 현재 데이터 정책이면 empty count가 0이다.
- max instances 값은 각 split의 실제 최대값이다.
- `split_hash`는 비어 있지 않고 같은 분할에서 재현 가능하다.

`split_hash`는 정렬한 image ID 또는 group ID와 분할 방식 버전을 직렬화한 뒤 SHA-256으로 계산하는 방식을 권장한다.

## 11. augmentation 요구사항

### 학습

첫 baseline은 bbox 변환이 검증된 범위에서 다음을 허용한다.

- horizontal flip
- vertical flip
- 제한적인 brightness/contrast 또는 photometric distortion
- float32 변환과 `[0, 1]` scaling

geometry 변환은 이미지와 bbox에 동시에 적용해야 한다. Albumentations를 사용한다면 bbox format 입력과 출력이 올바른지 샘플 시각화로 확인한다.

### Validation

Validation에는 random transform을 적용하지 않는다.

```text
RGB 변환 → float32 → [0, 1] scaling
```

SSDLite가 내부에서 resize와 normalization을 수행하므로 DataLoader에서 중복 적용하지 않는다.

## 12. 추론 DataLoader

정답이 없는 이미지 추론에는 `targets`를 생략하고 다음 형식을 사용한다.

```python
batch = {
    "images": list[torch.Tensor],
    "meta": list[dict],
}
```

`images`와 `meta` 조건은 학습 batch와 같다. `meta`에는 최소한 다음 정보가 있어야 한다.

```python
{
    "image_id": int,
    "file_name": str,
    "image_path": str,
    "original_size": [height, width],
}
```

## 13. 탐지기 출력 형식

모델의 이미지별 raw prediction은 다음 형식이다.

```python
prediction = {
    "boxes": boxes,
    "labels": labels,
    "scores": scores,
}
```

| 필드 | dtype | shape | 조건 |
| --- | --- | --- | --- |
| `boxes` | `torch.float32` | `[P, 4]` | 원본 이미지 좌표의 XYXY |
| `labels` | `torch.int64` | `[P]` | pill 예측은 `1` |
| `scores` | `torch.float32` | `[P]` | `0.0..1.0` |

저장용 예측은 bbox를 원본 annotation과 같은 XYWH로도 함께 변환한다.

```python
{
    "image_id": 0,
    "detection_id": "0-0001",
    "bbox_xyxy": [644.0, 845.0, 833.0, 1035.0],
    "bbox_xywh": [644.0, 845.0, 189.0, 190.0],
    "label": 1,
    "label_name": "pill",
    "detector_score": 0.91,
    "detector_name": "ssdlite320_mobilenet_v3_large"
}
```

## 14. 기존 분류기로 전달할 정보

예측 bbox에서 crop을 만든 뒤 기존 `classifier.predict_detections()`에 다음 batch를 전달한다.

```python
batch = {
    "images": classifier_images,
    "meta": [
        {
            "image_id": "0",
            "detection_id": "0-0001",
            "bbox_xywh": [644.0, 845.0, 189.0, 190.0],
            "detector_score": 0.91,
            "detector_name": "ssdlite320_mobilenet_v3_large",
        }
    ],
}
```

`classifier_images`는 기존 분류기 명세에 맞는 `float32 [B, 3, 224, 224]` tensor다. detector용 `[0, 1]` 이미지 tensor를 그대로 전달하지 않고 다음 처리를 적용한다.

1. 예측 bbox clip
2. RGB crop
3. RGB `(124, 116, 104)` 중앙 정사각형 padding
4. `224×224` resize
5. ImageNet normalization

연결 어댑터는 crop과 batch 생성을 담당하지만 분류기 checkpoint를 자동으로 선택하거나 로드하지 않는다.

## 15. 실행 전 입력 검증 예시

```python
from detector import (
    DetectorConfig,
    select_device,
    validate_detection_batch,
    validate_detector_data_summary,
    validate_detector_label_map,
)


device = select_device()
config = DetectorConfig(...)
train_loader, val_loader, label_map, data_summary = build_detector_inputs(
    config=config,
    device=device,
)

validate_detector_label_map(label_map)
validate_detector_data_summary(data_summary, label_map)
validate_detection_batch(next(iter(train_loader)), label_map)
validate_detection_batch(next(iter(val_loader)), label_map)
print("탐지기 입력 검증 완료")
```

첫 batch 확인으로 sampler 또는 augmentation 난수 상태가 바뀔 수 있으므로 실제 학습 전 loader를 다시 생성하는 것을 권장한다.

## 16. 데이터 구성 완료 체크리스트

- [ ] 함수가 `config`, `device` keyword argument를 받는다.
- [ ] 반환 순서가 loader 두 개, label map, data summary 순서다.
- [ ] 각 이미지 annotation에 보이는 모든 pill bbox가 포함되어 있다.
- [ ] JSON의 전체 annotation 목록이 한 이미지 target으로 변환된다.
- [ ] `source_annotation_count`가 target bbox 수와 같다.
- [ ] annotation ID가 이미지 안에서 중복되지 않는다.
- [ ] boxes, labels, area와 iscrowd의 첫 번째 차원 길이가 모두 같다.
- [ ] 평가용 `image_id`가 Dataset 전체에서 유일하다.
- [ ] train과 validation의 `group_id`가 겹치지 않는다.
- [ ] image는 RGB `float32 [3, H, W]`, 값 범위 `[0, 1]`이다.
- [ ] images를 강제로 batch tensor로 stack하지 않는다.
- [ ] source XYWH bbox를 target XYXY bbox로 변환했다.
- [ ] 모든 bbox를 경계로 clip했고 양의 면적을 가진다.
- [ ] 모든 target label이 `1`이다.
- [ ] target별 `boxes`, `labels`, `area`, `iscrowd` 길이가 같다.
- [ ] image count와 instance count를 구분해 summary에 기록했다.
- [ ] validation transform에는 무작위 처리가 없다.
- [ ] 입력 검증 호출이 모두 통과한다.

## 17. 자주 발생할 수 있는 실패

| 증상 | 확인할 내용 |
| --- | --- |
| 다른 알약이 background로 학습됨 | 전달 annotation에서 이미지 속 pill bbox가 빠지지 않았는지 확인 |
| source annotation 수 불일치 | JSON annotation이 Dataset 변환 중 누락되거나 중복되지 않았는지 확인 |
| target 필드 길이 불일치 | boxes, labels, area와 iscrowd를 같은 annotation 목록에서 생성했는지 확인 |
| `labels`에 0 또는 약품 ID가 있음 | 모든 pill label을 정수 1로 변환 |
| bbox 너비 또는 높이가 음수 | XYWH를 XYXY로 변환한 식과 좌표 순서 확인 |
| bbox가 이미지 밖에 있음 | 변환과 augmentation 이후 `[0, W]`, `[0, H]`로 clip |
| `images` shape 오류 | `[B, C, H, W]`가 아니라 `list[[C, H, W]]`인지 확인 |
| 입력 값 범위 오류 | uint8을 float32 `[0, 1]`로 변환했는지 확인 |
| 탐지가 전혀 학습되지 않음 | ImageNet normalization을 DataLoader와 모델에서 중복 적용하지 않았는지 확인 |
| COCO 평가에서 image ID 충돌 | raw JSON ID 대신 Dataset 전체 고유 ID를 생성했는지 확인 |
| group overlap 오류 | 파일 단위가 아니라 촬영 그룹 단위로 split했는지 확인 |
| instance count 불일치 | source annotation 수와 변환된 target box 수 비교 |

## 18. 공식 API 참고

torchvision의 SSDLite builder는 입력 크기 320 기반 모델이며 `num_classes`에 background를 포함한다. torchvision detection tutorial은 image list와 XYXY box, label 0의 background 의미를 설명한다.

- [torchvision SSDLite320 MobileNetV3-Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
- [TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

