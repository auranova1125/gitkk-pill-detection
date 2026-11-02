# 분류 라벨 2차 검수 보고서

> 실행 기록: 2026-10-30. 대상은 v1 분류 패키지의 crop 14,988개와 원본 이미지 4,048장이다.

## 목적

v1 기준 모델의 validation 오분류 5건을 원본 이미지에서 확인한 결과, 모두 품목 라벨 교환 또는 잘못 놓인 bbox였다. 기존 원천 검증은 bbox 범위·크기·중복·겹침 같은 수치 규칙을 중심으로 하므로, 좌표 자체는 정상 범위지만 다른 알약이나 빈 배경에 놓인 bbox와 서로 뒤바뀐 이름표를 놓칠 수 있다.

2차 검수는 코드가 후보를 좁히고 사람이 원본과 검수 이미지를 확인해 최종 판정하는 방식으로 진행했다. 자동 점수만으로 데이터를 제외하지 않는다.

## 검수 방법

### bbox 위치 검수

`etl.classifier.label_audit.detect_bbox_errors`는 조명 그라데이션을 보정한 뒤 배경과 다른 영역 및 윤곽선으로 알약 후보를 찾고, 후보와 annotation bbox를 일대일로 대응한다.

주요 점수는 다음과 같다.

| 점수 | 의미 |
| --- | --- |
| `match_iou` | 대응한 알약 후보와 bbox의 IoU. 0이면 대응 후보가 없음 |
| `fg_in` | bbox 내부에서 알약 후보가 차지하는 비율 |
| `ring_fg` | bbox 바깥의 좁은 띠에서 검출된 알약 후보 비율 |
| `unmatched_blobs` | bbox와 대응하지 않은 알약 후보 수 |

### 클래스 검수

`etl.classifier.label_audit.detect_class_errors`는 기준 checkpoint에서 crop embedding을 추출해 다음 두 결과를 함께 본다.

1. 같은 조합 그룹의 다른 촬영 각도에서 동일 품목과 더 닮았는지 비교한다.
2. 그룹 단위 교차 검증으로 embedding 기반 품목을 예측한다.
3. 같은 이미지의 두 crop이 서로 상대 품목으로 판단되면 라벨 swap 후보로 표시한다.

기준 모델은 train 라벨을 외웠을 수 있으므로 checkpoint의 직접 예측만으로 train 오류를 판단하지 않는다.

### 사람 판정과 조치

| 판정 | 기준 |
| --- | --- |
| 오류 | bbox 안에 알약이 없거나 다른 알약에 놓임, 알약이 크게 잘림, bbox가 지나치게 큼, 실제 품목이 다름 |
| 정상 | 대상 알약이 bbox 안에 적절히 들어 있음. 경미한 경계 차이는 정상 처리 |

| 원천 | 조치 | 이유 |
| --- | --- | --- |
| AI Hub 조합 이미지 | 이미지 제외 | 대체 후보가 충분하고 다른 bbox도 함께 의심할 수 있음 |
| Kaggle 이미지 | bbox 제외 | 원천 이미지가 적어 정상 crop은 유지 |
| 정답을 확정할 수 있는 Kaggle swap | 라벨 교환 | 각인과 동일 그룹의 다른 각도로 품목을 확인할 수 있음 |

## 결과

### bbox 위치 검수

| 우선순위 | 후보 bbox | 오류 | 정상 |
| --- | ---: | ---: | ---: |
| 1 | 59 | 42 | 17 |
| 2 | 31 | 0 | 31 |
| 3 | 45 | 10 | 35 |
| 합계 | 135 | 52 | 83 |

오류는 48개 이미지의 bbox 52개였으며 train 36개, validation 16개였다. 주된 형태는 bbox 밀림과 다른 알약 위에 놓인 bbox였다. 붙어 있는 알약을 하나의 영역으로 인식하거나 색이 배경과 비슷한 알약을 찾지 못하는 경우는 자동 검출의 오탐이었다.

### 클래스 검수

| 우선순위 | 후보 crop | 오류 | 정상 |
| --- | ---: | ---: | ---: |
| 1 | 16 | 16 | 0 |
| 2 | 19 | 10 | 9 |
| 합계 | 35 | 26 | 9 |

오류 26개 중 19개는 위치 검수에서도 오류로 확인됐다. 클래스 검수에서만 확인한 오류는 crop 7개였으며, 라벨이 서로 바뀐 이미지와 다른 품목이 배치된 이미지가 포함됐다. 최종 병합 결과는 이미지 49장 제외, bbox 2개 제외, crop 2개 라벨 교환이다.

## 탐지 데이터 전달

동일한 원본 이미지를 탐지 학습에도 사용할 수 있으므로 `export_detector_actions`가 확정 조치를 `detector_actions.csv`로 변환한다. 현재 탐지기는 단일 `pill` 클래스이므로 분류 라벨 교환은 탐지 class 변경이 아니라 `drug_code` 메타데이터 수정으로 전달한다. bbox 제외는 해당 알약을 배경으로 학습하지 않도록 bbox 수정 또는 이미지 제외로 전달한다.

## 한계

- 코드가 고른 후보만 사람이 확인했으므로 후보 기준 아래의 오류가 남아 있을 수 있다.
- 같은 그룹의 다른 각도 crop이 없는 샘플은 그룹 비교를 사용할 수 없다.
- 클래스 embedding은 오류가 포함된 v1으로 학습한 checkpoint에서 추출했다.
- 경미한 bbox 밀림이나 잘림은 사람에 따라 판정이 달라질 수 있다.
- v2 학습 후 새 checkpoint로 후보 탐색을 반복하면 남은 오류를 추가로 찾을 수 있다.

## 재실행

```bash
uv run python -m etl.classifier.label_audit.detect_bbox_errors
uv run python -m etl.classifier.label_audit.detect_class_errors --checkpoint data/best.pt

# records/position_suspects.csv와 class_suspects.csv를 사람이 검토한 뒤 실행
uv run python -m etl.classifier.label_audit.merge_review_actions
uv run python -m etl.classifier.label_audit.export_detector_actions
```

재실행 시 기존 사람 판정 열은 `sample_id`를 기준으로 보존한다. embedding과 검수 이미지는 `data/classifier/meta/label_audit/`에 저장하고, 사람 판정 및 최종 조치 CSV는 `etl/classifier/label_audit/records/`에서 Git으로 관리한다.
