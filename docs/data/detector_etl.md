# 탐지 데이터 ETL

> **문서 상태:** 현재 저장소가 제공하는 탐지 데이터 오프라인 처리 범위를 설명한다.

## 책임

탐지 ETL은 원천 이미지와 이미지별 전체 bbox annotation을 읽어, runtime DataLoader가 소비하는 고정 학습 패키지를 만드는 오프라인 과정이다. 학습 중 이미지 증강이나 tensor 변환은 담당하지 않는다.

```text
Codeit 원천 → 원천 검증 ─┐
                         ├→ 전달된 학습 패키지 → 제외 목록 적용 → 최종 검증
기존 학습 패키지 ────────┘
```

## 현재 구현

| 파일 | 상태 | 책임 |
| --- | --- | --- |
| `etl/detector/codeit_source.py` | `codeit_train_validated`의 이미지·JSON 연결과 이미지별 전체 bbox 검증 |
| `etl/detector/apply_image_exclusions.py` | 확정 제외 목록을 패키지 복사본에 적용하고 metadata 갱신 |
| `etl/detector/training_package.py` | 완성 패키지를 runtime parser로 검증하고 label map·summary 기록 |

최초 SELECT 원천 변환, 여러 출처 병합과 최초 그룹 분할은 현재 저장소 범위에 포함하지
않는다. 이미 확정된 `split_manifest.csv`를 전달받아 보존하며 임의로 다시 나누지 않는다.

## 확정된 Codeit 입력

```text
codeit_train_validated/
├── annotations/
│   └── 이미지별 JSON
└── images/
    └── codeit_K-...png
```

각 JSON은 다음 조건을 만족해야 한다.

- `images`에는 정확히 한 이미지가 있다.
- `annotations`에는 그 이미지 파일명에 포함된 모든 약품 코드가 정확히 한 번씩 있다.
- 각 `bbox`는 `XYWH`이며 실제 이미지 경계 안에 있다.
- `area`는 bbox 면적과 일치하고 `iscrowd`는 `0`이다.
- `category_id`, `drug_code`, category 이름이 같은 약품 코드를 나타낸다.
- 전체 파일에서 이미지 ID와 annotation ID는 중복되지 않는다.

`Archive.zip`은 형식 확인 자료일 뿐 저장소 실행 의존성이 아니다. 실제 ETL 호출 시 원천 디렉터리 경로를 명시적으로 전달한다.

## 최종 학습 패키지

완성 출력은 [탐지 데이터 사용 안내](detector_data_guide.md)의 구조와 [탐지기 입력 명세](../specs/detector_input_spec.md)를 따라야 한다. `validate_training_package`는 runtime parser와 같은 내부 검사 경로로 annotation·manifest·이미지 연결을 검증하고 metadata를 계산한다. 완성된 패키지를 읽는 runtime DataLoader는 전달된 metadata와 같은 계산 결과를 비교한다. 이로써 ETL 생성과 실제 학습 입력의 검증 기준을 하나로 유지한다.

## Ultralytics YOLO용 파생 데이터

YOLO 학습은 기존 `annotations.json`과 다른 label text 형식을 요구한다. 이를 원천 데이터나 별도 split으로 관리하지 않고, 검증된 detector 학습 패키지에서 다시 만들 수 있는 파생 view로 취급한다.

```bash
uv run python -m data_loading.yolo \
  --data-dir data/detector
```

이미지는 복사하거나 이동하지 않는다. `split_manifest.csv`의 기존 train/validation 배정을 그대로 사용하면서 다음 파일만 학습 패키지 안에 생성한다.

```text
data/detector/
├── images/                    # 기존 이미지, 변경 없음
├── labels/                    # class 0(pill)의 정규화 YOLO bbox
├── yolo_train.txt             # 기존 train 이미지 절대 경로 목록
├── yolo_validation.txt        # 기존 validation 이미지 절대 경로 목록
├── yolo_dataset.yaml          # Ultralytics data 설정
└── yolo_export_summary.json   # source split hash와 변환 개수
```

다시 생성하려면 `--overwrite`를 명시한다. 이 옵션은 `yolo_export_summary.json`으로 exporter 소유권이 확인된 label과 목록만 교체하며, 원본 이미지·annotation·split manifest는 수정하지 않는다.
