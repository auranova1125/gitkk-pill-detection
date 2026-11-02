# 분류 데이터 ETL

> **문서 상태:** 원천 이미지와 annotation에서 분류기 학습 패키지를 만드는 현재 오프라인 ETL 안내다.

## 처리 흐름

`etl/classifier/`는 다음 단계를 순서대로 실행한다.

| 순서 | 모듈 | 책임 |
| --- | --- | --- |
| 1 | `validate_source_labels.py` | 원천 annotation과 bbox를 검증하고 제외 대상을 기록 |
| 2 | `select_source_images.py` | 클래스·촬영 각도 기준으로 사용할 이미지 선정 |
| 3 | `create_pill_crops.py` | bbox에 여백을 적용해 확인용 crop 생성 |
| 4 | `assign_data_splits.py` | 탐지기 split manifest 기준으로 train/validation 배정 |
| 5 | `summarize_crops.py` | 클래스별 crop 수와 분할 결과 집계 |
| 6 | `export_training_package.py` | runtime Dataset이 읽을 학습 패키지 생성 |

`build_classifier_dataset.py`는 위 단계를 연결하는 실행 진입점이다. 단계 구현은 공통 crop 함수가 있는 `data_loading.image_transforms`를 재사용하지만, runtime 학습 코드에서는 ETL 모듈을 import하지 않는다.

## 원천 데이터 배치

기본 경로는 `etl/classifier/paths.py`에 정의되어 있다. 데이터를 다른 곳에 보관하면 `AIHUB_DIR`, `KAGGLE_DIR` 환경 변수로 위치를 지정한다.

```text
data/
├── aihub/                         AI Hub 이미지·라벨 zip
├── kaggle/
│   ├── train_images/
│   └── train_annotations/
└── detector/
    └── training_package_detector.zip
```

탐지기 패키지의 `split_manifest.csv`를 사용하면 동일한 원본 그룹이 분류기 train과 validation에 나뉘는 것을 막을 수 있다.

## 실행

프로젝트 루트에서 전체 ETL을 실행한다.

```bash
uv run python -m etl.classifier.build_classifier_dataset
```

주요 옵션은 다음과 같다.

- `--detector-package`: 탐지기 학습 패키지 경로
- `--target`: 클래스별 선정 목표 수
- `--skip-png-validation`: 원천 PNG 존재 여부 검사를 생략
- `--overwrite`: 기존 crop·export 결과를 명시적으로 교체

전체 옵션은 다음 명령으로 확인한다.

```bash
uv run python -m etl.classifier.build_classifier_dataset --help
```

`--overwrite` 없이 기존 산출물 폴더를 삭제하거나 교체하지 않는다. 일부 단계만 다시 확인할 때는 각 모듈의 `main()`을 직접 실행할 수 있다.

## 라벨 2차 검수와 데이터 v2

기본 ETL로 만든 v1 패키지는 bbox 수치 규칙과 원천 라벨 구성을 검증한다. 알약이 아닌 곳에 놓인 bbox나 서로 뒤바뀐 품목 라벨처럼 수치 규칙만으로 찾기 어려운 오류는 별도의 2차 검수로 확인한다.

| 순서 | 모듈 | 책임 |
| --- | --- | --- |
| 1 | `label_audit/detect_bbox_errors.py` | 원본 이미지에서 알약 후보와 bbox의 위치 관계를 비교해 검수 후보 생성 |
| 2 | `label_audit/detect_class_errors.py` | 기준 모델 embedding과 그룹 비교로 클래스 오류 후보 생성 |
| 3 | `label_audit/records/*.csv` | 사람이 후보를 확인해 판정·조치·메모 기록 |
| 4 | `label_audit/merge_review_actions.py` | 두 검수 결과를 crop별 최종 조치로 병합 |
| 5 | `build_classifier_v2.py` | 검수한 v1 hash를 확인하고 제외·라벨 교환을 적용해 v2 생성 |
| 6 | `label_audit/export_detector_actions.py` | 같은 원본 이미지에 대한 탐지 데이터 조치 목록 생성 |

```bash
uv run python -m etl.classifier.label_audit.detect_bbox_errors
uv run python -m etl.classifier.label_audit.detect_class_errors

# records/의 후보 CSV를 사람이 검토한 뒤 실행
uv run python -m etl.classifier.label_audit.merge_review_actions
uv run python -m etl.classifier.build_classifier_v2 --no-eval
uv run python -m etl.classifier.label_audit.export_detector_actions
```

기준 checkpoint로 v1과 v2 validation을 다시 평가하려면 `--no-eval`을 빼고 `--checkpoint`를 지정한다. `build_classifier_v2`는 감사한 v1의 `split_hash`가 다르면 중단하고, 임시 디렉터리에서 실제 DataLoader 입력 검증까지 통과한 결과만 최종 폴더와 ZIP으로 옮긴다. 기존 결과를 교체하려면 `--overwrite`를 명시해야 한다.

사람 판정과 최종 조치 CSV는 재생성할 수 없는 검수 기록이므로 Git으로 관리한다. embedding, 점수와 검수용 이미지는 `data/classifier/meta/label_audit/`에 생성하며 Git에 포함하지 않는다. 방법과 확정 결과는 [라벨 2차 검수 보고서](../../reports/data/classifier/label_audit_report.md), v1과 v2 비교는 [분류 데이터 v2 보고서](../../reports/data/classifier/data_v2_report.md)에 기록한다.

## 산출물

최종 패키지는 [분류 데이터 사용 안내](classifier_data_guide.md)의 형식을 따른다.

```text
training_package_classifier/
├── class_map.json
├── data_summary.json
├── samples.csv
└── crops/<pill_code>/<sample_id>.png
```

2차 검수 패키지는 같은 구조의 `training_package_classifier_v2/`와 ZIP으로 생성한다. 남은 crop의 `sample_id`, split, group과 `class_map.json`은 v1과 호환된다.

검수·집계 결과는 `reports/data/classifier/`에 기록한다. 데이터 분포와 crop은 [분류 데이터 품질 Notebook](../../notebooks/eda/classifier_data_quality.ipynb)에서 확인한다.

현재 ETL의 기본 선정 목표와 데이터 예시는 118개 클래스에 맞춰져 있지만, 분류 모델의 클래스 수를 고정하는 검증값은 아니다. 데이터 구성이 바뀌면 class map, summary, target과 모델 설정을 함께 맞춘다.
