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

`build_classifier_dataset.py`는 위 단계를 연결하는 실행 진입점이다. 단계 구현은 공통 crop 함수가 있는 `pilldata.transforms`를 재사용하지만, runtime 학습 코드에서는 ETL 모듈을 import하지 않는다.

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

## 산출물

최종 패키지는 [분류 데이터 사용 안내](classifier_data_guide.md)의 형식을 따른다.

```text
training_package_classifier/
├── class_map.json
├── data_summary.json
├── samples.csv
└── crops/<pill_code>/<sample_id>.png
```

검수·집계 결과는 `reports/data/classifier/`에 기록한다. 데이터 분포와 crop은 [분류 데이터 품질 Notebook](../../notebooks/eda/classifier_data_quality.ipynb)에서 확인한다.

현재 ETL의 기본 선정 목표와 데이터 예시는 118개 클래스에 맞춰져 있지만, 분류 모델의 클래스 수를 고정하는 검증값은 아니다. 데이터 구성이 바뀌면 class map, summary, target과 모델 설정을 함께 맞춘다.
