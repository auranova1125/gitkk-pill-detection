# 탐지 데이터 사용 안내

> **문서 상태:** 단일 `pill` 클래스 탐지기용 현재 runtime DataLoader와 데이터 배치 안내다.

## 책임과 구현 위치

| 파일 | 책임 |
| --- | --- |
| `data_loading/detector_package.py` | 데이터 경로 해석, JSON·split manifest 검증과 이미지별 bbox 연결 |
| `data_loading/detector.py` | Dataset, DataLoader와 factory 제공 |

DataLoader는 이미지 한 장과 그 이미지의 모든 알약 bbox를 함께 반환한다. 모든 target label은 `pill=1`이며 background는 모델 내부에서 `0`으로 처리한다. 원본 약품 코드는 추적용 metadata로만 보존한다. 자세한 형식은 [탐지기 입력 명세](../specs/detector_input_spec.md)를 따른다.

원천 데이터에서 이 학습 패키지를 만드는 책임과 현재 구현 범위는 [탐지 ETL](detector_etl.md)에 분리해 기록한다.

## 기본 데이터 배치

기본 설정은 저장소의 `data/`를 기준으로 한다.

```text
data/
├── annotations/
│   ├── train.json
│   └── validation.json
├── split_manifest.csv
├── class_mapping.csv
└── images/
    ├── select_aihub/
    └── select_codeit/
```

Annotation의 각 이미지는 해당 이미지 안의 모든 알약 bbox를 포함해야 한다. 이미지별 bbox 누락, 잘못된 XYWH 좌표, 분할 불일치와 실제 파일 누락은 학습 전에 오류로 처리한다.

## 저장소 밖 데이터 사용

데이터를 다른 위치에 둘 때 JSON 설정 파일을 만들고 `DetectorConfig.data_settings_path`에 전달한다.

```json
{
  "package_dir": "/absolute/path/to/detector_package",
  "image_roots": {
    "select": "/absolute/path/to/select_images",
    "codeit": "/absolute/path/to/codeit_images"
  },
  "augmentation": {
    "horizontal_flip": 0.0,
    "vertical_flip": 0.0,
    "brightness": 0.0,
    "contrast": 0.0
  }
}
```

환경 변수로 묵시적으로 경로를 바꾸지 않는다. CLI와 Notebook 모두 같은 설정 파일을 명시적으로 사용한다.

## 입력 검사

전체 라벨, 분할과 이미지 경로를 확인하고 JSON 보고서를 남긴다.

```bash
uv run python main.py check-detector-data \
  --settings /absolute/path/to/detector_data_settings.json \
  --report outputs/detector_data/check_report.json
```

기본 `data/` 배치를 사용하면 `--settings`를 생략한다. 이 명령은 모든 이미지 픽셀을 읽거나 탐지 정확도를 평가하지 않는다. 코드 연결과 합성 데이터 학습 1 step은 별도의 smoke test로 확인한다.

```bash
uv run pytest -q -m smoke
```

## 학습

[탐지기 baseline Notebook](../../notebooks/baseline/train_detector.ipynb) 또는 CLI에서 같은 factory를 사용한다.

```bash
uv run python main.py train-detector --config configs/detector.yaml
```

기본 factory는 `data_loading.detector:build_detector_inputs`다. Notebook에서는 `DetectorConfig`를 직접 만들어 `run_name`, batch size, epoch, device와 `data_settings_path`를 셀에서 바꿀 수 있다.
