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

기본 설정은 저장소의 `data/detector/training_package_detector/`를 기준으로 한다.

```text
data/detector/training_package_detector/
├── annotations/
│   ├── train.json
│   └── validation.json
├── split_manifest.csv
├── detector_label_map.json
├── detector_data_summary.json
└── images/                       기본 설정에서 이미지를 함께 둘 때 사용
```

Annotation의 각 이미지는 해당 이미지 안의 모든 알약 bbox를 포함해야 한다. 이미지별 bbox 누락, 잘못된 XYWH 좌표, 분할 불일치와 실제 파일 누락은 학습 전에 오류로 처리한다.

`detector_label_map.json`과 `detector_data_summary.json`도 완성 패키지의 필수
입력이다. runtime DataLoader는 두 파일을 대신 생성하지 않는다. annotation과
`split_manifest.csv`에서 클래스 구성과 집계를 다시 계산해 전달된 두 파일과
비교하며, 누락되거나 실제 데이터와 다르면 학습 전에 오류로 처리한다.

`images.file_name`과 `split_manifest.csv`의 `image_path`에는 이미지 root를 기준으로 한 같은 상대 경로를 기록한다. `dataset`은 경로 문자열에 넣지 않고 manifest의 별도 열로 관리한다.

```text
dataset:    sprint
image_path: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
```

로더는 `image_roots[dataset] / image_path`로 실제 파일을 찾는다. 절대 경로와 `..`로 root를 벗어나는 상대 경로는 허용하지 않는다.

## 저장소 밖 데이터 사용

데이터를 다른 위치에 둘 때 JSON 설정 파일을 만들고 `DetectorConfig.data_settings_path`에 전달한다.

```json
{
  "package_dir": "/absolute/path/to/detector_package",
  "image_roots": {
    "sprint": "/absolute/path/to/sprint_train_images",
    "additional_source": "/absolute/path/to/additional_images"
  }
}
```

`image_roots`의 키는 특정 데이터셋 이름으로 고정되지 않는다. manifest의 `dataset` 값과 일치하는 비어 있지 않은 이름이면 된다. 이미지가 한 출처뿐이면 항목도 하나만 두면 된다.

이 JSON은 데이터 위치만 정의한다. 학습 증강은 데이터 자체의 속성이 아니므로
`DetectorConfig`의 `horizontal_flip`, `vertical_flip`, `brightness`, `contrast`로
설정한다. CLI는 `configs/detector.yaml`, Notebook은 같은 설정 객체의 값을 사용한다.

환경 변수로 묵시적으로 경로를 바꾸지 않는다. CLI와 Notebook 모두 같은 데이터
경로 설정 파일을 명시적으로 사용한다.

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
