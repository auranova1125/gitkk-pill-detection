# 분류기 데이터 구성

조합 이미지(Kaggle train + AI Hub 경구약제 조합 1, 3~8번 세트)에서 알약을 bbox대로 잘라 118종 분류기 데이터를 만든다.
라벨 검수 → 이미지 선정 → crop → train/val 분할 → 집계 → 분류기 입력 패키지(export) 순서다.
만든 데이터를 학습에 쓰는 방법은 [`docs/classifier_data.md`](../docs/classifier_data.md)에 있다.

| 파일 | 내용 |
|---|---|
| `config.py`, `label_checklist.py`, `classes.csv` | 경로 설정·원본 zip 찾기, 「알약 데이터 검증 체크리스트」 항목 목록, 118종 클래스 표 (`class_id` = 모델 출력 번호 `class_index`) |
| `validate_labels.py` … `export_classifier_inputs.py` | 데이터 생성 스크립트 (아래 '실행') |
| `reports/` | 라벨 검수, 클래스별 개수, train/val 분할 결과 (스크립트가 자동 생성) |
| `classifier_data_check.ipynb` | 개수 그래프, 각도 분포, 제외 현황, crop 샘플, 분류기 입력 형식 확인 |

crop 함수(`pill_square_crop.py`)와 로더(`classifier_inputs.py`)는 학습·추론 코드도 쓰므로 `classifier_data/`에 있다.

## 준비
원본은 git에 올리지 않는 `data/` 아래에 둔다. 이미 다른 곳에 받아 두었다면 옮기지 말고 심볼릭 링크를 걸거나,
환경 변수 `AIHUB_DIR`, `KAGGLE_DIR`로 위치를 지정한다.

```
data/
├── aihub/     AI Hub zip (하위 폴더 이름은 상관없음)
│              이미지 TS_{1,3,4,5,6,7,8}_조합*.zip, 라벨 TL_{1,3,4,5,6,7,8}_조합.zip
├── kaggle/    Kaggle 데이터 (train_images/, train_annotations/)
└── detector/  탐지기 팀 학습 패키지 training_package_detector.zip (train/val 분할 기준)
```
```bash
ln -s <AI Hub zip 폴더> data/aihub
ln -s <Kaggle 데이터 폴더> data/kaggle
```

## 실행 (프로젝트 루트에서)
```bash
uv run python classifier_data_build/validate_labels.py # 라벨 검수          → data/classifier/meta/label_issues.csv, classifier_data_build/reports/validation_report.md
uv run python classifier_data_build/select_images.py   # 이미지 선정        → data/classifier/meta/
uv run python classifier_data_build/make_crops.py      # crop 생성          → data/classifier/crops/<약 코드>/*.jpg
uv run python classifier_data_build/split_dataset.py   # train/val 분할      → crops.csv에 split 열 (탐지기 분할 기준)
uv run python classifier_data_build/count_crops.py     # 클래스별 개수      → data/classifier/crops/crop_counts.csv, classifier_data_build/reports/crop_counts.md
uv run python classifier_data_build/export_classifier_inputs.py      # 분류기 입력 패키지 → data/classifier/export/training_package_classifier(.zip)
```
`split_dataset.py`는 `data/detector/training_package_detector.zip`(탐지기 학습 패키지)의 `split_manifest.csv`로 train/val을 나눈다.

## 결과 문서
이 문서에는 **방법**만 적는다. 숫자는 데이터를 다시 만들 때마다 바뀌므로 스크립트가 `reports/`에 자동으로 쓴다 (직접 고치지 않는다).

| 문서 | 내용 | 만드는 스크립트 |
|---|---|---|
| [`validation_report.md`](reports/validation_report.md) | 체크리스트 항목별 검수 결과, 제외한 이미지·bbox 수 | `validate_labels.py` |
| [`crop_counts.md`](reports/crop_counts.md) | **crop 총 개수**(맨 위 요약), 클래스별 crop 개수 (Kaggle/조합) | `count_crops.py` |
| [`split_report.md`](reports/split_report.md) | train/val 분할 기준, 클래스별 train·val crop 수, 유사 사진 확인 | `split_dataset.py` |
| [`classifier_data_check.ipynb`](classifier_data_check.ipynb) | 개수 그래프, 각도 분포, 제외 현황, crop 샘플, 분류기 입력 형식 확인 | 직접 실행 |

## 규칙
| 단계 | 내용 |
|---|---|
| 사용 세트 | 조합 1, 3~8번(TS 이미지 + TL 라벨). 2번 세트(TS_2·TL_2) 제외   |
| 촬영 조건 | 연회색 배경, 주백색 조명, 70/75/90° |
| 라벨 검수 | 「알약 데이터 검증 체크리스트」 항목 전체 검사. 라벨 오류가 있는 조합 이미지는 이미지째, Kaggle은 해당 bbox만 제외. 결과: [`validation_report.md`](reports/validation_report.md) |
| 선정 방법 | Kaggle 전부 + 조합은 목표 초과를 최소화하는 정수계획, 각 각도 30% 이상 |
| crop | bbox 사방 5% 여백 → 이미지 경계에서 clip → 고정색 padding으로 정사각형 (아래 'crop 기준과 방법') |
| 분할 | 탐지기 `split_manifest.csv`의 그룹(파일명의 알약 조합) 분할을 그대로 따름. 탐지기 목록에 없는 그룹은 train |
| 증강 | 학습 로더에서 회전(-180°~180°)만 한다. flip, color jitter, grayscale, blur, random erasing은 쓰지 않는다. val은 무작위 처리 없음 |

## crop 기준과 방법

### 어떤 bbox를 자르나
`select_images.py`가 고른 bbox만 자른다. 라벨이 틀린 bbox는 그 전 단계에서 이미 빠졌다.

| 단계 | 하는 일 | 기준 |
|---|---|---|
| 1. 라벨 검수 `validate_labels.py` | 체크리스트대로 라벨 오류를 찾는다 | 빈·0 크기 bbox, 이미지 밖 30% 넘게 나간 bbox, 같은 자리에 두 약 라벨(동일 bbox·IoU > 0.7·작은 bbox의 90% 이상 겹침), bbox 크기가 실제 알약 크기와 안 맞음(긴 변 1mm당 18~33px 밖, 짧은 변 18px 미만. 실제 크기 값이 없는 약은 클래스 중앙값 대비 긴 변 0.7배·짧은 변 0.5배 미만), 파일명과 라벨 구성 다름, PNG 없음 |
| 2. 제외 | 오류를 뺀다 | 조합 이미지는 **이미지째**(다른 이미지로 대체 가능), Kaggle은 **해당 bbox만**(대체 불가) |
| 3. 선정 `select_images.py` | 남은 이미지에서 고른다 | 클래스별 bbox 108개 목표, 각도 70/75/90° 고르게 |

### 어떻게 자르나 (`pill_square_crop.py`)
1. bbox 사방에 **5% 여백**을 더한다. (가로는 w의 5%, 세로는 h의 5%씩)
2. 이미지 밖으로 나간 부분은 잘라 낸다(clip). 이 범위만 원본에서 RGB로 자른다. (1~2: `cut_pill`)
3. 짧은 쪽 양옆(또는 위아래)을 **고정색 RGB (124, 116, 104)**로 채워 **정사각형**을 만든다. 알약은 가운데에 둔다. (`pad_square`, 1~3을 한 번에: `crop_pill`)
4. 224×224로 줄이고(bilinear) ImageNet mean/std로 정규화한다.

예) bbox 가로 200, 세로 100 → 220×110을 잘라, 위아래에 55px씩 고정색을 붙여 220×220.

crop은 두 곳에 저장한다.

| 위치 | 저장 상태 | 용도 |
|---|---|---|
| `data/classifier/crops/<약 코드>/<원본 파일명>_<번호>.jpg` (`make_crops.py`) | 3단계까지(정사각형), JPEG 95 | 레포 안 확인용: 개수 세기, 노트북 샘플 |
| 패키지 `crops/<약 코드>/<sample_id>.png` (`export_classifier_inputs.py`) | 2단계까지(padding 전 직사각형), PNG | **분류기 학습 입력**. 명세서 6.2대로 회전을 padding 전에 넣어야 해서 padding 전 상태로 둔다 |

분류기 로더(`classifier_inputs.py`)는 패키지 crop에 다음을 적용한다.
- train: `RandomRotation(-180°~180°, expand=True, bilinear)` (빈 곳은 (124, 116, 104)) → 3단계 → 4단계
- val: 3단계 → 4단계 (무작위 처리 없음)

### 왜 이렇게 하나
| 선택 | 이유 |
|---|---|
| 정사각형 | 224×224로 줄일 때 원형 알약이 타원이 되는 등 모양이 찌그러지지 않게 한다 |
| 사진이 아니라 padding으로 정사각형 | 사진에서 정사각형으로 넓게 자르면 옆 알약이 들어온다 (같은 이미지의 다른 bbox와 겹치는 crop: 정사각형 1.2배 7.9% → 여백 5% 2.1%) |
| 여백 5% | 탐지 bbox가 조금 작게 잡혀도 알약 테두리가 잘리지 않게 한다. 10%는 옆 알약이 더 들어온다(3.5%) |
| 고정색 (124, 116, 104) | ImageNet 평균색이라 정규화하면 0(정보 없음)이 된다. 모든 crop과 추론 때 같은 색이라 이미지마다 달라지는 잡음이 없다. 학습 때 회전으로 생기는 빈 곳도 같은 색으로 채운다 |
| 확인용 crop은 JPEG 95 | PNG보다 용량이 약 1/4.7이고, 224로 줄이면 압축 손실이 거의 보이지 않는다. 학습 패키지는 원본 화질 그대로 PNG로 둔다 |
| 오류 bbox는 crop 전에 제외 | 검수 기준을 `validate_labels.py` 한 곳에만 두어, 단계마다 기준이 어긋나지 않게 한다 |

`crops.csv`에는 원본 이미지 기준 bbox(`x, y, w, h`, **여백 추가 전**)도 남겨, 분류기 입력 `meta.bbox_xywh`를 바로 만들 수 있다.
