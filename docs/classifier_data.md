# 분류기 데이터

「분류기 데이터 및 함수 입력 명세서」(`docs/classifier_input_spec.md`)에 변경 사항 2가지를 더한 분류기 데이터 구성 함수다.
- 변경 1: bbox 사방 5% 여백 + 이미지 경계 clip
- 변경 2: 탐지 결과도 `crop_pill`로 자른다

| 파일 | 내용 |
|---|---|
| `classifier_inputs.py` | `build_classifier_inputs(*, config, device)` → `train_loader, val_loader, class_map, data_summary` |
| `crop.py` | `cut_pill`(여백 5% + clip), `pad_square`(정사각형 padding), `crop_pill`(둘을 한 번에, 탐지 결과용) |
| `config.py`, `checks.py`, `classes.csv` | 경로 설정, 검수 공용 함수, 118종 클래스 표 (`class_id` = `class_index`) |
| `validate_labels.py` → `select_images.py` → `make_crops.py` → `split_dataset.py` → `count_crops.py` → `export_classifier_inputs.py` | 데이터를 다시 만들 때 쓰는 스크립트 (아래 '데이터를 다시 만들 때') |

- 결과 문서: `reports/classifier_data/` (라벨 검수, 클래스별 개수, train/val 분할)
- 확인 노트북: `notebooks/classifier_data_check.ipynb` (개수 그래프, 각도 분포, 제외 현황, crop 샘플, 입력 형식)

학습에는 `classifier_inputs.py`, `crop.py`와 드라이브의 데이터만 있으면 된다. 나머지는 데이터를 다시 만들거나 기준을 확인할 때만 쓴다.

## 데이터
구글 드라이브 `gitkk_archive/training_package_classifier.zip` (탐지기 `training_package_detector.zip`과 같은 폴더)

zip을 풀면 맨 위에 바로 다음이 있다.
```
class_map.json        모델 출력 번호 → category_id, item_seq, 품목명
data_summary.json     분할·클래스 수 요약 (split_hash 포함)
samples.csv           샘플(알약 하나) 하나당 한 줄. meta 필드 전부 + split, crop_path, source, near_dup_in_train
crops/<약 코드>/<sample_id>.png   원본에서 bbox 사방 5% 여백 + 경계 clip으로 자른 직사각형 (무손실 PNG)
```

| 항목 | 값 |
|---|---|
| 클래스 | 118종 |
| 샘플 | train 12,125 / val 2,863 |
| 그룹(파일명의 알약 조합) | train 1,786 / val 429, 겹침 0 |
| split_hash | `49d2afee86a4ec8f48c25488eac9b1f9081b36066e5533eba40670087f60430c` |

## 사용법
`notebooks/train_classifier.ipynb` 설정 셀에서 다음을 바꾼다. 로더는 `class_map_path`와 같은 폴더에서 `samples.csv`와 `crops/`를 읽는다.
```python
# 드라이브에서 바로 읽으면 느리므로 Colab 로컬에 푼다
!unzip -q "/content/drive/MyDrive/gitkk_archive/training_package_classifier.zip" -d /content/training_package_classifier

data_factory_path = "classifier_data.classifier_inputs:build_classifier_inputs"
config = TrainConfig(
    ...,
    class_map_path="/content/training_package_classifier/class_map.json",
    data_summary_path="/content/training_package_classifier/data_summary.json",
)
```
- 명세서 예시의 `data.classifier_data`는 `/data/`가 `.gitignore`에 있어 레포에 둘 수 없으므로, 위 경로를 쓴다
- 입력 검증만 따로: 레포 루트에서 `python -m classifier_data.classifier_inputs /content/training_package_classifier/class_map.json`
- `data_summary.json`이 `samples.csv`와 맞지 않으면(다른 패키지의 파일을 섞은 경우) 학습 전에 멈춘다

## 로더 구성
| 항목 | 내용 |
|---|---|
| batch | `images` float32 `[B, 3, 224, 224]`, `targets` int64 `[B]`, `meta` `list[dict]` |
| train 전처리 | `RandomRotation(-180°~180°, expand=True, bilinear)` → 중앙 정사각형 padding → 224 resize(bilinear) → ImageNet mean/std 정규화 |
| val 전처리 | 중앙 정사각형 padding → 224 resize → 정규화 (무작위 처리 없음) |
| padding·회전 빈 곳 색 | RGB (124, 116, 104). ImageNet 평균색이라 정규화하면 0이 된다 |
| train 순서 | `WeightedRandomSampler`(샘플 가중치 = 1 / 그 클래스의 train 샘플 수, `replacement=True`, `config.seed`), `shuffle=False` |
| worker | CUDA: `num_workers=2`, `pin_memory=True` / MPS·CPU: 0, False |

flip, color jitter, grayscale, blur, random erasing은 쓰지 않는다.
- 회전을 padding 전에 해야 하므로 PNG는 padding 전 직사각형으로 저장했다
- `class_map.json`의 `class_index`는 약 코드 순서다(폴더 정렬 순서가 아님). `category_id`는 약 코드의 숫자, `item_seq`는 문자열이다

## 탐지 결과 분류
탐지기가 찾은 bbox를 학습 데이터와 **같은 함수**로 자른다.
```python
from classifier_data.crop import crop_pill
from classifier_data.classifier_inputs import detection_transform
image = detection_transform()(crop_pill(original_image, (x, y, w, h)))   # float32 [3, 224, 224]
```
- bbox는 원본 사진 픽셀 좌표 `[x, y, w, h]`다. 탐지기 출력(XYXY)은 `(x1, y1, x2 - x1, y2 - y1)`로 바꿔 넣는다
- ⚠️ 지금 `detector/classifier_bridge.py`는 5% 여백 없이 자른다. 학습 입력과 맞도록 `crop_pill`을 쓰거나 여백 5%를 더해야 한다

## 평가할 때
`near_dup_in_train=True`인 val 샘플 799개는 그룹은 다르지만 거의 같은 사진(알약 하나만 다른 조합)이 train에 있다.
전체 val과 이 샘플을 뺀 val(2,064개)의 Macro F1을 함께 보면, train 사진을 외워서 맞힌 몫을 확인할 수 있다.

## 분류기 팀과 확인할 것 (실험 후 결정)
지금은 명세서대로 두고, 학습 결과를 보고 바꿀지 정한다.

| 항목 | 현재 (명세서) | 확인할 점 | 비교할 대안 |
|---|---|---|---|
| padding 색 | RGB (124, 116, 104) | 배경은 푸른 회색(중앙값 (106, 124, 149))이라 색이 다르다. crop의 34%는 padding이 면적의 30% 이상이다 | 배경 중앙값 색. `crop.py`의 `PAD_COLOR`만 바꾸면 되고 데이터 재생성은 필요 없다 |
| 회전 시 알약 크기 | `RandomRotation(-180°~180°, expand=True)` | 캔버스를 키운 뒤 224로 줄여, 45° 근처에서 알약이 0°보다 71~85% 크기로 작아진다. val·추론은 항상 0° 크기다 | 회전 후 크기 맞추기, 90° 단위 + ±15° |

## 데이터를 다시 만들 때
### 준비
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

### 실행 (프로젝트 루트에서)
```bash
uv run python classifier_data/validate_labels.py # 라벨 검수          → data/classifier/meta/label_issues.csv, reports/classifier_data/validation_report.md
uv run python classifier_data/select_images.py   # 이미지 선정        → data/classifier/meta/
uv run python classifier_data/make_crops.py      # crop 생성          → data/classifier/crops/<약 코드>/*.jpg
uv run python classifier_data/split_dataset.py   # train/val 분할      → crops.csv에 split 열 (탐지기 분할 기준)
uv run python classifier_data/count_crops.py     # 클래스별 개수      → data/classifier/crops/crop_counts.csv, reports/classifier_data/crop_counts.md
uv run python classifier_data/export_classifier_inputs.py      # 분류기 입력 패키지 → data/classifier/export/training_package_classifier(.zip)
```
`split_dataset.py`는 `data/detector/training_package_detector.zip`(탐지기 학습 패키지)의 `split_manifest.csv`로 train/val을 나눈다.

### 결과 문서
이 문서에는 **방법**만 적는다. 숫자는 데이터를 다시 만들 때마다 바뀌므로 스크립트가 `reports/classifier_data/`에 자동으로 쓴다 (직접 고치지 않는다).

| 문서 | 내용 | 만드는 스크립트 |
|---|---|---|
| [`validation_report.md`](../reports/classifier_data/validation_report.md) | 체크리스트 항목별 검수 결과, 제외한 이미지·bbox 수 | `validate_labels.py` |
| [`crop_counts.md`](../reports/classifier_data/crop_counts.md) | **crop 총 개수**(맨 위 요약), 클래스별 crop 개수 (Kaggle/조합) | `count_crops.py` |
| [`split_report.md`](../reports/classifier_data/split_report.md) | train/val 분할 기준, 클래스별 train·val crop 수, 유사 사진 확인 | `split_dataset.py` |
| `notebooks/classifier_data_check.ipynb` | 개수 그래프, 각도 분포, 제외 현황, crop 샘플, 분류기 입력 형식 확인 | 직접 실행 |

### 규칙
| 단계 | 내용 |
|---|---|
| 사용 세트 | 조합 1, 3~8번(TS 이미지 + TL 라벨). 2번 세트(TS_2·TL_2) 제외   |
| 촬영 조건 | 연회색 배경, 주백색 조명, 70/75/90° |
| 라벨 검수 | 「알약 데이터 검증 체크리스트」 항목 전체 검사. 라벨 오류가 있는 조합 이미지는 이미지째, Kaggle은 해당 bbox만 제외. 결과: [`validation_report.md`](../reports/classifier_data/validation_report.md) |
| 선정 방법 | Kaggle 전부 + 조합은 목표 초과를 최소화하는 정수계획, 각 각도 30% 이상 |
| crop | bbox 사방 5% 여백 → 이미지 경계에서 clip → 고정색 padding으로 정사각형 (아래 'crop 기준과 방법') |
| 분할 | 탐지기 `split_manifest.csv`의 그룹(파일명의 알약 조합) 분할을 그대로 따름. 탐지기 목록에 없는 그룹은 train |
| 증강 | 학습 로더에서 회전(-180°~180°)만 한다. flip, color jitter, grayscale, blur, random erasing은 쓰지 않는다. val은 무작위 처리 없음 |

### crop 기준과 방법

#### 어떤 bbox를 자르나
`select_images.py`가 고른 bbox만 자른다. 라벨이 틀린 bbox는 그 전 단계에서 이미 빠졌다.

| 단계 | 하는 일 | 기준 |
|---|---|---|
| 1. 라벨 검수 `validate_labels.py` | 체크리스트대로 라벨 오류를 찾는다 | 빈·0 크기 bbox, 이미지 밖 30% 넘게 나간 bbox, 같은 자리에 두 약 라벨(동일 bbox·IoU > 0.7·작은 bbox의 90% 이상 겹침), bbox 크기가 실제 알약 크기와 안 맞음(긴 변 1mm당 18~33px 밖, 짧은 변 18px 미만. 실제 크기 값이 없는 약은 클래스 중앙값 대비 긴 변 0.7배·짧은 변 0.5배 미만), 파일명과 라벨 구성 다름, PNG 없음 |
| 2. 제외 | 오류를 뺀다 | 조합 이미지는 **이미지째**(다른 이미지로 대체 가능), Kaggle은 **해당 bbox만**(대체 불가) |
| 3. 선정 `select_images.py` | 남은 이미지에서 고른다 | 클래스별 bbox 108개 목표, 각도 70/75/90° 고르게 |

#### 어떻게 자르나 (`crop.py`)
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

#### 왜 이렇게 하나
| 선택 | 이유 |
|---|---|
| 정사각형 | 224×224로 줄일 때 원형 알약이 타원이 되는 등 모양이 찌그러지지 않게 한다 |
| 사진이 아니라 padding으로 정사각형 | 사진에서 정사각형으로 넓게 자르면 옆 알약이 들어온다 (같은 이미지의 다른 bbox와 겹치는 crop: 정사각형 1.2배 7.9% → 여백 5% 2.1%) |
| 여백 5% | 탐지 bbox가 조금 작게 잡혀도 알약 테두리가 잘리지 않게 한다. 10%는 옆 알약이 더 들어온다(3.5%) |
| 고정색 (124, 116, 104) | ImageNet 평균색이라 정규화하면 0(정보 없음)이 된다. 모든 crop과 추론 때 같은 색이라 이미지마다 달라지는 잡음이 없다. 학습 때 회전으로 생기는 빈 곳도 같은 색으로 채운다 |
| 확인용 crop은 JPEG 95 | PNG보다 용량이 약 1/4.7이고, 224로 줄이면 압축 손실이 거의 보이지 않는다. 학습 패키지는 원본 화질 그대로 PNG로 둔다 |
| 오류 bbox는 crop 전에 제외 | 검수 기준을 `validate_labels.py` 한 곳에만 두어, 단계마다 기준이 어긋나지 않게 한다 |

`crops.csv`에는 원본 이미지 기준 bbox(`x, y, w, h`, **여백 추가 전**)도 남겨, 분류기 입력 `meta.bbox_xywh`를 바로 만들 수 있다.
