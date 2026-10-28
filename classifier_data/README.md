# 분류기 입력 데이터

「분류기 데이터 및 함수 입력 명세서」(`docs/classifier_input_spec.md`)에 변경 사항 2가지를 더한 분류기 데이터 구성 함수다.
- 변경 1: bbox 사방 5% 여백 + 이미지 경계 clip
- 변경 2: 탐지 결과도 `crop_pill`로 자른다

| 파일 | 내용 |
|---|---|
| `classifier_inputs.py` | `build_classifier_inputs(*, config, device)` → `train_loader, val_loader, class_map, data_summary` |
| `crop.py` | `cut_pill`(여백 5% + clip), `pad_square`(정사각형 padding), `crop_pill`(둘을 한 번에, 탐지 결과용) |
| [`reports/split_report.md`](reports/split_report.md) | train/val 분할 기준, 클래스별 train·val 수, 유사 사진(`near_dup_in_train`) 확인 |

데이터를 만드는 코드(라벨 검수·이미지 선정·crop·분할·export)와 검수 결과는 `6-combo-crop-dataset` 브랜치에 보관한다(#6).

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
# 드라이브에서 바로 읽으면 느리므로 프로젝트의 data/ 아래(git 미포함)에 푼다
!unzip -q "/content/drive/MyDrive/gitkk_archive/training_package_classifier.zip" -d data/classifier

data_factory_path = "classifier_data.classifier_inputs:build_classifier_inputs"
config = TrainConfig(
    ...,
    class_map_path=project_root / "data" / "classifier" / "class_map.json",
    data_summary_path=project_root / "data" / "classifier" / "data_summary.json",
)
```
- 명세서 예시의 `data.classifier_data`는 `/data/`가 `.gitignore`에 있어 레포에 둘 수 없으므로, 위 경로를 쓴다
- 입력 검증만 따로: `uv run python -m classifier_data.classifier_inputs data/classifier/class_map.json`
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

## 평가할 때
`near_dup_in_train=True`인 val 샘플 799개는 그룹은 다르지만 거의 같은 사진(알약 하나만 다른 조합)이 train에 있다.
전체 val과 이 샘플을 뺀 val(2,064개)의 Macro F1을 함께 보면, train 사진을 외워서 맞힌 몫을 확인할 수 있다.

## 분류기 팀과 확인할 것 (실험 후 결정)
지금은 명세서대로 두고, 학습 결과를 보고 바꿀지 정한다.

| 항목 | 현재 (명세서) | 확인할 점 | 비교할 대안 |
|---|---|---|---|
| padding 색 | RGB (124, 116, 104) | 배경은 푸른 회색(중앙값 (106, 124, 149))이라 색이 다르다. crop의 34%는 padding이 면적의 30% 이상이다 | 배경 중앙값 색. `crop.py`의 `PAD_COLOR`만 바꾸면 되고 데이터 재생성은 필요 없다 |
| 회전 시 알약 크기 | `RandomRotation(-180°~180°, expand=True)` | 캔버스를 키운 뒤 224로 줄여, 45° 근처에서 알약이 0°보다 71~85% 크기로 작아진다. val·추론은 항상 0° 크기다 | 회전 후 크기 맞추기, 90° 단위 + ±15° |
