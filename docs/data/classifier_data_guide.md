# 분류기 데이터

> **문서 상태:** 현재 분류기 데이터 사용 안내다. runtime Dataset과 전처리의 기준 구현은 `data_loading/`에 있다.

[분류기 데이터 및 함수 입력 명세서](../specs/classifier_input_spec.md)에 변경 사항 2가지를 더한 분류기 데이터 구성 함수다.
- 변경 1: bbox 사방 5% 여백 + 이미지 경계 clip
- 변경 2: 탐지 결과도 `crop_pill`로 자른다

| 파일 | 내용 |
|---|---|
| `data_loading/classifier.py` | `build_classifier_inputs(*, config, device)` → `train_loader, val_loader, class_map, data_summary` |
| `data_loading/image_transforms.py` | crop, 여백 5%, clip, padding, resize와 정규화의 공통 구현 |

학습에는 위 runtime 모듈과 드라이브의 데이터만 있으면 된다.
데이터를 만든 방법·기준, 검수 결과, 확인 Notebook은 [분류 데이터 ETL](classifier_etl.md)에 있다.

## 데이터 v1
구글 드라이브 `gitkk_archive/training_package_classifier.zip`

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

위 수치는 현재 기준 데이터셋의 기록이며 분류기가 강제하는 고정값은 아니다. 다른 클래스 구성이나 분할을 사용하면 해당 패키지의 `class_map.json`과 `data_summary.json`을 기준으로 검증한다.

## 데이터 v2

`training_package_classifier_v2.zip`은 v1에 사람 확인을 마친 2차 라벨 검수 조치를 적용한 패키지다. 파일 구조와 `class_map.json`은 v1과 같다.

| 항목 | 값 |
|---|---|
| 샘플 | train 11,990 / val 2,808 |
| 그룹 | train 1,774 / val 424, 겹침 0 |
| 적용 조치 | 오류 crop 190개 제외, crop 2개의 품목 라벨 교환 |
| split_hash | `a2eb581a04604ff1e0b156980965a49650ecb4dec919e3def500c35acf8241e0` |

새 학습의 기본 후보는 v2다. 기존 기준 모델 재현이나 변경 전후 비교에는 v1을 사용하고, 실행 기록에는 사용한 패키지와 `split_hash`를 함께 남긴다. 상세 변경은 [분류 데이터 v2 보고서](../../reports/data/classifier/data_v2_report.md)에서 확인한다.

## 사용법
`notebooks/baseline/train_classifier.ipynb` 설정 셀에서 다음을 바꾼다. 로더는 `data_dir`에서 `class_map.json`, `data_summary.json`, `samples.csv`와 `crops/`를 함께 읽는다.
```python
# 드라이브에서 바로 읽으면 느리므로 Colab 로컬에 푼다
!unzip -q "/content/drive/MyDrive/gitkk_archive/training_package_classifier_v2.zip" -d /content/training_package_classifier_v2

data_dir = Path("/content/training_package_classifier_v2")
config = replace(base_config, data_dir=data_dir)
```
- 입력 검증만 따로: 레포 루트에서 `uv run python -m data_loading.classifier /content/training_package_classifier_v2`
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
탐지기가 찾은 bbox를 학습 데이터와 **같은 공통 모듈**로 자른다.
```python
from data_loading.image_transforms import (
    crop_pill,
    build_classifier_inference_transform,
)

transform = build_classifier_inference_transform()
image = transform(crop_pill(original_image, (x, y, w, h)))
```
- bbox는 원본 사진 픽셀 좌표 `[x, y, w, h]`다. 탐지기 출력(XYXY)은 `(x1, y1, x2 - x1, y2 - y1)`로 바꿔 넣는다
- `pipeline/pill_recognition.py`도 `data_loading.image_transforms`를 사용하므로 bbox 사방 5% 여백과 동일한 padding·정규화 설정이 적용된다

## 평가할 때
`near_dup_in_train=True`인 val 샘플 799개는 그룹은 다르지만 거의 같은 사진(알약 하나만 다른 조합)이 train에 있다.
전체 val과 이 샘플을 뺀 val(2,064개)의 Macro F1을 함께 보면, train 사진을 외워서 맞힌 몫을 확인할 수 있다.

## AI Hub val 외부 평가셋

v2 val은 기준 모델이 모두 맞혀서 실험끼리 비교할 수 없다. 그래서 **학습에 쓰지 않는** AI Hub val(조합)로 평가 전용 패키지를 만들었다. 만드는 방법은 [분류 데이터 ETL](classifier_etl.md#ai-hub-val-외부-평가셋), 구성은 [평가셋 보고서](../../reports/data/classifier/aihub_val_eval_package_report.md).

| 데이터 | 역할 | 사용 시점 |
| --- | --- | --- |
| v2 train | 학습 | 매 epoch |
| v2 val | best epoch 선택 | 매 epoch |
| **AI Hub val** | **실험 간 비교** | 학습이 끝난 checkpoint |

```text
eval_package_aihub_val/
├── class_map.json      v2와 같음
├── eval_summary.json   crop·사진·그룹 수, 클래스별 수, eval_hash
├── samples.csv         v2와 같은 열. split은 모두 "eval"
├── excluded.csv        뺀 사진·crop과 사유
└── crops/<약 코드>/<sample_id>.png
```

평가:
```bash
uv run python -m classifier.evaluate_external \
  --checkpoint <best.pt> \
  --package data/classifier/export/eval_package_aihub_val \
  --output-dir outputs/classifier_eval/aihub_val/<run_name>
```

- 로더는 `data_loading.classifier.build_classifier_eval_loader`다. train 없이 평가셋만 읽고, 전처리는 val과 같다
- 평가셋에는 118종 중 54종만 있다. **Macro F1(평가셋에 있는 54종 평균)**을 주 지표로 보고, Top-1과 클래스별 Recall은 개수와 함께 읽는다
- `near_dup_in_train=True` crop을 뺀 점수도 `evaluation_report.md`에 함께 나온다
- 실험 기간에는 AI Hub val을 학습에 넣지 않는다. 이 점수를 보며 설정을 반복해서 고치면 평가셋에도 과적합되므로, 실험 선택용으로만 쓴다
- 실험 기록에는 checkpoint와 함께 `eval_hash`를 적는다

## 분류기 팀과 확인할 것 (실험 후 결정)
지금은 명세서대로 두고, 학습 결과를 보고 바꿀지 정한다.

| 항목 | 현재 (명세서) | 확인할 점 | 비교할 대안 |
|---|---|---|---|
| padding 색 | RGB (124, 116, 104) | 배경은 푸른 회색(중앙값 (106, 124, 149))이라 색이 다르다. crop의 34%는 padding이 면적의 30% 이상이다 | 배경 중앙값 색. `data_loading.image_transforms.PADDING_COLOR_RGB`를 바꾸면 되고 데이터 재생성은 필요 없다 |
| 회전 시 알약 크기 | `RandomRotation(-180°~180°, expand=True)` | 캔버스를 키운 뒤 224로 줄여, 45° 근처에서 알약이 0°보다 71~85% 크기로 작아진다. val·추론은 항상 0° 크기다 | 회전 후 크기 맞추기, 90° 단위 + ±15° |
