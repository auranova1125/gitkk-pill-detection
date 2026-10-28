# 단일 클래스 알약 탐지 데이터 구성

검증된 SELECT·대회 데이터를 고정된 Train/Validation 분할로 읽어, 단일 클래스 알약 탐지용 Dataset과 DataLoader를 구성한다.

## 1. 최종 데이터

| 항목 | Train | Validation |
|---|---:|---:|
| 이미지 수 | 8,256장 | 2,063장 |
| 박스 수 | 31,673개 | 7,915개 |
| 조합 그룹 수 | 2,870개 | 717개 |

- 두 분할의 조합 그룹 중복은 0개이다.
- 기존 공통 분할을 사용하며 담당자가 개별적으로 재분할하지 않는다.
- 원본 약품 클래스 118종은 추적 정보로 보존한다.
- 탐지 정답은 모든 알약에 대해 `pill=1`이다.
- `background=0`이며, 모델 클래스 수는 background를 포함해 `2`이다.

## 2. 코드 파일

| 파일 | 역할 |
|---|---|
| `detector_inputs.py` | Dataset·DataLoader 구성 함수 |
| `__init__.py` | Python 패키지 구성 |
| `detector_data_settings.example.json` | 팀원용 경로 설정 예시 |
| `detector_data_settings.json` | 각 컴퓨터의 실제 설정, Git 제외 |
| `.gitignore` | 개인 설정·캐시·백업 파일 제외 |

## 3. 준비할 데이터

Google Drive의 `gitkk_archive`에서 다음 자료를 준비한다.

| 다운로드할 자료 | 사용할 폴더 |
|---|---|
| 기존 `select` 폴더 | 내부에 `images/`가 있는 `select/` |
| `codeit_train_validated.zip` | 압축 해제한 `codeit_train_validated/` |
| **`training_package_detector.zip`** | ZIP 내부의 **`training_package/`** |

**기존 `training_package.zip`과 혼용하지 않는다.**

오늘 탐지용 ZIP도 내부 폴더 이름은 `training_package`이므로, 다운로드한 ZIP 이름과 내부의 `data/detector_data.py` 존재 여부를 확인한다.

이미지와 대용량 라벨은 Google Drive에서 공유하고 GitHub에는 올리지 않는다.

## 4. 경로 설정

1. `detector_data_settings.example.json`을 같은 폴더에 복사한다.
2. 복사본의 이름을 `detector_data_settings.json`으로 바꾼다.
3. 아래 세 항목을 본인 컴퓨터의 실제 경로로 수정한다.

| 설정 | 지정할 위치 |
|---|---|
| `package_dir` | 최종 `training_package` 폴더 |
| `image_roots.select` | `images`의 상위 `select` 폴더 |
| `image_roots.codeit` | `images`의 상위 대회 이미지 폴더 |

세 폴더는 서로 다른 위치에 있어도 된다.

실제 설정 파일은 Git 제외 대상이다. 예시 파일에 개인 절대 경로를 넣지 않는다.

## 5. 학습 코드에서 사용

탐지기 저장소를 Python에서 불러올 수 있는 상태에서 다음 함수 경로를 사용한다.

```python
data_factory_path = (
    "detector_data.detector_inputs:build_detector_inputs"
)
```

직접 호출하는 경우:

```python
from detector_data.detector_inputs import build_detector_inputs

train_loader, val_loader, label_map, data_summary = (
    build_detector_inputs(config=config, device=device)
)
```

`config`는 팀원의 `DetectorConfig`, `device`는 선택한 장치이다.

반환 순서는 다음과 같다.

1. `train_loader`
2. `val_loader`
3. `label_map`
4. `data_summary`

## 6. 입력 검증

```python
from detector import (
    validate_detection_batch,
    validate_detector_data_summary,
    validate_detector_label_map,
)

validate_detector_label_map(label_map)
validate_detector_data_summary(data_summary, label_map)

validate_detection_batch(
    next(iter(train_loader)), label_map
)
validate_detection_batch(
    next(iter(val_loader)), label_map
)

# 첫 배치 검사 후 학습 전에 로더 재생성
train_loader, val_loader, label_map, data_summary = (
    build_detector_inputs(config=config, device=device)
)
```

검증이 통과하면 팀원의 모델 생성·학습 코드로 이어간다. 데이터 구성 함수 자체는 모델을 생성하거나 학습을 시작하지 않는다.

## 7. DataLoader 설정

| 항목 | 설정 |
|---|---|
| 배치 크기 | `config.batch_size`, 현재 검증값 8 |
| 난수 시드 | `config.seed`, 현재 검증값 42 |
| CUDA | `num_workers=2`, `pin_memory=True` |
| MPS·CPU | `num_workers=0`, `pin_memory=False` |
| Train 순서 | 셔플 |
| Validation 순서 | 고정 |
| 마지막 작은 배치 | 유지 |
| 이미지 | CPU, RGB, float32, `[3,H,W]`, 값 범위 0~1 |
| 배치 구성 | `images`, `targets`, `meta` 각각 리스트 |
| 박스 좌표 | XYXY |
| 박스 면적 | 최종 좌표에서 재계산 |
| 외부 resize·ImageNet 정규화 | 적용하지 않음 |
| 현재 증강 | 적용하지 않음 |

약품별 `category_id`는 탐지 target label로 사용하지 않는다. 원본 약품 코드와 annotation ID는 `meta`에 보존한다.

## 8. 검증 결과와 한계

다음 사항을 확인했다.

- 전체 라벨·분할 목록·이미지 경로 연결
- 이미지 및 annotation ID 중복 없음
- Train/Validation 조합 그룹 중복 없음
- 전체 bbox 수치 및 양의 면적
- 팀원의 실제 클래스 매핑·집계 검증 통과
- 팀원의 실제 Train·Validation 첫 배치 검증 통과
- 새 GitHub용 코드와 공유 패키지의 집계·클래스 매핑 일치
- 분할별 SELECT·대회 표본 총 4장 시각 검토 통과

분할과 라벨 기록의 식별값:

```text
992d7f0416335b49891640b257cb4c2f3a82a9faa6c14cfce32391861a606010
```

현재는 전달받은 **단일 클래스 SSDLite 입력 명세**를 기준으로 검증했다. 다른 모델의 입력 형식과 학습 코드 호환성은 별도로 확인해야 한다.

모델의 순전파·역전파·실제 학습은 아직 검증하지 않았다. 첫 배치 검증과 표본 시각 검토가 전체 이미지의 annotation 정확성을 보장하지는 않는다.

## 9. 기존 공유 노트북과의 구분

공유 ZIP의 `START_HERE.ipynb`는 다음 경로를 사용한다.

```text
data.detector_data:build_detector_inputs
```

이 GitHub 폴더의 함수 경로는 다음과 같다.

```text
detector_data.detector_inputs:build_detector_inputs
```

    두 위치의 코드가 동일한 데이터 집계와 클래스 매핑을 반환하고 입력 검증을 통과함을 확인했다. 학습 코드에서는 선택한 함수 경로를 일관되게 사용한다.