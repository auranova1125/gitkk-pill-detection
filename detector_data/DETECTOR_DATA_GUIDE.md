# 탐지 데이터 준비 및 실행 안내

공유 자료를 저장소 안의 `datasets/`에 배치한 뒤 `START_DETECTOR_DATA.ipynb`를 실행합니다. 기본 사용에서는 코드에 개인 경로를 적지 않습니다.

## 1. 어떤 자료를 사용하는가?

| 자료 | 역할 |
|---|---|
| 기존 `select/images/` 안의 이미지 파일 | 선정한 AI Hub 이미지 |
| `codeit_train_validated.zip`에서 꺼낸 `images/` 안의 이미지 파일 | 검증한 대회 Train 이미지 |
| `training_package_detector.zip` 안의 `training_package/annotations/train.json` | Train 이미지와 모든 알약 bbox |
| 같은 `training_package/annotations/validation.json` | Validation 이미지와 모든 알약 bbox |
| 같은 `training_package/split_manifest.csv` | 이미지별 고정 분할·출처·원본 정보 |
| 같은 `training_package/class_mapping.csv` | 공통 약품 클래스 번호 |

학습 라벨과 분할 목록의 출처는 **`training_package_detector.zip`**입니다. 이름이 다른 `training_package.zip` 또는 `training_package_classifier.zip`의 파일을 대신 사용하지 않습니다.

탐지 정답은 모든 알약에 대해 `pill=1`입니다. background는 `0`, 모델 클래스 수는 `2`입니다. `class_mapping.csv`의 약품별 118종 번호는 추적·분류기 연결 기준이며 단일 클래스 탐지 target label로 사용하지 않습니다.

## 2. 이미 자료를 받은 팀원: 무엇을 어디로 옮기는가?

다음 표는 **공유 저장소의 최상위 폴더를 기준으로 한 이동 목적지**입니다. 예를 들어 저장소가 `gitkk-pill-detection`이라면 그 안에 `datasets/`를 만듭니다.

| 기존 자료의 위치 | 옮길 내용 | 저장소 안의 새 위치 |
|---|---|---|
| `select/images/` | 폴더 안의 이미지 파일 전체 | `datasets/images/select_aihub/` |
| `codeit_train_validated/images/` | 폴더 안의 이미지 파일 전체 | `datasets/images/select_codeit/` |
| `training_package_detector.zip` 내부 `training_package/annotations/train.json` | `train.json` 파일 | `datasets/annotations/train.json` |
| 같은 ZIP 내부 `training_package/annotations/validation.json` | `validation.json` 파일 | `datasets/annotations/validation.json` |
| 같은 ZIP 내부 `training_package/split_manifest.csv` | `split_manifest.csv` 파일 | `datasets/split_manifest.csv` |
| 같은 ZIP 내부 `training_package/class_mapping.csv` | `class_mapping.csv` 파일 | `datasets/class_mapping.csv` |

**이미지 파일만 꺼내 옮깁니다.** 기존 `select` 또는 `codeit_train_validated` 폴더 전체를 새 이미지 폴더에 넣지 않습니다. `select_aihub/images/사진.png`가 아니라 `select_aihub/사진.png`처럼 이미지 파일이 바로 보여야 합니다.

파일명을 바꾸지 않습니다. SELECT 파일의 `source_…`, 대회 파일의 `codeit_…` 접두사도 유지합니다. 기존 이미지 폴더의 제외 사진이 함께 이동해도 됩니다. 폴더에 있는 모든 사진을 학습 대상으로 수집하는 방식이 아닙니다.

기존 `select/annotations/`, 기존 `select` 안의 CSV, `training_package_classifier.zip`은 이번 이미지 폴더로 옮기지 않습니다. 원본 보관 위치에 그대로 두면 됩니다. `class_mapping.csv`는 위 표에 적힌 ZIP 내부 파일을 사용합니다.

이미 위 파일들을 다운로드했다면 이미지를 다시 다운로드할 필요가 없습니다. 단, 이전 프로그램이나 Notebook이 기존 위치를 계속 사용한다면 실제 이동 전에 영향을 확인하세요.

## 3. Google Drive에서 새 구조로 받는 팀원

공유 Drive의 **`gitkk_archive/datasets/`** 폴더를 받은 뒤 저장소 최상위에 `datasets/`라는 이름으로 배치합니다. 2번의 기존 자료 정리 작업은 반복하지 않습니다.

배치 후 확인할 경로는 다음과 같습니다.

| 경로 | 확인할 내용 |
|---|---|
| `datasets/images/select_aihub/사진.png` | AI Hub 이미지 파일이 바로 보임 |
| `datasets/images/select_codeit/사진.png` | 대회 이미지 파일이 바로 보임 |
| `datasets/annotations/train.json` | Train 라벨 |
| `datasets/annotations/validation.json` | Validation 라벨 |
| `datasets/split_manifest.csv` | 공통 분할 목록 |
| `datasets/class_mapping.csv` | 약품 클래스 매핑 |

`datasets/datasets/`처럼 같은 폴더가 중첩되지 않았는지 확인합니다. 위 표의 `사진.png`는 위치를 설명하는 예시이며 실제 파일명은 기존 이름 그대로입니다.

`dataset_info.json`은 이번 구성에서 만들거나 다운로드할 필요가 없습니다.

## 4. 기존 JSON의 경로를 편집해야 하는가?

**이 버전의 코드를 쓰면 편집할 필요가 없습니다.** JSON과 CSV의 기존 출처별 경로를 새 이미지 폴더에 연결합니다.

| JSON·분할 목록에 기록된 값 | 이 코드가 읽는 실제 위치 |
|---|---|
| `select/images/source_…png` | `datasets/images/select_aihub/source_…png` |
| `codeit/images/codeit_…png` | `datasets/images/select_codeit/codeit_…png` |

위 표의 `…`는 파일명 일부를 생략한 표시입니다. 실제 파일은 이름 전체를 유지합니다.

이 연결은 `settings.py`의 기본 설정과 경로 처리 함수가 담당합니다. 팀원이 JSON·CSV에서 문자열을 직접 바꾸지 않습니다. Train/Validation 배정, 이미지 ID, annotation ID, bbox는 유지합니다.

라벨·분할 목록에 없는 사진은 이미지 폴더에 남아 있어도 읽지 않습니다. `split_manifest.csv`의 경로를 다른 프로그램에서 직접 사용할 때에도 위 경로 연결을 적용해야 합니다. 분류기가 이 탐지용 경로 함수를 자동으로 사용하는 것은 아닙니다.

## 5. 실행 환경 준비

공유 저장소의 `detector/`와 `detector_data/` 폴더가 있는 위치에서 실행합니다. 저장소가 컴퓨터의 어디에 있든 내부 구조만 같으면 됩니다.

팀 저장소의 학습 환경을 사용합니다. 기존 `.venv`를 쓰는 경우의 활성화 예시:

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

패키지 버전은 팀 저장소의 의존성 설정을 따릅니다. 이 안내서에서는 새로운 버전을 임의로 지정하지 않습니다. Notebook도 같은 학습 환경을 커널로 선택합니다.

## 6. 실행 순서

처음 사용하는 팀원은 **`detector_data/START_DETECTOR_DATA.ipynb`**를 열고 셀을 위에서부터 실행하면 됩니다. 이 Notebook은 데이터 검사와 로더 준비까지만 수행합니다.

터미널에서 단계별로 실행할 수도 있습니다.

### 6-1. 전체 라벨·분할·파일 경로 검사

```bash
python -m detector_data check
```

전체 이미지 파일 존재 여부, 라벨·분할 연결, ID 중복, 조합 그룹 중복, bbox 경계·면적을 확인합니다. Python 기본 기능만 사용합니다. 이미지 픽셀을 전부 열거나 실제 알약 위치를 판정하지는 않습니다.

성공 출력: `[통과] 탐지 데이터 검사 완료`

기록: `outputs/detector_data/check_report.json`

`training_package_detector.zip`을 생성한 뒤 사용자 환경에서 확인했던 수량은 다음과 같습니다. 현재 배치가 맞는지는 위 명령의 실제 출력으로 확인합니다.

| 항목 | Train | Validation |
|---|---:|---:|
| 이미지 | 8,256장 | 2,063장 |
| 박스 | 31,673개 | 7,915개 |
| 조합 그룹 | 2,870개 | 717개 |

조합 그룹 중복은 0개입니다. 이 표는 학습에 등록된 수량이며 이미지 폴더 안의 파일 전체 수와 같을 필요는 없습니다.

### 6-2. 실제 이미지 4장으로 입력 검사

```bash
python -m detector_data smoke
```

Train 2장과 Validation 2장을 기존 배정에서 선택합니다. 가능한 경우 각 분할에서 SELECT와 대회 이미지를 한 장씩 선택합니다. 각 이미지의 모든 annotation을 유지합니다.

팀원의 검증 함수로 클래스 매핑, 표본 집계, 이미지·target·meta를 확인하고 첫 배치 재현성도 확인합니다. 표본 라벨은 임시 생성 후 제거하며 원본 파일은 수정하지 않습니다. 모델은 아직 실행하지 않습니다.

기록: `outputs/detector_data/smoke_report.json`

표본 집계와 표본 해시는 전체 데이터의 집계·해시와 다릅니다.

### 6-3. 모델 연결 검사

Notebook과 터미널 중 **한 가지 방법만** 사용합니다. 이미 Notebook에서 입력 준비까지 통과했다면 방법 A를 권장합니다.

#### 방법 A: Notebook의 새 코드 셀에서 실행

앞 단계에서 선택한 학습 커널을 그대로 사용합니다. 별도 터미널을 열거나 활성화할 필요가 없습니다. 첫 번째 코드 셀에서 불러온 `subprocess`, `sys`를 사용합니다.

```python
subprocess.run(
    [
        sys.executable, "-m", "detector_data", "smoke",
        "--model-step",
        "--report", "outputs/detector_data/model_smoke_report.json",
    ],
    check=True,
)
```

`sys.executable`은 현재 Notebook 커널의 Python을 사용합니다. 커널을 재시작했다면 첫 번째 코드 셀부터 다시 실행합니다.

#### 방법 B: 저장소 터미널에서 실행

Notebook 커널을 선택한 것과 터미널의 학습 환경 활성화는 별개입니다. 새 터미널에서는 먼저 **공유 저장소 최상위**로 이동하고 환경을 활성화합니다. 터미널을 새로 열었다면 다시 활성화해야 할 수 있습니다.

Mac/Linux:

```bash
source .venv/bin/activate
python -m detector_data smoke --model-step --report outputs/detector_data/model_smoke_report.json
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m detector_data smoke --model-step --report outputs/detector_data/model_smoke_report.json
```

`zsh: command not found: python`은 터미널에서 `python` 명령을 찾지 못했다는 뜻입니다. 이 경우 데이터·모델 검사 자체는 시작되지 않았습니다. 저장소의 `.venv`가 있고 위 활성화를 수행했는지 확인합니다. `python3`로만 바꾸면 팀 학습 환경과 다른 Python이 실행될 수 있으므로, 먼저 학습 환경을 확인합니다.

#### 통과 기준과 검사 범위

출력에 `모델 연결 검사: passed`가 나오고 명령이 오류 없이 종료되면 통과입니다. 기록은 `outputs/detector_data/model_smoke_report.json`입니다.

SSDLite로 Train 한 배치의 손실 계산·역전파·optimizer 1회 갱신과 Validation 한 배치의 예측 형식을 확인합니다. 기본 장치는 CPU입니다. 사전 학습 가중치는 다운로드하지 않으며 checkpoint도 저장하지 않습니다. 학습 성능 평가용 결과는 아닙니다.

### 6-4. 전체 DataLoader 검사

```bash
python -m detector_data loaders
```

전체 Dataset의 첫 Train/Validation 배치를 검증합니다. 기본 배치는 8이며 다음처럼 변경할 수 있습니다.

```bash
python -m detector_data loaders --batch-size 4 --device cpu
```

기록: `outputs/detector_data/loaders_report.json`

## 7. Notebook에서 실제 학습 입력 준비

저장소가 Python에서 보이는 상태에서 다음 셀을 실행합니다. `START_DETECTOR_DATA.ipynb`에도 포함되어 있습니다.

```python
from detector_data.start import prepare_training_inputs

config, device, train_loader, val_loader, label_map, data_summary = (
    prepare_training_inputs(
        run_name="pill_baseline_01",
        batch_size=8,
        seed=42,
        max_epochs=50,
    )
)
```

이 함수는 전체 라벨·경로와 첫 배치를 검사하고 학습용 로더를 다시 생성해 반환합니다. 학습은 아직 시작하지 않습니다. 배치 크기를 바꿔도 공통 분할은 유지합니다.

실행 예시의 기본 `learning_rate`는 `0.001`, `weight_decay`는 `0.0005`입니다. 실제 실험 설정은 팀 합의에 맞춥니다.

팀원의 SSDLite 학습 코드로 이어갈 때만 별도 셀에서 실행합니다.

```python
from detector import run_training

result = run_training(
    config, train_loader, val_loader, label_map, data_summary,
    device=device,
)
```

이 셀부터 실제 학습을 시작합니다. 기본 호출은 사전 학습 backbone을 사용하므로 첫 실행 때 가중치를 다운로드할 수 있습니다. 결과는 `outputs/detector/pill_baseline_01/`에 저장됩니다. 같은 이름의 학습 결과가 이미 있으면 새로운 `run_name`을 사용합니다.

기존 학습 Notebook에 전달할 factory 경로는 다음과 같습니다.

```text
detector_data.detector_inputs:build_detector_inputs
```

함수는 `config`, `device`를 keyword argument로 받습니다. 반환 순서는 `train_loader`, `val_loader`, `label_map`, `data_summary`입니다. `detector_inputs.py`를 직접 실행하지 않습니다.

## 8. 입력 및 설정 기준

| 항목 | 내용 |
|---|---|
| 이미지 | CPU RGB `float32 [3,H,W]`, 범위 `[0,1]` |
| 배치 | `images`, `targets`, `meta` 각각 리스트 |
| bbox | 원본 XYWH를 XYXY로 변환 |
| target label | 모든 알약이 정수 `1` |
| area | 변환된 bbox에서 계산 |
| 추적 정보 | 원본 약품 코드·ID·bbox·출처를 `meta`에 보존 |
| 기본 증강 | 사용하지 않음 |
| Validation | 무작위 변환 없음 |
| 외부 resize·ImageNet 정규화 | 사용하지 않음. SSDLite 내부 처리 사용 |
| CUDA 로더 | workers 2, pin_memory 사용 |
| MPS·CPU 로더 | workers 0, pin_memory 사용하지 않음 |

현재 모델 연결 검사는 제공된 SSDLite를 대상으로 합니다. 다른 모델은 입력 명세를 따로 확인합니다.

## 9. 오류 확인

| 증상 | 확인할 내용 |
|---|---|
| 필요한 데이터 파일이 없음 | `datasets/annotations/`와 `datasets/split_manifest.csv` 배치 확인 |
| 이미지 파일이 없음 | 이미지가 `select_aihub`·`select_codeit` 바로 아래에 있는지 확인. 중간 `images/` 폴더를 추가하지 않음 |
| 이미지 출처 형식 오류 | JSON·CSV를 임의 편집했는지 확인하고 `training_package_detector.zip` 내부 파일과 비교 |
| `No module named detector_data` | 저장소 최상위에서 실행하고 작업 브랜치 파일 확인 |
| `zsh: command not found: python` | 터미널에서 `.venv`를 활성화하거나 Notebook의 방법 A 사용 |
| PyTorch 등 불러오기 실패 | 현재 Python 또는 Notebook 커널이 팀 학습 환경인지 확인 |
| bbox·분할 연결 검사 실패 | 출력과 검사 JSON을 데이터 담당자에게 전달. 임의로 좌표를 고치지 않음 |
| 메모리 부족 | 배치 크기를 줄임. 모델 소량 검사는 배치 2 이상 유지 |

성공 시 종료 코드 0과 `status: passed`, 실패 시 종료 코드 1과 `status: failed`를 기록합니다. 입력 검사 통과와 모델 검사 통과는 구분합니다. 검사 JSON에는 개인 경로가 포함될 수 있으므로 공개 전 확인합니다.

## 10. 공통 위치를 쓸 수 없는 경우에만

기본 사용자는 개인 설정 파일이 필요 없습니다. `detector_data_settings.json`을 자동으로 읽지 않습니다.

외부 디스크 등의 경로를 사용하려면 `detector_data_settings.example.json`을 복사해 개인 설정 파일로 만들고 `DETECTOR_DATA_SETTINGS` 환경 변수로 명시합니다.

| 설정 항목 | 지정할 위치 |
|---|---|
| `package_dir` | `annotations/`, `split_manifest.csv`, `class_mapping.csv`가 있는 `datasets` 폴더 |
| `image_roots.select` | 이미지 파일이 바로 있는 `select_aihub` 폴더 |
| `image_roots.codeit` | 이미지 파일이 바로 있는 `select_codeit` 폴더 |

이 버전의 `image_roots`는 기존 `images/`의 상위 자료 폴더를 뜻하지 않습니다. 이전 개인 설정 파일이 있다면 위 기준으로 수정해야 합니다. 상대 경로의 기준은 저장소 최상위입니다.

Mac/Linux:

```bash
export DETECTOR_DATA_SETTINGS=detector_data/detector_data_settings.json
python -m detector_data check
```

Windows PowerShell:

```powershell
$env:DETECTOR_DATA_SETTINGS="detector_data/detector_data_settings.json"
python -m detector_data check
```

공통 기본 경로로 되돌리려면 환경 변수를 해제합니다.

Mac/Linux:

```bash
unset DETECTOR_DATA_SETTINGS
```

Windows PowerShell:

```powershell
Remove-Item Env:DETECTOR_DATA_SETTINGS -ErrorAction SilentlyContinue
```

개인 설정 파일은 Git에 올리지 않습니다. Google Drive의 데이터를 재배치했다면 로컬에도 같은 폴더 구조를 적용합니다.
