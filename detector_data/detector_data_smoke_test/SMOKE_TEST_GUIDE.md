# 같은 이미지로 스모크 테스트 공유하기

코드 폴더 위치: `공유 저장소/detector_data/detector_data_smoke_test/`. 기존 폴더를 밖으로 옮길 필요가 없습니다.

이 안내서는 팀원 A가 검사한 **같은 이미지 4장·같은 라벨·같은 코드**로 팀원 B도 재실행하는 방법을 설명합니다. 전체 이미지 데이터와 전체 분할 파일은 B의 컴퓨터에 없어도 됩니다.

이미지·라벨 ZIP은 Google Drive에 공유하고 실행 코드·가이드는 GitHub에 공유합니다. 이미지는 Git에 올리지 않습니다.

## 1. 추가할 코드

`detector_data_smoke_test.zip` 안의 `detector_data_smoke_test/` 내부 파일 3개를 현재 `detector_data/detector_data_smoke_test/` 폴더에 추가·교체합니다. `tests/` 안의 파일은 저장소의 `tests/`에 반영합니다. 폴더를 한 겹 더 넣지 않습니다.

| 파일 | 역할 |
|---|---|
| `detector_data/detector_data_smoke_test/__init__.py` | 스모크 테스트 Python 패키지 구성 |
| `detector_data/detector_data_smoke_test/smoke_bundle.py` | 검사 표본 ZIP 생성·무결성 확인·재실행 |
| `detector_data/detector_data_smoke_test/SMOKE_TEST_GUIDE.md` | 이 안내서 |
| `tests/test_smoke_bundle.py` | 공유 패키지 관련 회귀 검사 |

기존 `detector_data/` 안의 `detector_data_smoke_test/` 폴더에 이 파일들을 둡니다. 이 폴더에 빈 `__init__.py`도 포함되어 있습니다. 기존 `detector_data/`의 파일은 수정하지 않습니다. 기반 코드는 앞서 제공한 `datasets/images/select_aihub/`, `datasets/images/select_codeit/` 구조를 사용하는 버전입니다. `detector/`와 `detector_data/`가 있는 같은 저장소에서 실행합니다.

## 2. 팀원 A: 이미 통과한 표본 묶기

저장소 최상위에서 학습 환경을 활성화합니다.

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

생성 명령:

```bash
python -m detector_data.detector_data_smoke_test.smoke_bundle export
```

이 명령은 기본적으로 **`outputs/detector_data/model_smoke_report.json`**을 읽습니다. 새로 표본을 고르거나 50 epoch 학습을 시작하지 않습니다.

- 통과한 `smoke` 검사 기록에 나온 Train 2장·Validation 2장의 ID를 사용합니다.
- 검사 기록의 라벨·분할 파일 해시가 현재 자료와 같은지 확인합니다.
- 각 이미지의 모든 annotation과 기존 이미지·annotation ID, bbox, 분할 배정을 유지합니다.
- 원본 이미지를 ZIP에 넣기 위해 임시 복사하고 원본 자료를 이동·수정하지 않습니다.
- 패키지 생성 시점의 이미지·라벨 파일 해시와 코드 해시를 기록합니다. 코드 해시의 기준 시점도 패키지 생성 시점입니다. 이전 검사 당시의 이미지 픽셀 해시는 기록되어 있지 않으므로 이미지 해시의 기준 시점은 패키지 생성 시점입니다.
- 개인 절대 경로가 담긴 원본 검사 기록 전체를 공유하지 않고 필요한 결과 항목만 복사합니다.

생성 파일:

```text
datasets/smoke_test.zip
```

완료 문구가 나오면 이 ZIP을 Google Drive에 올립니다. 코드는 같은 GitHub 작업 브랜치에 commit·push하여 팀원이 받을 수 있도록 합니다. 공유자가 실제로 사용한 `detector/`, `detector_data/`의 Python 코드와 수신자의 코드가 같아야 합니다. 공유 전에 A도 아래 4번 절차로 ZIP을 풀고 재실행해 보는 것을 권장합니다.

### 검사 기록이 없거나 실패한 기록인 경우

다음 명령으로 모델 연결 검사부터 실행하고 통과 여부를 확인합니다.

```bash
python -m detector_data smoke --model-step --report outputs/detector_data/model_smoke_report.json
```

통과 후 `export`를 실행합니다. `run_training(...)`은 실행하지 않습니다.

다른 통과 기록을 명시하려면:

```bash
python -m detector_data.detector_data_smoke_test.smoke_bundle export --from-report outputs/detector_data/model_smoke_report.json
```

같은 이름의 ZIP을 자동으로 덮어쓰지 않습니다. 다시 만들 때에는 `--output datasets/smoke_test_02.zip`처럼 새 파일명을 지정하거나 기존 ZIP을 별도로 보관한 뒤 실행합니다.

## 3. ZIP에 들어 있는 내용

| 경로 | 내용 |
|---|---|
| `smoke_test/images/select_aihub/` | 검사 기록에 나온 AI Hub 이미지 |
| `smoke_test/images/select_codeit/` | 검사 기록에 나온 대회 이미지 |
| `smoke_test/annotations/train.json` | Train 2장의 모든 bbox |
| `smoke_test/annotations/validation.json` | Validation 2장의 모든 bbox |
| `smoke_test/split_manifest.csv` | 위 4장의 기존 분할·출처 기록 |
| `smoke_test/reference_result.json` | A의 입력·모델 검사 결과 중 공유용 항목 |
| `smoke_test/smoke_package_manifest.json` | 표본 ID·수량·파일 해시·코드 해시·환경 정보 |
| `smoke_test/README.md` | B의 실행 안내 |

실제 객체 수는 이미지마다 다르므로 총 박스 수를 16개로 고정하지 않습니다. 원본에 등록된 모든 박스를 그대로 보존합니다. 단일 클래스 탐지 검사이므로 약품별 `class_mapping.csv`가 없어도 됩니다.

## 4. 팀원 B: 같은 자료로 검사

1. A가 사용한 GitHub 코드 버전을 받습니다. 코드만 있고 `detector/`가 없는 상태에서는 실행할 수 없습니다.
2. Drive에서 A가 만든 `smoke_test.zip`을 받습니다.
3. 압축을 풀어 저장소의 **`datasets/smoke_test/`**에 배치합니다. `datasets/smoke_test/smoke_test/`처럼 중첩하지 않습니다.
4. 저장소 최상위에서 팀의 학습 환경을 활성화합니다.
5. 아래 두 명령을 순서대로 실행합니다.

```bash
python -m detector_data.detector_data_smoke_test.smoke_bundle check
python -m detector_data.detector_data_smoke_test.smoke_bundle run --model-step
```

- `check`: 코드·입력 파일 해시, 이미지·라벨·분할 연결을 검사합니다. PyTorch 실행은 포함하지 않습니다.
- `run --model-step`: 바로 그 패키지의 표본으로 팀원의 입력 검증 함수와 SSDLite 한 배치 순전파·역전파·optimizer 1회·Validation 예측 검사를 실행합니다.
- 기본 장치는 CPU이며 사전 학습 가중치를 다운로드하지 않습니다. checkpoint를 저장하거나 전체 학습을 시작하지 않습니다.

통과 기준:

```text
모델 연결 검사: passed
[통과] 공유 스모크 패키지 검사 완료
```

B의 기록:

```text
outputs/detector_data/shared_smoke_report.json
```

B가 전체 학습 데이터가 있는 컴퓨터를 사용해도 이 명령은 `datasets/smoke_test/` 안의 표본만 읽습니다. 이전의 개인 데이터 경로 설정도 사용하지 않습니다. 전체 데이터를 읽는 `python -m detector_data smoke`와 공유 패키지 전용 명령을 구분합니다.

## 5. Notebook에서 실행하고 싶은 경우

기존 시작 Notebook의 첫 코드 셀로 저장소 위치와 학습 커널을 확인한 상태에서 새 셀에 넣습니다.

A의 ZIP 생성:

```python
subprocess.run(
    [sys.executable, "-m", "detector_data.detector_data_smoke_test.smoke_bundle", "export"],
    check=True,
)
```

B의 모델 연결 검사:

```python
subprocess.run(
    [sys.executable, "-m", "detector_data.detector_data_smoke_test.smoke_bundle", "run", "--model-step"],
    check=True,
)
```

같은 검사를 터미널과 Notebook에서 두 번 실행할 필요는 없습니다.

## 6. 무엇을 비교하는가?

- A와 B의 패키지 식별값 `package_hash`가 같은지 확인합니다.
- 입력 검증 통과와 모델 검사 통과 여부를 비교합니다.
- PyTorch·torchvision·장치가 다르면 손실값까지 정확히 같지 않을 수 있습니다. 성능 비교용 실험 결과가 아닙니다.
- B는 패키지에 보이는 A의 기록만 읽는 것이 아니라 자신의 환경에서 실제로 재실행합니다.
- 코드를 수정했다면 같은 코드인지 확인하는 검사에서 멈출 수 있습니다. A가 사용한 브랜치의 코드를 받아 다시 실행합니다. Windows/Mac 줄바꿈 차이는 허용합니다.

## 7. 개발 측 검사 범위

공유 패키지 도구는 테스트용 PNG·라벨로 검사했습니다. 실제 사용자의 알약 이미지 ZIP 생성이나 실제 PyTorch 모델 재실행은 개발 측에서 수행하지 않았습니다. A의 실제 `export` 결과와 B의 실제 `run` 결과로 완료 여부를 확인합니다.
