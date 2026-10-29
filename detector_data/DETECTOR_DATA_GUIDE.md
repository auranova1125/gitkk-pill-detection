# 탐지 데이터 사용 안내

이미지와 라벨을 읽어 팀원의 탐지 모델에 전달하는 코드입니다.
저장된 Train/Validation 배정을 그대로 사용합니다.

## 1. 먼저 할 일

팀 저장소에서 이 PR이 반영된 코드를 받으세요. 저장소에 `detector/`와 `detector_data/`가 있어야 합니다.
Notebook 커널은 저장소의 학습 환경을 선택합니다.
이미지 데이터는 GitHub에 올리지 않고 Google Drive에서 받습니다.

먼저 **표본 4장으로 연결 검사**, 이후 **전체 데이터로 학습 준비** 순서로 진행하세요.

## 2. 표본 4장으로 연결 검사

Drive의 기존 `smoke_test.zip`을 받아 압축을 풀면 아래 위치여야 합니다.

```text
datasets/smoke_test/smoke_package_manifest.json
datasets/smoke_test/annotations/train.json
datasets/smoke_test/annotations/validation.json
datasets/smoke_test/split_manifest.csv
datasets/smoke_test/images/select_aihub/...
datasets/smoke_test/images/select_codeit/...
```

`detector_data/detector_data_smoke_test/smok_test.ipynb`을 열어 위에서부터 실행합니다.
이 검사는 전체 학습 데이터를 요구하지 않습니다. 팀원이 ZIP을 새로 만들거나 다시 업로드할 필요도 없습니다.

검사하는 내용:

1. ZIP 생성 시 기록된 파일 해시와 현재 이미지·라벨이 일치하는지 확인합니다.
2. Train 2장·Validation 2장을 배치로 읽고 팀원의 입력 검증 함수를 실행합니다.
3. SSDLite로 손실 계산, 역전파, 가중치 갱신 1회와 Validation 예측을 수행합니다.

`[3/3 통과]`와 `[통과] 검사 완료`가 나오고, 보고서의 `status`가 `passed`이면 완료입니다.
가중치는 저장하지 않습니다. 탐지 정확도나 전체 학습 성공을 평가하는 검사는 아닙니다.

보고서: `outputs/detector_data/shared_smoke_report.json`

기존 ZIP의 코드 해시와 현재 코드를 같게 만드는 검사는 제거했습니다.
ZIP 내부 파일은 계속 해시로 검사하고, 실행한 현재 코드 해시는 새 보고서에 기록합니다.
ZIP 안의 예전 README 대신 이 안내와 저장소의 Notebook을 따르세요.

## 3. 전체 데이터 배치

기존에 받은 자료에서 아래 파일만 옮깁니다. 이미지 파일명은 바꾸지 않습니다.

| 기존 자료 | 저장소 안에 둘 위치 |
|---|---|
| `select/images/` 안의 이미지 파일 | `datasets/images/select_aihub/` |
| `codeit_train_validated.zip`의 `images/` 안의 이미지 파일 | `datasets/images/select_codeit/` |
| `training_package_detector.zip` 안 `training_package/annotations/train.json` | `datasets/annotations/train.json` |
| 같은 패키지의 `annotations/validation.json` | `datasets/annotations/validation.json` |
| 같은 패키지의 `split_manifest.csv` | `datasets/split_manifest.csv` |
| 같은 패키지의 `class_mapping.csv` | `datasets/class_mapping.csv` |

추가 5장을 제외한 라벨 기준: Train 8,256장·31,673개 박스, Validation 2,063장·7,915개 박스입니다.
다운로드한 라벨이 이 수량과 맞는지 확인하세요.

이미지 폴더에 제외된 사진이 남아 있어도 라벨과 분할 목록에 없는 이미지는 읽지 않습니다.
JSON의 `select/images/...`, `codeit/images/...` 경로는 `settings.py`에서 위 폴더로 연결합니다.
JSON 경로를 직접 수정할 필요가 없습니다.
`class_mapping.csv`는 원본 약품 클래스 연결 정보입니다. 단일 클래스 탐지 라벨 생성에는 사용하지 않습니다.
`dataset_info.json`은 필요하지 않습니다.

## 4. 전체 데이터로 학습 준비

`detector_data/START_DETECTOR_DATA.ipynb`을 열어 위에서부터 실행합니다.

1. 저장소와 Python 위치 확인
2. 전체 라벨·분할·이미지 파일 존재 검사
3. 전체 DataLoader 생성과 첫 배치 검증

3번에서 `config`, `device`, `train_loader`, `val_loader`, `label_map`, `data_summary`가 준비됩니다.
이 Notebook은 학습을 시작하지 않습니다.
실제 학습을 시작할 때만 새 셀에 다음 코드를 작성합니다.

```python
from detector import run_training

result = run_training(
    config, train_loader, val_loader, label_map, data_summary,
    device=device,
)
```

이 호출은 전체 학습을 시작하고 사전 학습 가중치를 다운로드할 수 있습니다.
같은 `run_name`의 학습 결과가 있으면 새로운 이름을 지정하세요.
현재 모델 연결 검사는 팀원의 SSDLite 구현 기준입니다. 다른 탐지 모델은 입력 형식이 같은지 별도로 확인해야 합니다.

## 5. 코드에서 볼 부분

| 파일 | 필요한 이유 |
|---|---|
| `settings.py` | 두 출처의 이미지 경로를 실제 폴더로 연결 |
| `data_records.py` | JSON·CSV를 한 번의 검증 로직으로 읽고 이미지별 모든 박스를 연결 |
| `detector_inputs.py` | Dataset, XYWH→XYXY 변환, pill 라벨 1, 배치 구성 |
| `start.py` | 팀원의 설정과 검증 함수를 연결하고 학습용 로더 반환 |
| `smoke_test.py` | 공유 표본의 무결성과 모델 연결 검사 |
| `__main__.py` | Notebook에서 호출하는 검사 명령의 시작점 |

Dataset은 이미지 한 장과 해당 이미지의 모든 박스를 반환합니다.
DataLoader는 이를 배치로 묶습니다. 배치는 `images`, `targets`, `meta` 세 리스트입니다.
이미지는 CPU의 RGB float32 `[3,H,W]`, 값 범위는 `[0,1]`입니다.
118개 약품 코드는 추적 정보로 남기고, 탐지 target은 모든 알약을 `pill=1`로 처리합니다.
Validation에는 무작위 변환이 없고 현재 기본 학습 증강도 꺼져 있습니다.
정규화와 크기 변경은 모델 내부에서 수행합니다.

## 6. 검사 보고서 공유

실행 보고서는 `outputs/detector_data/`에 저장됩니다.
이 폴더는 개인 실행 결과이므로 자동으로 Git에 올라가지 않을 수 있습니다.
리뷰 근거로 공유할 때는 직접 확인한 보고서를 `reports/`로 복사해 커밋합니다.

`reports/previous_shared_smoke_report.json`은 변경 전 코드의 실행 기록입니다.
이번 변경 코드의 모델 검사 통과를 증명하는 파일로 사용하면 안 됩니다.
새로 실행한 `shared_smoke_report.json`을 공유하고, 보고서에서 표본 수·장치·모델 검사 상태를 보여주세요.

## 7. 터미널에서 실행할 경우

저장소 루트에서 학습 환경을 먼저 활성화합니다.
Mac/Linux: `source .venv/bin/activate`
Windows PowerShell: `.venv\Scripts\Activate.ps1`

```bash
python -m detector_data check
python -m detector_data smoke --report outputs/detector_data/shared_smoke_report.json
```

`check`는 전체 라벨과 경로만 검사합니다. 전체 이미지 픽셀을 다시 읽거나 실제 알약 위치를 판단하지 않습니다.
`smoke`는 항상 공유 표본으로 모델 연결까지 검사합니다.
모델 실행 없이 표본 파일만 확인하려면 `python -m detector_data smoke-check`를 사용합니다.
기존 긴 `smoke_bundle check/run` 명령은 짧은 명령으로 연결하는 호환 코드만 남겼습니다.

데이터를 저장소 밖에 두어야 하는 팀원만 `detector_data_settings.example.json`을 복사해 경로를 설정하고
`DETECTOR_DATA_SETTINGS` 환경 변수로 지정합니다. 위 기본 배치를 사용하면 개인 설정은 필요 없습니다.
