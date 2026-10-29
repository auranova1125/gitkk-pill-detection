# 탐지 데이터 실행 구성 개선 및 검증 기록

작성일: 2026-10-28

## 변경 목적

다운로드한 자료를 어디에 배치하고 무엇을 실행할지 팀원이 확인할 수 있도록 공통 경로·실행 명령·시작 Notebook을 구성했다. 기본 코드는 `detector_data_datasets.zip`, 공유 검사 도구는 `detector_data_smoke_test.zip`을 기준으로 한다. 공유 검사 도구의 위치는 `detector_data/detector_data_smoke_test/`이다.

## 사용할 파일과 배치 위치

자료의 식별에는 ZIP·파일명을 사용한다.

| 기존 위치 | 옮길 내용 | 저장소 안의 위치 |
|---|---|---|
| `select/images/` | 이미지 파일 전체 | `datasets/images/select_aihub/` |
| `codeit_train_validated.zip`에서 꺼낸 `codeit_train_validated/images/` | 이미지 파일 전체 | `datasets/images/select_codeit/` |
| `training_package_detector.zip` 내부 `training_package/annotations/train.json` | `train.json` 파일 | `datasets/annotations/train.json` |
| 같은 ZIP 내부 `training_package/annotations/validation.json` | `validation.json` 파일 | `datasets/annotations/validation.json` |
| 같은 ZIP 내부 `training_package/split_manifest.csv` | `split_manifest.csv` 파일 | `datasets/split_manifest.csv` |
| 같은 ZIP 내부 `training_package/class_mapping.csv` | `class_mapping.csv` 파일 | `datasets/class_mapping.csv` |

이미지 파일은 `select_aihub/`, `select_codeit/` 바로 아래에 둔다. 기존 자료 폴더 전체를 넣거나 그 안에 `images/`를 한 겹 더 만들지 않는다. 파일명과 `source_…`, `codeit_…` 접두사는 유지한다.

`class_mapping.csv`는 약품별 번호를 보존하는 자료이며 탐지 모델의 정답은 `pill=1` 하나다. `dataset_info.json`은 이번 구성에서 사용하지 않는다.

## 경로 처리와 보존 범위

| JSON·분할 목록의 기존 경로 | 코드가 읽는 이미지 위치 |
|---|---|
| `select/images/source_…png` | `datasets/images/select_aihub/source_…png` |
| `codeit/images/codeit_…png` | `datasets/images/select_codeit/codeit_…png` |

코드의 경로 연결을 수정했으므로 팀원이 JSON·CSV 내용을 직접 편집할 필요는 없다. 이미지 ID, annotation ID, bbox, Train/Validation 배정과 해시 계산 방식을 유지한다.

이미지 폴더에는 제외 사진이 남아 있을 수 있다. 학습 대상은 폴더 전체가 아니라 라벨과 `split_manifest.csv`에 등록된 이미지다. 등록되지 않은 사진은 Dataset이나 소량 검사에 추가하지 않는다.

기본 사용에서는 개인 설정 파일이 필요 없다. `DETECTOR_DATA_SETTINGS` 환경 변수로 외부 설정을 지정한 경우에만 해당 설정을 적용한다. 이전 개인 설정이 지정되어 있다면 새 경로로 수정하거나 환경 변수를 해제해야 한다.

## 실행 코드

| 실행 방법 | 역할 |
|---|---|
| `python -m detector_data check` | 전체 라벨·분할·이미지 파일 존재 검사 |
| `python -m detector_data smoke` | 고정 분할에서 실제 이미지 표본을 읽고 팀원의 입력 검증 함수 실행 |
| `python -m detector_data smoke --model-step --report outputs/detector_data/model_smoke_report.json` | SSDLite 손실 계산·역전파·optimizer 1회·Validation 예측 형식 검사 |
| `python -m detector_data loaders` | 전체 DataLoader 구성 및 첫 배치 검사 |
| `detector_data/START_DETECTOR_DATA.ipynb` | 데이터 경로 확인부터 입력 검사·학습용 로더 준비까지 순서대로 실행 |
| `detector_data.start.prepare_training_inputs` | 설정 생성과 입력 검증 후 학습용 로더 반환. 학습은 시작하지 않음 |

공개 factory 경로는 `detector_data.detector_inputs:build_detector_inputs`다. `config`, `device` keyword argument와 로더 2개·label map·data summary의 반환 순서를 유지한다.

## 이전에 사용자 환경에서 확인한 데이터 수량

`training_package_detector.zip`을 구성한 뒤 사용자가 실행해 확인한 수량이다. 이번 보고서 갱신에서 실제 데이터로 다시 계산한 값은 아니다.

| 항목 | Train | Validation |
|---|---:|---:|
| 이미지 | 8,256장 | 2,063장 |
| 박스 | 31,673개 | 7,915개 |
| 조합 그룹 | 2,870개 | 717개 |

두 분할의 조합 그룹 중복은 0개이다. 이 수량은 학습에 등록된 이미지 기준이며 이미지 폴더의 전체 파일 수와 다를 수 있다.

기존 전체 분할·라벨 식별값:

```text
992d7f0416335b49891640b257cb4c2f3a82a9faa6c14cfce32391861a606010
```

실제 자료의 검사 결과와 비교해야 한다. 소량 검사 해시는 표본에 대한 값이므로 위 전체 해시와 비교하지 않는다.

## 이번 갱신에서 실제로 수행한 검사

개발 측 작업 환경에는 PyTorch·torchvision과 실제 학습 이미지·라벨이 없어 직접 모델을 실행하지 않았다. 이후 사용자가 학습 환경에서 실행한 출력과 첨부한 `shared_smoke_report.json`을 확인했으며, 실제 결과는 아래 별도 항목에 기록한다.

프로그램으로 생성한 테스트용 PNG·라벨을 사용해 다음 회귀 검사 17개를 실행했다.

1. 정상 라벨·분할·파일 경로의 수량 집계.
2. 작업 디렉터리와 무관한 저장소 기준 상대 경로 해석.
3. 명시한 사용자 설정 파일 적용.
4. 이미지 파일 누락 거부.
5. annotation ID 중복 거부.
6. bbox 이미지 경계 초과 거부.
7. bbox 면적 불일치 거부.
8. 이미지 경로의 기준 폴더 이탈 거부.
9. Train/Validation 조합 그룹 중복 거부.
10. 소량 검사에서 출처·전체 annotation 보존과 원본 불변 확인.
11. 소량 검사 표본 선택의 재현성.
12. `check` 명령의 성공 종료 코드·JSON 기록.
13. `check` 명령의 실패 종료 코드·JSON 기록.
14. 기존 출처 경로를 이미지 파일만 있는 `select_aihub`·`select_codeit` 폴더에 연결.
15. 폴더 안의 미등록 이미지가 학습·소량 검사 대상에 들어가지 않음.
16. 심볼릭 링크를 통한 이미지 기준 폴더 이탈 거부.
17. 기존 출처 이름의 폴더가 없어도 새 경로를 사용하며 JSON·CSV 내용을 보존.

실행 명령:

```bash
python -m unittest discover -s tests -p 'test_detector_data_preflight.py' -v
```

결과: **17개 통과, 실패 0개, 오류 0개**. 코드·Notebook 코드 셀의 Python 구문 검사도 통과했다. 기계 판독용 기록은 `reports/detector_data_regression_result.json`이다.

이 검사는 코드의 경로·라벨 처리 검증이며 실제 알약 annotation의 정확성이나 PyTorch 실행 성공을 보장하지 않는다.

## 사용자 환경에서 확인한 실제 검사 결과

근거: 사용자 제공 Notebook·터미널 출력과 첨부 `shared_smoke_report.json`. 아래 실제 모델 검사는 개발 측 환경에서 대신 실행한 것이 아니라 사용자의 학습 환경에서 실행한 결과다.

| 검사 | 상태 | 근거 |
|---|---|---|
| 전체 라벨·분할·이미지 파일 연결 | 통과 | 사용자 출력: Train 8,256장·31,673개 박스, Validation 2,063장·7,915개 박스 |
| 실제 이미지 소량 입력 검사 | 통과 | 사용자 실행 후 통과 확인 |
| 전체 학습 입력 준비 및 첫 배치 검증 | 통과 | Notebook `prepare_training_inputs` 실행 후 사용자 통과 확인 |
| 동일 표본 ZIP 생성 | 통과 | 생성 출력 및 패키지 식별값 확인 |
| 공유 패키지 파일·코드 동일성 검사 | 통과 | 사용자 check 출력 확인 |
| 공유 패키지 모델 순전파·역전파·Validation 예측 | 통과 | 첨부 `shared_smoke_report.json`의 model_step.status = passed |
| 다른 팀원 컴퓨터에서 재실행 | 미확인 | 이번 재실행은 사용자가 같은 컴퓨터에서 팀원 절차로 실행 |
| 전체 학습 완료·성능 검증 | 범위 밖 | 50 epoch 학습은 중단했으며 완료나 성능을 주장하지 않음 |

### 공유 패키지 실행 수치

| 항목 | 값 |
|---|---|
| 장치 | CPU |
| 배치 크기 | 2 |
| 표본 | Train 2장·8개 박스, Validation 2장·8개 박스 |
| 모델 | ssdlite320_mobilenet_v3_large |
| 입력 검증 | passed |
| 첫 배치 재현성 | passed |
| 그룹 중복 | 0개 |
| 경계 보정된 박스 | 0개 |
| 모델 연결 검사 | passed |
| optimizer 갱신 | 1회 |
| 손실값 | 9.117768287658691 |
| 사전 학습 backbone | 사용하지 않음 |
| checkpoint 저장 | 하지 않음 |
| 증강 | Train·Validation 모두 사용하지 않음 |

환경: Python 3.12.15, torch 2.14.1, torchvision 0.29.1, torchmetrics 1.9.0, numpy 2.5.4, Pillow 13.0.0. 버전은 첨부 결과 파일에 기록된 값을 그대로 옮긴 것이며 설치 권장 버전 목록이 아니다.

패키지 식별값:

```text
9178e655119902736ead03a862db6338cd77509f9c0d29364a85c08105a94b72
```

표본 분할·라벨 식별값:

```text
c1f87ba50ac5666ee7069061dc7980acd6973a7a292503027a2052ec567cc14a
```

위 표본 해시와 전체 데이터 해시는 대상이 다르다. 손실값은 연결 확인 기록이며 모델 성능 평가 수치가 아니다. 장치·라이브러리가 다른 환경에서 손실값까지 같아야 통과하는 것은 아니다.

공유 패키지 도구의 회귀 검사도 테스트용 자료로 9개를 별도 실행해 통과했다. 기본 데이터 코드의 회귀 검사 17개와 구분한다.

## 팀원의 재실행 방법

코드는 GitHub에서, `smoke_test.zip`은 Google Drive에서 받는다. ZIP을 `datasets/smoke_test/`에 풀고 학습 환경을 활성화한 뒤 저장소 최상위에서 실행한다.

```bash
python -m detector_data.detector_data_smoke_test.smoke_bundle check
python -m detector_data.detector_data_smoke_test.smoke_bundle run --model-step
```

전체 학습 데이터 없이 같은 표본을 사용한다. 팀원은 자신의 환경에서 실제로 실행하고 패키지 식별값과 통과 상태를 확인한다.

## PR 작성 및 머지 상태

현재 수정 파일은 사용자가 실행 결과를 확인한 상태다. 이 문서 작성 과정에서 GitHub push·PR 작성·머지는 수행하지 않았다.

1. 보고서와 PR 초안을 작업 브랜치에 반영한다.
2. 시작 Notebook과 테스트 Notebook의 실행 출력·실행 번호를 지운 뒤 저장한다. 코드 셀은 보존한다.
3. 개인 설정, 이미지, `datasets/`, `outputs/`, 모델 가중치를 제외하고 변경 파일만 커밋·push한다.
4. GitHub에서 PR을 작성하고 CI와 팀원의 리뷰 조건을 확인한다.
5. 승인 조건을 충족하면 머지한다.
