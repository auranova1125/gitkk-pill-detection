# 탐지기 데이터셋 사용 안내서

## 사용 전 조건

`detector_data/`와 모델 복원 수정이 `auranova1125/gitkk-pill-detection`의 main에 병합된 이후 사용합니다. Colab GPU 런타임에서 `detector_data/train_detector_colab.ipynb`를 열고 순서대로 실행합니다. Notebook은 main 코드를 clone합니다. 저장소 접근 권한과 데이터 폴더 접근 권한은 각각 필요합니다.

## 준비물과 데이터 위치

- Colab GPU 런타임(T4에서 실행 확인)
- Google Drive 내 `gitkk_archive/datasets`
- 기존 분할과 이미지 경로를 유지한 데이터

| 상대 경로 | 내용 |
|---|---|
| images/select_aihub/ | AI-Hub 이미지 |
| images/select_codeit/ | Codeit 이미지 |
| annotations/train.json | 학습 라벨 |
| annotations/validation.json | 검증 라벨 |
| split_manifest.csv | 기존 분할 목록 |
| class_mapping.csv | 원본 클래스 대응 정보 |

기본 DATA_DIR은 `/content/drive/MyDrive/gitkk_archive/datasets`입니다. 공유 폴더는 내 드라이브에 바로가기를 추가하거나 실제 위치에 맞게 DATA_DIR을 수정합니다. `smoke_test.zip`은 이번 전체 데이터 2 epoch 실행에 사용하지 않습니다. Train/Validation을 다시 나누지 않습니다.

## 실행 순서

1. Drive 연결, main 코드 clone, 의존성 설치 및 GPU 확인
2. 데이터 경로 확인과 학습 설정
3. 데이터 연결과 네 가지 반환값 확인
4. 실제 Train/Validation 배치 검증, 정답 bbox 확인
5. 전체 데이터 2 epoch 학습
6. 지표 그래프와 저장 파일 확인
7. best.pt 복원, 전체 검증 평가·예측 저장, 정답·예측 비교

기본값은 batch_size=8, max_epochs=2, seed=42입니다. 증강은 모두 0으로 유지합니다. 입력 이미지는 원본 크기의 float32, 0~1 Tensor이며 bbox는 XYXY로 전달됩니다. 모델 내부에서 입력 크기 변환을 처리합니다.

## 반환값

| 반환값 | 역할 |
|---|---|
| train_loader | 학습 이미지·bbox 배치, shuffle 적용 |
| val_loader | 검증 배치, 순서 고정 |
| label_map | 배경 0, 알약 pill=1; 모델 클래스 수 2 |
| data_summary | 이미지·박스·그룹 수, 분할 해시, 증강 설정 |

`detector_data.detector_inputs:build_detector_inputs`가 네 값을 반환합니다. label_map과 data_summary는 원본 라벨·분할 목록으로부터 생성하며, JSON을 사전에 별도로 준비하지 않습니다. 학습 시작 시 실행 폴더에 저장됩니다. 이는 118종을 구분하는 분류기 class_map과 다른 단일 클래스 탐지용 정의입니다.

## 데이터 읽기 방식

### A. Drive 직접 연결 — 이번에 실행 확인

Notebook 기본 방식입니다. 복사 없이 바로 시작할 수 있지만 많은 이미지를 읽는 동안 Drive I/O가 병목이 될 수 있습니다. 이번 T4 실행에서는 학습·검증 2 epoch 진행 표시 시간 합계가 약 73분이었고, 최종 재평가·예측 시간은 별도입니다. 다른 환경에서 같은 시간을 보장하지 않습니다.

### B. ZIP을 Colab 로컬에 복사·압축 해제 — 대안, 이번 실행에서는 미검증

전체 datasets.zip을 Drive에 올린 후 ZIP 하나를 `/content`로 복사하고 로컬에서 압축을 풉니다. 이미지별 Drive 접근을 줄일 수 있습니다. ZIP과 압축 해제 데이터가 함께 들어갈 디스크 여유를 먼저 확인합니다. ZIP 내부에 datasets/가 포함되는지 확인한 뒤, 풀린 폴더의 images/와 annotations/가 있는 위치를 DATA_DIR로 지정합니다. settings 생성 및 데이터 연결은 경로 변경 후 다시 실행합니다.

사용자가 만든 datasets.zip 용량은 약 18.93GB이며 내부 구조는 아직 확인되지 않았습니다. 이 ZIP을 사용한 실행 시간과 성공 여부는 검증 전입니다. 이미 학습 중인 로더의 경로를 중간에 바꾸지 않습니다.

## 결과 저장

기본 위치: `/content/drive/MyDrive/gitkk_runs/detector/<run_name>/`

| 파일 | 내용 |
|---|---|
| best.pt | 검증 기준 최적 체크포인트 |
| last.pt | 마지막 체크포인트 |
| resolved_config.json | 실행 설정 |
| label_map.json, data_summary.json | 사용한 데이터 정의와 요약 |
| metrics.csv | epoch별 학습·검증 지표 |
| val_predictions.json | 최종 예측 저장 결과 |
| error_samples.json | 추론 코드가 기록한 오류 정보; 빈 파일이 오탐·미탐 0을 의미하지 않음 |
| visualizations/image-*.png | 정답·예측 비교 이미지 |

동일한 실행 폴더에 신규 학습을 덮어쓰지 않도록 Notebook이 차단합니다. 새 실험은 설정 셀에서 새 run_name을 생성합니다. 이 Notebook은 신규 2 epoch 검증용이며 이어서 학습하는 절차는 별도로 구성해야 합니다.

## 확인된 실행 결과

작업 브랜치 `10-탐지-데이터-구성n`, 최초 clone commit `6e875f9ef7184e02cbf2ada7b708c98f1b4d0a9d`에서 실행했습니다. 이후 Colab에서 inference.py 복원 함수를 수정했습니다. 따라서 최초 commit만으로 최종 성공 코드 전체를 재현할 수는 없으며, 함께 제공되는 수정 파일의 반영이 필요합니다.

- Python 3.13.16 / torch 2.11.0+cu130 / torchvision 0.26.0+cu130 / torchmetrics 1.9.0
- GPU: Tesla T4
- Train 8,256장 / Validation 2,063장
- train_loss: 0.9024 → 0.4481
- mAP: 0.8950 → 0.9119
- mAP@0.75:0.95: 0.8043 → 0.8381
- 모델 복원 및 예측·시각화 저장 완료

공유본은 실행 완료본에서 중복 평가·임시 재로딩·오류 출력을 정리하고 main을 기본으로 설정했습니다. main 병합 이후 새 세션에서의 실행은 추가 확인 대상입니다. pyproject.toml의 로컬 의존성 조건과 이 Colab 환경은 같지 않으므로, Colab에서는 Notebook의 설치 셀을 사용합니다. 성능은 현재 Validation 기준이며 새로운 촬영 조건에 대한 일반화 검증은 별도입니다.

## 복원 오류 수정

기존 load_inference_model은 pretrained_backbone=False를 고정해 학습 때와 채널 수가 달라질 수 있었습니다. 수정 파일은 체크포인트의 model_config.pretrained_backbone을 읽어 모델을 생성한 다음 학습된 state_dict를 적용합니다. True인 경우 ImageNet 가중치를 내려받거나 캐시에서 읽을 수 있습니다. 기존 best.pt로 복원이 가능하며 이 수정 때문에 재학습할 필요는 없습니다.

## 문제 발생 시

- detector_data import 실패: main 병합 여부와 clone한 코드 위치 확인
- 파일 경로 오류: Drive 접근 권한과 DATA_DIR 확인
- CUDA 사용 불가: GPU 런타임 선택
- state_dict 크기 불일치: 수정된 inference.py가 반영됐는지 확인
- 오래 걸리는 실행: 현재 셀, 진행률, 경과 시간을 함께 기록

오류가 나면 임의로 데이터 분할·bbox·라벨을 변경하지 말고 발생 셀과 전체 오류를 공유합니다.
