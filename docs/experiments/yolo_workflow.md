# YOLO11s 학습·공통 분류기 e2e·Kaggle 제출

YOLO는 단일 `pill` 위치 탐지, ResNet50은 118종 품목 분류를 담당한다.
YOLO backend의 호출 방식·checkpoint는 torchvision 탐지기와 분리한다.
기존 SSDLite의 학습·통합 추론 코드는 유지한다.

## 실행 환경

프로젝트 루트에서 `uv sync --locked`로 설치한다. 이 adapter는 실험에 사용한
Ultralytics 8.4.183 내부 API를 사용하므로 다른 버전이면 명시적으로 중단한다.
Colab 노트북은 해당 버전과 최소 실행 의존성을 설치한다. 전체 프로젝트 개발 환경은
README의 Python 3.12·uv 기준을 따른다. 실제 기존 Colab 실험 환경은 결과 보고서에 따로 기록한다.

Notebook은 저장소 체크아웃을 사용하며 소스 ZIP 업로드가 필요하지 않다.
Colab에 체크아웃이 없으면 설정한 GitHub ref를 clone한다. PR 검토 중에는
`SOURCE_REF = "37-yolo11s-training-e2e"`, 병합 후에는 `main`을 사용한다.
기존 체크아웃의 브랜치는 자동으로 바꾸지 않는다. Notebook 출력은 커밋하지 않는다.

## 1. 탐지기 학습

[학습 노트북](../../notebooks/experiments/train_detector_yolo11s_colab.ipynb)을 사용한다.

- 원본 v2 패키지의 annotation·manifest·label map·summary·그룹 분리를 검증한다.
- 기존 분할을 유지하고, label 1을 YOLO class 0으로 변환한다.
- 이미지 고유 ID를 파일명으로 사용해 출처 간 파일명 충돌을 막는다.
- 이미지 크기, 변환 bbox, symlink와 split_hash를 학습 전에 대조한다.
- YOLO 데이터 생성은 명시적인 준비 단계이며 원본 annotation을 바꾸지 않는다.
- 매 epoch val 손실·COCO 지표·고 IoU mAP·MAR·threshold 전후 recall/개수 오차를 기록한다.
- Best는 고 IoU mAP@[0.75:0.95], 동률이면 MAR@100으로 선택한다.
- 원래 checkpoint schema에 모델/설정/데이터 hash와 선택 상태를 추가한다.
- 최종 optimizer 제거를 생략해 `weights/last.pt`를 재개용으로 보존한다.

```bash
uv run python -m detector.yolo_backend --config configs/detector_yolo11s.yaml
uv run python -m detector.yolo_backend --config /path/to/same-run-config.yaml \
  --resume /path/to/run/weights/last.pt
```

YAML의 `data_yaml`은 준비 완료된 YOLO export를 가리켜야 한다. CLI는 이미지·라벨을
묵시적으로 생성하지 않는다. Notebook의 준비 셀 또는 `export_yolo_package()`로 생성한다.
설정·데이터·모델이 다른 재개와 기존 신규 실행 폴더 덮어쓰기는 거부한다.
초기 batch 0.7은 GPU 메모리 목표 비율이며 실제 batch는 `actual_train_args.json`에 남는다.

데이터 복사가 중단되면 불완전한 로컬 복사본을 확인하고 다시 준비해야 한다.
원본 Drive 데이터를 삭제하지 않는다. Epoch 도중 중단되면 마지막 저장 epoch 이후를 재계산한다.
`resume`은 optimizer/EMA/epoch/선택 상태를 복원하지만 실행 간 난수 상태까지 동일한
bitwise 재현을 보장하지는 않는다.

## 2. 공통 분류기 e2e

[e2e 노트북](../../notebooks/experiments/evaluate_yolo11s_shared_resnet50_colab.ipynb)을 사용한다.

YOLO 예측 박스로 crop하고 기존 `build_classifier_batch()` 및
`classify_detections()`를 재사용한다. Margin 5%, RGB 정사각형 padding,
224 resize, ImageNet normalization과 탐지 score * 분류 확률을 유지한다.
확장 crop 영역이 아니라 원래 탐지 bbox를 채점한다. GT bbox와 약품 라벨은 채점에만 사용한다.
분류기는 공통 `best_v2_slim.pt` 및 같은 `class_map.json`을 쓰고 재학습하지 않는다.

```bash
uv run python -m pipeline.yolo_e2e --config configs/yolo_e2e.json
```

설정 예시는 기존 실험 재현용이며 파일 경로와 SHA-256을 기록한다. 다른 checkpoint를
평가하려면 실행명·탐지기 이름·경로와 해당 파일의 SHA-256을 함께 바꾼다.
공통 분류기 checkpoint의 class map hash도 검증한다. 대용량 파일은 Drive 또는 로컬
`models/`에 두고 커밋하지 않는다.

평가는 **전체 v2 val 2,026장**을 대상으로 한다. 탐지가 없는 이미지도 제외하지 않는다.
GT에 등장하는 category별 AP만 평균한다. AP@0.95는 별도로 계산한다.
점수 파일에서 `all_val_detector`와 `all_val_final`을 구분한다.

- `detector`: 탐지 score >= 0.25 예측을 평가한다.
- `final`: 탐지 score * 분류 확률 >= 0.25 예측을 평가한다.
- 두 조건의 AP 순위 점수는 모두 탐지 score * 분류 확률이다.
- `detector_only_reference`는 약품명을 무시하고 탐지 score로 평가한다.

Reference scorer는 torchmetrics/pyCOCOtools, IoU=[0.75,0.80,0.85,0.90,0.95],
recall grid 0.00~1.00(0.01 간격), maxDets=[1,10,100]이다.
팀/대회 scorer와 동일한지는 미확인이다. Threshold 위치도 합의 전에는 두 결과를 혼용하지 않는다.

Drive의 패키지를 검증하고 필요한 val 이미지만 로컬로 복사한다. 완료 이미지 예측은
원자적으로 저장한다. 같은 설정·모델 hash·코드로 재실행하면 완료 cache를 재사용한다.
모델·설정·코드가 바뀌면 새 실행 이름을 사용한다. 결과 확인만 할 때는 Notebook의
독립 결과 셀을 사용하며 모델·데이터를 다시 로드할 필요가 없다.

## 3. Test 예측과 Kaggle CSV

[제출 노트북](../../notebooks/experiments/make_yolo11s_kaggle_submission_colab.ipynb)을 사용한다.

```bash
uv run python -m pipeline.yolo_submission --config configs/yolo_submission.json
```

정답 없는 test 이미지를 예측하고 기존 `save_submission_csv()`를 재사용한다.
파일명 stem은 기존 프로젝트 계약처럼 숫자 image ID여야 한다. 임의로 ID를 재지정하지 않는다.
출력은 `annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score`다.
이미지/약품 ID, bbox 범위·양의 면적, score, annotation ID 중복을 검증한다.
Sample submission이 있으면 컬럼도 대조한다. 예측 0개인 이미지는 임의 박스로 채우지 않는다.

이미지별 cache로 중단 후 재개하며, CSV 생성과 Kaggle 업로드는 별도 단계다.
이 코드는 Kaggle 인증이나 업로드를 수행하지 않는다. 웹 업로드 후 확인한 Public score를
Notebook 마지막 셀에 직접 기록하면 val과 Kaggle 결과를 구분한 Markdown을 만든다.

## 공유·검증

가중치, 데이터, cache, 전체 예측, 실행 metadata와 원시 결과는 `outputs/` 또는 Drive에서
관리하고 Git에는 코드·설정·출력 없는 Notebook·테스트·결과 요약만 올린다.
결과 보고서는 [YOLO11s v2 실험 기록](../../reports/experiments/detector_yolo11s_v2.md)에 있다.

```bash
uv run python -m pytest -q
uv run ruff check main.py classifier configs detector etl data_loading pipeline utils tests
uv run ruff format --check main.py classifier configs detector etl data_loading pipeline utils tests
```

모델 구현 비교 시 입력 크기·사전학습 차이와 평가 precision/batch/전처리/threshold를 함께 기록한다.
탐지 단독 AP와 품목별 e2e AP는 평균 대상이 달라 직접 성능 증감으로 해석하지 않는다.
