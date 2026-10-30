# 단일 클래스 알약 탐지기 결과 보고서

> **문서 상태:** 실제 데이터 학습 결과를 기록하기 위한 양식이다. `TBD`는 폐기된 내용이 아니라 아직 측정되지 않은 항목이다.

> 실제 DataLoader와 기준 학습이 완료되면 `TBD` 항목을 실행 결과로 갱신한다.

## 1. 실험 개요

- 목표: 조합 이미지에서 모든 알약을 단일 foreground 클래스 `pill`로 탐지
- 모델: SSDLite320 MobileNetV3-Large
- 모델 출력 클래스: background `0`, pill `1`
- 실행 이름: `TBD`
- 데이터 분할 hash: `TBD`
- best epoch: `TBD`

품목 수는 탐지기 출력 클래스 수와 무관하다. 탐지된 bbox crop은 별도 분류기로 전달해 세부 약품명을 결정한다.

## 2. 데이터 구성

| 항목 | 결과 |
| --- | --- |
| 학습 이미지 수 | TBD |
| Validation 이미지 수 | TBD |
| 학습 annotation 수 | TBD |
| Validation annotation 수 | TBD |
| 학습 그룹 수 | TBD |
| Validation 그룹 수 | TBD |
| 그룹 중복 수 | 0 확인 필요 |
| 이미지당 최대 객체 수 | TBD |

각 이미지의 source annotation 수와 변환된 target bbox 수가 일치해야 한다. 이미지 픽셀을 보고 원본 annotation 자체의 누락을 판별하지는 않으므로 학습 전에 무작위 bbox 시각화를 별도로 확인한다.

## 3. 모델과 학습 조건

| 항목 | 설정 |
| --- | --- |
| Detector | SSDLite320 MobileNetV3-Large |
| Backbone 초기값 | MobileNetV3-Large `IMAGENET1K_V1` |
| Detection head | 무작위 초기화, 2 classes |
| Optimizer | SGD |
| Scheduler | CosineAnnealingLR |
| Learning rate | pilot 실행 후 확정, Notebook 시작값 `5e-3` |
| Momentum | `0.9` |
| Weight decay | `5e-4` |
| Score threshold | validation에서 확정, Notebook 시작값 `0.25` |
| NMS threshold | validation에서 확정, Notebook 시작값 `0.45` |
| Seed | `42` |
| 실행 장치 | TBD |

CUDA에서는 AMP와 GradScaler를 사용하고 MPS와 CPU에서는 float32를 사용한다.

## 4. Validation 결과

| 지표 | 결과 |
| --- | --- |
| COCO mAP@[0.50:0.95] | TBD |
| AP@0.50 | TBD |
| AP@0.75 | TBD |
| 고 IoU mAP@[0.75:0.95] | TBD |
| Mean average recall | TBD |
| Score threshold 적용 전 recall | TBD |
| Score threshold 적용 후 recall | TBD |
| 이미지당 예측 개수 절대 오차 | TBD |

Best checkpoint의 1차 기준은 고 IoU `mAP@[0.75:0.95]`이며, 동률일 때 mean average recall을 비교한다.

## 5. 오류 분석

- bbox가 누락된 이미지: TBD
- 중복 탐지가 많은 이미지: TBD
- 위치 오차가 큰 이미지: TBD
- score threshold 적용 후 누락된 알약: TBD
- 촬영 방향·배경·조명별 경향: TBD

상세 결과는 실행 디렉터리의 `val_predictions.json`, `error_samples.json`, `visualizations/`를 사용한다.

## 6. 분류기 연결 결과

```text
final_score = detector_score × classifier_score
```

| 항목 | 결과 |
| --- | --- |
| Detector bbox recall | TBD |
| GT bbox 분류 성능 | TBD |
| 예측 bbox 분류 성능 | TBD |
| 최종 category_id mAP@[0.75:0.95] | TBD |

탐지기 bridge는 crop batch만 생성하며 분류기 checkpoint나 class map을 자동 선택하지 않는다.

## 7. 합성 데이터 smoke test

2026-10-27 기준 저장소의 합성 데이터 테스트 결과다.

| 항목 | 결과 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.14.1 |
| torchvision | 0.29.1 |
| 실행 장치 | CPU |
| SSDLite train forward/backward | 통과 |
| SSDLite eval forward/NMS | 통과 |
| SSDLite 모델 smoke test | `15 passed` |
| 전체 단위 테스트 | `86 passed` |

현재 자동화 실행 환경에서는 PyTorch가 MPS 지원 빌드임을 확인했지만 `torch.backends.mps.is_available()`은 `False`를 반환했다. 장치 선택의 CUDA, MPS, CPU 경로는 각각 모의 테스트로 검증했다. 실제 로컬 MPS 한 epoch는 DataLoader 연결 후 다시 확인한다.

## 8. 한계와 후속 작업

- 실제 DataLoader는 저장소에 연결됐지만, 전체 데이터 baseline 학습 수치는 아직 미기재 상태다.
- 입력 검증은 source annotation에서 target으로 변환되는 누락은 찾지만 원본 annotation 자체의 누락은 찾지 못한다.
- score와 NMS threshold는 validation 결과를 확인한 뒤 확정해야 한다.
- SSDLite의 높은 IoU 위치 정밀도가 부족하면 같은 split에서 SSD300 또는 FPN 계열 모델을 비교한다.
- 실제 MPS와 CUDA에서 한 epoch 및 checkpoint 재개 smoke test가 필요하다.

## 9. 재현 방법

1. [탐지 데이터 사용 안내](../docs/data/detector_data_guide.md)에 맞춰 이미지와 annotation을 배치한다.
2. `notebooks/baseline/train_detector.ipynb`에서 경로와 실행 설정을 확인한다.
3. 입력 검증 셀과 bbox 시각화 셀로 annotation을 확인한다.
4. 신규 학습 또는 `last.pt` 재개 셀을 실행한다.
5. `best.pt` 평가와 예측 저장 셀을 실행한다.
6. 본 보고서의 `TBD` 항목을 실제 결과로 갱신한다.
