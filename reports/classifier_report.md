# 경구약제 118종 분류기 결과 보고서

> **문서 상태:** 실제 데이터 학습 결과를 기록하기 위한 양식이다. `TBD`는 폐기된 내용이 아니라 아직 측정되지 않은 항목이다.

> 실제 데이터 로더와 기준 학습이 완료되면 `TBD` 항목을 실행 결과로 갱신한다.

## 1. 실험 개요

- 목표: 탐지 또는 정답 bbox에서 추출한 알약을 118개 품목 중 하나로 분류
- 모델: ImageNet V2 사전학습 ResNet-50
- 출력: 118-way 단일 softmax
- 실행 이름: `TBD`
- 데이터 분할 hash: `TBD`
- best epoch: `TBD`

## 2. 데이터 구성

| 항목 | 결과 |
| --- | --- |
| 학습 샘플 수 | TBD |
| Validation 샘플 수 | TBD |
| 학습 그룹 수 | TBD |
| Validation 그룹 수 | TBD |
| 그룹 중복 수 | 0 확인 필요 |
| Validation 클래스 coverage | TBD |

분류 입력은 bbox crop, 정사각형 패딩, 224×224 resize, ImageNet normalization을 적용한다. 학습 데이터에는 회전 증강을 적용하고 validation에는 random transform을 사용하지 않는다.

## 3. 모델과 학습 조건

| 항목 | 설정 |
| --- | --- |
| Backbone | ResNet-50, `IMAGENET1K_V2` |
| Head-only | epoch 0~2, FC LR `1e-3` |
| Fine-tuning | epoch 3~29, layer4 LR `1e-4`, FC LR `5e-4` |
| Optimizer | AdamW, weight decay `1e-4` |
| Scheduler | CosineAnnealingLR, `T_max=27` |
| Loss | CrossEntropyLoss |
| Early stopping | Fine-tuning에서 macro F1 7 epoch 미개선 |
| Seed | 42 |
| 실행 장치 | TBD |

모든 BatchNorm 파라미터와 running statistics는 두 학습 단계에서 동결했다. CUDA에서는 AMP와 GradScaler를 사용하고 MPS와 CPU에서는 float32를 사용한다.

## 4. Holdout validation 결과

| 지표 | 결과 |
| --- | --- |
| Validation loss | TBD |
| Top-1 accuracy | TBD |
| Top-5 accuracy | TBD |
| Macro F1 | TBD |
| Validation class coverage | TBD |

성능 수치는 holdout validation의 `best.pt`에서 산출한다. Validation support가 없는 클래스는 macro F1 계산에서 제외하고 클래스별 F1을 `null`로 기록한다.

## 5. 클래스별 분석

- Recall이 낮은 클래스: TBD
- 가장 자주 혼동되는 품목 쌍: TBD
- 소수 클래스 성능: TBD
- 앞·뒷면 또는 각인에 따른 오류 경향: TBD

상세 결과는 실행 디렉터리의 `val_predictions.json`과 `error_samples.json`을 사용한다.

## 6. 탐지 bbox 연결 결과

```text
final_score = detector_score × classifier_score
```

| 항목 | 결과 |
| --- | --- |
| GT bbox 분류 성능 | TBD |
| 예측 bbox end-to-end mAP@[0.75:0.95] | TBD |
| 연결한 detector | TBD |

## 7. 한계와 후속 작업

- 실제 DataLoader가 아직 저장소에 연결되지 않아 기준 학습 수치는 미기재 상태다.
- GT bbox와 예측 bbox 사이의 crop domain gap을 end-to-end 평가해야 한다.
- 클래스별 표본 수가 적어 macro F1과 소수 클래스 recall의 변동성을 함께 확인해야 한다.
- 전체 허용 데이터 재학습과 `final.pt` 생성 여부는 detector 연결 결과와 제출 일정에 따라 결정한다.

## 8. 재현 방법

1. 데이터 구성 모듈에서 입력 검증 형식에 맞는 loader와 JSON 정보를 생성한다.
2. `notebooks/train_classifier.ipynb`의 `data_factory_path`를 설정한다.
3. 새 학습 또는 `last.pt` 재개 셀을 실행한다.
4. best checkpoint 평가 셀로 예측과 오류 artifact를 저장한다.
5. 본 보고서의 `TBD` 항목을 실행 결과로 갱신한다.
