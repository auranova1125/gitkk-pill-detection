# 알약 탐지·분류 End-to-End Validation 분석 보고서

> 작성일: 2026-11-04
> 대상 실행: `20261103-ssdlite320-resnet50-infer-test-full-02`
> 상태: 제외 데이터 반영 전 detector v1 validation 결과 분석

## 1. 목적

이 보고서는 조합 이미지에서 알약을 탐지하고, 탐지 bbox를 분류기에 전달해 세부 약품을 예측하는 전체 흐름을 평가한다.

```text
validation image
  → SSDLite320 pill detection
  → predicted bbox crop
  → ResNet-50 medicine classification
  → category_id + final score
```

개별 탐지기와 분류기의 성능뿐 아니라 다음 항목을 함께 확인한다.

- 예측 bbox가 정답 알약과 올바르게 연결되는가
- 올바르게 탐지한 crop을 분류기가 정확히 분류하는가
- bbox 위치와 category를 모두 만족하는 최종 예측 성능은 어느 정도인가
- 오류가 탐지기, 분류기, annotation 중 어디에서 발생했는가

## 2. 실행 구성

| 구분 | 설정 |
| --- | --- |
| Detector 실행 | `20261103-ssdlite320-det-default-baseline-01` |
| Detector | SSDLite320 MobileNetV3-Large |
| Classifier 실행 | `20261103-resnet50-cls-v1-baseline-01` |
| Classifier | ImageNet V2 사전학습 ResNet-50 |
| 실행 장치 | MPS (`device=auto`에서 선택) |
| 추론 batch size | 4 |
| Detector score threshold | 0.25 |
| 진단용 prediction-GT matching IoU | 0.50 |
| 고 IoU mAP 범위 | 0.75, 0.80, 0.85, 0.90, 0.95 |
| 최종 점수 | `detector_score × classifier_score` |

평가 데이터는 detector v1 validation 2,063장과 정답 bbox 7,915개다. 이 결과를 만든 뒤 annotation 오류 이미지가 제외되어 현재 detector 데이터는 validation 2,026장·bbox 7,771개로 변경됐다. 따라서 본 문서의 수치는 당시 실행을 기록하는 기준선이며, 현재 데이터셋에서의 최종 성능은 아니다.

## 3. 평가 방법

### 3.1 IoU 0.50 기준 오류 진단

이미지별 예측을 최종 점수 순서로 정렬한 뒤 아직 매칭되지 않은 GT 중 IoU가 가장 높은 대상을 선택한다.

| status | 의미 |
| --- | --- |
| `correct` | IoU 0.50 이상이고 category도 일치 |
| `misclassified` | IoU 0.50 이상이지만 category 불일치 |
| `false_positive` | 어떤 미매칭 GT와도 IoU 0.50 이상이 아닌 예측 |
| `false_negative` | 끝까지 예측과 연결되지 않은 GT |

여기서 `false_positive`는 분류기가 해당 영역을 배경으로 분류했다는 뜻이 아니다. 탐지 예측 bbox가 GT와 IoU 0.50 이상으로 연결되지 않았다는 뜻이다.

### 3.2 category-aware mAP

최종 mAP는 detector의 단일 `pill` label이 아니라 분류기가 예측한 `category_id`와 최종 점수를 사용한다. 따라서 bbox 위치와 약품 분류가 모두 맞아야 true positive가 된다.

## 4. 전체 결과

### 4.1 주요 지표

| 지표 | 결과 |
| --- | ---: |
| Validation 이미지 | 2,063 |
| GT 알약 | 7,915 |
| 최종 예측 | 7,908 |
| IoU 0.50 이상 탐지 매칭 | 7,869 |
| category까지 맞춘 최종 정답 | 7,866 |
| Detection precision | 0.9951 |
| Detection recall | 0.9942 |
| 매칭 bbox 분류 정확도 | 0.9996 |
| End-to-end precision | 0.9947 |
| End-to-end recall | 0.9938 |
| End-to-end F1 | 0.9942 |
| 고 IoU mAP@[0.75:0.95] | 0.9088 |
| Mean average recall | 0.9370 |

7,869개의 올바른 위치 매칭 중 분류 오답은 3개뿐이다. 예측 bbox가 정상적으로 들어오면 분류기는 99.96%를 맞혔으며, 전체 오차의 중심은 분류보다 bbox-GT 매칭에 있다.

### 4.2 IoU threshold별 AP

| IoU threshold | AP |
| --- | ---: |
| 0.75 | 0.9840 |
| 0.80 | 0.9835 |
| 0.85 | 0.9820 |
| 0.90 | 0.9649 |
| 0.95 | 0.6294 |
| 평균 0.75:0.95 | 0.9088 |

AP@0.90까지는 높지만 AP@0.95에서 0.6294로 크게 하락한다. 이는 대부분의 알약을 안정적으로 찾지만, bbox가 GT와 거의 완전히 겹쳐야 하는 매우 엄격한 기준에서는 위치·크기 차이가 남는다는 뜻이다. 다만 이후 육안 검수에서 잘못된 GT bbox가 다수 확인됐으므로 이 하락을 전부 모델의 위치 보정 한계로 해석하면 안 된다.

매칭된 bbox의 IoU 분포는 다음과 같다.

| 구간 | IoU |
| --- | ---: |
| 최솟값 | 0.5029 |
| 1 percentile | 0.8932 |
| 5 percentile | 0.9265 |
| 중앙값 | 0.9637 |
| 95 percentile | 0.9842 |
| 최댓값 | 0.9974 |

## 5. 오류 분석

### 5.1 오류 구성

| status | 개수 |
| --- | ---: |
| Correct | 7,866 |
| Misclassified | 3 |
| False positive | 39 |
| False negative | 46 |

오류 행은 총 88개이며 오류가 포함된 이미지는 38장이다.

| 이미지별 오류 구성 | 이미지 수 |
| --- | ---: |
| FP 1 + FN 1 | 25 |
| FN만 1개 | 7 |
| FP 3 + FN 3 | 2 |
| FP 4 + FN 4 | 2 |
| 분류 오답 1개 | 1 |
| 분류 오답 2개 | 1 |

FP 또는 FN이 발생한 이미지는 36장이고 분류 오답만 발생한 이미지는 2장이다.

### 5.2 FP/FN은 threshold 부족보다 annotation 위치 오류의 영향이 큼

FP 39개의 detector score는 최솟값도 0.9973이고 평균은 0.9999다. classifier score 평균도 0.9998이다. 따라서 현재 FP는 score threshold를 0.25에서 조금 높이는 방식으로 제거할 수 없다.

같은 이미지의 FP와 FN을 가장 가까운 순서로 연결하면 39개 중 28개는 예측 category와 FN의 target category가 같다. 즉 모델은 실제 알약을 매우 높은 확신도로 탐지·분류했지만 정답 bbox가 다른 위치에 있어 IoU 0.50을 넘지 못한 사례가 많다.

`false_positive_grid.png` 육안 확인에서도 다음 패턴이 반복됐다.

- 빨간 예측 bbox는 실제 알약을 정확히 감쌈
- 초록 GT bbox는 빈 배경, 다른 알약 또는 알약 일부에 위치
- 동일 알약 category에 FP와 FN이 동시에 기록됨
- 일부 이미지는 여러 GT가 함께 밀려 FP와 FN이 3~4개씩 발생

따라서 이 85개 FP/FN 행은 순수한 모델 오류 수로 사용하기 어렵다. 이후 detector v2 구성에서 validation 오류 이미지 37장이 제외됐으며, 새 데이터 버전으로 다시 평가해야 정제된 모델 성능을 확인할 수 있다.

### 5.3 분류 오답 3건

| image_id | 정답 | 예측 | IoU | classifier score | 해석 |
| --- | --- | --- | ---: | ---: | --- |
| 9095 | 디카테오엘씨정 | 리렉스펜정 | 0.9747 | 0.9469 | 위치는 정상이며 추가 원본 검토 필요 |
| 10312 | 트윈스타정 40/5MG | 아토젯정 10/40MG | 0.9393 | 1.0000 | 두 알약의 label swap으로 확인된 데이터 오류 |
| 10312 | 아토젯정 10/40MG | 트윈스타정 40/5MG | 0.9687 | 1.0000 | 두 알약의 label swap으로 확인된 데이터 오류 |

image 10312의 두 오답은 분류기 자체의 혼동이 아니라 annotation의 두 라벨이 서로 바뀐 사례다. image 9095는 높은 IoU로 올바른 알약 crop이 전달됐으므로 실제 외형 혼동인지 라벨 문제인지 별도 확인이 필요하다.

### 5.4 품목별 오류 빈도 해석 주의

FN은 쿠에타핀정 25MG 7건, 리피토정 20MG 4건, 이튼큐정 3건 순으로 많았다. 그러나 FP 쪽에서도 같은 품목이 유사한 횟수로 나타나고, 시각화에서 GT 위치 오류가 확인됐다. 이 빈도만으로 특정 품목의 탐지 난도가 높다고 결론 내릴 수 없다.

## 6. 모델 단계별 해석

### Detector

- 알약 존재 여부와 대략적인 위치 탐지는 매우 안정적이다.
- 정상 GT에 매칭된 bbox의 중앙 IoU는 0.9637이다.
- AP@0.95가 상대적으로 낮아 엄격한 위치 정밀도에는 개선 여지가 있다.
- 다만 평가 오류의 상당 부분이 GT bbox 오류이므로 정제 데이터 재평가가 우선이다.

### Classifier

- 정상 매칭 crop 7,869개 중 7,866개를 맞혔다.
- 두 건은 확정 label swap이며, 나머지 한 건만 추가 검토 대상이다.
- 탐지 crop domain에서도 기존 GT crop 기반 분류 성능이 대부분 유지됐다.

### End-to-end

- precision, recall, F1이 모두 0.99 이상으로 전체 파이프라인 연결은 정상이다.
- 최종 성능의 현재 병목은 약품 분류보다 bbox annotation 품질과 엄격한 위치 정밀도다.
- score threshold 조정만으로는 잘못된 GT와의 위치 불일치를 해결할 수 없다.

## 7. 데이터 정제 이후 영향

본 평가 후 detector 데이터에서 train 113장, validation 37장, 총 bbox 579개가 제외됐다.

| 분할 | 평가 당시 v1 | 현재 v2 | 차이 |
| --- | ---: | ---: | ---: |
| Train 이미지 | 8,256 | 8,143 | -113 |
| Validation 이미지 | 2,063 | 2,026 | -37 |
| Train bbox | 31,673 | 31,238 | -435 |
| Validation bbox | 7,915 | 7,771 | -144 |

오류 이미지 수와 이후 validation 제외 수가 가깝지만, 본 보고서의 image ID와 최종 제외표를 단순히 개수만으로 일대일 대응시키지는 않는다. 동일한 모델을 v2 validation에서 재평가해 실제 잔여 오류를 다시 산출해야 한다.

## 8. 결론과 후속 작업

1. SSDLite320과 ResNet-50의 end-to-end 연결은 정상이며 F1 0.9942를 기록했다.
2. 매칭 bbox의 분류 정확도는 0.9996으로, 현재 파이프라인의 주요 병목은 분류기가 아니다.
3. FP/FN 대부분은 높은 score의 실제 알약 예측과 잘못된 GT 위치가 충돌한 형태다.
4. AP@0.95의 하락은 모델 위치 정밀도와 annotation 오류가 섞인 결과이므로 정제 데이터 재평가가 필요하다.
5. detector v2에서 SSDLite320을 다시 평가해 정제된 기준선을 만들고, 같은 split에서 Faster R-CNN과 비교해야 한다.
6. 남은 실제 오류는 원본 이미지, 예측 bbox, GT bbox를 함께 시각화해 모델 오류와 데이터 오류를 분리한다.

## 9. 결과물 위치

```text
outputs/inference/20261103-ssdlite320-resnet50-infer-test-full-02/
├── pill_predictions.json
├── submission.csv
├── resolved_config.json
└── validation/
    ├── validation_metrics.json
    ├── validation_details.csv
    └── false_positive_grid.png
```

루트의 `pill_predictions.json`과 `submission.csv`는 Kaggle test image 추론 결과이고, 본 보고서의 validation 수치는 `validation/` 아래 두 파일에서 산출했다.

평가 구현은 `pipeline/evaluation.py`, 실행과 시각화는 `notebooks/baseline/run_pill_recognition.ipynb`에서 확인한다.
