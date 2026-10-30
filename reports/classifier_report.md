# 경구약제 118종 분류기 결과 보고서

> 기준 모델 `resnet50_baseline_v1`의 holdout validation 결과다. 탐지 bbox 연결(6절)은 탐지기 결과가 나온 뒤 갱신한다.

## 1. 실험 개요

- 목표: 탐지 또는 정답 bbox에서 추출한 알약을 118개 품목 중 하나로 분류
- 모델: ImageNet V2 사전학습 ResNet-50
- 출력: 118-way 단일 softmax
- 실행 이름: `resnet50_baseline_v1`
- 데이터: `training_package_classifier.zip` (bbox 사방 5% 여백 crop)
- 데이터 분할 hash: `49d2afee86a4ec8f48c25488eac9b1f9081b36066e5533eba40670087f60430c`
- best epoch: 9번째 epoch (`metrics.csv`의 `epoch=8`, fine-tuning 6번째 epoch). 16 epoch 뒤 early stopping으로 종료

## 2. 데이터 구성

| 항목 | 결과 |
| --- | --- |
| 학습 샘플 수 | 12,125 |
| Validation 샘플 수 | 2,863 |
| 학습 그룹 수 | 1,786 |
| Validation 그룹 수 | 429 |
| 그룹 중복 수 | 0 |
| Validation 클래스 coverage | 1.0 (118/118) |

출처는 AI Hub 조합 14,232개, Kaggle train 756개다. 분할은 탐지기 `split_manifest.csv`의 조합 그룹 분할을 따른다. 클래스별 train 샘플은 33~373개이며 `WeightedRandomSampler`로 균형을 맞춘다.

분류 입력은 bbox 사방 5% 여백 crop, 정사각형 패딩, 224×224 resize, ImageNet normalization을 적용한다. 학습 데이터에는 회전 증강을 적용하고 validation에는 random transform을 사용하지 않는다.

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
| 실행 장치 | Colab NVIDIA L4 GPU (VS Code Colab 커널) |

모든 BatchNorm 파라미터와 running statistics는 두 학습 단계에서 동결했다. CUDA에서는 AMP와 GradScaler를 사용하고 MPS와 CPU에서는 float32를 사용한다.

## 4. Holdout validation 결과

| 지표 | 결과 |
| --- | --- |
| Validation loss | 0.0307 |
| Top-1 accuracy | 0.9983 (2,858 / 2,863) |
| Top-5 accuracy | 0.9986 |
| Macro F1 | 0.9983 |
| Validation class coverage | 1.0 |

`near_dup_in_train`(train에 거의 같은 사진이 있는 val 샘플)으로 나눈 결과는 다음과 같다.

| Validation | 샘플 수 | Top-1 | Macro F1 |
| --- | ---: | ---: | ---: |
| 전체 | 2,863 | 0.9983 | 0.9983 |
| 유사 사진 제외 | 2,064 | 0.9976 | 0.9973 |
| 유사 사진만 | 799 | 1.0000 | 1.0000 |

오분류 5건은 모두 유사 사진이 아닌 샘플에서 나왔으므로, train 사진을 외워서 점수가 높아진 영향은 확인되지 않았다.

성능 수치는 holdout validation의 `best.pt`에서 산출한다. Validation support가 없는 클래스는 macro F1 계산에서 제외하고 클래스별 F1을 `null`로 기록한다.

## 5. 클래스별 분석

- Recall이 1.0 미만인 클래스: 넥시움정 40MG (0.933), 트윈스타정 40/5MG (0.955), 액티프롤정 (0.966), 카듀엣정 5/20MG (0.968), 아토젯정 10/40MG (0.974). 나머지 113종은 recall 1.0
- 혼동 품목 쌍: 5쌍 모두 1건씩 (트윈스타 ↔ 아토젯, 넥시움 → 신바로, 카듀엣 → 하이트린, 액티프롤 → 쿠에타핀)
- 소수 클래스 성능: train 샘플이 가장 적은 기넥신에프정 40MG(33개)도 recall 1.0
- 앞·뒷면 또는 각인에 따른 오류 경향: 확인되지 않음. 오분류 5건을 원본 사진에서 확인한 결과 모두 라벨 또는 bbox 오류였다

| 샘플 (원본 사진) | 라벨 | 예측 | 원인 |
| --- | --- | --- | --- |
| Kaggle `K-003483-025367-027733-035206_0_2_0_2_75_000_200` | 트윈스타정 / 아토젯정 | 아토젯정 / 트윈스타정 | 두 알약의 라벨이 서로 바뀜 (같은 그룹의 70°·90° 사진은 정상). 예측 확신도 1.00 |
| AI Hub `K-010224-016548-033009-033878_0_2_0_2_70_000_200` | 넥시움정 | 신바로정 | 넥시움 bbox가 신바로정("S300") 위에 그려짐. 확신도 1.00 |
| AI Hub `K-016235-019861-020238-031885_0_2_0_2_90_000_200` | 카듀엣정 | 하이트린정 | bbox 안에 알약이 없음 (빈 배경). 실제 알약은 bbox 아래쪽에 있음 |
| AI Hub `K-006835-020004-024941-029711_0_2_0_2_90_000_200` | 액티프롤정 | 쿠에타핀정 | bbox가 다른 알약(이튼큐정) 가장자리에 그려짐. 실제 알약은 오른쪽 아래에 있음 |

이 5건을 빼면 validation 2,858개를 모두 맞혔다. 기존 라벨 검수(크기·중복·겹침 규칙)는 위치만 틀린 bbox와 서로 바뀐 라벨을 잡지 못한다.

분석 그림과 표는 `notebooks/classifier_baseline_analysis.ipynb`로 만들며, 실행 디렉터리의 `analysis/`에 저장된다.

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

- Validation 오분류 5건이 모두 라벨·bbox 오류여서, 현재 validation으로는 모델 간 차이를 구분하기 어렵다. 같은 종류의 오류가 train에도 있을 수 있으므로, 교차 검증 예측으로 전체 데이터의 라벨 오류를 찾아 데이터 다음 버전에서 제외한다.
- Kaggle 사례는 탐지기 데이터에도 같은 이미지가 있으므로 탐지기 담당과 공유한다.
- GT bbox와 예측 bbox 사이의 crop domain gap을 end-to-end 평가해야 한다.
- 클래스별 표본 수가 적어 macro F1과 소수 클래스 recall의 변동성을 함께 확인해야 한다.
- 전체 허용 데이터 재학습과 `final.pt` 생성 여부는 detector 연결 결과와 제출 일정에 따라 결정한다.

## 8. 재현 방법

1. Drive의 `gitkk_archive/training_package_classifier.zip`을 준비한다 (`split_hash` 위와 같음).
2. `notebooks/train_classifier.ipynb`를 Colab GPU에서 열고 설정 셀의 `run_name`을 정한 뒤 `max_epochs`, `head_only_epochs` 줄을 지워 기본값(30, 3)으로 실행한다.
3. 끊기면 같은 `run_name`으로 `config.resume_from`에 `last.pt`를 지정해 재개한다.
4. `notebooks/classifier_baseline_analysis.ipynb`에 같은 `run_name`을 넣어 분석 그림과 표를 만든다.
5. 결과를 `reports/classifier_experiments.md`에 한 줄로 기록한다.
