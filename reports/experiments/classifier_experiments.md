# 분류기 실험 기록

실험마다 **바꾸는 것 하나만** 바꾸고 나머지는 아래 고정 조건을 따른다. 기준 모델 상세는 [분류기 기준 결과](../baseline/classifier_report.md)에 있다.

## 고정 조건

| 항목 | 기준 |
| --- | --- |
| 데이터 | `training_package_classifier.zip`, `split_hash` `49d2afee…` (`data_summary.json`) |
| 학습 | `notebooks/baseline/train_classifier.ipynb`, `ClassifierConfig` 기본값 (seed 42, batch 32, 최대 30 epoch, head-only 3 epoch, early stopping 7) |
| 모델 선택 | `best.pt` (val macro F1 최대, 같으면 Top-1이 높은 epoch) |
| 평가 | macro F1은 val support가 1 이상인 클래스만. 전체 val과 `near_dup_in_train` 제외 val을 함께 기록 |
| 분석 | `notebooks/experiments/classifier_baseline_analysis.ipynb`에 같은 `run_name`으로 실행 |

데이터를 바꾸는 실험(라벨 오류 제거, crop 여백 변경 등)은 `split_hash`가 달라져 같은 val로 비교하는 것이 아니다. 아래 표의 `split_hash` 열에 데이터 버전을 꼭 적는다.

| 데이터 | 패키지 | split_hash | v1 기준 모델의 val Top-1 |
| --- | --- | --- | ---: |
| v1 | `training_package_classifier.zip` | `49d2afee…` | 0.9983 (오분류 5) |
| v2 (라벨 2차 검수 적용) | `training_package_classifier_v2.zip` | `a2eb581a…` | 1.0000 (오분류 0) |

v2 validation에서는 v1 기준 모델이 모든 샘플을 맞혔다. 이후 모델은 Top-1·macro F1만으로 차이를 구분하기 어려우므로 validation loss, 확신도와 탐지 bbox 연결 평가를 함께 본다. 수치의 의미와 주의점은 [분류 데이터 v2 보고서](../data/classifier/data_v2_report.md)에 기록한다.

### 외부 평가셋: AI Hub val

v2 val이 포화되어, 학습에 쓰지 않는 **AI Hub val(조합) 평가셋**으로 실험끼리 비교한다. best epoch는 v2 val로 고르고, 학습이 끝난 `best.pt`를 이 평가셋으로 평가한다 (`python -m classifier.evaluate_external`).
평가셋에는 118종 중 54종만 있으므로 **macro F1(54종 평균)**을 주 지표로 본다. 구성은 [AI Hub val 평가셋 보고서](../data/classifier/aihub_val_eval_package_report.md).

| 평가셋 | eval_hash | crop | 클래스 |
| --- | --- | ---: | ---: |
| AI Hub val (1·2차 라벨 검수 반영) | `6f8d9b09…` | 5,428 | 54 |

## 실험 순서

1. 바꿀 것 하나를 정한다.
2. 새 `run_name`으로 학습한다. 이전 결과를 덮어쓰지 않는다.
3. 분석 노트북에서 `run_name`만 바꿔 실행한다. 결과는 `<run_dir>/analysis/`에 저장된다.
4. 아래 표에 한 줄을 추가한다. 실제 설정은 각 실행의 `resolved_config.json`에 남는다.

## 기록

| run_name | 바꾼 것 | split_hash | best epoch | val Top-1 | val macro F1 | 유사 사진 제외 macro F1 | 메모 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| resnet50_baseline_v1 | 기준 | 49d2afee… | 9 / 16 | 0.9983 | 0.9983 | 0.9973 | 오분류 5건 모두 라벨·bbox 오류 |
| resnet50_baseline_v2 | 데이터 v2 (라벨 2차 검수 반영) | a2eb581a… | 8 / 15 | 1.0000 | 1.0000 | 1.0000 | v2 val은 v1 모델도 100%라 비교 불가. 비교는 아래 AI Hub val |

AI Hub val 평가 (eval_hash `6f8d9b09…`)

| run_name | AI Hub val Top-1 | AI Hub val macro F1 | 유사 사진 제외 macro F1 | 오분류 | 메모 |
| --- | ---: | ---: | ---: | ---: | --- |
| resnet50_baseline_v1 | 0.9963 | 0.9951 | 0.9939 | 20 / 5,428 | 20건 모두 라벨은 맞음. 브린텔릭스 "20" 면→제미메트 8, 흐린 크레스토→자트랄 5, 베시케어 각인 없는 면→리피로우 3, 졸로푸트 3, 글리아티린→콜리네이트 1 |
| resnet50_baseline_v2 | 0.9980 | 0.9981 | 0.9985 | 11 / 5,428 | 11건 모두 v1도 틀린 crop (새로 틀린 것 0). 흐린 크레스토→자트랄 5, 베시케어 각인 없는 면→리피로우 3, 브린텔릭스 "20" 면→넥시움 3. v1의 브린텔릭스→제미메트 5, 졸로푸트 3, 글리아티린 1은 맞힘 |

- v1과 v2는 seed 42로 한 번씩 학습한 결과다. 오분류 차이(20 → 11)가 데이터 v2 효과인지 학습 무작위성 때문인지는 seed를 바꿔 반복해야 확실히 말할 수 있다
- 두 모델 모두 남은 오분류는 같은 세 종류(흐린 사진, 각인 없는 면, 학습에 적은 "20" 면)다. 데이터를 더 보태거나 증강을 바꾸는 실험의 대상이다
