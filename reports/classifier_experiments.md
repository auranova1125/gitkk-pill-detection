# 분류기 실험 기록

실험마다 **바꾸는 것 하나만** 바꾸고 나머지는 아래 고정 조건을 따른다. 기준 모델 상세는 [`classifier_report.md`](classifier_report.md)에 있다.

## 고정 조건

| 항목 | 기준 |
| --- | --- |
| 데이터 | `training_package_classifier.zip`, `split_hash` `49d2afee…` (`data_summary.json`) |
| 학습 | `notebooks/train_classifier.ipynb`, `TrainConfig` 기본값 (seed 42, batch 32, 최대 30 epoch, head-only 3 epoch, early stopping 7) |
| 모델 선택 | `best.pt` (val macro F1 최대, 같으면 Top-1이 높은 epoch) |
| 평가 | macro F1은 val support가 1 이상인 클래스만. 전체 val과 `near_dup_in_train` 제외 val을 함께 기록 |
| 분석 | `notebooks/classifier_baseline_analysis.ipynb`에 같은 `run_name`으로 실행 |

데이터를 바꾸는 실험(라벨 오류 제거, crop 여백 변경 등)은 `split_hash`가 달라져 같은 val로 비교하는 것이 아니다. 아래 표의 `split_hash` 열에 데이터 버전을 꼭 적는다.

## 실험 순서

1. 바꿀 것 하나를 정한다.
2. 새 `run_name`으로 학습한다. 이전 결과를 덮어쓰지 않는다.
3. 분석 노트북에서 `run_name`만 바꿔 실행한다. 결과는 `<run_dir>/analysis/`에 저장된다.
4. 아래 표에 한 줄을 추가한다. 실제 설정은 각 실행의 `resolved_config.json`에 남는다.

## 기록

| run_name | 바꾼 것 | split_hash | best epoch | val Top-1 | val macro F1 | 유사 사진 제외 macro F1 | 메모 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| resnet50_baseline_v1 | 기준 | 49d2afee… | 9 / 16 | 0.9983 | 0.9983 | 0.9973 | 오분류 5건 모두 라벨·bbox 오류 |
