# 분류 데이터 v2 결과

> v1 분류 학습 패키지에 사람이 확정한 라벨 2차 검수 조치를 적용한 결과다. 현재 데이터셋의 기록이며 클래스 수나 샘플 수를 일반적인 고정값으로 정의하지 않는다.

## 요약

| 항목 | v1 | v2 |
| --- | ---: | ---: |
| crop | 14,988 | 14,798 |
| train | 12,125 | 11,990 |
| validation | 2,863 | 2,808 |
| 그룹 train / validation | 1,786 / 429 | 1,774 / 424 |
| validation 클래스 coverage | 118 / 118 | 118 / 118 |
| split_hash | `49d2afee…` | `a2eb581a…` |

- v1 전체 hash: `49d2afee86a4ec8f48c25488eac9b1f9081b36066e5533eba40670087f60430c`
- v2 전체 hash: `a2eb581a04604ff1e0b156980965a49650ecb4dec919e3def500c35acf8241e0`
- 남은 crop은 v1과 `sample_id`, 원본 crop 파일, split과 group이 같다.
- `class_map.json`도 같으므로 v1에서 학습한 checkpoint를 v2 validation에 평가할 수 있다.
- `near_dup_in_train`은 v1 값을 유지한다. 짝이 되는 train 이미지가 빠진 경우에도 `True`로 남을 수 있어 유사 이미지 제외 평가는 보수적일 수 있다.

## 적용 조치

| 조치 | 사진 | crop | train | validation |
| --- | ---: | ---: | ---: | ---: |
| 이미지 제외 | 49 | 188 | 134 | 54 |
| bbox 제외 | 2 | 2 | 1 | 1 |
| 라벨 교환 | 1 | 2 | 0 | 2 |

라벨 교환은 Kaggle validation 이미지 한 장의 K-027733과 K-035206 crop에 적용했다. 두 crop은 제외하지 않고 코드, class index, 품목 메타데이터와 crop 경로만 올바른 클래스로 옮겼다. 확정 crop별 조치는 `etl/classifier/label_audit/records/label_actions.csv`에 있다.

## 기준 모델 재평가

v1에서 학습한 `resnet50_baseline_v1`의 같은 checkpoint를 두 validation에 평가했다.

| validation | 샘플 | 오분류 | Top-1 | Top-5 | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 2,863 | 5 | 0.9983 | 0.9986 | 0.9983 |
| v2 | 2,808 | 0 | 1.0000 | 1.0000 | 1.0000 |

v1 오분류 5건 중 두 건은 라벨 교환 후 정답이 되었고 세 건은 오류 이미지 제외로 v2에서 빠졌다. v2에서 새로 생긴 오분류는 없었다.

이 결과는 v2로 다시 학습한 모델이 완전한 성능을 낸다는 뜻이 아니다. 기준 모델이 v2 validation을 이미 모두 맞혀 Top-1과 macro F1만으로 후속 모델을 비교하기 어렵다. validation loss, 예측 확신도, GT bbox와 예측 bbox 사이의 차이 및 end-to-end 평가를 함께 사용해야 한다.

## 생성과 검증

```bash
uv run python -m etl.classifier.build_classifier_v2 --no-eval

# 기준 checkpoint가 있을 때 v1·v2 validation 재평가
uv run python -m etl.classifier.build_classifier_v2 \
  --checkpoint data/best.pt \
  --overwrite
```

`build_classifier_v2`는 감사 대상 v1 hash, 조치 메타데이터와 이미지 단위 제외의 완전성을 확인한다. 패키지를 임시 경로에 만든 뒤 현재 `data_loading.classifier`와 분류 입력 검증을 통과한 경우에만 최종 폴더와 ZIP으로 옮긴다.

검수 방법은 [라벨 2차 검수 보고서](label_audit_report.md), 조치 집계는 [라벨 2차 검수 적용 요약](label_audit_summary.md)에서 확인한다.
