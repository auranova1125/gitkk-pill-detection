# 라벨 2차 검수 적용 요약

완료된 bbox 위치·클래스 사람 검수에서 `오류`로 판정한 항목만 병합한 결과다. 상세 crop별 조치는 `etl/classifier/label_audit/records/label_actions.csv`가 기준이다.

## 조치별 건수

| 조치 | 사진 | crop | train | validation |
| --- | ---: | ---: | ---: | ---: |
| 이미지 제외 | 49 | 188 | 134 | 54 |
| bbox 제외 | 2 | 2 | 1 | 1 |
| 라벨 교환 | 1 | 2 | 0 | 2 |

- 전체 crop 14,988개에서 190개를 제외해 14,798개가 남는다.
- 라벨 교환 crop 2개는 제외하지 않으며, 남는 수에도 포함된다.
- 결과 패키지는 train 11,990개와 validation 2,808개다.
- 현재 데이터셋의 모든 클래스가 train과 validation에 남아 있다.
- 가장 적은 train 클래스는 K-003483의 33개, 가장 적은 validation 클래스는 K-025438의 9개다.

## 기록 위치

| 파일 | 내용 |
| --- | --- |
| `position_suspects.csv` | bbox 위치 후보, 사람 판정과 메모 |
| `class_suspects.csv` | 클래스 후보, 사람 판정과 메모 |
| `label_actions.csv` | 중복 후보를 병합한 crop별 최종 조치 |
| `detector_actions.csv` | 탐지 패키지 이미지와 연결한 전달 조치 |

파일은 모두 `etl/classifier/label_audit/records/`에 있다. 데이터 분포의 세부 변화는 [분류 데이터 v2 결과](data_v2_report.md), 후보 생성과 판정 기준은 [분류 라벨 2차 검수 보고서](label_audit_report.md)에서 확인한다.

이 문서는 확정 데이터의 요약본이다. 후보 CSV의 판정을 변경했다면 다음 명령으로 `label_actions.csv`와 이 요약을 다시 만든다.

```bash
uv run python -m etl.classifier.label_audit.merge_review_actions
```
