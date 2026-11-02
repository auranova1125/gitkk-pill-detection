# 탐지 데이터 ETL

> **문서 상태:** Codeit 원천 형식과 최종 학습 패키지 검증은 확정됐으며, SELECT 원천 형식과 최초 분할 규칙은 미확정이다.

## 책임

탐지 ETL은 원천 이미지와 이미지별 전체 bbox annotation을 읽어, runtime DataLoader가 소비하는 고정 학습 패키지를 만드는 오프라인 과정이다. 학습 중 이미지 증강이나 tensor 변환은 담당하지 않는다.

```text
SELECT 원천 ─┐
             ├→ 원천 검증 → 그룹 단위 분할 → 학습 패키지 → 최종 검증
Codeit 원천 ─┘
```

## 현재 구현

| 파일 | 상태 | 책임 |
| --- | --- | --- |
| `etl/detector/codeit_source.py` | 구현 | `codeit_train_validated`의 이미지·JSON 연결과 이미지별 전체 bbox 검증 |
| `etl/detector/select_source.py` | 플레이스홀더 | SELECT 원천 형식 확정 후 adapter 구현 |
| `etl/detector/training_package.py` | 구현 | 완성 패키지를 runtime parser로 검증하고 label map·summary 기록 |
| `etl/detector/build_detector_dataset.py` | 플레이스홀더 | 두 출처 병합, 최초 그룹 분할과 최종 export 조립 |

플레이스홀더는 빈 결과를 반환하지 않고 `NotImplementedError`를 발생시킨다. 따라서 미완성 경로가 정상 처리된 것처럼 학습으로 넘어가지 않는다.

## 확정된 Codeit 입력

```text
codeit_train_validated/
├── annotations/
│   └── 이미지별 JSON
└── images/
    └── codeit_K-...png
```

각 JSON은 다음 조건을 만족해야 한다.

- `images`에는 정확히 한 이미지가 있다.
- `annotations`에는 그 이미지 파일명에 포함된 모든 약품 코드가 정확히 한 번씩 있다.
- 각 `bbox`는 `XYWH`이며 실제 이미지 경계 안에 있다.
- `area`는 bbox 면적과 일치하고 `iscrowd`는 `0`이다.
- `category_id`, `drug_code`, category 이름이 같은 약품 코드를 나타낸다.
- 전체 파일에서 이미지 ID와 annotation ID는 중복되지 않는다.

`Archive.zip`은 형식 확인 자료일 뿐 저장소 실행 의존성이 아니다. 실제 ETL 호출 시 원천 디렉터리 경로를 명시적으로 전달한다.

## 최종 학습 패키지

완성 출력은 [탐지 데이터 사용 안내](detector_data_guide.md)의 구조와 [탐지기 입력 명세](../specs/detector_input_spec.md)를 따라야 한다. `validate_training_package`는 별도 규칙을 다시 구현하지 않고 runtime의 `data_loading.detector_package.read_records`를 호출한다. 이로써 ETL 완료 판정과 실제 학습 입력의 검증 기준을 하나로 유지한다.

## 구현 보류 조건

다음 정보가 전달되면 플레이스홀더를 실제 구현으로 교체한다.

1. 검증 완료된 SELECT 원천 디렉터리 구조와 이미지별 annotation 예시
2. 최초 train/validation 그룹 분할을 재현하는 규칙 또는 기존 split manifest
3. 두 출처에서 ID가 겹칠 때 사용할 재번호 부여 규칙

기존 split manifest가 있다면 임의로 다시 나누지 않고 그대로 보존하는 것을 우선한다.
