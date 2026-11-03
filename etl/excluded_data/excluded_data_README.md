# 학습 데이터 제외 기록

이 폴더는 detector 학습 데이터에서 제외하기로 확정한 기록과 그 검수 근거를 보관한다. 현재 detector 학습 패키지에 직접 적용하는 기준은 `final/detector_excluded_images.csv`다.

## 현재 적용 기준

### Detector

`final/detector_excluded_images.csv`는 detector v1 10,319장에서 추가로 제외할 이미지 150장을 기록한다.

- train: 113장
- validation: 37장
- annotation: 579개
- 적용 후 예상 결과: 이미지 10,169장, annotation 39,009개

사진 하나에 잘못된 bbox가 있으면 그 이미지와 연결된 annotation 전체를 제외한다. `etl/classifier/label_audit/records/detector_actions.csv`의 48장과 `audit/detector_position_review.csv`에서 오류가 확인된 102장의 합집합이다.

### Classifier

Classifier crop에 적용할 최종 조치 192건은 기존 정식 기록인 `etl/classifier/label_audit/records/label_actions.csv`에서 관리한다.

- `이미지 제외`: 해당 이미지에 연결된 crop 전체 제외
- `bbox 제외`: 지정 crop만 제외
- `라벨 교환 (수정)`: crop은 유지하고 품목 라벨 수정

Detector는 이미지 단위로 제외하지만 classifier는 crop 단위 조치와 라벨 수정을 구분한다. 두 모델의 최종 제외 수가 같을 필요는 없다. 같은 파일의 중복 사본은 이 폴더에 두지 않는다.

## 폴더 구조

```text
excluded_data/
├── README.md
├── final/
│   └── detector_excluded_images.csv
├── audit/
│   └── detector_position_review.csv
├── reports/
│   └── detector_v2_report.md
└── archive/
    ├── legacy_exclusion_guide.md
    ├── legacy_excluded_image_details.csv
    ├── legacy_class_exclusion_summary.csv
    └── legacy_*_issues.csv
```

- `final/`: detector ETL과 현재 학습 패키지가 직접 사용하는 확정 입력
- `audit/`: 확정 판정의 근거가 된 세부 검수 기록
- `reports/`: detector v2 구성 결과와 한계
- `archive/`: detector v1 10,319장을 만들기 전에 이미 반영된 153장 제외 기록과 이전 검수 산출물

`archive/legacy_excluded_image_details.csv`의 153장과 `final/detector_excluded_images.csv`의 150장은 서로 겹치지 않는다. 앞의 153장은 detector v1에 이미 반영되어 있고, 뒤의 150장은 v1에서 v2를 만들 때 추가로 적용한다. 최신 150장 중 분류기 검수에서 전달된 48장의 근거는 `etl/classifier/label_audit/records/detector_actions.csv`에 있다.

## 적용 원칙

1. 원본 데이터 폴더 전체를 직접 학습 입력으로 사용하지 않는다.
2. 제외 결과는 annotation, split manifest와 요약 metadata에 함께 반영한다.
3. classifier는 `samples.csv`와 `data_summary.json`을 같은 기준으로 갱신한다.
4. 제외 전 train/validation 배정과 기존 ID는 유지한다.
5. 중간 검수 후보를 자동으로 제외하지 않고 `final/`에 확정된 조치만 적용한다.
