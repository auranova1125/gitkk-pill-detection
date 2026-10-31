# 프로젝트 구조

> **문서 상태:** 현재 구현에 대한 구조·책임 기준이다. 초기 리팩터링 계획서를 실제 병합 결과에 맞게 갱신했다.

## 1. 설계 목표

- 분류기와 탐지기는 같은 역할에 같은 구조와 명명 규칙을 사용한다.
- 모델별 학습 loop, 평가 지표와 checkpoint schema는 억지로 합치지 않는다.
- 오프라인 ETL, runtime Dataset, 모델 코드와 통합 추론을 분리한다.
- Notebook과 CLI는 같은 Python 모듈을 호출한다.
- 공통화는 device·RNG·artifact I/O·이미지 geometry 같은 저수준 기능에 한정한다.

## 2. 현재 구조

```text
gitkk-pill-detection/
├── main.py
├── configs/
│   ├── classifier.py / classifier.yaml
│   ├── detector.py / detector.yaml
│   └── pill_recognition.py / pill_recognition.yaml
├── classifier/
│   ├── model.py
│   ├── training_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── input_validation.py
│   └── checkpoint.py
├── detector/
│   └── classifier와 같은 역할별 파일
├── data_loading/
│   ├── classifier.py
│   ├── detector.py / detector_package.py
│   ├── recognition.py
│   ├── image_transforms.py / validation.py / factory.py
│   └── synthetic.py
├── etl/classifier/
│   ├── validate_source_labels.py
│   ├── select_source_images.py
│   ├── create_pill_crops.py
│   ├── assign_data_splits.py
│   ├── summarize_crops.py
│   ├── export_training_package.py
│   └── build_classifier_dataset.py
├── etl/detector/
│   ├── codeit_source.py
│   ├── select_source.py
│   ├── training_package.py
│   └── build_detector_dataset.py
├── pipeline/pill_recognition.py
├── utils/runtime.py / io.py
├── notebooks/
│   ├── baseline/
│   ├── experiments/
│   └── eda/
└── tests/
    ├── classifier/ / detector/ / data_loading/
    ├── etl/ / pipeline/ / configs/ / utils/ / notebooks/ / documentation/
    └── smoke/
```

## 3. 데이터 경계

```text
원천 데이터
  → etl/                         오프라인 검증·변환·패키징
  → data/ 학습 패키지
  → data_loading/                Dataset·DataLoader·tensor 변환
  → classifier/ 또는 detector/ 학습·평가·추론
  → pipeline/                    탐지 bbox를 품목 분류로 연결
```

`etl/`은 학습 실행 중 import하지 않는다. `data_loading/`는 준비된 학습 패키지를 읽을 뿐 원천 데이터를 재생성하지 않는다.

탐지 ETL은 확정된 Codeit 원천 adapter와 최종 패키지 검증만 구현한다. 형식이 확정되지 않은 SELECT adapter와 전체 조립 함수는 조용히 임의 처리하지 않고 `NotImplementedError`를 발생시키는 플레이스홀더다.

## 4. 분류기·탐지기 대칭성

| 역할 | 분류기 | 탐지기 |
| --- | --- | --- |
| 설정 | `ClassifierConfig` | `DetectorConfig` |
| 모델 생성 | `build_classifier_model` | `build_detector_model` |
| 학습 데이터 | `prepare_classifier_data` | `prepare_detector_data` |
| 학습 | `train_classifier` | `train_detector` |
| checkpoint 평가 | `evaluate_classifier_checkpoint` | `evaluate_detector_checkpoint` |
| 추론 모델 로드 | `load_classifier_inference_model` | `load_detector_inference_model` |
| 학습 batch 검증 | `validate_classifier_training_batch` | `validate_detector_training_batch` |

분류기의 단계별 fine-tuning과 탐지기의 bbox metric처럼 도메인이 다른 기능은 각 패키지에 남긴다.

## 5. 실행 진입점

- Notebook: 실험 설정, 샘플 확인, 시각화
- `main.py`: 자동화, 재현, CLI smoke
- `configs/*.yaml`: Notebook과 CLI가 공유할 수 있는 기본값
- Python 모듈: 실제 학습·평가·추론 로직

## 6. 안전 규칙

- 신규 학습은 기존 `best.pt`, `last.pt`, `metrics.csv`를 묵시적으로 덮어쓰지 않는다.
- ETL은 기존 crop·export 폴더를 `--overwrite` 없이 삭제하지 않는다.
- DataLoader, checkpoint, 모델 출력과 pipeline 경계에서 입력 검증을 유지한다.
- 분류 클래스 수 118은 기본값이며 설정으로 변경 가능하다.

## 7. 후속 브랜치 범위

다음 항목은 이 구조 리팩터링에 포함하지 않았다.

- 분류기·탐지기 모델 registry 확장
- 필수·실험 의존성 계층 분리
- W&B 실험 추적
- 실제 탐지 결과를 사용한 end-to-end 평가

## 8. 관련 문서

- [분류 데이터 사용](../data/classifier_data_guide.md)
- [분류 ETL](../data/classifier_etl.md)
- [탐지 데이터 사용](../data/detector_data_guide.md)
- [탐지 ETL](../data/detector_etl.md)
- [분류기 입력 명세](../specs/classifier_input_spec.md)
- [탐지기 입력 명세](../specs/detector_input_spec.md)
