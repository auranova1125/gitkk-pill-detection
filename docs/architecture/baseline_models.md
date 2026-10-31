# 기준 모델과 교체 기준

> **문서 상태:** 현재 코드에 구현된 기준 모델과 후속 모델 비교 원칙을 설명한다. 실행값의 단일 기준은 `configs/*.yaml`, 실제 성능은 `reports/`의 결과 문서다.

## 1. 현재 기준 모델

| 구분 | 분류기 | 탐지기 |
| --- | --- | --- |
| 목적 | bbox crop의 세부 약품 분류 | 조합 이미지의 모든 알약 위치 탐지 |
| 모델 | ResNet-50 | SSDLite320 MobileNetV3-Large |
| 초기 가중치 | ImageNet `IMAGENET1K_V2` | MobileNetV3-Large `IMAGENET1K_V1` backbone |
| 출력 | 설정 가능한 품목 클래스 수 | background `0`, pill `1` |
| Optimizer | AdamW | SGD with momentum |
| Scheduler | fine-tuning 단계의 CosineAnnealingLR | 전체 학습의 CosineAnnealingLR |
| Best 기준 | validation macro F1, 동률이면 Top-1 | 고 IoU mAP@[0.75:0.95], 동률이면 mean average recall |

품목 수의 현재 기본값은 118이지만 분류기 구조의 절대값은 아니다. `ClassifierConfig.expected_num_classes`와 전달된 class map을 함께 변경할 수 있다. 탐지기는 품목 수와 무관하게 단일 foreground 클래스만 예측한다.

## 2. 분류기 기준

### 선택 이유

ResNet-50은 torchvision에서 안정적으로 제공되고, ImageNet 사전학습 특징을 작은 품목 데이터에 전이하기 쉽다. 단계별 동결 범위가 명확해 첫 기준 모델과 오류 분석에 적합하다.

### 현재 학습 방식

1. Head-only 단계에서는 backbone과 BatchNorm을 동결하고 FC만 학습한다.
2. Fine-tuning 단계에서는 `layer4`와 FC만 학습하며 서로 다른 learning rate를 사용한다.
3. 두 단계 모두 BatchNorm 파라미터와 running statistics를 동결한다.
4. `CrossEntropyLoss`를 사용하고, 클래스 불균형은 DataLoader의 sampler에서 처리한다.
5. validation support가 있는 클래스만 macro F1 계산에 포함하고 coverage를 함께 기록한다.

구현 위치는 다음과 같다.

- 모델 생성과 동결: `classifier/model.py`
- optimizer와 학습 단계: `classifier/train.py`
- 지표: `classifier/evaluate.py`
- 기본 실행값: `configs/classifier.yaml`
- 실제 기준 결과: `reports/classifier_report.md`

현재 분류기는 모델 선택 registry를 제공하지 않는다. 다른 backbone 지원은 별도 모델 선택 작업에서 추가한다.

## 3. 탐지기 기준

### 선택 이유

SSDLite320 MobileNetV3-Large는 single-stage SSD 계열이라 학습·추론 흐름이 단순하고, 비교적 가벼워 MPS와 CUDA에서 반복 실험하기 좋다. torchvision detection의 공통 호출 규격을 사용하므로 동일 계열 모델을 비교하기도 쉽다.

COCO detector 전체 가중치는 사용하지 않는다. ImageNet 사전학습 backbone과 단일 `pill` 클래스에 맞춘 새 detection head를 사용한다.

### 현재 학습 방식

1. 모든 학습 가능 파라미터를 SGD로 최적화한다.
2. momentum과 weight decay를 설정으로 전달한다.
3. 전체 epoch를 주기로 CosineAnnealingLR을 적용한다.
4. validation에서는 COCO mAP와 고 IoU mAP, threshold 전후 recall, 객체 수 오차를 기록한다.
5. 이미지당 알약 수를 고정하거나 예측을 임의로 최대 네 개로 자르지 않는다.

구현 위치는 다음과 같다.

- 모델 builder와 현재 registry: `detector/model.py`
- optimizer와 scheduler: `detector/train.py`
- 지표와 threshold 평가: `detector/evaluate.py`
- 기본 실행값: `configs/detector.yaml`
- 결과 기록 양식: `reports/detector_report.md`

현재 registry에는 SSDLite 하나만 등록돼 있다. registry가 존재한다는 사실과 여러 모델의 호환이 검증됐다는 사실을 혼동하지 않는다.

## 4. 모델 비교 기록

후속 모델을 추가할 때 아래 표에 상태와 결과 문서를 연결한다. 실제 성능 수치는 동일한 데이터 분할에서 측정한 결과 문서에 기록한다.

| 구분 | 모델 | 상태 | 비교 목적 |
| --- | --- | --- | --- |
| 분류기 | ResNet-50 | 기준 구현·학습 완료 | 품목 분류 기준선 |
| 탐지기 | SSDLite320 MobileNetV3-Large | 기준 구현, 실제 전체 학습 결과 대기 | 속도와 위치 정확도 기준선 |
| 탐지기 | SSD300 VGG16 | 후보 | 같은 SSD 계열에서 고 IoU 위치 정확도 비교 |

후보는 결과 없이 registry에 미리 추가하지 않는다. 기준 모델의 한계가 실제 validation에서 확인되고 비교 실험이 정해졌을 때 구현한다.

## 5. 교체와 비교 원칙

모델 추가는 다음 조건을 만족해야 한다.

1. 같은 train/validation split과 동일한 데이터 버전을 사용한다.
2. 입력 전처리가 달라지면 모델 차이와 데이터 차이를 분리해 기록한다.
3. 공통 학습·평가 인터페이스를 유지하거나 차이를 adapter 내부에 격리한다.
4. checkpoint에 모델 이름과 생성 설정을 저장하고 다른 모델로 잘못 재개하지 못하게 한다.
5. 합성 batch forward/backward와 checkpoint 복원 테스트를 추가한다.
6. 정확도뿐 아니라 추론 시간, peak memory와 checkpoint 크기도 함께 비교한다.

분류기는 Top-1, Top-5, macro F1과 클래스별 오류를 비교한다. 탐지기는 COCO mAP, 고 IoU mAP, recall과 객체 수 오차를 비교한다. 탐지기 변경은 최종적으로 예측 bbox를 분류기에 연결한 end-to-end 결과도 확인한다.

Ultralytics처럼 torchvision과 호출 규격이 다른 구현은 기존 학습 loop에 조건문을 누적하지 않고 별도 backend adapter가 필요한지 먼저 판단한다. 모든 모델을 지원하는 범용 계층보다 실제 비교 대상 몇 개를 단순하게 지원하는 것을 우선한다.
