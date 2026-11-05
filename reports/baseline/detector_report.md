# 알약 탐지기 모델 비교 요약

이 문서는 탐지기 기준선과 비교 모델의 핵심 결과를 한곳에서 확인하기 위한 요약이다. 모든 모델은 탐지기 데이터 v2의 동일한 train/validation 분할을 사용하고, 공통 ResNet-50 분류기와 연결해 최종 품목 예측을 평가했다. 세부 학습 조건과 오류 사례는 모델별 보고서에 기록한다.

## 비교 결과

| 탐지기 | Category-aware mAP@[0.75:0.95] | Kaggle Public | 해석 |
| --- | ---: | ---: | --- |
| SSDLite320 MobileNetV3-Large | 0.9331 | 0.58765 | 빠른 경량 기준선이지만 높은 IoU 구간의 위치 오차가 남음 |
| Faster R-CNN ResNet50-FPN v2 | 0.9577 | 0.59375 | 기준선보다 위치 정밀도가 개선됐지만 학습 비용과 특정 품목 누락이 큼 |
| YOLO11s | **0.9910** | 0.60831 | 높은 로컬 위치 정밀도와 안정적인 객체 수 예측을 기록함 |
| YOLO26s | 0.9841 | **0.61804** | 로컬 지표는 YOLO11s보다 낮지만 Kaggle Public 점수가 가장 높음 |

로컬 validation과 Kaggle Public은 데이터 분포와 평가 환경이 다르므로 수치를 직접 환산하지 않는다. 최종 제출 모델 판단에는 공개 점수뿐 아니라 재현 가능한 학습 경로, 추론 비용, 오류 유형도 함께 고려한다.

## 상세 보고서

- [SSDLite320 학습·통합 평가](../experiments/ssdlite320_training_report.md)
- [Faster R-CNN 학습·통합 평가](../experiments/faster_rcnn_training_report.md)
- [YOLO11s 학습·통합 평가](../experiments/yolo11s_training_report.md)
- [YOLO26s 학습·통합 평가](../experiments/yolo26s_training_report.md)
- [Kaggle 제출 기록과 진단](../experiments/kaggle_submissions.md)

## 공통 재현 경로

- torchvision 탐지기: `notebooks/baseline/train_detector.ipynb`
- Ultralytics 탐지기: `notebooks/experiments/train_yolo26s_detector.ipynb`
- 탐지·분류 통합 평가와 제출: `notebooks/baseline/run_pill_recognition.ipynb`
- 실행 설정: `configs/detector.yaml`, `configs/yolo_detector.yaml`, `configs/pipeline.yaml`

가중치와 전체 실행 산출물은 Git에 포함하지 않고 실행별 output 디렉터리에 보관한다.
