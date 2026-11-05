# 보고서 안내

보고서는 목적에 따라 기준 요약, 모델 실험, 데이터 검수 기록으로 구분한다. 실행 가중치와 전체 예측 파일은 저장소 밖의 실행 폴더에 보관하고, Git에는 비교에 필요한 지표·그림·해석만 남긴다.

## 기준 결과

- [분류기 기준 결과](baseline/classifier_report.md): ResNet-50 기준선의 데이터, 학습 조건, 품목별 오류
- [탐지기 모델 비교](baseline/detector_report.md): 네 탐지기의 통합 평가와 Kaggle 결과 요약

## 모델 실험

- [분류기 실험 기록](experiments/classifier_experiments.md)
- [SSDLite320](experiments/ssdlite320_training_report.md)
- [Faster R-CNN](experiments/faster_rcnn_training_report.md)
- [YOLO11s](experiments/yolo11s_training_report.md)
- [YOLO26s](experiments/yolo26s_training_report.md)
- [Kaggle 제출 기록과 진단](experiments/kaggle_submissions.md)

## 데이터 검수

- 분류기: [최종 입력 검증](data/classifier/validation_report.md), [데이터 v2 구성](data/classifier/data_v2_report.md), [AI Hub 외부 평가셋](data/classifier/aihub_val_eval_package_report.md)
- 탐지기: [현재 v2 구성·검수](data/detector/data_v2_report.md), [과거 v1 입력 검증](data/detector/data_v1_validation_report.md)

`data_v1_validation_report.md`는 데이터 정제 전 상태를 재현하기 위한 과거 기록이다. 현재 학습과 모델 비교는 v2 데이터와 해당 split hash를 기준으로 한다.
