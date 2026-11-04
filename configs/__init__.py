"""분류기, 탐지기와 통합 인식의 실행 설정."""

from configs.classifier import ClassifierConfig, load_classifier_config
from configs.detector import DetectorConfig, load_detector_config
from configs.pill_recognition import (
    PillRecognitionConfig,
    load_pill_recognition_config,
)
from configs.yolo_detector import YoloDetectorConfig, load_yolo_detector_config

__all__ = [
    "ClassifierConfig",
    "DetectorConfig",
    "PillRecognitionConfig",
    "YoloDetectorConfig",
    "load_classifier_config",
    "load_detector_config",
    "load_pill_recognition_config",
    "load_yolo_detector_config",
]
