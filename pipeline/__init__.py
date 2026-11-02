"""알약 탐지 결과를 세부 품목 분류로 연결하는 인식 인터페이스."""

from pipeline.pill_recognition import (
    PillPrediction,
    PillRecognitionResult,
    build_classifier_batch,
    plot_pill_predictions,
    recognize_pills,
    recognize_pills_from_config,
    save_pill_recognition_artifacts,
)

__all__ = [
    "PillPrediction",
    "PillRecognitionResult",
    "build_classifier_batch",
    "plot_pill_predictions",
    "recognize_pills",
    "recognize_pills_from_config",
    "save_pill_recognition_artifacts",
]
