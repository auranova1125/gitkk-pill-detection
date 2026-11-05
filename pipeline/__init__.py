"""알약 탐지 결과를 세부 품목 분류로 연결하는 인식 인터페이스."""

from pipeline.evaluation import (
    PillRecognitionEvaluation,
    PillRecognitionEvaluationDetail,
    evaluate_pill_recognition,
    save_pill_recognition_evaluation,
)
from pipeline.pill_recognition import (
    PillPrediction,
    PillRecognitionData,
    PillRecognitionResult,
    build_classifier_batch,
    load_configured_detector,
    plot_pill_predictions,
    prepare_pill_recognition_data,
    recognize_pills,
    recognize_pills_from_config,
    save_pill_recognition_artifacts,
    save_submission_csv,
)

__all__ = [
    "PillPrediction",
    "PillRecognitionData",
    "PillRecognitionEvaluation",
    "PillRecognitionEvaluationDetail",
    "PillRecognitionResult",
    "build_classifier_batch",
    "evaluate_pill_recognition",
    "load_configured_detector",
    "plot_pill_predictions",
    "prepare_pill_recognition_data",
    "recognize_pills",
    "recognize_pills_from_config",
    "save_pill_recognition_artifacts",
    "save_pill_recognition_evaluation",
    "save_submission_csv",
]
