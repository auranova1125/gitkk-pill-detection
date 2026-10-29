"""단일 클래스 알약 탐지기 공용 인터페이스."""

from configs.detector import DetectorConfig
from detector.checkpoint import (
    CHECKPOINT_VERSION,
    hash_label_map,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from detector.evaluate import (
    EvaluationResult,
    evaluate_detector_checkpoint,
    evaluate_model,
    filter_predictions_by_score,
)
from detector.inference import (
    DetectorInferenceError,
    DetectorInferenceResult,
    DetectorPrediction,
    load_inference_model,
    plot_detection_comparison,
    predict_detector,
    save_inference_artifacts,
    validate_inference_batch,
)
from detector.input_validation import (
    InputValidationError,
    validate_detector_data_summary,
    validate_detector_label_map,
    validate_detector_training_batch,
)
from detector.model import (
    MODEL_BUILDERS,
    available_detector_models,
    build_detector_model,
    build_ssdlite320,
)
from detector.train import (
    EarlyStopping,
    TrainingEpochResult,
    TrainingResult,
    build_optimizer,
    build_scheduler,
    run_training,
    train_detector,
    train_one_epoch,
)
from detector.training_data import (
    DetectorTrainingData,
    move_detection_batch_to_device,
    prepare_detector_data,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "MODEL_BUILDERS",
    "DetectorConfig",
    "DetectorInferenceError",
    "DetectorInferenceResult",
    "DetectorPrediction",
    "DetectorTrainingData",
    "EarlyStopping",
    "EvaluationResult",
    "InputValidationError",
    "TrainingEpochResult",
    "TrainingResult",
    "available_detector_models",
    "build_detector_model",
    "build_optimizer",
    "build_scheduler",
    "build_ssdlite320",
    "evaluate_detector_checkpoint",
    "evaluate_model",
    "filter_predictions_by_score",
    "hash_label_map",
    "load_checkpoint",
    "load_inference_model",
    "move_detection_batch_to_device",
    "plot_detection_comparison",
    "predict_detector",
    "prepare_detector_data",
    "restore_checkpoint_state",
    "run_training",
    "save_checkpoint",
    "save_inference_artifacts",
    "train_detector",
    "train_one_epoch",
    "validate_checkpoint_compatibility",
    "validate_detector_data_summary",
    "validate_detector_label_map",
    "validate_detector_training_batch",
    "validate_inference_batch",
]
