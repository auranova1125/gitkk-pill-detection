"""알약 탐지기와 분류기를 연결하는 end-to-end 추론 인터페이스."""

from pipeline.inference import (
    EndToEndInferenceResult,
    EndToEndPrediction,
    plot_end_to_end_predictions,
    run_end_to_end_inference,
    save_end_to_end_artifacts,
)

__all__ = [
    "EndToEndInferenceResult",
    "EndToEndPrediction",
    "plot_end_to_end_predictions",
    "run_end_to_end_inference",
    "save_end_to_end_artifacts",
]
