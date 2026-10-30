"""ResNet-50 분류 모델과 단계별 학습 설정."""

from enum import StrEnum

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import ResNet


class TrainingStage(StrEnum):
    """분류 모델의 학습 단계."""

    HEAD_ONLY = "head_only"
    LAYER4_FINE_TUNING = "layer4_fine_tuning"


def build_classifier_model(num_classes: int, pretrained: bool = True) -> ResNet:
    """ImageNet ResNet-50의 분류 head를 지정한 클래스 수로 교체한다."""

    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a boolean")

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            _set_requires_grad(module, False)
            module.eval()


def configure_model_for_stage(
    model: ResNet,
    stage: TrainingStage | str,
) -> None:
    """
    Head-only에서는 FC만 학습 가능하게 만듭니다.
    Fine-tuning에서는 layer4와 FC만 학습 가능하게 만듭니다.
    두 단계 모두 BatchNorm 파라미터는 동결합니다.
    """

    try:
        resolved_stage = TrainingStage(stage)
    except ValueError as error:
        valid_stages = ", ".join(stage.value for stage in TrainingStage)
        raise ValueError(f"stage must be one of: {valid_stages}") from error

    _set_requires_grad(model, False)
    _set_requires_grad(model.fc, True)
    if resolved_stage is TrainingStage.LAYER4_FINE_TUNING:
        _set_requires_grad(model.layer4, True)

    _freeze_batch_norm(model)


def set_model_train_mode(model: nn.Module) -> None:
    """모델을 train mode로 전환하고 frozen BatchNorm은 eval mode로 유지한다."""

    model.train()
    for module in model.modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()
