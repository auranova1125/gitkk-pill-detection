# 실행 이름·태그·결과 저장 규칙

학습과 통합 추론 결과를 서로 덮어쓰지 않고, 설정과 결과를 나중에 다시 찾을 수 있도록
`run_name`, tag와 출력 폴더 규칙을 통일한다. 현재 tag는 Notebook과 YAML에 주석으로
기록하고, W&B를 연결할 때 같은 문자열 목록을 그대로 전달한다.

## 1. 결과 폴더

새 실행은 작업 종류별 폴더 아래에 저장한다.

```text
outputs/
├── classifier/
│   └── <run_name>/
├── detector/
│   └── <run_name>/
└── inference/
    └── <run_name>/
```

| 작업 | `output_dir` 또는 결과 경로 | 실행 폴더 |
| --- | --- | --- |
| 분류기 학습 | `outputs/classifier` | `outputs/classifier/<run_name>` |
| 탐지기 학습 | `outputs/detector` | `outputs/detector/<run_name>` |
| 탐지·분류 통합 추론 | `outputs/inference/<run_name>` | 해당 경로 자체 |

분류기 학습, 탐지기 학습과 통합 추론 모두 `output_dir / run_name`을 사용한다.
`output_dir`에는 작업별 결과 루트를 전달하고 `run_dir`은 설정 객체에서 계산한다.

`outputs/`는 Git에 포함하지 않는다. 공유할 기준 결과와 비교표는 `reports/`에 기록한다.

## 2. `run_name` 형식

### 학습

```text
<YYYYMMDD>-<model>-<data>-<purpose>-<NN>
```

### 통합 추론

```text
<YYYYMMDD>-<detector>-<classifier>-<data>-<purpose>-<NN>
```

각 항목의 의미는 다음과 같다.

| 항목 | 규칙 | 예시 |
| --- | --- | --- |
| `YYYYMMDD` | 실행을 시작한 한국 시간 날짜 | `20261102` |
| `model` | 문서에서 합의한 짧은 모델 별칭 | `resnet50`, `ssdlite320`, `fasterrcnn-r50-fpn-v2` |
| `data` | 데이터 패키지 또는 입력 묶음의 짧은 별칭 | `cls-v2`, `det-mini`, `infer-mini` |
| `purpose` | 실행의 주목적 하나 | `baseline`, `smoke`, `tune`, `ablation` |
| `NN` | 같은 조건의 당일 순번, 두 자리 숫자 | `01`, `02` |

예시는 다음과 같다.

```text
# 분류기 학습
20261102-resnet50-cls-v2-baseline-01

# 탐지기 smoke 학습
20261102-ssdlite320-det-mini-smoke-01

# 두 모델의 통합 smoke 추론
20261102-ssdlite320-resnet50-infer-mini-smoke-01
```

다음 문자 규칙을 지킨다.

- 영문 소문자, 숫자와 하이픈(`-`)만 사용한다.
- 공백, 한글, underscore와 경로 구분자는 사용하지 않는다.
- 세부 hyperparameter를 모두 이름에 넣지 않는다. 상세 조건은 설정 파일과 tag에 둔다.
- 한 실행에서 비교할 핵심 변경은 하나만 두고 `purpose`와 tag에 표시한다.
- 중단된 실행을 checkpoint로 재개할 때는 같은 `run_name`을 사용한다.
- 처음부터 다시 학습할 때는 `NN`을 올려 새 폴더를 만든다.

## 3. Tag 형식

Tag는 W&B의 `tags: list[str]`에 그대로 넣을 수 있도록 다음 형태를 사용한다.

```text
<key>:<value>
```

key와 value도 영문 소문자, 숫자와 하이픈을 사용한다. 같은 의미에 서로 다른 표현을
만들지 않는다. 예를 들어 `mps`, `apple-mps`, `mac-gpu`를 섞지 않고 `device:mps`로
통일한다.

### 학습 실행의 필수 tag

| key | 값 예시 | 의미 |
| --- | --- | --- |
| `task` | `classifier`, `detector` | 작업 종류 |
| `stage` | `baseline`, `smoke`, `experiment` | 실행 단계 |
| `model` | `resnet50`, `ssdlite320`, `fasterrcnn-r50-fpn-v2` | 모델 별칭 |
| `data` | `cls-v2`, `det-mini` | 데이터 별칭 |
| `change` | `baseline`, `horizontal-flip`, `lr` | 기준 대비 핵심 변경 하나 |
| `device` | `cpu`, `mps`, `cuda` | 실제 실행 장치 |

### 통합 추론의 필수 tag

| key | 값 예시 |
| --- | --- |
| `task` | `inference` |
| `stage` | `smoke`, `evaluation` |
| `detector` | `ssdlite320`, `fasterrcnn-r50-fpn-v2` |
| `classifier` | `resnet50` |
| `data` | `infer-mini` |
| `device` | `cpu`, `mps`, `cuda` |

원본 실행을 추적해야 하면 다음 tag를 추가한다.

```text
detector-run:20261102-ssdlite320-det-mini-smoke-01
classifier-run:20261102-resnet50-cls-v2-baseline-01
```

## 4. 현재 주석 작성 방법

### Notebook

설정 셀에서 `run_name` 바로 아래에 둔다.

```python
run_name = "20261102-ssdlite320-det-mini-smoke-01"

# tags = [
#     "task:detector",
#     "stage:smoke",
#     "model:ssdlite320",
#     "data:det-mini",
#     "change:baseline",
#     "device:mps",
# ]
```

### YAML

`run_name`과 가까운 위치에 같은 값을 주석으로 둔다.

```yaml
run_name: 20261103-ssdlite320-det-mini-smoke-01
output_dir: outputs/detector
device: cuda
# tags:
#   - task:detector
#   - stage:smoke
#   - model:ssdlite320
#   - data:det-mini
#   - change:baseline
#   - device:cuda
```

W&B를 연결할 때 주석의 배열을 설정 필드로 옮기고 다음처럼 그대로 전달한다.

```python
# wandb.init(name=config.run_name, tags=config.tags, ...)
```

## 5. Notebook 경로 예시

### 분류기 학습

```python
output_root = project_root / "outputs" / "classifier"
run_name = "20261102-resnet50-cls-v2-smoke-01"
```

### 탐지기 학습

```python
output_root = project_root / "outputs" / "detector"
run_name = "20261102-ssdlite320-det-mini-smoke-01"
```

### 통합 추론

```python
output_root = project_root / "outputs"
output_dir = output_root / "inference"
run_name = (
    "20261102-ssdlite320-resnet50-infer-mini-smoke-01"
)

detector_run_dir = output_root / "detector" / detector_run_name
classifier_run_dir = output_root / "classifier" / classifier_run_name
result_dir = output_dir / run_name
```

## 6. 기존 실행 결과 전환

현재 `outputs/<run_name>`에 만들어진 결과는 이번 연결 확인이 끝날 때까지 그대로 둔다.
기존 checkpoint를 사용하는 통합 추론에서는 기존 경로를 계속 사용한다.

```python
# 기존 실행 결과를 사용할 때
detector_run_dir = output_root / detector_run_name
classifier_run_dir = output_root / classifier_run_name

# 통합 추론 결과만 새 규칙으로 저장
output_dir = output_root / "inference"
result_dir = output_dir / run_name
```

새 학습부터 `outputs/classifier`와 `outputs/detector`를 사용한다. 기존 완료 폴더를 옮겨야
한다면 checkpoint와 metadata 파일을 함께 폴더 단위로 옮기고, 이를 참조하는 Notebook과
YAML 경로도 동시에 수정한다.

## 7. 실행 전 확인

- 같은 `run_name`의 기존 학습 artifact가 없는가?
- `purpose`가 이번 실행의 핵심 변경 하나를 나타내는가?
- 데이터 별칭과 실제 `split_hash` 또는 입력 묶음이 일치하는가?
- Notebook/YAML 주석의 tag가 실제 설정과 일치하는가?
- 통합 추론이 참조하는 detector/classifier run과 checkpoint가 맞는가?
- `resolved_config.json`, class/label map과 checkpoint를 개별 파일이 아닌 실행 폴더로 관리하는가?
