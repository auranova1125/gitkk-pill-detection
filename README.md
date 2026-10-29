# 알약 이미지 Object Detection

팀명: 깃크크크(4팀)
화이팅.
사진 속 최대 4개 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 협업 프로젝트입니다.

Kaggle: [\[AI15\] 경구약제 이미지 객체 검출(Object Detection) 프로젝트](https://www.kaggle.com/competitions/ai15-level1-project/overview)

## 개발 환경

| 항목 | 기준 |
| --- | --- |
| Python | 3.12 |
| 패키지 관리 | uv |
| 로컬 학습 | macOS, PyTorch MPS |
| 원격 학습 | Google Colab, CUDA |

`pyproject.toml`, `uv.lock`, `.python-version`은 Git으로 공유합니다.

## 시작하기

### 1. uv와 Python 설치

```bash
brew install uv
uv python install 3.12
```

Homebrew를 사용하지 않는 경우 [uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 따릅니다.

### 2. 저장소 복제 및 의존성 설치

GitHub 저장소의 **Code > HTTPS**에서 주소를 복사해 저장소를 clone한 뒤 프로젝트 디렉터리로 이동합니다.

```bash
cd gitkk-pill-detection
uv sync --locked
```

`uv sync --locked`는 `uv.lock`을 변경하지 않고 잠긴 의존성으로 `.venv`를 구성합니다. `pyproject.toml`과 `uv.lock`이 일치하지 않으면 실패합니다.

### Google Colab

Colab에서 저장소를 clone하고 프로젝트 디렉터리로 이동한 뒤 다음 셀을 실행합니다.

```python
%pip install -q uv
!uv sync --locked
```

이후 **런타임 > 런타임 유형 변경 > GPU**를 선택합니다.

## GPU 확인

로컬 환경에서는 다음 명령으로 실제 사용 장치를 확인합니다.

```bash
uv run python - <<'PY'
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.ones(1, device=device)
print(f"device={device}, tensor={x}")
PY
```

- Apple Silicon Mac: `device=mps`
- GPU를 사용할 수 없는 환경: `device=cpu`

Colab에서는 다음 셀로 CUDA 연결을 확인합니다.

```python
!uv run python -c 'import torch; print(f"cuda={torch.cuda.is_available()}")'
```

정상적인 GPU 런타임에서는 `cuda=True`가 출력됩니다.

## Jupyter Notebook Git 설정

`nbdime`은 Notebook diff/merge를 읽기 쉽게 만들고, `nbstripout`은 커밋되는 Notebook에서 출력값을 제거합니다. 저장소를 복제한 뒤 팀원마다 한 번씩 설정합니다.

```bash
uv tool install nbdime
uv tool install nbstripout
```

명령을 찾지 못하면 `uv tool update-shell`을 실행하고 터미널을 다시 연 뒤 계속 진행합니다.

```bash
# .gitattributes 제외 규칙을 정확히 한 줄로 유지
sed -i '' '/^[[:space:]]*\.gitattributes[[:space:]]*$/d' .git/info/exclude
printf '%s\n' '.gitattributes' >> .git/info/exclude

# 현재 저장소에 Git 연동
nbdime config-git --enable
nbstripout --install

# Notebook 규칙을 Git 내부 파일에 저장
printf '%s\n' \
  '*.ipynb filter=nbstripout diff=jupyternotebook merge=jupyternotebook' \
  '*.zpln filter=nbstripout' \
  > .git/info/attributes

# nbdime이 생성한 루트 파일 제거
rm -f .gitattributes
```

루트에는 `.gitattributes`를 남기지 않습니다. 규칙은 Git 내부의 `.git/info/attributes`에서 적용되며, `.git/info/exclude`를 통해 `.gitattributes`가 다시 생성돼도 추적되지 않습니다. 두 파일은 로컬 Git 설정이므로 다른 팀원에게 자동으로 전달되지 않습니다.

설정 결과를 확인합니다.

```bash
test ! -e .gitattributes && echo 'root .gitattributes: 없음'
git check-ignore .gitattributes
git check-attr filter diff merge -- notebooks/example.ipynb
```

정상 결과는 다음과 같습니다.

```text
root .gitattributes: 없음
.gitattributes
notebooks/example.ipynb: filter: nbstripout
notebooks/example.ipynb: diff: jupyternotebook
notebooks/example.ipynb: merge: jupyternotebook
```

## 협업 방식

`main` 브랜치에는 직접 Commit하거나 Push하지 않습니다. Issue를 생성한 뒤 최신 `main`에서 작업 브랜치를 만듭니다.

```bash
git switch main
git pull --ff-only origin main

# 예시: Issue #12의 클래스 분포 분석 작업
git switch -c 12-eda-class-distribution
```

브랜치명은 `<이슈번호>-<작업명>` 형식을 사용하며, 하나의 브랜치에서는 하나의 Issue만 처리합니다.

Commit 메시지는 변경 목적에 따라 `feat:`, `fix:`, `chore:` 접두어를 사용합니다.

```bash
git add -p
git commit -m "feat: 클래스 분포 분석"
git push -u origin "$(git branch --show-current)"
```

작업 중 `main`의 변경 사항이 현재 작업에 영향을 주거나 PR 제출 전 동기화가 필요하다면 작업 브랜치에서 다음을 실행합니다.

```bash
git fetch origin
git merge origin/main
```

충돌을 해결한 뒤 관련 기능을 다시 확인합니다. 의존성 파일이 변경됐다면 `uv sync --locked`도 실행합니다.

모든 변경은 `main` 대상 PR로 제출합니다. 최소 1명의 리뷰와 승인을 받은 뒤 **Squash Merge**하고 작업 브랜치를 삭제합니다. 다음 작업은 다시 최신 `main`에서 시작합니다.

상세 협업 규칙은 프로젝트 Notion을 참고합니다.

## 분류기 구현

`classifier/`에는 현재 118종을 대상으로 하는 분류기의 입력 검증, ResNet-50, 학습·평가, checkpoint 재개, 탐지 결과 연동 코드가 있습니다. 클래스 수는 실행 시 `expected_num_classes`로 변경할 수 있습니다. 실제 데이터 구성 함수가 준비되면 `notebooks/train_classifier.ipynb`의 `data_factory_path`를 설정해 학습을 실행합니다.

데이터 구성 함수와 DataLoader가 제공해야 하는 정확한 형식은 [`docs/classifier_input_spec.md`](docs/classifier_input_spec.md)를 참고합니다.

전체 단위 테스트와 코드 검사는 다음 명령으로 실행합니다.

```bash
uv run pytest -q
uv run ruff check classifier common tests
uv run ruff format --check classifier common tests
```

학습 결과와 checkpoint는 `outputs/{run_name}/` 아래에 저장되며 Git에는 포함되지 않습니다. 실험 결과 요약은 `reports/classifier_report.md`에 기록합니다.

## 구현 문서

- [아키텍처 및 리팩터링 문서](docs/architecture/README.md)
- [프로젝트 구조 리팩터링 계획](docs/architecture/project_structure_refactor_plan.md)
- [분류기 구현 계획서](docs/classifier_implementation_plan.md)
- [단일 클래스 탐지기 구현 계획서](docs/detector_implementation_plan.md)
- [단일 클래스 탐지기 입력 명세서](docs/detector_input_spec.md)

## 단일 클래스 알약 탐지기

`detector/`에는 이미지 속 모든 알약을 `pill` 하나의 foreground 클래스로 찾는 SSDLite320 MobileNetV3-Large baseline이 있습니다. 모델 출력 클래스는 background `0`과 pill `1`을 포함한 2개이며, 세부 약품명은 탐지 bbox crop을 별도 분류기에 전달해 결정합니다.

실제 데이터 구성 함수는 아직 저장소에 포함하지 않습니다. 함수가 반환해야 하는 DataLoader, label map과 data summary 형식은 [탐지기 입력 명세서](docs/detector_input_spec.md)를 따릅니다. 전체 설계와 단계별 결정은 [탐지기 구현 계획서](docs/detector_implementation_plan.md)에서 확인할 수 있습니다.

데이터 구성 모듈이 준비되면 [학습 Notebook](notebooks/train_detector.ipynb)의 `data_factory_path`를 다음과 같은 경로로 설정합니다.

```python
data_factory_path = "data.detector_data:build_detector_inputs"
```

Notebook은 입력 검증, 신규 학습 또는 `last.pt` 재개, 학습 곡선, `best.pt` 평가, bbox 시각화와 분류기 연결 batch 확인을 순서대로 실행합니다. 결과는 `outputs/{run_name}/` 아래에 저장하며 Git에는 포함하지 않습니다. 실험 결과는 [탐지기 결과 보고서](reports/detector_report.md)의 `TBD` 항목에 기록합니다.

전체 단위 테스트와 코드 검사는 다음 명령으로 실행합니다.

```bash
uv run python -m pytest -q
uv run ruff check detector tests
uv run ruff format --check detector tests
```

실제 SSDLite의 합성 이미지 forward/backward와 NMS smoke test만 다시 실행하려면 다음 명령을 사용합니다.

```bash
uv run python -m pytest -q tests/test_detector_model.py
```

## 탐지·분류 통합 추론

[통합 추론 Notebook](notebooks/run_end_to_end_inference.ipynb)은 학습된 탐지기와 분류기의 `best.pt`를 불러와 원본 이미지에서 알약 bbox와 세부 품목을 순서대로 예측합니다. 각 모델의 학습은 기존 개별 Notebook에서 수행합니다.

통합 Notebook의 `inference_factory_path`에는 `factory(batch_size=...)` 호출로 추론 DataLoader를 반환하는 함수 경로를 지정합니다. DataLoader batch는 [탐지기 입력 명세서의 추론 형식](docs/detector_input_spec.md#12-추론-dataloader)을 따라야 합니다.

결과는 `outputs/end_to_end_inference/` 아래에 저장됩니다.

- `end_to_end_predictions.json`: bbox, 품목 정보, 탐지·분류·결합 점수
- `end_to_end_errors.json`: 탐지 및 분류 단계별 오류
- `visualizations/`: 품목명과 최종 점수를 표시한 이미지
