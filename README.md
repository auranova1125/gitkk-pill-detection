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

## 실행 진입점

Notebook은 데이터와 예측을 직접 확인하는 주 실행 환경으로 유지합니다. 반복 실행이나 자동화가 필요할 때는 같은 Python 학습·평가 함수를 호출하는 `main.py`를 사용합니다.

```bash
# 설치와 장치, 공통 전처리, 분류기·탐지기 학습 1 step 확인
uv run python main.py smoke

# YAML 설정을 사용한 학습
uv run python main.py train-classifier --config configs/classifier.yaml
uv run python main.py train-detector --config configs/detector.yaml

# checkpoint 평가와 통합 추론
uv run python main.py evaluate classifier \
  --config configs/classifier.yaml \
  --checkpoint outputs/resnet50_baseline/best.pt
uv run python main.py recognize-pills --config configs/pill_recognition.yaml
```

`configs/*.yaml`은 `configs/classifier.py`, `configs/detector.py`, `configs/pill_recognition.py`의 설정 객체로 변환되므로 Notebook과 CLI의 값 검증 기준이 같습니다. CLI는 YAML 값을 사용하고, Notebook에서는 같은 설정 객체를 직접 생성해 셀에서 실험값을 바꿀 수 있습니다. 상대 경로는 프로젝트 루트에서 명령을 실행하는 것을 기준으로 합니다.

기본 YAML은 저장소가 제공하는 `pilldata` factory를 사용합니다. 데이터가 저장소 밖에 있으면 탐지기는 `data_settings_path`, 분류기는 `class_map_path`와 `data_summary_path`, 통합 추론은 `input_dir`을 실제 경로에 맞게 바꿉니다.

## 전체 품질 검사

프로젝트 루트에서 전체 테스트와 정적 검사를 실행합니다.

```bash
uv run python -m pytest -q
uv run ruff check main.py classifier configs detector etl pilldata pipeline utils tests
uv run ruff format --check main.py classifier configs detector etl pilldata pipeline utils tests
```

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

## 프로젝트 구조

| 경로 | 책임 |
| --- | --- |
| `classifier/` | 품목 분류 모델, 학습·평가·추론, checkpoint |
| `detector/` | 단일 `pill` 클래스 탐지 모델, 학습·평가·추론, checkpoint |
| `pilldata/` | 학습·추론 중 사용하는 Dataset, DataLoader, transform |
| `etl/` | 원천 데이터에서 학습 패키지를 만드는 오프라인 ETL |
| `pipeline/` | 탐지 bbox를 분류기에 연결하는 알약 인식 |
| `configs/` | 검증된 실행 설정과 기본 YAML |
| `utils/` | 장치, RNG, artifact 저장 등 저수준 공통 기능 |

설계 기준과 전체 tree는 [프로젝트 구조](docs/architecture/project_structure.md)에서 확인합니다.

## 데이터 흐름

```text
원천 데이터 → etl/ → data/ 학습 패키지 → pilldata/ → classifier/ 또는 detector/
                                                     └→ pipeline/ 알약 인식
```

분류 학습 패키지 ETL은 다음 명령으로 실행합니다. 기존 출력을 교체하려면 `--overwrite`를 명시해야 합니다.

```bash
uv run python -m etl.classifier.build_classifier_dataset
```

탐지 학습 패키지는 다음 명령으로 라벨·분할·경로를 검사합니다.

```bash
uv run python main.py check-detector-data
```

## Notebook

### Baseline

- [분류기 학습](notebooks/baseline/train_classifier.ipynb)
- [탐지기 학습](notebooks/baseline/train_detector.ipynb)
- [탐·분 통합 알약 인식](notebooks/baseline/run_pill_recognition.ipynb)

### 실험·EDA

- [분류기 baseline 분석](notebooks/experiments/classifier_baseline_analysis.ipynb)
- [분류 데이터 품질 확인](notebooks/eda/classifier_data_quality.ipynb)

Notebook은 설정과 시각화를 담당하며, 학습·평가 핵심 로직은 Python 모듈을 호출합니다. 자동화나 재현이 필요한 경우 `main.py`의 `train-classifier`, `train-detector`, `evaluate`, `recognize-pills`, `smoke` 명령을 사용합니다.

## 검증

```bash
# 전체 테스트
uv run pytest -q

# 합성 데이터 소형 smoke
uv run pytest -q -m smoke

# 코드 스타일
uv run ruff check classifier detector pipeline pilldata configs utils etl tests main.py
uv run ruff format --check classifier detector pipeline pilldata configs utils etl tests main.py
```

학습 결과와 checkpoint는 `outputs/<run_name>/`에 저장하며 Git에 포함하지 않습니다. 분류 클래스 수의 기본값은 118이지만 `expected_num_classes`로 변경할 수 있으며, 검증 로직이 118을 절대값으로 강제하지 않습니다.

## 문서

루트 `README.md`를 유일한 README로 사용합니다.

- 구조: [프로젝트 구조](docs/architecture/project_structure.md)
- 데이터: [분류 데이터](docs/data/classifier_data_guide.md), [분류 ETL](docs/data/classifier_etl.md), [탐지 데이터](docs/data/detector_data_guide.md)
- 입력 명세: [분류기](docs/specs/classifier_input_spec.md), [탐지기](docs/specs/detector_input_spec.md)
- 초기 계획·결정: [분류기](docs/plans/classifier_implementation_plan.md), [탐지기](docs/plans/detector_implementation_plan.md)
- 결과: [분류기](reports/classifier_report.md), [탐지기](reports/detector_report.md)
