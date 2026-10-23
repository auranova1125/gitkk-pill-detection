# 알약 이미지 Object Detection

팀명: 깃크크크크(4팀)

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
