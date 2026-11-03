# 탐지기 데이터셋 v2 구성 및 검수 보고서

작성일: 2026-11-03
상태: v2 생성·검증 및 Drive 데스크톱 폴더 복사·내용 검증 완료. 서버 동기화 상태는 Drive 앱에서 확인 필요.

## 목적과 처리 원칙

분류기 검수에서 오류가 확인된 사진과 추가 bbox 검수에서 오류로 판정한 사진을 탐지기 v1에서 제외했다.
사진 속 알약 하나만 잘못 라벨링되어 있어도 사진과 연결된 모든 bbox를 제외했다. bbox 수정이나 재분할은 하지 않았다.

## 기준 데이터와 결과

기준은 Google Drive `gitkk_archive/detector_data/1. selected_data/v1/`이다.
Colab의 v2 폴더 위치는 유지했다. 최종 게시 전 최신 main의 계약에 맞춰 JSON·manifest를 `train/images/...`, `val/images/...` 상대 경로로 정규화하고 출처는 manifest의 dataset 열로 관리한다. 경로 설정에는 package_dir와 image_roots만 남겼고, 작업 상태 기록은 학습 summary에서 검수 요약으로 분리했다. 로컬 검증 시에만 메모리에서 루트 경로를 대체한다.

| 분할 | v1 이미지 | 기존 목록 제외 후 | 최종 v2 이미지 | v1 bbox | 최종 v2 bbox |
|---|---:|---:|---:|---:|---:|
| train | 8,256 | 8,223 | 8,143 | 31,673 | 31,238 |
| validation | 2,063 | 2,048 | 2,026 | 7,915 | 7,771 |
| 합계 | 10,319 | 10,271 | 10,169 | 39,588 | 39,009 |

| 제외 단계 | train 이미지 | val 이미지 | 전체 이미지 | 전체 bbox |
|---|---:|---:|---:|---:|
| 분류기에서 전달받은 기존 오류 | 33 | 15 | 48 | 184 |
| 추가 bbox 육안 검수 | 80 | 22 | 102 | 395 |
| 합계 | 113 | 37 | 150 | 579 |

기존 목록의 대회 이미지 3장도 수정 없이 전체 제외했다. 제외 후에도 약 코드 118종이 유지된다.

## 추가 검사 대상과 자동 검사

이미 검수한 범위는 분류기 `3. split_metadata/v1/samples.csv`의 원본 이미지 식별자로 확인했다.
탐지 파일명에서 `codeit_`, `source_N_` 접두어를 제거하여 대조했다. 기존 제외 목록은 `file_name`으로 연결했다.

- 검사 대상: AI Hub 이미지 6,302장, bbox 24,694개
- train: 5,012장, bbox 19,633개
- validation: 1,290장, bbox 5,061개
- 가이드의 6,307장·24,714개와 달리 현재 Drive v1과 중간 v2에서 모두 위 수량이 확인됐다. 차이의 원인은 확정하지 않았다.
- 자동 검사 결과: 의심 bbox 330개, 중복 제거 사진 263장, 검수 그림 28개

기존 `etl/classifier/label_audit/detect_bbox_errors.py`의 점수·등급·그림 생성 함수를 재사용했다.
로컬 이미지 경로를 선택적으로 전달할 수 있게 확장하여 원본 ZIP을 다시 읽지 않고 검사했다.

## 사용자 최종 판정

검수 그림을 등급별로 제시하고 각 판정을 사용자에게 확인받아 CSV에 기록했다.

| 등급 | 의심 bbox | 오류 bbox | 정상 bbox | 오류 포함 이미지 |
|---|---:|---:|---:|---:|
| 1 | 138 | 97 | 41 | 87 |
| 2 | 62 | 1 | 61 | 1 |
| 3 | 130 | 16 | 114 | 15 |
| 합계 | 330 | 114 | 216 | 중복 제거 102 |

등급 간 동일 사진이 중복되므로 이미지 수를 단순 합산하지 않았다.
오류 사진 102장과 연결된 bbox 395개 전체를 제외했다. 정상 판정을 받은 bbox가 있어도 같은 사진에 오류 bbox가 있으면 사진 전체를 제외했다.

판정 기준:

- 오류: 위치가 밀려 배경·다른 알약을 가리킴, 대상 알약을 상당 부분 잘라냄, 선처럼 좁거나 지나치게 큰 bbox
- 정상: 대상 알약을 적절히 포함함. 작은 경계 차이, bbox끼리의 겹침, 자동 분홍 영역의 그림자·반사 오검출만으로 제외하지 않음

이 결과는 자동으로 선별된 후보에 대한 육안 검수다. 6,302장 전체를 육안 전수 검수했다는 의미가 아니며, 자동 후보에 포함되지 않은 오류가 남을 수 있다.

## 파일과 재현 방법

| 파일 | 역할 |
|---|---|
| `etl/detector/build_detector_v2.py` | 기존 48장 제외 생성 및 확정 판정 반영 |
| `etl/detector/detect_bbox_errors.py` | 미검수 입력표 생성·자동 검사·그림 생성 |
| `etl/detector/label_audit/detector_actions.csv` | 분류기에서 전달받은 기존 48장 목록 |
| `etl/detector/label_audit/position_suspects.csv` | 의심 bbox 330건의 사람 판정 |
| `etl/detector/label_audit/excluded_images.csv` | 최종 제외 이미지 150장과 사유 |
| `datasets/select/v2/` | 최종 데이터셋; Git 추적 제외 |
| `datasets/select/position_audit/` | 입력·점수·검수 그림 및 1단계 백업; Git 추적 제외 |

새 작업공간에서 실행한다. 원본·분류기 목록 경로는 실행 환경에 맞게 지정한다. Drive v1의 구형 JSON 경로는 원본을 변경하지 않고 임시 메타데이터로 변환하여 검사한다.

```bash
# 1. 기존 48장 제외. output/excluded/report는 기존 파일을 덮어쓰지 않는다.
.venv/bin/python -m etl.detector.build_detector_v2 \
  --source-dir "/path/to/detector/v1"

# 2. 입력표와 후보·그림 생성. 기존 CSV의 사람 판정은 sample_id로 보존한다.
.venv/bin/python -m etl.detector.detect_bbox_errors \
  --classifier-samples "/path/to/classifier/v1/samples.csv"

# 3. position_suspects.csv의 사용자 판정을 확인한 뒤 반영한다.
.venv/bin/python -m etl.detector.build_detector_v2 --apply-reviews
```

중단된 1단계 복사는 `--resume`으로 재개할 수 있다. 최종 반영은 검사 입력의 split_hash를 대조하고, 이미 반영한 패키지에 다시 적용하는 것을 거부한다.
반영 전 메타데이터와 추가 제외 이미지는 `datasets/select/position_audit/stage1_backup/`에 보존했다.

## 검증 결과

- JSON·split_manifest·실제 PNG 목록 일치, 제외 사진 150장의 최종 폴더 부재 확인
- 남은 이미지의 split·image ID·annotation ID·bbox·약 코드 동일성 확인; 최신 main 경로 정규화 전후 로더 기록 동일성 확인
- 전체 로더 검증: 고유 ID, bbox 좌표·면적, annotation 개수, 빈 이미지, train/val 조합 중복 검사 통과
- 약 코드 118종 유지, Colab 폴더 위치와 label map 보존; 최신 main의 strict 패키지 로더 검증 통과
- 1단계 이미지 복사 시 원본과 내용 비교 수행
- 최신 main 기준 관련 테스트 37개·subtest 17개와 메타데이터 호환 테스트 2개 통과 (총 39개); 변경 코드 정적 검사 통과

## split_hash

- v1: `bc02aabd8d0d8d34f243568a5c6286feb2d73126aae7efe7f7ca64a022a00bed`
- 기존 48장 제외 후: `44d9e05df134af169168a5d92470ff54b6fbfe8e1b14fd45eb0e275a2988ee6b`
- 구형 경로 계약에서의 최종 v2: `1005a210f54830390892f6e9588e43244edd15d6c99c7aa0bcb58b24082822bb`
- 최신 main 계약으로 정규화한 최종 v2: `ac6618cdf2fd69352b0d3848b69064eadbc6527d6153795c3cc2de1c2dfa7d32`

경로도 hash 입력에 포함되므로 이미지·판정·분할·bbox가 같아도 경로 정규화로 hash가 달라진다. 변경 전 메타데이터는 `datasets/select/position_audit/pre_main_metadata/`에 보존했다.

## 배포 구조

최종 데이터셋은 v1과 동일한 폴더 구조다. metadata 내부 계약은 최신 main에 맞췄다.

```text
v2/
├── annotations/
│   ├── train.json
│   └── validation.json
├── split_manifest.csv
├── data_settings.json
├── detector_label_map.json
├── detector_data_summary.json
├── train/images/
└── val/images/
```

Drive 데스크톱 폴더의 파일 10,186개에 대해 원본과 SHA-256 일치를 확인했다. 서버 동기화 완료 여부는 이 검증과 별개이며 Drive 앱에서 확인해야 한다.

- 전체 데이터셋: `gitkk_archive/detector_data/1. selected_data/v2/`
- 메타데이터 사본: `gitkk_archive/detector_data/3. split_metadata/v2/`
- 조치·판정·제외 CSV 및 보고서: `gitkk_archive/detector_data/2. excluded_data/v2/`
- 로컬 복사 검증 manifest: `datasets/select/position_audit/output/drive_copy_manifest.json`
- 노션은 게스트 접근 제약으로 직접 게시하지 않고, 사용자가 붙여넣을 문서를 제공한다.
