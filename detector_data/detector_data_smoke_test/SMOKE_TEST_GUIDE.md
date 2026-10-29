# 공유 표본 검사

1. Drive의 기존 `smoke_test.zip`을 받아 저장소의 `datasets/smoke_test/`에 압축을 풉니다.
2. 이 폴더의 `smok_test.ipynb`을 열고 저장소의 학습 환경을 커널로 선택합니다.
3. 위에서부터 실행하면 파일 확인 → 모델 연결 검사 → 보고서 확인 순서로 진행됩니다.

모델 검사는 Train 2장으로 가중치 갱신 1회, Validation 2장으로 예측 1회를 수행합니다.
전체 학습과 가중치 파일 저장은 수행하지 않습니다.
보고서는 `outputs/detector_data/shared_smoke_report.json`에 저장됩니다.
팀원은 ZIP을 생성하거나 다시 업로드하지 않습니다.

데이터 전체 배치와 코드 설명은 상위 폴더의 `DETECTOR_DATA_GUIDE.md`를 보세요.
