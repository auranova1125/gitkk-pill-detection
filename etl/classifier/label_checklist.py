"""분류 데이터 라벨 검증 항목. validate_source_labels.py가 사용한다."""

# 체크리스트 순서: (항목, 구분, 결과 성격)
CHECKS = [
    ("bbox 면적 불일치", "Annotation", "오류"),
    ("bbox 범위 초과", "Annotation", "오류"),
    ("bbox 음수·0 크기", "Annotation", "오류"),
    ("Annotation ID 중복", "Annotation", "오류"),
    ("동일 bbox 반복", "Annotation", "의심"),
    ("bbox 과도한 중첩", "Annotation", "의심"),
    ("ignore·iscrowd 이상", "Annotation", "의심"),
    ("이미지 ID 중복", "Image", "오류"),
    ("PNG 누락·손상", "Image", "오류"),
    ("PNG 크기 불일치", "Image", "오류"),
    ("중복·유사 이미지", "Image", "의심"),
    ("코드 매핑 불일치", "교차 검증", "의심"),
    ("파일명 ↔ 라벨 구성 불일치", "교차 검증", "의심"),
    ("같은 클래스 제품 정보 상충", "교차 검증", "의심"),
    ("같은 조합의 각도별 알약 수 다름", "교차 검증", "의심"),
    ("bbox 크기 ↔ 실제 알약 크기 불일치", "교차 검증", "의심"),
    ("클래스별 bbox 크기·종횡비 이상치", "EDA", "편향"),
    ("클래스별 bbox 위치 편향", "EDA", "편향"),
]
