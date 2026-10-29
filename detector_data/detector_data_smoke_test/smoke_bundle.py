"""이전 check/run 명령을 새 검사 코드에 연결한다. ZIP 생성 기능은 제거했다."""
import sys
from detector_data.__main__ import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {"check", "run"}:
        raise SystemExit("기존 smoke_test.zip을 사용하세요. 실행: python -m detector_data smoke")
    command = "smoke-check" if arguments[0] == "check" else "smoke"
    raise SystemExit(main([command, *arguments[1:]]))
