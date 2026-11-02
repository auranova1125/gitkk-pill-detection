"""자동 후보를 다시 만들 때 기존 사람 판정을 보존한다."""

from pathlib import Path

import pandas as pd

REVIEW_COLUMNS = ("판정", "조치", "메모", "검수")


def preserve_human_reviews(
    candidates: pd.DataFrame,
    review_path: Path,
) -> pd.DataFrame:
    """새 후보에 기존 판정을 연결하고 판정된 이전 후보도 보존한다."""

    if "sample_id" not in candidates:
        raise ValueError("candidates must contain sample_id")
    if not candidates["sample_id"].is_unique:
        raise ValueError("candidate sample_id values must be unique")

    result = candidates.copy()
    for column in REVIEW_COLUMNS:
        result[column] = ""
    review_path = Path(review_path)
    if not review_path.exists():
        return result

    previous = pd.read_csv(
        review_path,
        dtype=str,
        keep_default_na=False,
    )
    if "sample_id" not in previous:
        raise ValueError("existing review file must contain sample_id")
    if not previous["sample_id"].is_unique:
        raise ValueError("existing review sample_id values must be unique")
    previous = previous.set_index("sample_id")

    matched = result["sample_id"].isin(previous.index)
    for column in REVIEW_COLUMNS:
        if column in previous:
            result.loc[matched, column] = result.loc[matched, "sample_id"].map(
                previous[column]
            )

    reviewed = previous.get("판정", pd.Series("", index=previous.index)).ne("")
    missing_ids = previous.index[reviewed & ~previous.index.isin(result["sample_id"])]
    if len(missing_ids) == 0:
        return result

    missing = previous.loc[missing_ids].reset_index()
    for column in ("그림번호", "그림파일"):
        if column in missing:
            missing[column] = ""
    for column in result.columns:
        if column not in missing:
            missing[column] = ""
    return pd.concat([result, missing[result.columns]], ignore_index=True)
