"""위치·클래스 검수 판정을 데이터 v2 조치 목록으로 합친다."""

import argparse
from pathlib import Path

import pandas as pd

from etl.classifier.paths import (
    EXPORT,
    LABEL_AUDIT_RECORDS,
    REPORTS,
    save_table,
)

ACTION_PRIORITY = {
    "이미지 제외": 0,
    "bbox 제외": 1,
    "라벨 교환 (수정)": 2,
}
COMPLETED_REVIEW_STATES = {"사람 확인 완료", "위치 검수에서 확인"}
ACTION_COLUMNS = (
    "sample_id",
    "image_id",
    "code",
    "name",
    "new_code",
    "action",
    "split",
    "source",
    "출처",
    "메모",
)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _review_errors(
    frame: pd.DataFrame,
    *,
    audit_source: str,
    class_review: bool,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "image_id",
        "code",
        "판정",
        "조치",
        "메모",
        "검수",
    }
    if class_review:
        required.add("suspect_code")
    _require_columns(frame, required, name=audit_source)

    incomplete = ~frame["검수"].isin(COMPLETED_REVIEW_STATES)
    if incomplete.any():
        sample_ids = frame.loc[incomplete, "sample_id"].head(3).tolist()
        raise ValueError(f"{audit_source} contains incomplete reviews: {sample_ids}")

    errors = frame[frame["판정"] == "오류"].copy()
    invalid_actions = set(errors["조치"]) - set(ACTION_PRIORITY)
    if invalid_actions:
        raise ValueError(f"{audit_source} contains invalid actions: {invalid_actions}")
    errors["audit_source"] = audit_source
    errors["new_code"] = ""
    if class_review:
        swap = errors["조치"] == "라벨 교환 (수정)"
        errors.loc[swap, "new_code"] = errors.loc[swap, "suspect_code"]
    return errors


def build_label_actions(
    samples: pd.DataFrame,
    position_candidates: pd.DataFrame,
    class_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """완료된 사람 판정을 샘플별 최종 조치로 변환한다."""

    _require_columns(
        samples,
        {"sample_id", "image_id", "code", "name", "split", "source"},
        name="samples",
    )
    if not samples["sample_id"].is_unique:
        raise ValueError("samples.sample_id values must be unique")

    position_errors = _review_errors(
        position_candidates,
        audit_source="위치 검수",
        class_review=False,
    )
    class_errors = _review_errors(
        class_candidates,
        audit_source="클래스 검수",
        class_review=True,
    )
    errors = pd.concat([position_errors, class_errors], ignore_index=True)
    samples_by_id = samples.set_index("sample_id", drop=False)

    rows: list[dict[str, str]] = []
    for error in errors.to_dict("records"):
        action = error["조치"]
        if action == "이미지 제외":
            affected = samples[samples["image_id"] == error["image_id"]]
            if affected.empty:
                raise ValueError(
                    f"review references unknown image_id: {error['image_id']}"
                )
            for sample in affected.to_dict("records"):
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "image_id": sample["image_id"],
                        "code": sample["code"],
                        "new_code": "",
                        "action": action,
                        "audit_source": error["audit_source"],
                        "note": f"{error['code']} {error['메모']}".strip(),
                    }
                )
            continue

        sample_id = error["sample_id"]
        if sample_id not in samples_by_id.index:
            raise ValueError(f"review references unknown sample_id: {sample_id}")
        sample = samples_by_id.loc[sample_id]
        if sample["image_id"] != error["image_id"] or sample["code"] != error["code"]:
            raise ValueError(f"review metadata differs from samples: {sample_id}")
        if action == "라벨 교환 (수정)" and not error["new_code"]:
            raise ValueError(f"label swap has no target code: {sample_id}")
        rows.append(
            {
                "sample_id": sample_id,
                "image_id": sample["image_id"],
                "code": sample["code"],
                "new_code": error["new_code"],
                "action": action,
                "audit_source": error["audit_source"],
                "note": error["메모"],
            }
        )

    output_rows: list[dict[str, str]] = []
    action_rows = pd.DataFrame(rows)
    if action_rows.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    action_rows["priority"] = action_rows["action"].map(ACTION_PRIORITY)
    for sample_id, group in action_rows.groupby("sample_id", sort=True):
        ordered = group.sort_values("priority")
        selected = ordered.iloc[0]
        sample = samples_by_id.loc[sample_id]
        action = selected["action"]
        output_rows.append(
            {
                "sample_id": sample_id,
                "image_id": sample["image_id"],
                "code": sample["code"],
                "name": sample["name"],
                "new_code": selected["new_code"]
                if action == "라벨 교환 (수정)"
                else "",
                "action": action,
                "split": sample["split"],
                "source": sample["source"],
                "출처": ", ".join(sorted(set(ordered["audit_source"]))),
                "메모": " / ".join(dict.fromkeys(ordered["note"])),
            }
        )
    return pd.DataFrame(output_rows, columns=ACTION_COLUMNS)


def build_summary(samples: pd.DataFrame, actions: pd.DataFrame) -> str:
    """최종 조치와 조치 후 클래스별 개수를 Markdown으로 요약한다."""

    excluded = actions[actions["action"] != "라벨 교환 (수정)"]
    remaining = samples[~samples["sample_id"].isin(excluded["sample_id"])]
    before = samples.groupby(["code", "split"]).size().unstack(fill_value=0)
    after = (
        remaining.groupby(["code", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(before.index, fill_value=0)
    )
    lost = before - after
    lost = lost[(lost > 0).any(axis=1)]
    if "train" in lost:
        lost = lost.sort_values("train", ascending=False)

    lines = [
        "# 라벨 2차 검수 적용 요약",
        "",
        "완료된 위치·클래스 사람 검수에서 `오류`로 판정한 항목만 집계한다.",
        "",
        "## 조치별 건수",
        "",
        "| 조치 | 사진 | crop | train | val |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action in ACTION_PRIORITY:
        group = actions[actions["action"] == action]
        lines.append(
            f"| {action} | {group['image_id'].nunique()} | {len(group)} | "
            f"{(group['split'] == 'train').sum()} | {(group['split'] == 'val').sum()} |"
        )
    lines.extend(
        [
            "",
            f"- 전체 crop {len(samples):,}개 → 남는 crop {len(remaining):,}개",
            f"- 라벨 교환 {(actions['action'] == '라벨 교환 (수정)').sum()}개",
            "",
            "## crop이 줄어드는 클래스",
            "",
            "| 코드 | 품목 | train 감소 | val 감소 | 남는 train | 남는 val |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    names = samples.drop_duplicates("code").set_index("code")["name"]
    for code, row in lost.iterrows():
        lines.append(
            f"| {code} | {names[code]} | {row.get('train', 0)} | "
            f"{row.get('val', 0)} | {after.loc[code].get('train', 0)} | "
            f"{after.loc[code].get('val', 0)} |"
        )
    return "\n".join(lines) + "\n"


def merge_review_actions(
    *,
    package_dir: Path,
    records_dir: Path = LABEL_AUDIT_RECORDS,
    report_path: Path | None = None,
) -> pd.DataFrame:
    """저장된 사람 판정을 읽어 label_actions.csv와 요약을 갱신한다."""

    package_dir = Path(package_dir)
    records_dir = Path(records_dir)
    samples = pd.read_csv(
        package_dir / "samples.csv",
        dtype=str,
        keep_default_na=False,
    )
    position_candidates = pd.read_csv(
        records_dir / "position_suspects.csv",
        dtype=str,
        keep_default_na=False,
    )
    class_candidates = pd.read_csv(
        records_dir / "class_suspects.csv",
        dtype=str,
        keep_default_na=False,
    )
    actions = build_label_actions(samples, position_candidates, class_candidates)
    records_dir.mkdir(parents=True, exist_ok=True)
    save_table(actions, records_dir / "label_actions.csv")

    destination = report_path or REPORTS / "label_audit_summary.md"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_summary(samples, actions), encoding="utf-8")
    return actions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=EXPORT / "training_package_classifier",
    )
    parser.add_argument("--records-dir", type=Path, default=LABEL_AUDIT_RECORDS)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS / "label_audit_summary.md",
    )
    args = parser.parse_args(argv)
    actions = merge_review_actions(
        package_dir=args.package_dir,
        records_dir=args.records_dir,
        report_path=args.report,
    )
    print(f"조치 crop {len(actions)}개 (사진 {actions['image_id'].nunique()}장)")


if __name__ == "__main__":
    main()
