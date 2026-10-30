"""라벨 2차 검수 ③: 위치 검수와 클래스 검수의 판정을 합쳐 데이터 v2에 적용할 목록을 만든다.

입력 (사람 검수가 끝난 CSV)
    data/classifier/meta/label_audit/position_suspects.csv   audit_bbox_position.py 결과 + 판정
    data/classifier/meta/label_audit/class_suspects.csv      audit_class_label.py 결과 + 판정

조치 (판정이 "오류"인 줄의 조치 칸)
    이미지 제외      그 사진의 crop을 모두 뺀다 (AI Hub 조합 사진. 대신 쓸 사진이 많다)
    bbox 제외        그 crop만 뺀다 (Kaggle. 대신할 사진이 없다)
    라벨 교환 (수정)  같은 사진에서 서로 바뀐 두 crop의 라벨을 맞바꾼다 (class_suspects의 suspect_code가 새 라벨)
    한 crop에 조치가 여러 개면 이미지 제외 > bbox 제외 > 라벨 교환 순으로 하나만 남긴다

결과
    data/classifier/meta/label_audit/label_actions.csv   crop마다 한 줄 (sample_id, image_id, code, new_code, action, 출처, 메모)
    classifier_data_build/reports/label_audit_summary.md  건수 요약 (자동 생성)

사용:
    python classifier_data_build/merge_label_audit.py
"""
import pandas as pd

from config import EXPORT, META, REPORTS, save_table

OUT = META / "label_audit"
RANK = {"이미지 제외": 0, "bbox 제외": 1, "라벨 교환 (수정)": 2}


def load(name, source):
    d = pd.read_csv(OUT / name)
    assert (d.검수.isin(["사람 확인 완료", "위치 검수에서 확인"])).all(), f"{name}: 검수가 끝나지 않은 줄이 있다"
    d = d[d.판정 == "오류"].copy()
    assert d.조치.isin(RANK).all(), f"{name}: 알 수 없는 조치 {set(d.조치) - set(RANK)}"
    d["출처"] = source
    return d


samples = pd.read_csv(EXPORT / "training_package_classifier" / "samples.csv")
pos = load("position_suspects.csv", "위치 검수")
cls = load("class_suspects.csv", "클래스 검수")
cls["new_code"] = cls.suspect_code.where(cls.조치 == "라벨 교환 (수정)")
errors = pd.concat([pos, cls], ignore_index=True)

rows = []
for e in errors.itertuples():
    if e.조치 == "이미지 제외":           # 사진의 crop 전부
        for s in samples[samples.image_id == e.image_id].itertuples():
            rows.append((s.sample_id, s.image_id, s.code, None, e.조치, e.출처, f"{e.code} {e.메모}"))
    else:
        rows.append((e.sample_id, e.image_id, e.code, getattr(e, "new_code", None), e.조치, e.출처, e.메모))
act = pd.DataFrame(rows, columns=["sample_id", "image_id", "code", "new_code", "action", "출처", "메모"])
# 한 crop에 조치가 여러 개면 가장 강한 조치 하나. 출처·메모는 합친다
act["rank"] = act.action.map(RANK)
act = (act.sort_values(["sample_id", "rank"])
       .groupby("sample_id", as_index=False)
       .agg(image_id=("image_id", "first"), code=("code", "first"), new_code=("new_code", "first"),
            action=("action", "first"), 출처=("출처", lambda x: ", ".join(sorted(set(x)))),
            메모=("메모", lambda x: " / ".join(dict.fromkeys(x)))))
act.loc[act.action != "라벨 교환 (수정)", "new_code"] = None
act = act.merge(samples[["sample_id", "split", "source", "name"]], on="sample_id")
act = act[["sample_id", "image_id", "code", "name", "new_code", "action", "split", "source", "출처", "메모"]]
save_table(act, OUT / "label_actions.csv")

# ── 요약 ──
drop = act[act.action != "라벨 교환 (수정)"]
kept = samples[~samples.sample_id.isin(drop.sample_id)]
before = samples.groupby(["code", "split"]).size().unstack(fill_value=0)
after = kept.groupby(["code", "split"]).size().unstack(fill_value=0).reindex(before.index, fill_value=0)
lost = (before - after)
lost = lost[(lost > 0).any(axis=1)].sort_values("train", ascending=False)

lines = [
    "# 라벨 2차 검수 적용 요약",
    "",
    "`merge_label_audit.py`가 만든다. 사람 검수가 끝난 `position_suspects.csv`, `class_suspects.csv`의 \"오류\" 판정만 쓴다.",
    "",
    "## 조치별 건수",
    "",
    "| 조치 | 사진 | crop | train | val |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for a in RANK:
    g = act[act.action == a]
    lines.append(f"| {a} | {g.image_id.nunique()} | {len(g)} | {(g.split == 'train').sum()} | {(g.split == 'val').sum()} |")
lines += [
    "",
    f"- 전체 crop {len(samples):,}개 → 남는 crop {len(kept):,}개 (train {(kept.split == 'train').sum():,}, val {(kept.split == 'val').sum():,})",
    f"- 라벨 교환 {(act.action == '라벨 교환 (수정)').sum()}개는 남기고 라벨만 바꾼다",
    f"- 제외 후 val에 없는 클래스: {', '.join(after.index[after.get('val', 0) == 0]) or '없음'}",
    f"- 제외 후 train이 가장 적은 클래스: {after.train.idxmin()} ({after.train.min()}개)",
    "",
    "## 출처별 (한 crop이 두 검수에 모두 걸리면 둘 다 표시)",
    "",
    "| 출처 | crop |",
    "| --- | ---: |",
]
lines += [f"| {k} | {v} |" for k, v in act.출처.value_counts().items()]
lines += [
    "",
    "## crop이 줄어드는 클래스",
    "",
    "| 코드 | 품목 | train 감소 | val 감소 | 남는 train | 남는 val |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
names = samples.drop_duplicates("code").set_index("code").name
for code, r in lost.iterrows():
    lines.append(f"| {code} | {names[code]} | {r.get('train', 0)} | {r.get('val', 0)} | {after.loc[code, 'train']} | {after.loc[code].get('val', 0)} |")
lines += [
    "",
    "## 제외하는 사진",
    "",
    "탐지기 데이터에도 같은 사진이 있으면 함께 확인한다.",
    "",
    "| 사진 | 출처 | split | 조치 |",
    "| --- | --- | --- | --- |",
]
for img, g in act.groupby("image_id"):
    lines.append(f"| `{img}` | {g.source.iloc[0]} | {g.split.iloc[0]} | {', '.join(sorted(set(g.action)))} |")
(REPORTS / "label_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"조치 crop {len(act)}개 (사진 {act.image_id.nunique()}장)")
print(act.groupby(["action", "split"]).size().to_string())
print(f"남는 crop {len(kept)} (train {(kept.split == 'train').sum()}, val {(kept.split == 'val').sum()})")
print(f"요약: {REPORTS / 'label_audit_summary.md'}")
