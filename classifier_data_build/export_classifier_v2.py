"""분류기 데이터 v2: v1 입력 패키지에 라벨 2차 검수 조치(label_actions.csv)를 적용한다.

v1을 처음부터 다시 만들지 않고 v1 패키지에서 바로 고친다.
    - 다시 고르면(select_images.py) 빈자리를 2차 검수를 안 거친 사진으로 채우고, bbox 번호가 당겨져 sample_id가 바뀐다
    - v1에서 고치면 남는 crop은 v1과 sample_id·crop 파일이 같아서 1:1로 비교할 수 있다

조치 (label_actions.csv의 action)
    이미지 제외, bbox 제외   그 crop을 뺀다 (이미지 제외는 사진의 crop이 모두 목록에 있다)
    라벨 교환 (수정)         code를 new_code로 바꾸고 class_index, category_id, item_seq, name, crop_path를 맞춘다
v1의 나머지 열(split, group_id, near_dup_in_train 등)은 그대로 둔다. class_map.json도 같다 (best.pt와 호환).

입력
    data/classifier/export/training_package_classifier/          v1 패키지 (split_hash가 V1_SPLIT_HASH여야 한다)
    classifier_data_build/label_audit/label_actions.csv          merge_label_audit.py 결과
결과
    data/classifier/export/training_package_classifier_v2/(.zip) v1과 같은 구조
    classifier_data_build/reports/data_v2_report.md              변경 건수, split_hash, 클래스별 개수, 기준 모델 v2 val 점수
    data/classifier/meta/data_v2_eval/                           기준 모델의 v1·v2 val 예측 (--checkpoint가 있을 때)

사용:
    python classifier_data_build/export_classifier_v2.py [--checkpoint data/best.pt] [--no-eval]
"""
import argparse
import json
import os
import shutil
from types import SimpleNamespace

import pandas as pd
import torch

from config import AUDIT, DATA, EXPORT, META, REPORTS
from classifier_data import classifier_inputs as ci

V1 = EXPORT / "training_package_classifier"
V2 = EXPORT / "training_package_classifier_v2"
V1_SPLIT_HASH = "49d2afee86a4ec8f48c25488eac9b1f9081b36066e5533eba40670087f60430c"   # 2차 검수를 한 v1
DROP = {"이미지 제외", "bbox 제외"}
SWAP = "라벨 교환 (수정)"


def apply_actions(v1, act, class_map):
    """v1 samples(문자열 그대로) + 조치 → v2 samples"""
    unknown = set(act.action) - DROP - {SWAP}
    assert not unknown, f"알 수 없는 조치: {unknown}"
    missing = set(act.sample_id) - set(v1.sample_id)
    assert not missing, f"v1에 없는 sample_id {len(missing)}개 (다른 v1 패키지?): {sorted(missing)[:3]}"
    # 이미지 제외는 그 사진의 crop이 모두 목록에 있어야 한다
    img = act[act.action == "이미지 제외"].image_id
    assert set(v1[v1.image_id.isin(img)].sample_id) <= set(act.sample_id), "이미지 제외 사진의 crop이 목록에서 빠졌다"

    v2 = v1[~v1.sample_id.isin(act[act.action.isin(DROP)].sample_id)].copy()
    by_code = {f"K-{c['category_id']:06d}": c for c in class_map["classes"]}
    for r in act[act.action == SWAP].itertuples():
        c = by_code[r.new_code]
        i = v2.index[v2.sample_id == r.sample_id][0]
        assert v2.at[i, "code"] == r.code, f"{r.sample_id}: v1 라벨이 {v2.at[i, 'code']}로 조치 목록({r.code})과 다르다"
        v2.loc[i, ["code", "class_index", "category_id", "item_seq", "name"]] = [
            r.new_code, str(c["class_index"]), str(c["category_id"]), c["item_seq"], c["name"]]
        v2.at[i, "crop_path"] = f"crops/{r.new_code}/{r.sample_id}.png"
    return v2


def write_package(v1, v2):
    if V2.exists():
        shutil.rmtree(V2)
    V2.mkdir(parents=True)
    old_path = v1.set_index("sample_id").crop_path
    for r in v2.itertuples():                    # crop 파일은 v1 것을 그대로 복사 (라벨 교환은 새 코드 폴더로)
        out = V2 / r.crop_path
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V1 / old_path[r.sample_id], out)
    v2.to_csv(V2 / "samples.csv", index=False, encoding="utf-8")
    shutil.copy2(V1 / "class_map.json", V2 / "class_map.json")
    class_map = json.loads((V1 / "class_map.json").read_text(encoding="utf-8"))
    summary = ci.make_data_summary(ci.read_samples(V2), class_map["num_classes"])
    (V2 / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cfg = SimpleNamespace(batch_size=32, seed=42, class_map_path=V2 / "class_map.json", data_summary_path=V2 / "data_summary.json")
    ci.check_inputs(*ci.build_classifier_inputs(config=cfg, device=torch.device("cpu")),
                    expected_num_classes=class_map["num_classes"])
    zip_path = shutil.make_archive(str(EXPORT / V2.name), "zip", V2)
    print(f"입력 검증 완료: train {summary['num_train_samples']}, val {summary['num_val_samples']}")
    print("저장:", V2, zip_path, f"{os.path.getsize(zip_path) / 1e6:.0f}MB")
    return summary


def evaluate(checkpoint, root):
    """기준 모델을 패키지 root의 val로 평가한다. (지표, 예측 표)"""
    from classifier.checkpoint import hash_class_map, load_checkpoint
    from classifier.evaluate import evaluate_with_predictions
    from classifier.model import build_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = SimpleNamespace(batch_size=64, seed=42, class_map_path=root / "class_map.json", data_summary_path=root / "data_summary.json")
    _, val_loader, class_map, _ = ci.build_classifier_inputs(config=cfg, device=dev)
    ckpt = load_checkpoint(checkpoint, torch.device("cpu"))
    assert ckpt["class_map_hash"] == hash_class_map(class_map), "best.pt와 class_map이 다르다"
    model = build_model(class_map["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    art = evaluate_with_predictions(model, val_loader, class_map, torch.nn.CrossEntropyLoss(), dev,
                                    expected_num_classes=class_map["num_classes"])
    return art.metrics, pd.DataFrame([p.to_dict() for p in art.predictions])


def report(v1, v2, act, summary, evals):
    names = v1.drop_duplicates("code").set_index("code").name
    cnt = lambda d: d.groupby(["code", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val"], fill_value=0)
    c1, c2 = cnt(v1), cnt(v2).reindex(cnt(v1).index, fill_value=0)
    v1_hash = ci.split_hash(ci.read_samples(V1))
    lines = [
        "# 분류기 데이터 v2",
        "",
        "`export_classifier_v2.py`가 만든다. v1 입력 패키지에 라벨 2차 검수 조치(`label_audit/label_actions.csv`)를 적용했다.",
        "검수 방법과 판정은 [`label_audit_report.md`](label_audit_report.md), 조치 목록 요약은 [`label_audit_summary.md`](label_audit_summary.md).",
        "",
        "## 요약",
        "",
        "| | v1 | v2 |",
        "| --- | ---: | ---: |",
        f"| crop | {len(v1):,} | {len(v2):,} |",
        f"| train | {(v1.split == 'train').sum():,} | {summary['num_train_samples']:,} |",
        f"| val | {(v1.split == 'val').sum():,} | {summary['num_val_samples']:,} |",
        f"| 그룹 train / val | {v1[v1.split == 'train'].group_id.nunique()} / {v1[v1.split == 'val'].group_id.nunique()} "
        f"| {summary['train_group_count']} / {summary['val_group_count']} |",
        f"| val에 있는 클래스 | {v1[v1.split == 'val'].code.nunique()} | {len(summary['validation_class_indices'])} |",
        f"| split_hash | `{v1_hash[:8]}…` | `{summary['split_hash'][:8]}…` |",
        "",
        f"- v2 split_hash 전체: `{summary['split_hash']}`",
        "- 남는 crop은 v1과 sample_id, crop 파일, split, group_id가 같다. `class_map.json`도 같아서 v1으로 학습한 `best.pt`를 그대로 평가할 수 있다",
        "- `near_dup_in_train`은 v1 값을 그대로 둔다. 짝이 되는 train 사진이 빠졌어도 True로 남을 수 있다 (유사 사진 제외 평가가 조금 보수적이 된다)",
        "",
        "## 적용한 조치",
        "",
        "| 조치 | 사진 | crop | train | val |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for a in ["이미지 제외", "bbox 제외", SWAP]:
        g = act[act.action == a]
        lines.append(f"| {a} | {g.image_id.nunique()} | {len(g)} | {(g.split == 'train').sum()} | {(g.split == 'val').sum()} |")
    sw = act[act.action == SWAP]
    lines += ["", "라벨 교환:", ""] + [f"- `{r.sample_id}` ({r.split}): {r.code} → {r.new_code}" for r in sw.itertuples()]

    if evals:
        (m1, p1), (m2, p2) = evals
        lines += [
            "",
            "## 기준 모델(`resnet50_baseline_v1`, `best.pt`)의 v2 val 점수",
            "",
            "v1으로 학습한 같은 모델을 v1 val과 v2 val로 평가했다. 데이터 v2로 학습한 모델과 비교할 기준은 v2 val 점수다.",
            "",
            "| val | 샘플 | 오분류 | Top-1 | Top-5 | macro F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for n, m, p in (("v1", m1, p1), ("v2", m2, p2)):
            lines.append(f"| {n} | {m.num_samples:,} | {(~p.correct).sum()} | {m.top1_accuracy:.4f} | {m.top5_accuracy:.4f} | {m.macro_f1:.4f} |")
        err1 = p1[~p1.correct].sample_id
        fate = {s: ("v2에서 제외" if s not in set(v2.sample_id) else
                    "라벨 교환 후 정답" if s in set(sw.sample_id) and s in set(p2[p2.correct].sample_id) else
                    "v2에서도 오분류" if s in set(p2[~p2.correct].sample_id) else "v2에서 정답") for s in err1}
        lines += ["", "v1 val 오분류가 v2에서 어떻게 됐나:", ""]
        lines += [f"- `{s}`: {f}" for s, f in fate.items()]
        new = p2[~p2.correct & ~p2.sample_id.isin(err1)]
        lines += [f"- v2에서 새로 생긴 오분류: {len(new)}개"]
        if (~p2.correct).sum() == 0:
            lines += ["", "**주의: 기준 모델이 v2 val을 모두 맞혔다.** v2 val Top-1·macro F1으로는 v2 학습 모델과 차이를 볼 수 없다. "
                      "val loss, 확신도(`classifier_score`), 탐지 bbox를 연결한 평가 등 다른 기준이 필요하다."]

    changed = c1[(c1 != c2).any(axis=1)]
    lines += [
        "",
        "## 클래스별 개수 (바뀐 클래스만)",
        "",
        f"118개 중 {len(changed)}개 클래스의 개수가 바뀌었다. 나머지는 v1과 같다 ([`split_report.md`](split_report.md)).",
        "",
        "| 코드 | 품목 | v1 train | v2 train | v1 val | v2 val |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for code, r in changed.iterrows():
        lines.append(f"| {code} | {names[code]} | {r.train} | {c2.loc[code, 'train']} | {r.val} | {c2.loc[code, 'val']} |")
    lines += ["", f"- v2에서 train이 가장 적은 클래스: {c2.train.idxmin()} ({c2.train.min()}개), "
              f"val이 가장 적은 클래스: {c2.val.idxmin()} ({c2.val.min()}개)"]
    (REPORTS / "data_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서: {REPORTS / 'data_v2_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DATA / "best.pt"), help="v2 val로 다시 평가할 기준 모델")
    ap.add_argument("--no-eval", action="store_true", help="기준 모델 평가를 하지 않는다")
    args = ap.parse_args()

    v1 = pd.read_csv(V1 / "samples.csv", dtype=str, keep_default_na=False)
    assert ci.split_hash(ci.read_samples(V1)) == V1_SPLIT_HASH, "v1 패키지가 2차 검수를 한 v1과 다르다"
    act = pd.read_csv(AUDIT / "label_actions.csv", dtype=str, keep_default_na=False)
    class_map = json.loads((V1 / "class_map.json").read_text(encoding="utf-8"))

    v2 = apply_actions(v1, act, class_map)
    assert not set(v2.sample_id) & set(act[act.action.isin(DROP)].sample_id)
    print(f"v1 {len(v1)} → v2 {len(v2)} (제외 {len(v1) - len(v2)}, 라벨 교환 {(act.action == SWAP).sum()})")
    summary = write_package(v1, v2)

    evals = None
    if not args.no_eval and os.path.exists(args.checkpoint):
        out = META / "data_v2_eval"
        out.mkdir(parents=True, exist_ok=True)
        evals = []
        for name, root in (("v1", V1), ("v2", V2)):
            m, p = evaluate(args.checkpoint, root)
            p.to_csv(out / f"baseline_{name}_val_predictions.csv", index=False, encoding="utf-8-sig")
            (out / f"baseline_{name}_val_metrics.json").write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"기준 모델 {name} val: Top-1 {m.top1_accuracy:.4f}, macro F1 {m.macro_f1:.4f}, 오분류 {(~p.correct).sum()}")
            evals.append((m, p))
    elif not args.no_eval:
        print(f"기준 모델 없음({args.checkpoint}): 평가를 건너뛴다")
    report(v1, v2, act, summary, evals)
