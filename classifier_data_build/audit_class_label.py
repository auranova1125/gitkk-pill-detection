"""라벨 2차 검수 ②: bbox 위치는 맞는데 클래스(이름표)가 틀린 샘플을 찾는다.

audit_bbox_position.py는 사진 픽셀만 보므로, 두 알약의 이름표가 서로 바뀐 경우처럼
위치가 맞는 클래스 오류는 잡지 못한다. 여기서는 기준 모델(best.pt)의 embedding으로 생김새를 비교한다.

방법
    1. best.pt로 모든 crop의 embedding(2048차원, avgpool 출력)과 예측 확률을 구한다
       전처리는 학습의 val과 같다 (정사각형 padding → 224 resize → normalize)
    2. 같은 그룹 비교: 같은 조합 그룹은 같은 실물 알약을 각도만 바꿔 찍었다.
       각 crop을 같은 그룹, 다른 사진의 crop들과 비교해서
       같은 클래스보다 다른 클래스와 더 닮았으면 의심 (group_gap > 0)
    3. 교차 검증 예측: 그룹 단위 5-fold로 embedding 위에 로지스틱 회귀를 학습해
       학습에 안 쓴 crop의 클래스를 예측한다 (oof_pred, oof_label_prob)
       best.pt는 train 라벨을 이미 학습했으므로 train의 틀린 라벨도 외웠을 수 있다
    4. 서로 바뀜: 같은 사진의 두 crop이 서로 상대 클래스로 판단되면 swap

의심 등급
    1  강함   swap, 또는 같은 그룹 비교와 교차 검증이 둘 다 다른 클래스를 가리킴
    2  약함   둘 중 하나만 다른 클래스를 가리킴

결과
    data/classifier/meta/label_audit/class_embeddings.npz   embedding, best.pt 확률 (다시 계산하지 않도록 저장)
    data/classifier/meta/label_audit/class_scores.csv       crop마다 점수 (전체)
    classifier_data_build/label_audit/class_suspects.csv    의심 crop과 검수 칸(판정, 조치, 메모, 검수). git에 올린다
        이미 사람이 채운 파일이 있으면 sample_id로 판정을 이어받는다 (config.keep_review)
    data/classifier/meta/label_audit/review/c<등급>_review_<번호>.jpg  검수용 그림
        한 줄 = 의심 crop | 같은 그룹 다른 사진의 라벨 클래스 crop | 같은 그룹의 의심 클래스 crop
        사람 검수(위치·클래스)에서 이미 오류로 판정된 crop은 흐리게 X 표시

사용:
    python classifier_data_build/audit_class_label.py [--checkpoint data/best.pt] [--recompute]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from config import AUDIT, DATA, EXPORT, META, keep_review, save_table
from classifier.checkpoint import hash_class_map
from classifier.model import build_model
from classifier_data.classifier_inputs import PillCropDataset, read_samples, val_transform

ROOT = EXPORT / "training_package_classifier"
OUT = META / "label_audit"
FOLDS = 5


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stack_images(batch):
    return torch.stack([s["image"] for s in batch])


def extract(df, checkpoint, dev):
    """(embedding [N, 2048], best.pt 확률 [N, 118])"""
    class_map = json.loads((ROOT / "class_map.json").read_text())
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["class_map_hash"] == hash_class_map(class_map), "best.pt와 class_map이 다르다"
    model = build_model(class_map["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(dev)
    feats = {}
    model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("x", o.flatten(1)))
    loader = DataLoader(PillCropDataset(df, ROOT, val_transform()), batch_size=64, num_workers=4,
                        collate_fn=stack_images)
    embs, probs = [], []
    with torch.no_grad():
        for k, x in enumerate(loader):
            logits = model(x.to(dev))
            embs.append(feats["x"].cpu().numpy())
            probs.append(logits.softmax(1).cpu().numpy())
            if k % 50 == 0:
                print(f"  embedding {k * 64}/{len(df)}")
    return np.concatenate(embs).astype(np.float32), np.concatenate(probs).astype(np.float32)


def group_compare(df, emb):
    """같은 그룹, 다른 사진의 crop과 비교. 같은 클래스 평균 유사도 vs 가장 닮은 다른 클래스 평균 유사도"""
    z = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    own = np.full(len(df), np.nan)
    other = np.full(len(df), np.nan)
    other_cls = np.full(len(df), -1)
    for _, idx in df.groupby("group_id").indices.items():
        sim = z[idx] @ z[idx].T
        img = df.image_id.values[idx]
        cls = df.class_index.values[idx]
        for a in range(len(idx)):
            mask = img != img[a]                      # 같은 사진은 빼고 다른 각도 사진만
            if not mask.any():
                continue
            per = pd.Series(sim[a, mask]).groupby(cls[mask]).mean()
            if cls[a] in per.index:
                own[idx[a]] = per[cls[a]]
            rest = per.drop(cls[a], errors="ignore")
            if len(rest):
                other[idx[a]] = rest.max()
                other_cls[idx[a]] = rest.idxmax()
    return own, other, other_cls


def oof_predict(df, emb):
    """그룹 단위 5-fold 교차 검증 예측 확률 [N, 118]"""
    z = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    y = df.class_index.values
    prob = np.zeros((len(df), 118), np.float32)
    for k, (tr, te) in enumerate(GroupKFold(FOLDS).split(z, y, df.group_id.values)):
        clf = LogisticRegression(C=10, max_iter=2000)
        clf.fit(z[tr], y[tr])
        prob[np.ix_(te, clf.classes_)] = clf.predict_proba(z[te])
        print(f"  fold {k + 1}/{FOLDS}")
    return prob


def find_swaps(df, s):
    """같은 사진의 두 crop이 서로 상대 클래스로 판단되면 True"""
    swap = np.zeros(len(df), bool)
    for _, idx in df.groupby("image_id").indices.items():
        for a in idx:
            for b in idx:
                if a < b and s.suspect_class[a] == df.class_index[b] and s.suspect_class[b] == df.class_index[a]:
                    swap[[a, b]] = True
    return swap


def crop_tile(path, size, text, color, known_error=False):
    im = np.array(Image.open(ROOT / path).convert("RGB"))
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    tile = np.full((size + 22, size, 3), 255, np.uint8)
    tile[:im.shape[0], :im.shape[1]] = im
    cv2.putText(tile, text, (2, size + 16), 0, 0.42, color, 1)
    if known_error:                                   # 이미 오류로 판정된 crop: 비교 기준으로 쓰지 않는다
        tile[:size] = (tile[:size] * 0.4 + 255 * 0.6).astype(np.uint8)
        cv2.line(tile, (0, 0), (size, size), (120, 120, 120), 3)
        cv2.line(tile, (size, 0), (0, size), (120, 120, 120), 3)
        cv2.putText(tile, "known error", (4, 20), 0, 0.5, (90, 90, 90), 1)
    return tile


def known_errors():
    """사람 검수에서 오류로 판정된 crop의 (sample_id 집합, 이미지 제외 사진 집합). 위치 검수와 클래스 검수를 함께 본다"""
    ids, imgs = set(), set()
    for name in ("position_suspects.csv", "class_suspects.csv"):
        path = AUDIT / name
        if path.exists():
            d = pd.read_csv(path, dtype=str, keep_default_na=False)
            bad = d[d.판정 == "오류"]
            ids |= set(bad.sample_id)
            imgs |= set(bad[bad.조치 == "이미지 제외"].image_id)
    return ids, imgs


def draw_sheet(rows, df, path, size=150):
    """한 줄 = 번호 | 의심 crop | 라벨 클래스(같은 그룹 다른 사진) 2개 | 의심 클래스(같은 그룹 다른 사진) 2개
    사람 검수에서 이미 오류로 판정된 crop은 흐리게 X 표시 (known error)"""
    bad_ids, bad_imgs = known_errors()
    is_bad = lambda r: r.sample_id in bad_ids or r.image_id in bad_imgs
    blank = np.full((size + 22, size, 3), 255, np.uint8)
    lines = []
    for k, r in rows:
        g = df[(df.group_id == r.group_id) & (df.image_id != r.image_id)]
        same = list(g[g.class_index == r.class_index].head(2).itertuples())
        sus = list(g[g.class_index == r.suspect_class].head(2).itertuples())
        label = blank.copy()
        cv2.putText(label, str(k), (10, 60), 0, 1.6, (0, 0, 0), 3)
        cv2.putText(label, r.code[-6:], (10, 100), 0, 0.6, (0, 0, 0), 1)
        cv2.putText(label, f"-> {r.suspect_code[-6:]}", (10, 125), 0, 0.6, (200, 0, 0), 1)
        tiles = [label, crop_tile(r.crop_path, size, "suspect", (200, 0, 0))]
        tiles += [crop_tile(x.crop_path, size, f"label {r.code[-6:]}", (0, 140, 0), is_bad(x)) for x in same] + [blank] * (2 - len(same))
        tiles += [crop_tile(x.crop_path, size, f"other {r.suspect_code[-6:]}", (200, 0, 0), is_bad(x)) for x in sus] + [blank] * (2 - len(sus))
        line = np.hstack(tiles)
        line[-1:, :] = 180
        lines.append(line)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(lines), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DATA / "best.pt"))
    ap.add_argument("--recompute", action="store_true", help="저장된 embedding이 있어도 다시 계산한다")
    args = ap.parse_args()

    df = read_samples(ROOT).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "class_embeddings.npz"
    if cache.exists() and not args.recompute:
        z = np.load(cache)
        assert list(z["sample_id"]) == df.sample_id.tolist(), "저장된 embedding의 샘플 순서가 다르다 (--recompute)"
        emb, prob = z["emb"], z["prob"]
    else:
        emb, prob = extract(df, args.checkpoint, device())
        np.savez_compressed(cache, emb=emb, prob=prob, sample_id=df.sample_id.to_numpy(dtype=str))

    print("같은 그룹 비교")
    own, other, other_cls = group_compare(df, emb)
    print("교차 검증 예측")
    oof = oof_predict(df, emb)

    y = df.class_index.values
    idx = np.arange(len(df))
    code_of = df.drop_duplicates("class_index").set_index("class_index").code
    s = df[["sample_id", "image_id", "group_id", "code", "name", "split", "source", "class_index", "crop_path"]].copy()
    s["model_pred"] = prob.argmax(1)
    s["model_label_prob"] = prob[idx, y].round(4)
    s["oof_pred"] = oof.argmax(1)
    s["oof_label_prob"] = oof[idx, y].round(4)
    s["group_own_sim"] = np.round(own, 4)
    s["group_other_sim"] = np.round(other, 4)
    s["group_other_class"] = other_cls
    s["group_gap"] = np.round(other - own, 4)          # > 0 이면 다른 클래스와 더 닮음
    group_flag = s.group_gap > 0
    oof_flag = s.oof_pred != y
    # 의심 클래스: 같은 그룹 비교가 가리키는 클래스를 우선, 없으면 교차 검증 예측
    s["suspect_class"] = np.where(group_flag, s.group_other_class, s.oof_pred)
    s["suspect_code"] = s.suspect_class.map(code_of)
    s["swap"] = find_swaps(df, s)
    s["priority"] = np.select([s.swap | (group_flag & oof_flag), group_flag | oof_flag], [1, 2], 0)
    save_table(s, OUT / "class_scores.csv")

    sus = s[s.priority > 0].sort_values(["priority", "oof_label_prob"]).copy()
    sus["그림번호"] = sus.groupby("priority").cumcount()
    sus["그림파일"] = [f"review/c{p}_review_{n // 20 + 1}.jpg" for p, n in zip(sus.priority, sus.그림번호)]
    pos = pd.read_csv(AUDIT / "position_suspects.csv", dtype=str, keep_default_na=False) if (AUDIT / "position_suspects.csv").exists() else None
    pos_bad = pos[pos.판정 == "오류"] if pos is not None else pd.DataFrame(columns=["sample_id", "image_id", "조치"])
    bad_ids, bad_imgs = set(pos_bad.sample_id), set(pos_bad[pos_bad.조치 == "이미지 제외"].image_id)
    sus["위치검수와중복"] = sus.sample_id.isin(bad_ids) | sus.image_id.isin(bad_imgs)
    AUDIT.mkdir(exist_ok=True)
    sus = keep_review(sus, AUDIT / "class_suspects.csv")
    cols = ["그림번호", "그림파일"] + [c for c in sus.columns if c not in ("그림번호", "그림파일")]
    save_table(sus[cols], AUDIT / "class_suspects.csv")

    print(f"crop {len(s)}개 / best.pt가 라벨과 다르게 예측 {(s.model_pred != y).sum()}개 / 교차 검증이 다르게 예측 {oof_flag.sum()}개")
    for p in (1, 2):
        g = sus[sus.priority == p]
        print(f"  {p}등급: {len(g)}개 (train {(g.split == 'train').sum()}, val {(g.split == 'val').sum()}, swap {g.swap.sum()})")

    (OUT / "review").mkdir(exist_ok=True)
    for old in (OUT / "review").glob("c*_review_*.jpg"):
        old.unlink()
    for p in (1, 2):
        g = sus[sus.priority == p]
        for n in range(0, len(g), 20):
            draw_sheet(list(zip(g.그림번호[n:n + 20], g.iloc[n:n + 20].itertuples())), df,
                       OUT / "review" / f"c{p}_review_{n // 20 + 1}.jpg")
    print(f"검수 그림: {OUT / 'review'}")
