"""알약 하나를 bbox대로 잘라 정사각형 이미지로 만든다. 학습 데이터 생성과 추론(탐지 bbox)에서 같이 쓴다.
명세서 「분류기 데이터 및 함수 입력 명세서」 6.1 (수정: bbox 사방 5% 여백)과 같은 순서다.

    from classifier_data.crop import crop_pill
    pill = crop_pill(image, (x, y, w, h))   # PIL 이미지, bbox는 픽셀 [x, y, w, h]

학습 때 회전을 넣으려면 cut_pill → 회전 → pad_square 순서로 쓴다 (명세서 6.2).
"""
from PIL import Image

MARGIN = 0.05                 # bbox 사방에 더할 여백 (가로는 w, 세로는 h 대비)
PAD_COLOR = (124, 116, 104)   # ImageNet 평균색. 정규화하면 0이 된다


def cut_pill(img, bbox, margin=MARGIN):
    """bbox에 사방 margin만큼 여백을 더하고 이미지 경계로 clip해 자른 직사각형 (RGB)."""
    img = img.convert("RGB")
    x, y, w, h = bbox
    mx, my = w * margin, h * margin
    box = (max(0, round(x - mx)), max(0, round(y - my)),
           min(img.width, round(x + w + mx)), min(img.height, round(y + h + my)))
    return img.crop(box)


def pad_square(img, fill=PAD_COLOR):
    """짧은 쪽 양옆(또는 위아래)을 fill 색으로 채워 가운데 정렬한 정사각형."""
    s = max(img.size)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return out


def crop_pill(img, bbox, margin=MARGIN, fill=PAD_COLOR):
    """cut_pill + pad_square. 사진에서 더 넓게 잘라 정사각형을 만들지 않으므로 옆 알약이 덜 들어온다.
    결과는 항상 1:1이고, 224×224로 줄여도 알약 모양이 찌그러지지 않는다."""
    return pad_square(cut_pill(img, bbox, margin), fill)
