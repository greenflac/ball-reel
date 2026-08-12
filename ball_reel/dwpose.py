"""DWPose как штатный источник условий — вместо MediaPipe.

Две причины перейти, обе измерены, а не предположены.

ПОЛНОТА. На живом фитнес-референсе MediaPipe дал покрытие кадров 1.0 и при
этом потерял конечности в 32 условиях из 96: кисти, лежащие на мяче, уходят
ниже порога видимости, а вместе с локтем выбрасывается предплечье. Треть кадров
не ограничивала руку, и заметить это удалось только глазами. DWPose (RTMPose,
COCO-WholeBody) держит кисти заметно лучше — на них у него отдельные 42 точки.

НЕЗАВИСИМОСТЬ. GPU_BRANCH.md требует, чтобы кондиционирующий экстрактор
отличался от проверяющего: если позу для ControlNet снимать тем же MediaPipe,
которым потом меряется результат, гейт начнёт подтверждать собственные ошибки —
систематический промах попадёт и в условие, и в эталон, и разойтись они не
смогут. MediaPipe остаётся судьёй, DWPose становится источником.

СОСТОЯНИЕ: инференс НЕ ИСПОЛНЯЛСЯ НИ РАЗУ. `huggingface.co` закрыт egress-
политикой этой среды (403 на CONNECT), поэтому веса сюда не скачать. Написано
по документированному формату моделей; чистая часть — маппинг точек и
геометрия — покрыта тестами и работает без весов. Первый запуск на машине с
доступом и есть проверка; расхождения дописывать сюда, как дописывались
расхождения по Pollinations в POLLINATIONS_CONTRACT.md.

Веса (скачать там, где HuggingFace доступен):
    mkdir -p ~/.dwpose && cd ~/.dwpose
    curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
    curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx
"""

from __future__ import annotations

import os
from pathlib import Path

DET_ENV, POSE_ENV = "BALL_REEL_DWPOSE_DET", "BALL_REEL_DWPOSE_POSE"
DEFAULT_DET = "~/.dwpose/yolox_l.onnx"
DEFAULT_POSE = "~/.dwpose/dw-ll_ucoco_384.onnx"

#: COCO-WholeBody отдаёт 133 точки; первые 17 — тело в порядке COCO.
#: Здесь их индексы под именами, которыми пользуется остальной проект.
WHOLEBODY_INDEX = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

#: Ниже этого балла точка считается ненаблюдаемой. Порог сознательно ниже
#: MediaPipe-шного 0.5: RTMPose калиброван иначе, и именно излишняя строгость
#: выбрасывала кисти. Требует перекалибровки на первом живом прогоне — число
#: взято из практики DWPose, а не измерено здесь.
MIN_SCORE = 0.3

_SESSIONS: dict = {}


def _model_paths() -> tuple:
    det = Path(os.environ.get(DET_ENV, DEFAULT_DET)).expanduser()
    pose = Path(os.environ.get(POSE_ENV, DEFAULT_POSE)).expanduser()
    missing = [str(p) for p in (det, pose) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"нет весов DWPose: {', '.join(missing)}. Скачать один раз:\n"
            f"  mkdir -p ~/.dwpose && cd ~/.dwpose\n"
            f"  curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx\n"
            f"  curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx\n"
            f"или указать пути через {DET_ENV} / {POSE_ENV}. Молча падать на "
            f"MediaPipe нельзя: это вернёт кондиционер и верификатор в одно лицо.")
    return det, pose


def available() -> bool:
    """Есть ли веса — чтобы вызывающий мог решить ДО начала рендера."""
    try:
        _model_paths()
        return True
    except RuntimeError:
        return False


def _session(path: Path):
    import onnxruntime  # type: ignore

    key = str(path)
    if key not in _SESSIONS:
        _SESSIONS[key] = onnxruntime.InferenceSession(
            key, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _SESSIONS[key]


def _letterbox(img, size: tuple):
    """Вписать кадр в квадрат детектора, сохранив пропорции.

    Именно вписать, а не растянуть: растяжение искажает форму тела, и детектор
    начинает промахиваться по вытянутым фигурам. Возвращает (полотно, масштаб).
    """
    import numpy as np

    h, w = img.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
    from PIL import Image

    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas[:nh, :nw] = resized
    return canvas, scale


def _largest_person(boxes, scores, scale: float, thresh: float = 0.3):
    """Самый крупный человек в кадре, в координатах исходника.

    Крупнейший, а не первый: в кадре бывает отражение в зеркале или прохожий,
    и условие должно строиться по тому, кого снимают.
    """
    best, best_area = None, 0.0
    for box, score in zip(boxes, scores):
        if score < thresh:
            continue
        x0, y0, x1, y1 = (v / scale for v in box[:4])
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best, best_area = (x0, y0, x1, y1), area
    return best


def _decode_simcc(x_logits, y_logits, box, input_size: tuple) -> dict:
    """SimCC-выход RTMPose -> точки в координатах исходного кадра.

    RTMPose предсказывает не тепловые карты, а два одномерных распределения на
    сустав (по x и по y) с подпиксельным разрешением; координата — argmax,
    делённый на коэффициент разбиения, а уверенность — произведение пиков.
    """
    import numpy as np

    x_locs = np.argmax(x_logits, axis=-1)
    y_locs = np.argmax(y_logits, axis=-1)
    x_conf = np.max(x_logits, axis=-1)
    y_conf = np.max(y_logits, axis=-1)
    split = x_logits.shape[-1] / input_size[0]

    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    sx, sy = bw / input_size[0], bh / input_size[1]

    out = {}
    for name, i in WHOLEBODY_INDEX.items():
        conf = float(min(x_conf[i], y_conf[i]))
        out[name] = (float(x0 + (x_locs[i] / split) * sx),
                     float(y0 + (y_locs[i] / split) * sy), conf)
    return out


def to_normalised(points: dict, width: int, height: int) -> dict:
    """Пиксели -> нормированные координаты в формате, который ждёт skeleton.

    Тот же контракт, что у `skeleton.pose_points`: ``{имя: (x, y, видимость)}``
    в 0..1 плюс `__size__`, плюс синтезированные `neck` и `hip_c`, которых нет
    ни в одном из наборов и которые нужны для нормировки и ретаргетинга.
    """
    out = {name: (x / width, y / height, c)
           for name, (x, y, c) in points.items()}
    ls, rs = out.get("l_shoulder"), out.get("r_shoulder")
    lh, rh = out.get("l_hip"), out.get("r_hip")
    if ls and rs:
        out["neck"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2,
                       min(ls[2], rs[2]))
    if lh and rh:
        out["hip_c"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2,
                        min(lh[2], rh[2]))
    out["__size__"] = (float(width), float(height), 1.0)
    return out


def usable(points: dict, min_score: float = MIN_SCORE) -> bool:
    """Есть ли в наборе хоть одна наблюдаемая точка.

    Вынесено отдельной функцией, потому что решение «кадр пустой» не должно
    жить внутри вызова, требующего весов: иначе порог невозможно проверить, а
    непроверяемый порог сдвигается молча. Мутационный аудит показал это прямо —
    снятый MIN_SCORE не ронял ни одного теста.
    """
    return any(c >= min_score for _, _, c in points.values())


def pose_points(path: str | Path) -> dict | None:
    """Точки COCO-18 для одного кадра — замена `skeleton.pose_points`.

    НЕ ИСПОЛНЯЛОСЬ: см. докстринг модуля. Форма выхода совпадает с
    MediaPipe-версией специально, чтобы `skeleton.render_sequence` не знал,
    кто был источником.
    """
    import numpy as np
    from PIL import Image

    det_path, pose_path = _model_paths()
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    det = _session(det_path)
    canvas, scale = _letterbox(rgb, (640, 640))
    blob = canvas.transpose(2, 0, 1)[None].astype("float32")
    raw = det.run(None, {det.get_inputs()[0].name: blob})[0][0]
    person = _largest_person(raw[:, :4], raw[:, 4], scale)
    if person is None:
        return None

    pose = _session(pose_path)
    x0, y0, x1, y1 = person
    crop = Image.fromarray(rgb).crop((int(x0), int(y0), int(x1), int(y1)))
    inp = np.asarray(crop.resize((192, 256), Image.BILINEAR),
                     dtype="float32").transpose(2, 0, 1)[None]
    x_logits, y_logits = pose.run(None, {pose.get_inputs()[0].name: inp})
    points = _decode_simcc(x_logits[0], y_logits[0], person, (192, 256))

    if not usable(points):
        return None
    return to_normalised(points, w, h)
