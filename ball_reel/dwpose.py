"""DWPose как штатный источник условий — вместо MediaPipe.

ПОЛНОТА — выигрыш есть, но меньше, чем я обещал, когда писал этот модуль.
Замерено на одном и том же клипе из 96 кадров:

    источник     кадры   суставы   кадров с потерянной конечностью
    MediaPipe    1.000   0.917     32
    DWPose       0.979   0.915     19

Кадров с неограниченной конечностью стало на 40% меньше — это и была цель.
Но общее покрытие суставов сравнялось, а два кадра DWPose потерял целиком.
Гипотеза «у DWPose 42 точки на кисти, значит кисти на мяче будут держаться
заметно лучше» подтвердилась лишь частично. Порог тут ни при чём: прогон на
0.30/0.25/0.20 дал одинаковый результат, а распределения баллов у обоих
источников сопоставимы (при 0.5 проходит 91% против 92%).

НЕЗАВИСИМОСТЬ — после этих замеров именно она главное основание, а не полнота.
GPU_BRANCH.md требует, чтобы кондиционирующий экстрактор отличался от
проверяющего: если позу для ControlNet снимать тем же MediaPipe, которым потом
меряется результат, гейт начнёт подтверждать собственные ошибки —
систематический промах попадёт и в условие, и в эталон, и разойтись они не
смогут. MediaPipe остаётся судьёй, DWPose становится источником.

СОСТОЯНИЕ: инференс исполнялся, числа выше сняты с него. Веса приехали после
того, как домен открыли в вайтлисте среды. Первый прогон дал три расхождения
с документацией, все тихие (то есть код не падал, а выдавал правдоподобный
мусор): вход 288x384, а не 192x256; YOLOX отдаёт сырую сетку [1,8400,85], а не
готовые боксы; и главное — отсутствовала ImageNet-нормализация, из-за чего
расхождение с эталоном было 0.457 вместо 0.014. Все три исправлены (коммит
cb91a55). Дальнейшие расхождения дописывать сюда, как расхождения по
Pollinations дописываются в POLLINATIONS_CONTRACT.md.

Веса (скачать там, где HuggingFace доступен):
    mkdir -p ~/.dwpose && cd ~/.dwpose
    curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
    curl -sSLO https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx
"""

from __future__ import annotations

from . import cure

import os
from pathlib import Path

def _providers():
    """Провайдеры onnxruntime под то устройство, что реально есть.

    Раньше здесь был жёсткий CUDA-список, и onnxruntime МОЛЧА его игнорировал:
    на замерах латентности DWPose просил CUDA и выдал 438 мс на процессоре.
    Запрошенный, но отсутствующий провайдер не ошибка, а тихая деградация.
    """
    from .device import detect, onnx_providers

    want, _ = onnx_providers(detect())
    return want


DET_ENV, POSE_ENV = "BALL_REEL_DWPOSE_DET", "BALL_REEL_DWPOSE_POSE"
DEFAULT_DET = "~/.dwpose/yolox_l.onnx"
DEFAULT_POSE = "~/.dwpose/dw-ll_ucoco_384.onnx"

#: Откуда качаются, если их нет. Отдельной константой: адрес попадает в ТЕКСТ
#: ОТКАЗА, а вписанный туда вручную разъезжается с настоящим при первой правке.
WEIGHTS_ONLINE = "https://huggingface.co/yzd-v/DWPose/resolve/main/"

#: Разделитель пути ДЛЯ ОБОЛОЧКИ, а не для Python.
SEP = "\\" if cure.WINDOWS else "/"

#: COCO-WholeBody отдаёт 133 точки; первые 17 — тело в порядке COCO.
#: Здесь их индексы под именами, которыми пользуется остальной проект.
WHOLEBODY_INDEX = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

#: Ниже этого балла точка считается ненаблюдаемой.
#: Замерено на реальном клипе (204 точки): баллы лежат в 0.20..0.95, медиана
#: 0.84. Калибровать порог по «видимости» MediaPipe нельзя — именно его мы и
#: заменяем, и его пропуски были бы приняты за истину. Поэтому 0.3 отсекает
#: только явный хвост распределения, а судит о пользе перехода не порог, а
#: итоговое покрытие суставов в условиях. [проверено live]
MIN_SCORE = 0.3

#: Нормализация входа RTMPose (ImageNet). БЕЗ НЕЁ МОДЕЛЬ МОЛЧА ВРЁТ:
#: точки уезжают, ничего не падает. Замерено против MediaPipe на живом кадре —
#: среднее расхождение 0.457 кадра без нормализации против 0.014 с ней.
#: [проверено live]
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)

#: Кроп расширяется до пропорций входа модели и на 25% — так RTMPose обучался.
BOX_PADDING = 1.25

#: Вход модели позы (ширина, высота). Не 192x256, как я написал по памяти:
#: файл называется dw-ll_ucoco_384, и сигнатура ONNX подтверждает [3, 384, 288].
#: Ошибка в этом числе не падает, а тихо смещает все точки. [проверено live]
POSE_INPUT = (288, 384)

_SESSIONS: dict = {}


def _model_paths() -> tuple:
    det = Path(os.environ.get(DET_ENV, DEFAULT_DET)).expanduser()
    pose = Path(os.environ.get(POSE_ENV, DEFAULT_POSE)).expanduser()
    missing = [str(p) for p in (det, pose) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"нет весов DWPose: {', '.join(missing)}. Скачать один раз:\n"
            f"  {cure.mkdir(cure.home('.dwpose'))}\n"
            + "".join(
                f"  {cure.download(WEIGHTS_ONLINE + n, cure.home('.dwpose') + SEP + n)}\n"
                for n in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"))
            + f"или указать пути через {DET_ENV} / {POSE_ENV}. Молча падать на "
            f"MediaPipe нельзя: это вернёт кондиционер и верификатор в одно лицо.")
    return det, pose


def why_unavailable() -> str:
    """Причина отсутствия весов одной строкой. Пустая строка = всё на месте.

    `_model_paths` бросает исключение — это правильно на пути исполнения, но
    негодно там, где надо всего лишь объяснить в манифесте, почему условия
    сняты не тем экстрактором.
    """
    try:
        _model_paths()
        return ""
    except RuntimeError as exc:
        return str(exc)


def available() -> bool:
    """Есть ли веса — чтобы вызывающий мог решить ДО начала рендера.

    Выражено ЧЕРЕЗ `why_unavailable`, а не отдельной проверкой, намеренно: два
    независимых способа ответить на один вопрос расходятся при первой же
    правке, и тогда код может считать веса отсутствующими, а объяснение —
    сообщать, что всё на месте.
    """
    return not why_unavailable()


def _session(path: Path):
    import onnxruntime  # type: ignore

    key = str(path)
    if key not in _SESSIONS:
        _SESSIONS[key] = onnxruntime.InferenceSession(
            key, providers=list(_providers()))
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


def decode_yolox(raw, scale: float, strides=(8, 16, 32), size: int = 640):
    """Сырой выход YOLOX -> боксы в координатах исходника. [проверено live]

    Модель отдаёт НЕ готовые боксы, а сетку [1, 8400, 85]: 8400 якорей трёх
    масштабов (80x80 + 40x40 + 20x20), на каждый 4 смещения + objectness +
    80 классов COCO. Координаты — смещения относительно ячейки, размеры — в
    логарифме, и то и другое в единицах страйда. Я изначально написал этот
    код так, будто первые четыре числа уже боксы; форма модели это опровергла.

    Берётся только класс 0 (человек), балл — произведение objectness на балл
    класса, как принято в YOLOX.
    """
    import numpy as np

    grids, expanded = [], []
    for stride in strides:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        expanded.append(np.full((1, n * n, 1), stride))
    grid = np.concatenate(grids, 1)
    step = np.concatenate(expanded, 1)

    out = np.array(raw, dtype="float64")
    out[..., :2] = (out[..., :2] + grid) * step
    out[..., 2:4] = np.exp(out[..., 2:4]) * step

    preds = out[0]
    scores = preds[:, 4] * preds[:, 5]          # objectness x класс «человек»
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return boxes, scores


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


def expand_box(box, aspect: float = POSE_INPUT[0] / POSE_INPUT[1],
               pad: float = BOX_PADDING):
    """Расширить бокс до пропорций входа модели, с запасом.

    Плоский resize прямоугольного бокса в 288x384 сжимает одну ось и смещает
    все точки. Расширение вокруг центра сохраняет форму тела, а запас в 25%
    возвращает в кадр конечности, которые детектор обрезал по краю.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    w, h = w * pad, h * pad
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _peak(logits):
    """Пик распределения SimCC как балл сустава.

    Сырой максимум, а НЕ softmax по бинам: распределение размазано по 576/768
    бинам, и softmax даёт пики порядка 0.003 — на таком все точки оказываются
    ниже любого разумного порога, и кадр отбрасывается целиком (поймано живьём).
    С правильно нормализованным входом сырой максимум сам лежит в 0..1:
    измерено на 204 точках реального клипа — медиана 0.84. [проверено live]
    """
    import numpy as np

    return np.max(logits, axis=-1)


def _decode_simcc(x_logits, y_logits, box, input_size: tuple) -> dict:
    """SimCC-выход RTMPose -> точки в координатах исходного кадра.

    RTMPose предсказывает не тепловые карты, а два одномерных распределения на
    сустав (по x и по y) с подпиксельным разрешением; координата — argmax,
    делённый на коэффициент разбиения, а уверенность — произведение пиков.
    """
    import numpy as np

    x_locs = np.argmax(x_logits, axis=-1)
    y_locs = np.argmax(y_logits, axis=-1)
    x_conf = _peak(x_logits)
    y_conf = _peak(y_logits)
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
    raw = det.run(None, {det.get_inputs()[0].name: blob})[0]
    boxes, scores = decode_yolox(raw, scale)
    person = _largest_person(boxes, scores, scale)
    if person is None:
        return None

    pose = _session(pose_path)
    person = expand_box(person)
    x0, y0, x1, y1 = person
    crop = Image.fromarray(rgb).crop((int(x0), int(y0), int(x1), int(y1)))
    arr = np.asarray(crop.resize(POSE_INPUT, Image.BILINEAR), dtype="float32")
    arr = (arr - np.array(PIXEL_MEAN)) / np.array(PIXEL_STD)
    inp = arr.transpose(2, 0, 1)[None].astype("float32")
    x_logits, y_logits = pose.run(None, {pose.get_inputs()[0].name: inp})
    points = _decode_simcc(x_logits[0], y_logits[0], person, POSE_INPUT)

    if not usable(points):
        return None
    return to_normalised(points, w, h)
