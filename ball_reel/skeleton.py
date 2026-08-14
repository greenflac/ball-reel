"""Pose track -> ControlNet condition images, retargeted to the target body.

This is what turns a measured trajectory into something a generator must obey.
ControlNet takes a drawn skeleton, not a description, so the motion stops being
a request the model may reinterpret: the joints are where they are.

Three jobs, and only the first one is about drawing.

RENDER. MediaPipe's 33 landmarks become the 18-point COCO skeleton OpenPose
ControlNet expects, drawn with its colour convention — the colours are the
channel encoding, not decoration, and a differently-coloured skeleton conditions
badly.

RETARGET. The driving skeleton carries the DRIVING person's proportions. Feed it
unchanged and you have asked for the donor's body wearing the client's face —
the exact failure the whole subject package exists to prevent. So each bone
keeps its direction from the driving frame and takes its LENGTH from the
target's measured 3D proportions, walking outward from the hips so the change
propagates down each limb instead of tearing it apart.

FRAME. Третья работа, и она появилась не из вкуса, а из арифметики. Генерация
идёт на карте 6 ГБ, потолок 512x768. На ПОЛНОМ РОСТЕ лицо занимает 9.4% высоты
кадра (ИЗМЕРЕНО живьём, `animate.FULL_BODY_FACE_SHARE`) — на 768 это ~72 px, а
ArcFace судит видео от 100 px и старт-кадр от 70 (`identity_arcface`). То есть
главный критерий демо — «личность перенесена» — подтвердить нечем: гейт вернёт
не «плохо», а «не смог». Кадрировка ПО ПОЯС удваивает долю лица (19%, тоже
ИЗМЕРЕНО), и на тех же 512x768 лицо выходит ~146 px, что уже судится.

Поэтому окно строится по СКЕЛЕТУ, а не по доле кадра: «по пояс» — это от
макушки до таза с запасом, и границу проводят суставы. И поэтому же манифест
несёт ОЖИДАЕМУЮ долю лица для обеих кадрировок: решение «будет ли гейт судить
идентичность» принимается ДО генерации, а не после потраченных GPU-минут.

Runs on CPU, no GPU, no network. Which means the expensive half of the pipeline
can be prepared and checked before a single GPU-minute is rented.

A NOTE ON INDEPENDENCE. Экстрактор, который СТАВИТ УСЛОВИЯ, и тот, что их
ПРОВЕРЯЕТ, обязаны быть разными: иначе гейт подтверждает ошибки кондиционера.
Условия снимает DWPose, проверяет MediaPipe.

Это долго было заявлением, а не кодом. Умолчанием стоял MediaPipe, то есть судья
и судимый совпадали, и заявление держалось на одном предупреждении в манифесте.
Теперь умолчание — DWPose, если его веса на месте, а откат на MediaPipe остаётся
рабочим, но помечается в манифесте как ослабленный режим.

Расхождение двух экстракторов ИЗМЕРЕНО на живом кадре: медиана 0.0121
нормированной длины, худший сустав (ухо) 0.0280, при баре гейта по позе 0.25.
То есть в двадцать раз ниже порога — подмена экстрактора калибровку не ломает,
и это стоило проверить до подмены, а не после.
"""

from __future__ import annotations

from pathlib import Path

#: OpenPose COCO-18 joint order. ControlNet's openpose annotator emits exactly
#: this, and the model was trained on it.
COCO18 = ("nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
          "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
          "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear")

#: MediaPipe's 33-point indices for the joints COCO-18 needs. `neck` is absent
#: from MediaPipe and is synthesised as the shoulder midpoint.
MEDIAPIPE_INDEX = {
    "nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}

#: OpenPose limb connections, in its own drawing order.
LIMBS18 = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
           (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
           (0, 15), (15, 17))

#: The OpenPose palette, in the same order as LIMBS18. These exact colours are
#: how the model reads which limb is which.
LIMB_COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                (255, 0, 170))

JOINT_COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                 (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                 (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                 (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                 (255, 0, 170), (255, 0, 85))

#: Bones, as (parent, child), ordered outward from the hips. Retargeting walks
#: this order so that moving a shoulder carries its whole arm with it.
BONE_TREE = (("hip_c", "neck"), ("neck", "nose"),
             ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"),
             ("l_elbow", "l_wrist"),
             ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"),
             ("r_elbow", "r_wrist"),
             ("hip_c", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
             ("hip_c", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"))

#: Which measured proportion sets each bone's length. Bones with no entry keep
#: the driving length — the head and the hip half-widths, where the subject
#: package has nothing better to say.
BONE_TO_PROPORTION = {
    ("l_shoulder", "l_elbow"): "l_shoulder->l_elbow",
    ("l_elbow", "l_wrist"): "l_elbow->l_wrist",
    ("r_shoulder", "r_elbow"): "r_shoulder->r_elbow",
    ("r_elbow", "r_wrist"): "r_elbow->r_wrist",
    ("l_hip", "l_knee"): "l_hip->l_knee",
    ("l_knee", "l_ankle"): "l_knee->l_ankle",
    ("r_hip", "r_knee"): "r_hip->r_knee",
    ("r_knee", "r_ankle"): "r_knee->r_ankle",
}


#: Кадрировки, которые умеет строить `_window`. Третьей не предполагается: либо
#: в кадре вся фигура, либо её верхняя половина.
FRAMINGS = ("full_body", "waist_up")

#: Суставы верхней части тела — ими задаётся окно кадрировки «по пояс».
#: Горизонтальный габарит берётся по ним всем (иначе локоть уедет за край), а
#: НИЖНЯЯ граница — всегда по тазу, даже если запястья висят ниже: иначе
#: опущенные руки вернут кадр к полному росту и вся затея развалится.
UPPER_BODY = ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "neck",
              "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
              "l_wrist", "r_wrist", "l_hip", "r_hip", "hip_c")

#: Запас вокруг габарита фигуры, долей от самого габарита.
#: `full_body` 0.35 — ВЫБРАН, стоит здесь с первого дня и трогать его нельзя:
#: на нём откалибрована ИЗМЕРЕННАЯ доля лица 9.4%.
#: `waist_up` 0.25 — ВЫБРАН, но не наугад. На канонической фигуре (см. вывод
#: FACE_TO_TORSO) он даёт долю лица 0.192 при ИЗМЕРЕННЫХ живьём 0.19
#: (`animate.WAIST_UP_FACE_SHARE`); при 0.35 вышло бы 0.178, то есть запас
#: прямо тратится на лицо. Больше запаса — мельче лицо, и это не метафора.
FULL_BODY_MARGIN = 0.35
WAIST_UP_MARGIN = 0.25

#: Высота лица (то, что мерит детектор: примерно от бровей до подбородка) в
#: длинах торса neck->hip_c. ВЫБРАН по канону «рост = 7.5 голов», и вот сверка
#: с ИЗМЕРЕННЫМ, ради которой это число вообще можно держать в коде.
#:
#: Канон, в долях роста: нос 0.935, шея (середина плеч) 0.82, таз 0.53,
#: лодыжка 0.039. Торс = 0.29 роста, габарит нос->лодыжка = 0.896.
#: Окно полного роста = 1.35 * 0.896 = 1.210 роста, лицо = 0.39 * 0.29 = 0.113.
#: Доля = 0.113 / 1.210 = 0.0934 — против ИЗМЕРЕННЫХ 0.094.
#: Совпадение с точностью до третьего знака получено НЕ подгонкой: 0.39 взято
#: из канона, 0.094 снято с живых кадров, встретились они здесь.
FACE_TO_TORSO = 0.39

#: Насколько макушка выше носа, в длинах торса. ВЫБРАН по тому же канону:
#: (1.0 - 0.935) / 0.29 = 0.224. Нужен потому, что в COCO-18 макушки нет, а
#: кадрировка «по пояс» без неё срезает верх головы — то есть ровно то, ради
#: чего кадрировка и делается.
CROWN_ABOVE_NOSE = 0.224

#: Насколько геометрия здесь вправе разойтись с живым измерением, прежде чем об
#: этом стоит предупредить. 0.35 ВЫБРАН: на живом kit-кадре широкая стойка
#: (ноги врозь) роняет долю лица на полном росте с 0.094 до 0.074, то есть на
#: 21% — это норма позы, а не дефект, и лаять на неё не надо. Втрое большее
#: расхождение — уже повод посмотреть глазами.
FACE_SHARE_TOLERANCE = 0.35


def pose_points(path: str | Path) -> dict | None:
    """The COCO-18 joints for one image, in normalised image coordinates.

    Values are ``(x, y, visibility)`` in 0..1. Returns None when no body is
    found — an absent skeleton must not become a drawn guess.
    """
    import mediapipe as mp  # type: ignore
    import numpy as np
    from PIL import Image

    from .pose import _pose_model

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    res = _pose_model().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]
    pts = {name: (float(lm[i].x), float(lm[i].y), float(lm[i].visibility))
           for name, i in MEDIAPIPE_INDEX.items()}
    ls, rs = pts["l_shoulder"], pts["r_shoulder"]
    pts["neck"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2,
                   min(ls[2], rs[2]))
    lh, rh = pts["l_hip"], pts["r_hip"]
    pts["hip_c"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, min(lh[2], rh[2]))
    # The SOURCE aspect ratio travels with the points. Normalised coordinates
    # are relative to their own frame, so drawing them straight onto a canvas of
    # a different shape rescales x and y by different amounts and silently
    # changes the body's proportions — caught live: a 1280x720 driving frame
    # drawn onto 512x768 stretched the figure vertically about 2.7x, which would
    # have taught the generator a body nobody has.
    pts["__size__"] = (float(rgb.shape[1]), float(rgb.shape[0]), 1.0)
    return pts


def retarget(points: dict, proportions: dict, *,
             min_visibility: float = 0.5) -> dict:
    """Rescale bone lengths to the target's proportions, keeping directions.

    `proportions` is `metrics.body_metrics()["proportions"]` — 3D ratios in
    torso lengths, which is why they had to be measured in 3D: a projected
    ratio would carry the reference photo's camera angle into every frame.

    Directions come from the driving pose (that is the motion), lengths from the
    target (that is the body). Walking outward from the hips means a rescaled
    upper arm carries the forearm and hand with it, instead of detaching them.
    """
    import numpy as np

    if not proportions:
        return dict(points)
    neck, hip_c = points.get("neck"), points.get("hip_c")
    if neck is None or hip_c is None:
        return dict(points)
    torso = float(np.hypot(neck[0] - hip_c[0], neck[1] - hip_c[1]))
    if torso < 1e-6:
        return dict(points)

    out = {k: tuple(v) for k, v in points.items()}
    for parent, child in BONE_TREE:
        key = BONE_TO_PROPORTION.get((parent, child))
        if key is None or key not in proportions:
            continue
        p, c = out.get(parent), out.get(child)
        if p is None or c is None or c[2] < min_visibility:
            continue
        vec = np.array([c[0] - p[0], c[1] - p[1]])
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        target = float(proportions[key]) * torso
        shift = vec / length * (target - length)
        # Move the child and everything hanging off it, so the limb stays whole.
        for name in _descendants(child):
            if name in out:
                x, y, v = out[name]
                out[name] = (x + shift[0], y + shift[1], v)
    return out


def _descendants(joint: str) -> list:
    """`joint` and everything below it in BONE_TREE."""
    found = [joint]
    changed = True
    while changed:
        changed = False
        for parent, child in BONE_TREE:
            if parent in found and child not in found:
                found.append(child)
                changed = True
    return found


def _check_framing(framing: str) -> None:
    """Опечатка в имени кадрировки обязана падать, а не тихо давать полный рост.

    Стоило бы этому провалиться молча — и на демо приехали бы условия полного
    роста с манифестом, обещающим лицо по пояс. Это не гипотеза: `framing`
    приходит строкой из вызывающего кода и из CLI.
    """
    if framing not in FRAMINGS:
        raise ValueError(
            f"неизвестная кадрировка {framing!r}; бывают только {FRAMINGS}")


def _visible(points: dict, min_visibility: float = 0.5) -> dict:
    """{сустав: (x, y)} — только то, что видно. `__size__` не сустав."""
    return {k: (v[0], v[1]) for k, v in points.items()
            if k != "__size__" and v[2] >= min_visibility}


def _torso(vis: dict) -> float | None:
    """Длина торса neck->hip_c в тех же единицах, что и точки, или None.

    Единственный отрезок, на который здесь опирается вся арифметика лица: он
    длинный (значит, шум детектора по нему мал), он есть в обеих кадрировках, и
    ретаргет уже нормирует пропорции именно на него — то есть новых допущений
    не вводится. Плата известна: при сильном наклоне к камере торс укорачивается
    проекцией, и доля лица выходит ЗАНИЖЕННОЙ. Занижение здесь безопаснее
    завышения — оно ведёт к более крупной кадрировке, а не к слепому гейту.
    """
    import math

    n, h = vis.get("neck"), vis.get("hip_c")
    if n is None or h is None:
        return None
    t = math.hypot(n[0] - h[0], n[1] - h[1])
    return t if t > 1e-6 else None


def _upper_body_box(vis: dict, torso: float | None) -> tuple | None:
    """Габарит «по пояс»: от макушки до таза. None, если строить не из чего.

    Верх — самая высокая видимая точка верха тела, но не ниже МАКУШКИ, которой
    в COCO-18 нет и которая достраивается от носа (CROWN_ABOVE_NOSE). Руки над
    головой поднимают верх окна сами, потому что они в UPPER_BODY.

    Низ — строго линия таза. Запястье, висящее ниже бедра, на низ окна не
    влияет: иначе опущенные руки растянут окно до полного роста, и кадрировка
    молча перестанет работать ровно в тех кадрах, где она нужнее всего.
    """
    ups = {k: p for k, p in vis.items() if k in UPPER_BODY}
    hips = [vis[k][1] for k in ("l_hip", "r_hip", "hip_c") if k in vis]
    if not ups or not hips:
        return None
    xs = [p[0] for p in ups.values()]
    y0 = min(p[1] for p in ups.values())
    if torso and "nose" in vis:
        y0 = min(y0, vis["nose"][1] - CROWN_ABOVE_NOSE * torso)
    return (min(xs), max(xs), y0, max(hips))


def _window(points: dict, aspect: float, margin: float | None = None,
            framing: str = "full_body") -> tuple:
    """A crop window around the subject, in normalised coords, at `aspect`.

    Cropping rather than letterboxing, because the driving footage is usually
    landscape and the reel is vertical: letterboxing would leave the figure
    tiny in a band of black, wasting most of the conditioning resolution on
    nothing. The window is grown to the target aspect around the person, so the
    skeleton fills the frame AND keeps its proportions.

    `framing="waist_up"` строит окно по ВЕРХНЕЙ части тела (см.
    `_upper_body_box`). Если таза не видно, окно строится по всей фигуре —
    молча, потому что рисунок скелета обязан получиться в любом случае; о том,
    что доля лица при этом неизвестна, сообщает отдельно `face_share`, и оно же
    попадает в манифест как ТРЕТИЙ исход, а не как «всё хорошо».
    """
    vis = _visible(points)
    if not vis:
        return (0.0, 0.0, 1.0, 1.0)
    box = None
    if framing == "waist_up":
        box = _upper_body_box(vis, _torso(vis))
        if margin is None:
            margin = WAIST_UP_MARGIN
    if margin is None:
        margin = FULL_BODY_MARGIN
    waist = box is not None
    if box is None:
        xs, ys = [p[0] for p in vis.values()], [p[1] for p in vis.values()]
        box = (min(xs), max(xs), min(ys), max(ys))
    x0, x1, y0, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, 1e-3) * (1 + margin), max(y1 - y0, 1e-3) * (1 + margin)
    if waist:
        # ВЫСОТА окна «по пояс» задаётся телом и больше ничем: ширина следует за
        # ней по соотношению сторон, а что не влезло по бокам — обрезается.
        #
        # ПОЙМАНО ТЕСТОМ, а не рассуждением. По общему правилу (растить короткую
        # сторону до нужного соотношения) разведённые в стороны руки делали окно
        # шире торса, оно росло в высоту, и выигрыш кадрировки падал с 2.0x до
        # 1.08x — то есть кадрировка «по пояс» переставала быть кадрировкой
        # ровно там, где поза шире. На вертикальном холсте вместить размах рук и
        # оставить лицо крупным нельзя: это не выбор реализации, а арифметика.
        # Выбор один — что резать, и режется размах, потому что кадрировка
        # существует ради размера лица. Кисть, уходящую за край, ControlNet
        # читает как «продолжается за кадром»; лицо мельче 100 px читать нечем.
        w = h * aspect
    elif w / h > aspect:
        # Grow the short side until the window matches the target aspect.
        h = w / aspect
    else:
        w = h * aspect
    return (cx - w / 2, cy - h / 2, w, h)


def face_share(points: dict, *, width: int = 512, height: int = 768,
               framing: str = "full_body",
               min_visibility: float = 0.5) -> float | None:
    """Ожидаемая доля ВЫСОТЫ кадра, которую займёт лицо. None — не измерили.

    Ради этого числа кадрировка и добавлена. Порог идентичности — это на самом
    деле требование к РАЗМЕРУ ЛИЦА В ПИКСЕЛЯХ: ArcFace судит видео от 100 px и
    старт-кадр от 70 (`identity_arcface.MIN_FACE_PX`, `START_MIN_FACE_PX`, оба
    ИЗМЕРЕНЫ). Умножив эту долю на высоту холста, вызывающий узнаёт ДО
    генерации, будет ли гейту что судить, — вместо того чтобы выяснять это
    после потраченных GPU-минут и получать вердикт «не смог».

    Считается по геометрии ЭТОГО скелета в ЭТОМ окне, а не берётся из таблицы:
    широкая стойка растягивает габарит и роняет долю (на живом kit-кадре 0.074
    вместо 0.094), и знать надо фактическое число, а не типовое.

    None возвращается, когда не видно шеи или таза: длину лица тогда не от чего
    отсчитывать. Это ТРЕТИЙ исход — «не смогли измерить», — и он не должен
    сливаться ни с крупным лицом, ни с мелким.

    Размер холста входит сюда только через соотношение сторон: доля от высоты
    от числа пикселей не зависит, зависит форма окна. Это проверяется тестом.

    `min_visibility` управляет измерением лица, но не построением окна: окно
    строится ровно так же, как в `draw`, — иначе посчитанное число относилось бы
    к кадру, которого не будет.
    """
    _check_framing(framing)
    src_w, src_h, _ = points.get("__size__", (float(width), float(height), 1.0))
    # В ИСХОДНЫХ пикселях, иначе x и y живут в разных масштабах — та самая
    # ошибка, из-за которой фигура вытягивалась в 2.7 раза.
    px = {k: (v[0] * src_w, v[1] * src_h, v[2])
          for k, v in points.items() if k != "__size__"}
    vis = _visible(px, min_visibility)
    torso = _torso(vis)
    if torso is None:
        return None
    if framing == "waist_up" and _upper_body_box(vis, torso) is None:
        return None
    win_h = _window(px, aspect=width / height, framing=framing)[3]
    if win_h <= 0:
        return None
    return FACE_TO_TORSO * torso / win_h


def draw(points: dict, out_path: str | Path, *, width: int = 512,
         height: int = 768, min_visibility: float = 0.5,
         line_width: int = 4, framing: str = "full_body") -> str:
    """Draw the OpenPose skeleton on black — the ControlNet condition image.

    Joints below `min_visibility` are omitted along with their limbs rather than
    drawn at a guessed position: a confidently drawn wrong limb conditions the
    generator into a wrong pose, which is worse than leaving it unconstrained.

    Proportions are preserved by mapping through a crop window of the SOURCE
    aspect (see `_window`), not by stretching normalised coordinates onto the
    canvas — that stretch is a real bug this function used to have. Кадрировка
    на это не влияет и влиять не должна: окно меняет ГРАНИЦЫ, а не масштабы по
    осям, поэтому «по пояс» обязано сохранять пропорции ровно так же, как
    полный рост, — и на это есть свой тест.

    Умолчание `full_body` оставлено намеренно: смена кадрировки — решение
    вызывающего, а не побочный эффект обновления этого файла. Чем это решение
    обосновать, лежит в манифесте (`face_share`, `face_px`).
    """
    from PIL import Image, ImageDraw

    _check_framing(framing)
    img = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    src_w, src_h, _ = points.get("__size__", (float(width), float(height), 1.0))
    # Work in source PIXELS so x and y share one scale, then fit the window.
    win = _window({k: (v[0] * src_w, v[1] * src_h, v[2])
                   for k, v in points.items() if k != "__size__"},
                  aspect=width / height, framing=framing)
    wx, wy, ww, wh = win
    scale = width / ww if ww > 0 else 1.0

    def xy(name):
        p = points.get(name)
        if p is None or name == "__size__" or p[2] < min_visibility:
            return None
        return ((p[0] * src_w - wx) * scale, (p[1] * src_h - wy) * scale)

    for (a, b), colour in zip(LIMBS18, LIMB_COLOURS):
        pa, pb = xy(COCO18[a]), xy(COCO18[b])
        if pa and pb:
            d.line([pa, pb], fill=colour, width=line_width)
    for i, name in enumerate(COCO18):
        p = xy(name)
        if p:
            r = line_width
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                      fill=JOINT_COLOURS[i])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def _extractor(source=None):
    """Чем снимать скелет. Возвращает (функция, имя для манифеста).

    ЗДЕСЬ БЫЛО ДВА ФЛАГА ПРО ОДНО И ТО ЖЕ, И ОНИ МОГЛИ РАСХОДИТЬСЯ. `source`
    выбирал экстрактор, а `from_mediapipe` — подпись в манифесте, независимо от
    того, кто отработал на самом деле. Манифест лежит рядом с условиями и
    читается на другой машине через несколько часов; подпись, способная соврать
    о происхождении условий, хуже отсутствующей.

    Теперь имя выводится из того, что ДЕЙСТВИТЕЛЬНО исполнилось.

    Умолчание — DWPose, и это не вкусовщина. Условия и проверка обязаны идти от
    РАЗНЫХ моделей: если экстрактор ошибся, а проверяет его он же, гейт
    подтвердит собственную ошибку. Замерено на живом кадре — два экстрактора
    расходятся на медиану 0.0121 нормированной длины (худший сустав, ухо,
    0.0280) при баре гейта по позе 0.25, то есть в двадцать раз ниже порога:
    подмена экстрактора калибровку не ломает.
    """
    if source is not None:
        name = getattr(source, "__module__", "") or ""
        return source, (name.rsplit(".", 1)[-1] or "custom")
    from . import dwpose

    if not dwpose.why_unavailable():
        return dwpose.pose_points, "dwpose"
    return pose_points, "mediapipe"


def render_sequence(frames: list, out_dir: str | Path, *,
                    proportions: dict | None = None, width: int = 512,
                    height: int = 768, source=None,
                    framing: str = "full_body") -> dict:
    """A whole driving segment -> a folder of condition images.

    Returns the manifest a GPU run consumes, including `coverage` — the share of
    frames that produced a skeleton. A gap in the middle of a sequence means the
    generator gets no constraint for those frames, so it is reported rather than
    silently skipped.

    Скелет снимается DWPose, если его веса на месте, и MediaPipe иначе — с
    громким предупреждением в манифесте, потому что MediaPipe здесь ещё и
    проверяющий. Чем именно сняли, записано в `source` по факту исполнения.

    `framing` выбирает кадрировку, а манифест несёт ожидаемую долю лица для
    ОБЕИХ — чтобы решение «хватит ли лица гейту» принималось до генерации и по
    числу. Умолчание `full_body` сохраняет прежнее поведение; на потолке
    512x768 оно почти наверняка даст непроверяемую идентичность, и манифест
    скажет об этом прямо, вместе с тем, что дала бы кадрировка по пояс.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made, missing = [], []
    drawn_joints = []
    # Доли лица собираются по КАЖДОМУ кадру и по обеим кадрировкам: движение
    # меняет габарит фигуры (шаг в сторону — и лицо в кадре мельче), поэтому
    # осмысленна медиана по последовательности, а не число с первого кадра.
    shares: dict = {f: [] for f in FRAMINGS}
    unmeasured = 0
    # Какой исходный кадр стоит за каждым условием — собирается здесь же,
    # иначе после пропусков соответствие уже не восстановить.
    driving: dict = {}
    extract, source_name = _extractor(source)
    for i, f in enumerate(frames):
        pts = extract(f)
        if pts is None:
            missing.append(i)
            continue
        if proportions:
            pts = retarget(pts, proportions)
        # Сколько суставов реально попало в условие. Кадр со скелетом ещё не
        # означает скелет ЦЕЛИКОМ: невидимый локоть выбрасывает вместе с собой
        # предплечье, и в этом кадре рука ничем не ограничена. Поймано глазами
        # на живом референсе, где `coverage` показывал 1.0, а у фигуры не было
        # одной руки. Кадр — единица слишком крупная, чтобы это заметить.
        drawn_joints.append(sum(1 for n in COCO18
                                if pts.get(n) and pts[n][2] >= 0.5))
        for name in FRAMINGS:
            s = face_share(pts, width=width, height=height, framing=name)
            if s is not None:
                shares[name].append(s)
            elif name == framing:
                unmeasured += 1
        made.append(draw(pts, out_dir / f"{i:04d}.png",
                         width=width, height=height, framing=framing))
        driving[f"{i:04d}"] = str(f)
    coverage = round(len(made) / len(frames), 3) if frames else 0.0
    joint_cover = (round(sum(drawn_joints) / (len(drawn_joints) * len(COCO18)), 3)
                   if drawn_joints else 0.0)
    partial = sum(1 for n in drawn_joints if n < len(COCO18) - 2)
    import statistics

    by_framing = {f: (round(statistics.median(v), 4) if v else None)
                  for f, v in shares.items()}
    share = by_framing[framing]
    # px, а не доля, — потому что порог ArcFace задан в пикселях лица, и
    # переводить его туда-сюда должен один раз код, а не читатель манифеста.
    face_px = round(share * height, 1) if share is not None else None
    from .identity_arcface import MIN_FACE_PX, START_MIN_FACE_PX

    # None, а не False: «не смогли измерить» — отдельный исход. False здесь
    # означал бы «лицо точно мелкое», а мы этого не знаем.
    judgeable = None if face_px is None else bool(face_px >= MIN_FACE_PX)
    manifest = {
        "conditions": made, "missing_frames": missing, "coverage": coverage,
        "joint_coverage": joint_cover, "partial_frames": partial,
        "size": [width, height], "retargeted": bool(proportions),
        "source": source_name,
        # Кадрировка и её цена в пикселях лица. `face_share_by_framing` — обе
        # кадрировки сразу: вызывающий выбирает, ещё не заплатив за генерацию.
        "framing": framing,
        "face_share": share,
        "face_px": face_px,
        "face_share_by_framing": by_framing,
        "face_unmeasured_frames": unmeasured,
        "identity_judgeable": judgeable,
        # Какой исходный кадр стоит за каждым условием. Без этой карты условия
        # — набор палок на чёрном фоне, по которому уже не восстановить, ЧТО
        # они кодируют. А восстанавливать нужно: сгенерированный кейфрейм
        # сверяется по позе не с условием (детектор поз на рисунке скелета
        # ничего не находит — проверено), а с тем самым driving-кадром.
        "driving_frames": driving,
        "warnings": [],
    }
    if partial:
        manifest["warnings"].append(
            f"{partial} of {len(made)} condition(s) are missing more than two "
            f"joints ({joint_cover:.0%} of joints drawn overall): those limbs "
            f"are unconstrained in those frames, and the generator will invent "
            f"them.")
    if source_name == "mediapipe":
        from . import dwpose

        manifest["warnings"].append(
            "conditions were rendered from MediaPipe, which is also the "
            "verifier: the gate can no longer catch this extractor's own "
            "errors, because judge and judged are the same model. "
            + dwpose.why_unavailable())
    if coverage < 1.0:
        manifest["warnings"].append(
            f"{len(missing)} frame(s) produced no skeleton: the generator is "
            f"unconstrained there.")
    if not proportions:
        manifest["warnings"].append(
            "not retargeted: these are the DRIVING person's proportions, so "
            "the output will carry their body, not the target's.")
    # ЛИЦО. Три исхода, и они не сводятся к двум.
    if made and share is None:
        manifest["warnings"].append(
            f"долю лица НЕ УДАЛОСЬ ИЗМЕРИТЬ ни на одном из {len(made)} "
            f"условий (не видно шеи или таза, а от них отсчитывается размер "
            f"лица). Это третий исход: не «лицо крупное» и не «мелкое». "
            f"Планировать проверку идентичности на этом прогоне не на чем.")
    elif unmeasured:
        manifest["warnings"].append(
            f"на {unmeasured} из {len(made)} условий долю лица измерить не "
            f"удалось; медиана взята по остальным.")
    if share is not None and not judgeable:
        fix = []
        other = "waist_up" if framing == "full_body" else "full_body"
        if by_framing[other] is not None and by_framing[other] > share:
            fix.append(f"кадрировка {other} дала бы ~"
                       f"{by_framing[other] * height:.0f} px")
        fix.append(f"высота холста от {int(-(-MIN_FACE_PX // share))} px "
                   f"даёт judgeable при текущей кадрировке")
        manifest["warnings"].append(
            f"кадрировка {framing} на {width}x{height} даёт лицо ~"
            f"{face_px:.0f} px, а ArcFace судит видео от {MIN_FACE_PX} px "
            f"(старт-кадр от {START_MIN_FACE_PX}): ГЕЙТ ПО ИДЕНТИЧНОСТИ "
            f"ОСЛЕПНЕТ и вернёт «не смог», а не «похож». Лечится: "
            + "; ".join(fix) + ".")
    if share is not None:
        from .animate import FULL_BODY_FACE_SHARE, WAIST_UP_FACE_SHARE

        measured = {"full_body": FULL_BODY_FACE_SHARE,
                    "waist_up": WAIST_UP_FACE_SHARE}.get(framing)
        if measured and abs(share - measured) / measured > FACE_SHARE_TOLERANCE:
            manifest["warnings"].append(
                f"геометрия даёт долю лица {share:.3f}, а живьём для "
                f"{framing} ИЗМЕРЕНО {measured}: расхождение "
                f"{abs(share - measured) / measured:.0%}. Обычная причина — "
                f"поза: широкая стойка или наклон растягивают габарит фигуры "
                f"и роняют долю. Планировать по меньшему из двух и посмотреть "
                f"на условия глазами.")
    # Манифест кладётся РЯДОМ С УСЛОВИЯМИ, а не только возвращается. Условия
    # рендерятся дома, а используются на другой машине через несколько часов;
    # всё, что осталось в возвращённом словаре, к тому моменту потеряно, и
    # папка с png перестаёт объяснять саму себя. Здесь же лежат предупреждения,
    # ради которых её и стоит открыть.
    import json

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
