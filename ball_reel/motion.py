"""Whether the clip MOVES well: does it loop, and is the motion physical.

Identity asks "is this the same person". These ask the other two questions a
reel actually fails on — it cuts visibly when it repeats, or the motion is
generator soup: a limb teleports, the body morphs between frames, the subject
freezes for half the clip.

Both measures here are RATIOS against the clip's own motion, not absolute pixel
thresholds. That matters: a calm clip and a violent one have completely
different frame-to-frame magnitudes, so any fixed number would be tuned to one
and wrong for the other. Dividing by the clip's own median step makes the same
threshold mean the same thing across clips.

numpy only — no model, no network. The arithmetic is unit-tested on synthetic
sequences, so a sceptic can recompute every number here.
"""

from __future__ import annotations

from pathlib import Path

#: A loop is seamless when the first->last step is no larger than this multiple
#: of an ordinary step inside the clip. Measured live on seedance-2.0: passing
#: the start frame as BOTH keyframes gave 0.09, while the same prompt with only
#: a start frame gave 0.61 (a visible jump). 0.30 sits between those, closer to
#: the good one, so it accepts a real loop and rejects a merely-similar ending.
SEAMLESS_MAX = 0.30

#: A single step this many times the median step is a discontinuity, not motion
#: — the mark of a teleport or a morph rather than a body moving.
JUMP_MAX = 4.0

#: Below this, in the same normalised units, nothing is happening.
STILL_MIN = 0.15


def _gray(path: str | Path, side: int = 96):
    """A small grayscale array. Downscaled so the measure tracks BODY movement
    rather than sensor noise and compression shimmer."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        small = im.convert("L").resize((side, side), Image.BILINEAR)
    return np.asarray(small, dtype="float64")


def _steps(frames: list[str]) -> list[float]:
    """Mean absolute difference between each adjacent pair of frames."""
    import numpy as np

    arrs = [_gray(f) for f in frames]
    return [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]


def loop_seam(frames: list[str]) -> dict:
    """How visible the cut is when the clip repeats.

    ``ratio`` is the first->last difference over the MEDIAN adjacent step. Below
    1.0 the seam is smaller than an ordinary frame transition, i.e. the repeat
    is less visible than the motion already on screen.

    The median, not the mean, sets the scale: one morph artefact would inflate a
    mean and quietly make a bad loop look acceptable.
    """
    import numpy as np

    if len(frames) < 3:
        return {"ratio": None, "seam": None, "typical_step": None,
                "seamless": False, "note": "need at least 3 frames to judge a loop."}
    steps = _steps(frames)
    typical = float(np.median(steps))
    seam = float(np.abs(_gray(frames[-1]) - _gray(frames[0])).mean())
    if typical == 0:
        return {"ratio": None, "seam": round(seam, 3), "typical_step": 0.0,
                "seamless": False, "note": "the clip does not move at all."}
    ratio = seam / typical
    return {"ratio": round(ratio, 3), "seam": round(seam, 3),
            "typical_step": round(typical, 3),
            "seamless": ratio <= SEAMLESS_MAX,
            "note": (f"loop seam {ratio:.2f}x a typical frame step "
                     f"({'seamless' if ratio <= SEAMLESS_MAX else 'visible cut on repeat'}; "
                     f"bar {SEAMLESS_MAX}).")}


def motion_quality(frames: list[str]) -> dict:
    """Is the movement continuous and physical, or does it jump and morph.

    Returns ``worst_jump`` (largest step over the median), ``jumps`` (their
    indices), ``moving`` (there is motion at all) and ``smooth``. A generator
    that loses the thread produces one enormous step between two frames while
    the rest are ordinary — exactly what the ratio exposes.
    """
    import numpy as np

    if len(frames) < 3:
        return {"worst_jump": None, "jumps": [], "moving": False, "smooth": False,
                "activity": None, "note": "need at least 3 frames to judge motion."}
    steps = _steps(frames)
    typical = float(np.median(steps))
    if typical == 0:
        return {"worst_jump": None, "jumps": [], "moving": False, "smooth": False,
                "activity": 0.0, "note": "static clip: nothing moves."}
    ratios = [s / typical for s in steps]
    worst = max(ratios)
    jumps = [i for i, r in enumerate(ratios) if r > JUMP_MAX]
    # Activity is the typical step against the frame's own brightness scale, so
    # "nothing happens" is separated from "the camera is just dark".
    activity = typical / max(float(np.mean(_gray(frames[0]))), 1.0)
    moving = activity >= STILL_MIN / 10
    return {"worst_jump": round(worst, 3), "jumps": jumps, "moving": moving,
            "smooth": not jumps, "activity": round(activity, 4),
            "note": (f"largest frame step {worst:.1f}x the median"
                     + (f"; {len(jumps)} discontinuity(ies) at {jumps} — "
                        f"limbs teleport or the body morphs there"
                        if jumps else "; motion is continuous") + ".")}


def best_loop_window_pose(points: list, *, size: int, stride: int = 1) -> dict:
    """Замкнутость окна ПО СКЕЛЕТУ, без порога. Лучше пиксельной по двум причинам.

    ПОЧЕМУ СКЕЛЕТ, А НЕ ПИКСЕЛИ. Пиксельная разность меряет всё сразу: свет
    мигнул, фон дрогнул, компрессия дала другой шум — и кадр «не сошёлся», хотя
    человек вернулся ровно в ту же позу. Скелет от этого свободен: двенадцать
    точек в единицах длины торса. Измерено на нашем исходнике — поза нашлась на
    96 кадрах из 96, тогда как пиксельная мера считалась по всему, что попало в
    кадр.

    И главное: ControlNet потребляет ИМЕННО скелет. Замкнув его, мы замыкаем то,
    что доедет до генерации; пиксели финального рендера рисуются заново, и их
    расхождение к движению отношения не имеет.

    ПОЧЕМУ БЕЗ ПОРОГА, И ЭТО НЕ ЛЕНЬ. Порог пробовали вывести тем же способом,
    которым калибровался отбор FaceNet, — два облака и граница в разрыве.
    Облака ПЕРЕКРЫЛИСЬ, и отказ информативнее числа:

        соседние кадры   медиана 0.0530  p95 0.3985  max 0.5148
        далёкие кадры    min 0.0437      медиана 0.6080

    Минимум «далёких» МЕНЬШЕ медианы «соседних»: нашлись кадры в четверти
    ролика друг от друга и при этом ближе по позе, чем два соседних. Причина не
    в стенде, а в материале — упражнение циклично, и время не есть мера
    различия поз. Тот же факт есть доказательство, что луп вообще существует.

    Глобальный порог здесь не выводится в принципе: «далеко по позе» и есть то,
    что мы определяем, круг замыкается. Поэтому критерий ЛОКАЛЬНЫЙ: стык
    незаметен, если он не выбивается из ритма СОСЕДНИХ переходов. Зритель видит
    шов на фоне того, что происходит рядом с ним, а не на фоне среднего темпа
    всего ролика — и это различие измерено:

        окно 35..50  стык 0.0357, локальная медиана 0.0305 -> мельче  6 из 15
        окно 52..67  стык 0.0695, локальная медиана 0.1048 -> мельче 11 из 15

    Первое побеждало по глобальной мере и оказалось медленным участком, где
    маленький стык мал просто потому, что там всё мелкое. Второе — быстрый
    участок, где шов спрятан движением. Видно будет второе, а не первое.

    `points` — список поз (`pose.landmarks`), по одной на кадр; None там, где
    тела не нашли. Возвращает индекс начала и числа, по которым выбирали.
    """
    span = (size - 1) * stride + 1
    if size < 3 or stride < 1 or len(points) < span:
        return {"start": 0, "hidden": None, "seam": None,
                "note": (f"поз {len(points)}, а окно требует {span} — "
                         f"выбирать не из чего")}
    from .pose import pose_delta

    def gap(i: int, j: int):
        if not points[i] or not points[j]:
            return None
        got = pose_delta(points[i], points[j])
        return got["mean"] if got else None

    rows = []
    for s in range(len(points) - span + 1):
        seam = gap(s, s + span - 1)
        if seam is None:
            continue
        steps = [x for x in (gap(i, i + stride)
                             for i in range(s, s + span - stride, stride))
                 if x is not None]
        if len(steps) < 2:
            continue
        # СКОЛЬКО РЯДОВЫХ ШАГОВ НЕ МЕНЬШЕ СТЫКА — это и есть мера спрятанности.
        # Доля, а не разность: она не зависит ни от темпа, ни от длины окна, и
        # потому переносится на чужой драйвинг без перекалибровки.
        hidden = sum(1 for x in steps if x >= seam) / len(steps)
        rows.append({"start": s, "seam": round(seam, 4),
                     "hidden": round(hidden, 3), "steps": len(steps),
                     "local_median": round(sorted(steps)[len(steps) // 2], 4)})
    if not rows:
        return {"start": 0, "hidden": None, "seam": None,
                "note": "скелет не найден на нужных кадрах — судить нечем"}
    # Спрятанность важнее абсолютной величины стыка: см. разбор в докстринге.
    best = max(rows, key=lambda r: (r["hidden"], -r["seam"]))
    return {**best, "candidates": len(rows),
            "note": (f"лучшее окно {best['start']}..{best['start'] + span - 1} "
                     f"из {len(rows)}: стык {best['seam']} при локальной "
                     f"медиане {best['local_median']}, то есть мельче "
                     f"{round(best['hidden'] * best['steps'])} рядовых шагов "
                     f"из {best['steps']}. Порога здесь НЕТ намеренно: облака "
                     f"перекрылись, и сравнение идёт с самим материалом")}


def best_loop_window(frames: list[str], *, size: int, stride: int = 1) -> dict:
    """Какое ОКНО исходника замыкается лучше всех. Выбор до генерации, не после.

    ЗАЧЕМ ОТДЕЛЬНО ОТ `best_loop_cut`. Обрезка работает с уже отрисованным
    клипом и умеет только отбросить хвост — значит она ограничена тем, что
    внутри выбранного окна вообще есть кадр, похожий на первый. ИЗМЕРЕНО, что
    это ограничение бывает непреодолимым: на нашем драйвинге текущее окно даёт
    стык 4.374, а лучшая обрезка внутри него — 3.742 при баре 0.30. Движение за
    эти 16 кадров никуда не возвращается, и резать нечего.

    А исходник длиннее окна: 192 кадра против 16. Перебор всех окон на тех же
    данных даёт стык 1.245 (окно 66..81) против 4.374 у текущего — В 3.5 РАЗА
    лучше и БЕСПЛАТНО: та же генерация, другой отрезок. Бар 0.30 не берёт и оно,
    потому что движение в этом видео нециклично в принципе, — но выбирать окно
    наугад, имея замер, незачем.

    Порядок поэтому такой: сначала ВЫБРАТЬ окно (здесь), потом генерировать,
    потом при нужде подрезать (`best_loop_cut`). Обратный порядок оплачивает
    31 минуту генерации за отрезок, который заведомо хуже соседнего.

    Стоит один декод исходника и ноль генераций.
    """
    import numpy as np

    span = (size - 1) * stride + 1
    if size < 2 or stride < 1 or len(frames) < span:
        return {"start": 0, "ratio": None, "seamless": False,
                "note": (f"кадров {len(frames)}, а окно требует {span} — "
                         f"выбирать не из чего")}
    arrs = [_gray(f) for f in frames]
    steps = [float(np.abs(arrs[i + 1] - arrs[i]).mean())
             for i in range(len(arrs) - 1)]
    typical = float(np.median(steps)) or 1.0
    # Стык окна — расстояние между его ПОСЛЕДНИМ и ПЕРВЫМ кадром, в единицах
    # обычного шага. Та же шкала, что у `loop_seam`, иначе числа «до» и «после»
    # выбора окна нельзя сравнивать.
    scored = [(float(np.abs(arrs[s + span - 1] - arrs[s]).mean()) / typical, s)
              for s in range(len(frames) - span + 1)]
    ratio, start = min(scored)
    worst = max(scored)[0]
    return {"start": start, "ratio": round(ratio, 3),
            "seamless": ratio <= SEAMLESS_MAX,
            "worst_ratio": round(worst, 3), "candidates": len(scored),
            "note": (f"лучшее окно {start}..{start + span - 1} из "
                     f"{len(scored)} возможных: стык {ratio:.3f} при баре "
                     f"{SEAMLESS_MAX} (худшее окно дало бы {worst:.3f})"
                     + ("" if ratio <= SEAMLESS_MAX else
                        " — бар не взят: движение исходника нециклично, и "
                        "выбор окна это улучшает, а не чинит"))}


def best_loop_cut(frames: list[str], *, min_keep: float = 0.5) -> dict:
    """Find where to cut so the clip loops, without generating anything new.

    An end-frame keyframe is the cheap way to get a loop, but only models that
    DECLARE `end_frame` accept one — the rest ignore the second keyframe
    silently (measured: veo, which declares it, closed at 0.17; wan, which does
    not, came back at 1.71). When no end frame is available, the clip usually
    still PASSES THROUGH a pose close to its opening one — so the loop becomes a
    trimming problem rather than a generation one.

    Scans candidate end frames in the last ``1 - min_keep`` of the clip and
    returns the one closest to frame 0, with the seam ratio it achieves. Costs
    one decode, no tokens, and cannot make the clip worse: if nothing beats the
    original ending, ``cut_at`` is the last frame.
    """
    import numpy as np

    if len(frames) < 4:
        return {"cut_at": len(frames) - 1, "ratio": None, "seamless": False,
                "note": "too few frames to search for a loop point."}
    arrs = [_gray(f) for f in frames]
    steps = [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]
    typical = float(np.median(steps)) or 1.0
    first = int(len(frames) * min_keep)
    scores = {i: float(np.abs(arrs[i] - arrs[0]).mean()) / typical
              for i in range(max(first, 2), len(frames))}
    cut = min(scores, key=lambda i: scores[i])
    ratio = round(scores[cut], 3)
    kept = (cut + 1) / len(frames)
    return {"cut_at": cut, "ratio": ratio, "seamless": ratio <= SEAMLESS_MAX,
            "kept_fraction": round(kept, 3),
            "note": (f"best loop point is frame {cut}/{len(frames) - 1} "
                     f"(seam {ratio:.2f}x a typical step, keeps {kept:.0%} of the "
                     f"clip){'' if ratio <= SEAMLESS_MAX else ' — still visible'}.")}


def trim_to_loop(mp4_path: str | Path, frames: list[str], out_mp4: str | Path,
                 *, fps: int) -> dict:
    """Cut an mp4 at its best loop point with ffmpeg. Returns the cut report.

    ``fps`` must be the rate the frames were extracted at, since the frame index
    is converted back to a timestamp with it.
    """
    import subprocess

    cut = best_loop_cut(frames)
    if cut["ratio"] is None:
        return cut
    duration = (cut["cut_at"] + 1) / float(fps)
    subprocess.run(["ffmpeg", "-y", "-i", str(mp4_path), "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)],
                   check=True, capture_output=True)
    return {**cut, "out": str(out_mp4), "duration": round(duration, 3)}


#: Motion wording that describes the PHYSICS rather than the vibe. Video models
#: invent floaty, weightless bouncing when the prompt only names the action;
#: naming contact, compression and weight transfer is what produces a bounce
#: that reads as a real body on a real ball.
PHYSICAL_MOTION = (
    "The ball compresses under their weight and rebounds, driving the bounce. "
    "Their feet stay in contact with the ball, knees absorb the landing, arms "
    "counterbalance. Real weight and momentum, continuous single take, no cuts, "
    "no camera move."
)

#: Appended when the clip has to loop. The end-frame keyframe does the actual
#: work (see pollinations.video_loop); this stops the model from getting there
#: by freezing or fading, which technically matches the frame and looks dead.
LOOP_MOTION = (
    "One complete bounce cycle that ends exactly where it began, so the clip "
    "repeats seamlessly. Keep moving through the final frame — do not slow to a "
    "stop, freeze or fade."
)
