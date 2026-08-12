"""Сколько времени стоит каждый этап — и что из этого влезает в бюджет.

Счётчик появился раньше ручки, и это не формальность: заказчик назвал своей
болью 10 мс, а в проекте до сих пор не было НИ ОДНОГО замера времени. Спорить
о латентности, не измеряя её, — то же самое, что спрашивать генератор, хорошо
ли он сгенерировал.

ТРИ РЕШЕНИЯ, КОТОРЫЕ ЗДЕСЬ ЗАФИКСИРОВАНЫ.

**Медиана и p95, а не среднее.** Ровно та же причина, по которой гейт судит
личность по медиане: среднее прячет хвост, а латентность живёт в хвосте.
Пользователь не чувствует среднего времени отклика — он чувствует худшие
случаи. Поэтому `summary()` отдаёт p50 и p95, а не mean.

**Холодный старт считается отдельно.** Первый вызов MediaPipe или ArcFace
включает загрузку модели и медленнее последующих на порядки. Смешать его с
остальными — значит получить число, которое не описывает ни одно реальное
состояние системы: ни прогрев, ни работу. Поэтому первый вызов каждого этапа
уходит в `cold_ms` и не участвует в квантилях.

**У каждого этапа объявлено, на что он тратится: на прогон, на кадр или на
кейфрейм.** Без этого профиль нечитаем: «rendering 40 c» — это много или мало?
Смотря 40 секунд на весь прогон или на каждый из 71 кадра. И главное — только
это позволяет отличить ПРЕДПОДСЧЁТ от работы НА ЗАПРОС, а весь разговор про
10 мс именно об этом: то, что посчитано заранее, латентности не стоит вообще.

Ничего не генерирует, в сеть не ходит, денег не тратит. Работает на monotonic —
системные часы могут прыгнуть назад, и тогда этап займёт минус три миллисекунды.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

#: Бюджет реального времени, миллисекунды. Названная заказчиком боль, взятая
#: как порог по умолчанию. Это ВЫБРАННОЕ число, не измеренное: оно пришло из
#: разговора, а не из профиля, и держится здесь как цель, с которой сравнивают.
REALTIME_BUDGET_MS = 10.0

#: Во сколько раз этап должен промахнуться мимо бюджета, чтобы считаться
#: безнадёжным для онлайна, а не «почти влезающим». Десятикратный промах не
#: чинится настройками — он чинится другой архитектурой (предподсчётом,
#: дистилляцией, другой моделью), и путать эти два случая дорого: на первый
#: имеет смысл потратить день оптимизации, на второй — нет.
HOPELESS_FACTOR = 10.0

#: Сколько замеров нужно, чтобы p95 вообще что-то значил.
#:
#: Число не из головы, оно следует из арифметики самой квантили. p95 при n
#: замерах берётся в позиции 0.95*(n-1), то есть интерполирует между двумя
#: верхними значениями, и вес САМОГО МЕДЛЕННОГО замера в ответе равен дробной
#: части этой позиции. При n=5 он весит 80%, при n=10 — 55%, и только при n=20
#: падает до 5%, то есть до той доли, которую p95 и обязан описывать.
#:
#: Проверено на синтетике: [4.0]*4 + [40.0] даёт p95 = 32.8 при реальной
#: медиане 4. По такому числу этап уезжает в «безнадёжно» и отправляет
#: переделывать архитектуру там, где просто мало замеров. Поэтому ниже порога
#: мы не говорим «медленно» — мы говорим «судить рано».
MIN_TAIL_SAMPLES = 20

#: Полоса вокруг бюджета, внутри которой вердикт «влезает/не влезает» не
#: заслуживает доверия. ±20% — не вкус, а наблюдение: `skeleton.draw` дал p95
#: 9.90 мс в нашем профиле и 10.49 мс в независимом замере на той же машине,
#: то есть два честных прогона легли по РАЗНЫЕ стороны черты. Метрика, которая
#: меняет вердикт от прогона к прогону, обязана сказать «на границе», а не
#: выдать тот ответ, который выпал сегодня. Шум виртуалки легко даёт эти 20%.
BORDERLINE_BAND = 0.2

#: Единицы, в которых этап тратит время. Строки, а не enum: они попадают
#: в JSON-отчёт и читаются человеком.
PER_RUN, PER_FRAME, PER_KEYFRAME = "run", "frame", "keyframe"


class Timings:
    """Собиратель замеров. Один на прогон.

    Используется как контекстный менеджер на каждом этапе:

        t = Timings()
        with t.stage("pose", per=PER_FRAME):
            landmarks(path)

    Замер снимается даже если внутри блока произошло исключение: этап, который
    падает через десять секунд, стоит этих десяти секунд, и прятать их — значит
    описывать не ту систему, которая работает.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list] = {}
        self._per: dict[str, str] = {}
        self._cold: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, *, per: str = PER_RUN):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0, per=per)

    def record(self, name: str, ms: float, *, per: str = PER_RUN) -> None:
        """Записать замер вручную — когда время пришло со стороны.

        Нужно, например, для этапов, которые считает не наш код: длительность
        вызова шлюза приходит из его же заголовков.
        """
        self._per.setdefault(name, per)
        if name not in self._cold:
            # Первый вызов — это загрузка модели, а не работа этапа.
            self._cold[name] = ms
            self._samples[name] = []
            return
        self._samples[name].append(ms)

    def summary(self) -> dict:
        """Профиль по этапам. Пустой этап (был только холодный старт) сохраняет
        свои calls=0 и None вместо квантилей, а не притворяется нулём."""
        out = {}
        for name, samples in self._samples.items():
            out[name] = {
                "per": self._per[name],
                "calls": len(samples),
                "cold_ms": round(self._cold[name], 3),
                "p50_ms": _quantile(samples, 0.5),
                "p95_ms": _quantile(samples, 0.95),
                "total_ms": round(sum(samples), 3) if samples else 0.0,
            }
        return out


def _quantile(values: list, q: float):
    """Квантиль с линейной интерполяцией. None на пустом входе.

    None, а не ноль: «замеров не было» и «занимает ноль миллисекунд» — разные
    утверждения, и второе неправда никогда.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return round(ordered[low] * (1 - frac) + ordered[high] * frac, 3)


def latency_verdict(summary: dict, *, budget_ms: float = REALTIME_BUDGET_MS,
                    hopeless_factor: float = HOPELESS_FACTOR,
                    min_samples: int = MIN_TAIL_SAMPLES,
                    borderline_band: float = BORDERLINE_BAND) -> dict:
    """Что из этого может работать НА ЗАПРОС, а что обязано быть посчитано ЗАРАНЕЕ.

    Судит по p95, а не по медиане: бюджет, который держится в половине случаев,
    — это не бюджет.

    Четыре корзины, потому что чинятся они по-разному:

    * ``online``     — влезает в бюджет, может считаться в реальном времени;
    * ``optimise``   — промахивается, но не безнадёжно: настройки, разрешение,
                       батчинг могут вытянуть, и день на это потратить осмысленно;
    * ``precompute`` — промахивается кратно, и настройками это не чинится.
                       Такой этап либо считается заранее, либо не считается
                       вовсе; оптимизировать его — потерянное время;
    * ``thin``       — замеров слишком мало, чтобы p95 что-то значил. Это НЕ
                       вердикт о скорости, это отказ выносить вердикт: на малой
                       выборке одна случайная задержка весит десятки процентов
                       ответа (см. MIN_TAIL_SAMPLES) и уводит этап не в ту
                       корзину. Лечится не оптимизацией, а прогоном подлиннее.

    Этапы с ``per="run"`` в вердикт не попадают совсем: они по определению
    происходят один раз за прогон, а не на запрос, и мерить их бюджетом
    реального времени — категориальная ошибка.
    """
    online, optimise, precompute, unmeasured, thin = [], [], [], [], []
    borderline = []
    lo = budget_ms * (1 - borderline_band)
    hi = budget_ms * (1 + borderline_band)
    for name, s in sorted(summary.items()):
        if s.get("per") == PER_RUN:
            continue
        p95 = s.get("p95_ms")
        if p95 is None:
            unmeasured.append(name)
        elif s.get("calls", 0) < min_samples:
            thin.append((name, p95, s.get("calls", 0)))
        elif lo <= p95 <= hi:
            borderline.append((name, p95))
        elif p95 <= budget_ms:
            online.append((name, p95))
        elif p95 <= budget_ms * hopeless_factor:
            optimise.append((name, p95))
        else:
            precompute.append((name, p95))

    parts = [f"бюджет {budget_ms:g} мс по p95"]
    if online:
        parts.append("в бюджете: " + ", ".join(
            f"{n} {v:g}" for n, v in online))
    if optimise:
        parts.append("промах, но чинится настройками: " + ", ".join(
            f"{n} {v:g} ({v / budget_ms:.1f}x)" for n, v in optimise))
    if precompute:
        parts.append("считать ЗАРАНЕЕ, настройками не спасти: " + ", ".join(
            f"{n} {v:g} ({v / budget_ms:.0f}x)" for n, v in precompute))
    if borderline:
        parts.append(
            f"НА ГРАНИЦЕ (±{borderline_band:.0%} бюджета — вердикт меняется от "
            f"прогона к прогону, мерить дольше или на тихой машине): "
            + ", ".join(f"{n} {v:g}" for n, v in borderline))
    if thin:
        parts.append(
            f"СУДИТЬ РАНО (нужно {min_samples}+ замеров, иначе p95 держится на "
            f"одном случайном выбросе): " + ", ".join(
                f"{n} {v:g} по {c}" for n, v, c in thin))
    if unmeasured:
        parts.append("не измерено (был только холодный старт): "
                     + ", ".join(unmeasured))
    if not (online or optimise or precompute or unmeasured or thin
            or borderline):
        parts.append("поштучных этапов не замерено — судить не о чем")

    return {
        "budget_ms": budget_ms,
        "online": [n for n, _ in online],
        "optimise": [n for n, _ in optimise],
        "precompute": [n for n, _ in precompute],
        "borderline": [n for n, _ in borderline],
        "thin": [n for n, _, _ in thin],
        "unmeasured": unmeasured,
        "note": ". ".join(parts) + ".",
    }


#: Этапы, которые умеет прогнать `--profile`: (имя, что делает, на что тратит).
#: Только офлайн-часть — она и есть та половина пайплайна, которая в принципе
#: может жить в реальном времени. Генерация сюда не входит: она промахивается
#: мимо любого интерактивного бюджета на три порядка, и мерить её нечем, кроме
#: как самим прогоном.
PROFILE_STAGES = ("landmarks", "dwpose", "arcface", "draw", "pose_delta")


def profile(frames: list, *, stages=PROFILE_STAGES, repeats: int = 30,
            clock=None):
    """Прогнать офлайн-этапы на настоящих кадрах и вернуть Timings.

    Кадры берутся РАЗНЫЕ на каждом повторе (циклически), а не один и тот же:
    повторный вызов на одном файле меряет кэш декодера, а не работу этапа.
    """
    got = clock or Timings()
    if not frames:
        return got

    def _frame(i):
        return frames[i % len(frames)]

    if "landmarks" in stages:
        from .pose import landmarks

        for i in range(repeats + 1):
            with got.stage("landmarks", per=PER_FRAME):
                landmarks(_frame(i))

    if "dwpose" in stages:
        from . import dwpose

        if dwpose.available():
            for i in range(repeats + 1):
                with got.stage("dwpose", per=PER_FRAME):
                    dwpose.pose_points(_frame(i))

    if "arcface" in stages:
        from .identity_arcface import face_detail

        for i in range(repeats + 1):
            with got.stage("arcface", per=PER_FRAME):
                face_detail(_frame(i))

    if "draw" in stages or "pose_delta" in stages:
        import tempfile
        from pathlib import Path

        from .pose import landmarks, pose_delta
        from .skeleton import draw, pose_points

        # Точки снимаются ВНЕ таймера: иначе в замер отрисовки протечёт
        # MediaPipe, который дороже её в три раза, и число будет про него.
        pts = [p for p in (pose_points(_frame(i)) for i in range(8)) if p]
        if pts and "draw" in stages:
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(repeats + 1):
                    with got.stage("draw", per=PER_FRAME):
                        draw(pts[i % len(pts)], Path(tmp) / f"{i}.png")
        if len(pts) >= 2 and "pose_delta" in stages:
            poses = [landmarks(_frame(i)) for i in range(4)]
            poses = [p for p in poses if p]
            if len(poses) >= 2:
                for i in range(repeats * 5 + 1):
                    with got.stage("pose_delta", per=PER_FRAME):
                        pose_delta(poses[i % len(poses)],
                                   poses[(i + 1) % len(poses)])
    return got


def main(argv: list) -> int:
    """python3 -m ball_reel.timing --frames kit/driving

    Снимает профиль офлайн-этапов на СВОЁМ железе. Число без железа
    бессмысленно, поэтому команда печатает и то, на чём мерила.
    """
    import argparse
    import glob
    import json
    import platform
    from pathlib import Path

    ap = argparse.ArgumentParser(
        prog="ball_reel.timing",
        description="профиль латентности офлайн-этапов; ничего не генерирует")
    ap.add_argument("--frames", default="kit/driving",
                    help="каталог с настоящими кадрами")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--budget-ms", type=float, default=REALTIME_BUDGET_MS)
    ap.add_argument("--stage", action="append", default=[],
                    help="только этот этап; повторяемо. Запуск по одному "
                         "этапу на процесс даёт ЧЕСТНЫЙ холодный старт — "
                         "внутри одного процесса этапы прогревают друг другу "
                         "общие библиотеки")
    ap.add_argument("--json", default="", help="куда сложить профиль")
    args = ap.parse_args(argv)

    frames = sorted(glob.glob(str(Path(args.frames) / "*.jpg"))
                    + glob.glob(str(Path(args.frames) / "*.png")))
    if not frames:
        print(f"в {args.frames} нет кадров — указать каталог с jpg/png")
        return 1

    import multiprocessing

    print(f"железо: {platform.processor() or platform.machine()}, "
          f"{multiprocessing.cpu_count()} ядер, {platform.system()} "
          f"{platform.release()}, python {platform.python_version()}")
    try:
        import torch  # type: ignore

        cuda = ("есть: " + torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "НЕТ (всё ниже — CPU)")
        print(f"torch: {torch.__version__}, cuda {cuda}")
    except ImportError:
        print("torch: не установлен (всё ниже — CPU)")
    print(f"кадров {len(frames)}, повторов {args.repeats}\n")

    got = profile(frames, stages=tuple(args.stage) or PROFILE_STAGES,
                  repeats=args.repeats)
    summary = got.summary()
    print(render(summary))
    verdict = latency_verdict(summary, budget_ms=args.budget_ms)
    print(f"\n{verdict['note']}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"timing": summary, "latency": verdict}, indent=2,
            ensure_ascii=False))
        print(f"\nпрофиль: {args.json}")
    return 0


def render(summary: dict) -> str:
    """Профиль таблицей, для печати в консоль прогона."""
    if not summary:
        return "таймингов нет"
    rows = ["  этап                  на      вызовов   p50 мс   p95 мс  холодн."]
    for name, s in sorted(summary.items(),
                          key=lambda kv: -(kv[1].get("p95_ms") or 0)):
        p50 = "—" if s["p50_ms"] is None else f"{s['p50_ms']:8.2f}"
        p95 = "—" if s["p95_ms"] is None else f"{s['p95_ms']:8.2f}"
        rows.append(f"  {name:<20} {s['per']:<9} {s['calls']:>5} "
                    f"{p50} {p95} {s['cold_ms']:8.1f}")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
