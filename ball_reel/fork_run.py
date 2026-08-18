"""Точка входа форка: сквозной путь от фотографии до кадров условий, НА МОКЕ.

ЧТО ЭТО И ЧЕМ ОНО НЕ ЯВЛЯЕТСЯ. Здесь собирается путь — приём входов, условия,
маски, граф, приёмочные оси — и НЕ ЗАПУСКАЕТСЯ генерация. Генерации в спринте
нет: железо не выбрано, ComfyUI не установлен. Модуль отдаёт то, что можно
получить без карты: посчитанные последовательности, произведённый граф и отчёт
о том, какие проверки прошли, какие провалились и какие НЕ СМОГЛИ отработать.

ЗАЧЕМ ОН ВООБЩЕ НУЖЕН, КРОМЕ УДОБСТВА. `ball_reel/tests/test_reachable.py`
краснеет на модуле, который написан и не подключён ни к чему, — и он покраснел
на `fork_comfy`, `fork_leak`, `fork_mask` ровно тогда, когда они были дописаны.
Это правильное поведение сторожа: модуль, которого нет в конвейере, но есть в
отчёте и в справочнике, опаснее ненаписанного. Собрать их в путь — верный ответ
на это, а вписать в список самостоятельных — способ заглушить.

ТРИ ИСХОДА НА КАЖДОМ ШАГЕ (Р1), и итог не сворачивается в булев флаг: шаг,
который не смог отработать, отличается и от пройденного, и от провалившегося.
Печатается числами (Р2): проверено N, провалено M, не смогли K.
"""

from __future__ import annotations

from pathlib import Path

from . import (fork_build_route, fork_channels, fork_comfy, fork_leak,
               fork_lora_attach, fork_lora_dataset, fork_mask, fork_preflight,
               fork_props, fork_seam)
from .fork_identity import FAIL, PASS, UNMEASURED

#: Порядок шагов. Дешёвое раньше дорогого (П2): отсутствующий вход ловится за
#: миллисекунды, а поиск его после снятия условий по сотне кадров стоил бы
#: всего прогона. Длительность каждого шага печатается.
STEPS = ("предполёт", "входы", "корзина", "условия", "маски", "предметы",
         "граф", "адаптер", "протечка", "шов")


def _step(name: str, outcome: str, note: str, seconds: float) -> dict:
    return {"step": name, "outcome": outcome, "note": note,
            "seconds": round(seconds, 3)}


def run(photo: str | Path, driving_frames, out_dir: str | Path, *,
        grow_px: int = fork_mask.BLOCK,
        quant: str | None = None,
        props: str | Path | None = None,
        adapter: str | Path | None = None,
        mask_model=None) -> dict:
    """Сквозной путь на моке. Возвращает отчёт по шагам, а не «получилось».

    `photo` — ЗАГРУЖЕННАЯ фотография, она же якорь оси личности. Медоид сюда
    подать нельзя: `fork_identity.refuse_derived_anchor` уронит прогон, и это
    единственный способ не повторить дефект, из-за которого 0.2579 однажды
    прочли как успех.
    """
    import time

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [Path(p) for p in driving_frames]
    steps: list[dict] = []

    # Предполёт первым: он самый дешёвый и решает, поедет ли вообще что-то.
    t = time.perf_counter()
    pre = fork_preflight.report()
    steps.append(_step("предполёт", pre["outcome"], pre["note"],
                       time.perf_counter() - t))

    t = time.perf_counter()
    missing = [str(p) for p in [Path(photo), *frames] if not p.exists()]
    steps.append(_step(
        "входы", FAIL if missing else PASS,
        f"нет файлов: {missing}" if missing else
        f"фотография и {len(frames)} кадров драйвинга на месте",
        time.perf_counter() - t))
    if missing:
        return _report(steps, out)

    # Корзина — сразу после входов и до всего дорогого: она решает, КАКОЙ
    # темплейт (и какая LoRA) поедет, а узнать это после снятия условий по
    # сотне кадров значит снять их, возможно, зря.
    t = time.perf_counter()
    routed = fork_build_route.route(photo)
    steps.append(_step(
        "корзина",
        PASS if routed["bucket"] != fork_build_route.UNSURE else UNMEASURED,
        routed["note"], time.perf_counter() - t))

    t = time.perf_counter()
    if fork_channels.dwpose.available():
        cond = fork_channels.render_sequence(frames, out / "cond")
        steps.append(_step(
            "условия",
            PASS if cond["rendered"] == cond["total"] else
            UNMEASURED if cond["rendered"] else FAIL,
            cond["note"], time.perf_counter() - t))
    else:
        cond = None
        steps.append(_step("условия", UNMEASURED,
                           fork_channels.dwpose.why_unavailable().split(".")[0],
                           time.perf_counter() - t))

    t = time.perf_counter()
    from . import bodyparts

    if bodyparts.available():
        masks = fork_mask.sequence(frames, out / "mask", grow_px=grow_px,
                                   model=mask_model)
        steps.append(_step("маски", masks["outcome"], masks["note"],
                           time.perf_counter() - t))
    else:
        masks = None
        steps.append(_step("маски", UNMEASURED,
                           bodyparts.why_unavailable().split(".")[0],
                           time.perf_counter() - t))

    # ПРЕДМЕТЫ — сразу после масок, потому что они правят ИМЕННО маску, и до
    # графа, потому что граф эту маску получает. Разметка снимается на сборке
    # темплейта руками оператора и переиспользуется всеми клиентами: драйвинг у
    # темплейта один. Её отсутствие — не провал, а «нечего применять».
    t = time.perf_counter()
    if props is None:
        steps.append(_step("предметы", UNMEASURED,
                           "разметка предметов не подана — предметы остаются "
                           "там, куда их отнёс сегментатор тела, а он относит "
                           "предмет в руке к персонажу классом `others`",
                           time.perf_counter() - t))
    elif masks is None:
        steps.append(_step("предметы", UNMEASURED,
                           "масок нет — размечать поверх нечего",
                           time.perf_counter() - t))
    else:
        try:
            # Маски берутся ФАЙЛАМИ из каталога, куда их положил шаг выше:
            # `fork_mask.sequence` пишет на диск и массивов не возвращает.
            # Первая версия звала `masks["masks"]` и падала в KeyError, а шаг
            # печатал провал — то есть отчёт врал бы про разметку, которой
            # ничто не мешало отработать.
            written = sorted(Path(masks["dir"]).glob("*.png"))
            marked = fork_props.sequence(props, written)
            steps.append(_step("предметы", marked["outcome"], marked["note"],
                               time.perf_counter() - t))
        except (OSError, ValueError, KeyError) as exc:
            steps.append(_step("предметы", FAIL, str(exc)[:200],
                               time.perf_counter() - t))

    t = time.perf_counter()
    try:
        derived = fork_comfy.derive(quant=quant)
        audit = fork_comfy.audit(derived)
        # ВТОРАЯ ПРОВЕРКА ГРАФА, И ОНА НУЖНА ОТДЕЛЬНО ОТ ПЕРВОЙ. `audit` мерит
        # структуру — ноды, связи, питание входов, — и на состоянии, где граф
        # грузит fp8, а лок объявляет Q3_K_M, он честно печатает «ЧИСТО»:
        # структурно там всё в порядке. Расхождение по ФАЙЛАМ ловит только
        # `audit_weights`, и без него смена скачала бы по локу 15.338 ГиБ и
        # запустила граф, просящий другие 26.849. Худший из двух исходов идёт в
        # шаг: зелёная структура при разъехавшихся весах — это не «граф готов».
        weights = fork_comfy.audit_weights(derived)
        fork_comfy.write(derived, out / "fork_graph.json")
        worst = (FAIL if FAIL in (audit["outcome"], weights["outcome"])
                 else UNMEASURED if UNMEASURED in (audit["outcome"],
                                                   weights["outcome"])
                 else PASS)
        steps.append(_step("граф", worst,
                           f"{audit['note']} ВЕСА: {weights['note']}",
                           time.perf_counter() - t))
    except (OSError, ValueError) as exc:
        steps.append(_step("граф", UNMEASURED, str(exc)[:200],
                           time.perf_counter() - t))

    # АДАПТЕР — сразу после графа, потому что имена тензоров модели берутся из
    # того, что граф собирается грузить, и до всякой генерации: адаптер, который
    # не приложится, делает бессмысленным весь дорогой прогон. Числовой канал
    # здесь не работает — он требует трёх настоящих прогонов на карте.
    t = time.perf_counter()
    if adapter is None:
        steps.append(_step("адаптер", UNMEASURED,
                           "адаптер не подан — проверять нечего. Это НЕ «LoRA "
                           "не нужна»: базовая линия без адаптера штатна",
                           time.perf_counter() - t))
    else:
        try:
            keys = fork_lora_attach.read_tensor_names(adapter)
            static = {
                "outcome": UNMEASURED,
                "note": ("имён тензоров модели нет — они снимаются "
                         "`tools/fork_probe_lora_fit.py` с карты или из сети, "
                         "а на моке брать их неоткуда"),
                "dora": fork_lora_attach.dora_readiness(keys)}
            merged = fork_lora_attach.report(static=static)
            worst = (FAIL if static["dora"]["outcome"] == FAIL
                     else merged["outcome"])
            steps.append(_step(
                "адаптер", worst,
                f"ключей в адаптере {len(keys)}. "
                f"DoRA: {static['dora']['note']} {merged['note']}",
                time.perf_counter() - t))
        except (OSError, ValueError) as exc:
            steps.append(_step("адаптер", FAIL, str(exc)[:200],
                               time.perf_counter() - t))

    # Протечка на моке НЕ ИЗМЕРЯЕТСЯ, и это не пропуск: сравнивать выход с
    # драйвингом можно только когда выход есть, а генерации в спринте нет.
    #
    # ЗДЕСЬ СТОЯЛ ВЫЗОВ `fork_leak.noise_floor(frames)` — планки шума КОДЕКА.
    # Снято по §1a: вне маски не копия, а реконструкция через VAE и сэмплер,
    # и кодековая планка не просто неполна, а ЗАНИЖЕНА ниже неустранимого пола
    # представления. Поток B это замерил: VAE-круг даёт 0.007308 против
    # кодековых 0.004513, то есть старая планка была меньше того, что тракт
    # тратит ДО единого шага сэмплера. Функции больше нет в модуле нарочно —
    # заниженная планка, лежащая в коде, будет использована.
    #
    # Сводящий проход НЕ поднимает VAE сам: замер стоит минуты процессорного
    # времени (encode 109 с, decode 397 с на пяти кадрах половинной геометрии)
    # и на целевой геометрии здесь падает по памяти. Путь на моке обязан
    # оставаться дешёвым, поэтому здесь только называется, чего не хватает.
    t = time.perf_counter()
    steps.append(_step(
        "протечка", UNMEASURED,
        ("выхода нет — мерить нечего. Планка тоже не готова: судить вправе "
         f"только пол «{fork_leak.FLOOR_FULL_LOOP}» (полный круг при нулевой "
         f"маске), а он требует карты. VAE-пол снимается "
         f"`fork_leak.vae_roundtrip_floor`, но один он не судит."),
        time.perf_counter() - t))

    # Шов меряется на ОДНОМ кадре — внутри маски против снаружи. Но кадра
    # выхода нет, а мерить шов на кадре драйвинга бессмысленно: там внутри
    # маски те же пиксели, что снаружи, и ось по построению даст единицу.
    # Поэтому здесь честное «не смогли», а не подставной прогон.
    t = time.perf_counter()
    steps.append(_step(
        "шов", UNMEASURED,
        ("выхода нет — мерить шов не на чем. Планка тоже не откалибрована: "
         f"облака снимаются с живого выхода ({fork_seam.RING_PX}px кольцо)"),
        time.perf_counter() - t))

    return _report(steps, out)


def build_template(frame_paths, out_dir: str | Path, *, domain: str,
                   sources=None, from_driving: bool = False,
                   build_type: str = fork_lora_dataset.THE_ONLY_FIGURE) -> dict:
    """Сборка ТЕМПЛЕЙТА: набор под ОДНУ фигуру, один раз, не на клиента.

    Отдельная точка входа, а не шаг `run`, и это существенно: набор собирается
    ОДИН РАЗ при сборке темплейта, а `run` идёт на каждого клиента. Склеив их,
    мы получили бы обучение на клиента — ровно то, чего продукт не делает.

    `sources` И `from_driving` ПРОБРОШЕНЫ НАРОЧНО, а не забыты. Поток G
    сообщил, что без них вызов проходит на умолчаниях и честно возвращает
    «не смогли проверить»: учёт людей не подан, значит набор не признан
    годным. Это верное поведение модуля и негодное поведение точки входа —
    сводящий проход, который структурно не может собрать годный набор, выглядел
    бы как неудачная сборка вместо неполного вызова.

    `build_type` с умолчанием `THE_ONLY_FIGURE`, потому что LoRA обучается одна
    (§2a): второе плечо — базовая линия без LoRA, и набор ему не нужен.
    Подставить сюда имя базовой линии нельзя, `fork_lora_dataset` уронит вызов
    первой же строкой.
    """
    return fork_lora_dataset.build(frame_paths, out_dir,
                                   build_type=build_type, domain=domain,
                                   sources=sources, from_driving=from_driving)


def _report(steps: list, out: Path) -> dict:
    passed = sum(1 for s in steps if s["outcome"] == PASS)
    failed = sum(1 for s in steps if s["outcome"] == FAIL)
    unmeasured = sum(1 for s in steps if s["outcome"] == UNMEASURED)
    return {
        "steps": steps, "dir": str(out),
        "passed": passed, "failed": failed, "unmeasured": unmeasured,
        "outcome": FAIL if failed else (UNMEASURED if unmeasured else PASS),
        "note": (f"шагов {len(steps)}: пройдено {passed}, провалено {failed}, "
                 f"не смогли {unmeasured}. ГЕНЕРАЦИИ НЕ БЫЛО — путь собран на "
                 f"моке, продуктовое заявление им не проверяется."),
    }
