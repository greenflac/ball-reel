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

from . import (fork_backend, fork_build_route, fork_channels, fork_comfy,
               fork_leak, fork_lora_attach, fork_lora_dataset, fork_mask,
               fork_preflight, fork_props, fork_seam, fork_template,
               fork_video)
from .fork_comfy import SECONDS_MAX, SECONDS_MIN
from .fork_identity import FAIL, PASS, UNMEASURED

#: Порядок шагов. Дешёвое раньше дорогого (П2): отсутствующий вход ловится за
#: миллисекунды, а поиск его после снятия условий по сотне кадров стоил бы
#: всего прогона. Длительность каждого шага печатается.
STEPS = ("предполёт", "входы", "корзина", "условия", "маски", "предметы",
         "граф", "адаптер", "рендер", "протечка", "шов")


def seconds_for(frame_count: int, *, fps: float | None = None) -> float:
    """Длина ролика по числу кадров драйвинга, прижатая к границам продукта.

    `fps` — ЧАСТОТА ДРАЙВИНГА, А НЕ НАША ВЫХОДНАЯ. Это исправление дефекта,
    найденного 18.08 на стыке с раскодировщиком, и разница не словесная.
    Wan Animate потребляет кадры позы ОДИН К ОДНОМУ с выходными: сколько
    кадров драйвинга подано, столько кадров и родится. Значит длина сцены во
    времени определяется тем, с какой частотой драйвинг СНЯТ.

    ЧТО БЫЛО: делили на `WRAP_FPS = 30` независимо от источника. Драйвинг
    24 к/с длиной 10 с — это 240 кадров; 240/30 = 8, и путь молча объявлял
    восьмисекундный ролик. Движение при этом проигрывалось бы на четверть
    быстрее реального, и заметил бы это глазами клиент, а не отчёт.

    УМОЛЧАНИЕ ОСТАЛОСЬ `WRAP_FPS` НАРОЧНО: подавать кадры без указания их
    частоты — законный случай (каталог PNG без метаданных), и тогда считать
    их нашими 30 к/с — единственное, что вообще можно сделать. Но вызывающий,
    у которого частота ЕСТЬ, обязан её назвать, и `from_template` теперь
    называет.

    Границы 5..10 с — промышленный стандарт, названный владельцем; частота 30
    выбрана им же после того, как 24 были забракованы за реализм.
    """
    fps = fork_comfy.WRAP_FPS if fps is None else fps
    if not frame_count:
        return SECONDS_MIN
    if fps <= 0:
        raise ValueError(
            f"частота драйвинга {fps!r} — делить на это нельзя. Ноль или "
            f"отрицательное здесь означает, что метаданные не прочитаны, а "
            f"не что ролик мгновенный")
    return min(max(frame_count / fps, SECONDS_MIN), SECONDS_MAX)


def conditions_verdict(cond: dict) -> str:
    """Исход шага условий. Три, а не два, и ноль из нуля — не успех.

    ВЫНЕСЕНО ИЗ `run` (Т5) по той же причине: чтобы шаг сказал «годно», нужен
    настоящий детектор на настоящих кадрах, и потому обратная сторона правила
    Р2 не проверялась ничем. `rendered == total` на пустом списке даёт истину,
    и шаг печатал «годно» под текстом «снято 0 из 0».
    """
    if not cond["total"]:
        return UNMEASURED
    if cond["rendered"] > cond["total"]:
        # Снято больше, чем подано, — состояние, которого быть не может.
        # Пока здесь стояло `>=`, мутация `==` -> `>=` переживала весь сьют:
        # на достижимых входах они неразличимы. Но неразличимость и есть
        # ответ: несходящийся отчёт нельзя читать как успех, его надо читать
        # как «не смогли» — считалка сломана, и вердикт по ней недействителен.
        return UNMEASURED
    if cond["rendered"] == cond["total"]:
        return PASS
    return UNMEASURED if cond["rendered"] else FAIL


def _step(name: str, outcome: str, note: str, seconds: float) -> dict:
    return {"step": name, "outcome": outcome, "note": note,
            "seconds": round(seconds, 3)}


def run(photo: str | Path, driving_frames, out_dir: str | Path, *,
        grow_px: int | None = None,
        backend: bool = False,
        props: str | Path | None = None,
        adapter: str | Path | None = None,
        mask_model=None,
        seconds: float | None = None,
        driving_fps: float | None = None,
        lock: dict | None = None,
        wrap: dict | None = None) -> dict:
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
    # ОСТАЛЬНЫЕ ПОЛЯ КАРТОЧКИ тоже доезжают до графа, а не только длина.
    # Проверено прогоном: до 18.08 в граф уходила ПУСТАЯ строка промта при
    # том, что `fork_template.check` требует его непустым. Описание было
    # проверено на согласованность и после этого проигнорировано — это уже не
    # «ролик короче», это генерация без условия.
    wrap = dict(wrap or {})
    steps: list[dict] = []
    derived = None

    # Предполёт первым: он самый дешёвый и решает, поедет ли вообще что-то.
    t = time.perf_counter()
    pre = fork_preflight.report()
    steps.append(_step("предполёт", pre["outcome"], pre["note"],
                       time.perf_counter() - t))

    t = time.perf_counter()
    missing = [str(p) for p in [Path(photo), *frames] if not p.exists()]
    # СУЩЕСТВОВАНИЯ МАЛО, и это найдено прогоном: испытательное описание
    # кладёт заглушку `SYNTHETIC-NOT-A-PHOTO` с расширением .png, файл
    # существует, и путь падал трассировкой уже на шаге корзины — то есть
    # дорогой шаг оплачивался ради ошибки, читаемой за миллисекунду (П2).
    # Открытие картинки стоит микросекунды и отвечает на настоящий вопрос:
    # можно ли с этим работать.
    unreadable = []
    if not missing:
        from PIL import Image, UnidentifiedImageError
        for f in [Path(photo), *frames]:
            try:
                with Image.open(f) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                unreadable.append(f"{f.name}: {type(exc).__name__}")
    # НОЛЬ КАДРОВ ДРАЙВИНГА — НЕ «все читаются» (Р2). Достижимо штатно:
    # карточка называет драйвинг видеофайлом (`driving.mp4`), а раскодировщика
    # видео в форке НЕТ, и сбор кадров идёт `glob`-ом по каталогу. Оператор,
    # положивший mp4 — а поле так и называется, — получал три «годно» подряд и
    # граф на 150 кадров при нуле поданных.
    outcome = (FAIL if (missing or unreadable) else
               UNMEASURED if not frames else PASS)
    steps.append(_step(
        "входы", outcome,
        f"нет файлов: {missing}" if missing else
        f"файлы есть, но не читаются как изображения: {unreadable}"
        if unreadable else
        "кадров драйвинга подано 0 — судить не о чем. Форк принимает драйвинг "
        "КАДРАМИ В КАТАЛОГЕ; раскодировщика видео в нём нет, и поданный "
        "видеофайл даёт пустой список молча" if not frames else
        f"фотография и {len(frames)} кадров драйвинга на месте, все читаются",
        time.perf_counter() - t))
    if missing or unreadable or not frames:
        return _report(steps, out)

    # Корзина — сразу после входов и до всего дорогого: она решает, КАКОЙ
    # темплейт (и какая LoRA) поедет, а узнать это после снятия условий по
    # сотне кадров значит снять их, возможно, зря.
    t = time.perf_counter()
    routed = fork_build_route.route(photo)
    # ПЛЕЧО МАСКИ ВЫБИРАЕТСЯ ЗДЕСЬ ЖЕ, а не в шаге масок, и это существенно:
    # корзина роутера и ширина маски — одно решение, снятое с одной
    # фотографии. Разложенные по разным шагам, они разъехались бы (Е1), и
    # разъезд был бы не виден: обе величины выглядят самостоятельными.
    # Отображение живёт в `fork_template.resolve_arm` — второй копии здесь
    # нет нарочно.
    armed = fork_template.resolve_arm({"arm": fork_template.AUTO_ARM,
                                       "photo": str(photo), "_root": None},
                                      router=lambda _p: routed)
    arm_name = armed["arm"] if grow_px is None else None
    steps.append(_step(
        "корзина",
        PASS if (routed["bucket"] != fork_build_route.UNSURE
                 and armed["outcome"] == PASS) else UNMEASURED,
        f"{routed['note']} ПЛЕЧО: {armed['note']}", time.perf_counter() - t))

    t = time.perf_counter()
    if fork_channels.dwpose.available():
        cond = fork_channels.render_sequence(frames, out / "cond")
        steps.append(_step("условия", conditions_verdict(cond), cond["note"],
                           time.perf_counter() - t))
    else:
        cond = None
        steps.append(_step("условия", UNMEASURED,
                           fork_channels.dwpose.why_unavailable().split(".")[0],
                           time.perf_counter() - t))

    t = time.perf_counter()
    from . import bodyparts

    if bodyparts.available():
        if grow_px is None and arm_name is None:
            # РОУТЕР СКАЗАЛ «не выбрано» — И МАСКА НЕ РАСШИРЯЕТСЯ МОЛЧА.
            # `fork_mask.sequence` на двух None ставит `BLOCK` (32 px), то
            # есть фактически плечо `wide`, и в отчёте его имени нет. Оператор
            # читал «плечо не выбрано, выбирайте руками», получал `wide`,
            # выбирал руками `narrow` и считал, что сравнил два плеча.
            masks = None
            steps.append(_step("маски", UNMEASURED,
                               "плечо не выбрано роутером и не задано руками "
                               "— расширять маску нечем. Это НЕ «плечо ноль»: "
                               f"плечи {sorted(fork_mask.ARMS)} выбираются "
                               f"явно, а умолчание здесь выдало бы `wide` под "
                               f"именем «неизвестно»",
                               time.perf_counter() - t))
        else:
            masks = fork_mask.sequence(frames, out / "mask", grow_px=grow_px,
                                       arm_name=arm_name, model=mask_model)
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
            # ПРОТАГОНИСТ И ПРАВИЛО ПОСТАНОВКИ — отдельными числами в тот же
            # шаг. `sequence` отдаёт только `note` и `outcome`, и записанным
            # долгом (DEBT 18.08) было ровно это: разметка второго человека
            # снималась, а до отчёта не доезжала. Оператор увидел бы «предметы
            # годно» и не узнал бы, что протагонист не выбран, — то есть
            # заменён будет тот, кого сегментатор взял первым.
            loaded = fork_props.load_marking(props)
            rule = fork_props.directing_rule(loaded, len(written))
            worst = (FAIL if FAIL in (marked["outcome"], rule["outcome"])
                     else UNMEASURED if UNMEASURED in (marked["outcome"],
                                                       rule["outcome"])
                     else PASS)
            steps.append(_step("предметы", worst,
                               f"{marked['note']} ПОСТАНОВКА: {rule['note']}",
                               time.perf_counter() - t))
        except (OSError, ValueError, KeyError) as exc:
            steps.append(_step("предметы", FAIL, str(exc)[:200],
                               time.perf_counter() - t))

    t = time.perf_counter()
    try:
        # ГРАФ СОБИРАЕТСЯ ОБЁРТКОЙ, А НЕ ШТАТНЫМИ НОДАМИ, и это не смена вкуса.
        # Цикл по окнам у обёртки живёт ВНУТРИ сэмплера — длина ролика
        # задаётся числом `num_frames`. На штатных нодах его надо строить в
        # графе руками, и у нас его не было: всё, что длиннее 77 кадров,
        # конвейером не выражалось вовсе. Пять секунд при 30 к/с — это 150
        # кадров, то есть КАЖДЫЙ наш ролик длиннее одного окна.
        #
        # ПРОТИВОРЕЧИЕ, КОТОРОЕ ЗДЕСЬ ЗАКРЫВАЕТСЯ ВСЛУХ. В модуле два
        # производителя графа — `derive` (штатные ноды) и `derive_wrapper`.
        # Пока сводящий проход звал первый, решение владельца про обёртку
        # лежало в документах, а конвейер ехал по старому. Два способа узнать
        # одно и то же — дефект (Е1); здесь остаётся один, и это тот, который
        # поедет на карте.
        # ДЛИНА БЕРЁТСЯ ИЗ КАРТОЧКИ, ЕСЛИ ОНА ТАМ НАЗВАНА, и только иначе —
        # из числа поданных кадров. Прежде параметр `seconds` принимался и
        # МОЛЧА ВЫБРАСЫВАЛСЯ: карточка с «10 секунд» давала граф на 5, шаг
        # печатал «годно», и расхождение вдвое не называлось нигде. Оператор
        # платил за прогон и получал ролик вдвое короче заказанного.
        want = (seconds_for(len(frames), fps=driving_fps)
                if seconds is None else float(seconds))
        derived = fork_comfy.derive_wrapper(seconds=want, lock=lock, **wrap)
        # Один аудит вместо трёх ПОТОМУ, ЧТО он их в себя включает: внутри
        # `audit_wrapper` зовёт и `audit_weights` (имена файлов против лока), и
        # `audit_loader_formats` (прочитает ли загрузчик поданный ему формат).
        # Структурно чистый граф с разъехавшимися весами — не «граф готов», и
        # худший из вложенных исходов доезжает до шага.
        audit = fork_comfy.audit_wrapper(derived, lock=lock)
        fork_comfy.write(derived, out / "fork_graph.json")
        steps.append(_step(
            "граф", audit["outcome"],
            f"{want:.1f} с: {derived['params']['length']['note']}; "
            f"{derived['params']['windows']['note']}. {audit['note']}",
            time.perf_counter() - t))
    except (OSError, ValueError, KeyError) as exc:
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

    # РЕНДЕР — здесь и нигде раньше: он единственный шаг, который стоит
    # минут на чужой машине, и всё, что можно забраковать до него, уже
    # забраковано (П2). Сюда же приходит перевод графа в формат, который
    # сервер принимает: до 18.08 этого стыка не было вовсе — сборщик отдавал
    # формат интерфейса, клиент ждал формат API, и обе половины были зелёными
    # по отдельности.
    t = time.perf_counter()
    if not backend:
        steps.append(_step(
            "рендер", UNMEASURED,
            ("бэкенд не запрошен (`backend=False`) — генерации не было. Это "
             "НЕ «нечего рендерить»: граф собран и переведён, не хватает "
             "только машины с ComfyUI"),
            time.perf_counter() - t))
    else:
        try:
            if derived is None:
                raise ValueError(
                    "графа нет — переводить нечего. Причина названа шагом "
                    "«граф» выше; здесь она не повторяется, чтобы отчёт не "
                    "объявлял две разные беды")
            api = fork_comfy.to_api(derived)
            if api["outcome"] != PASS:
                steps.append(_step("рендер", api["outcome"],
                                   f"перевод графа: {api['note']}",
                                   time.perf_counter() - t))
            else:
                # Транспорт называется ЯВНО: у `fork_backend.run` он по
                # умолчанию `None`, и вызов без него падает атрибутной
                # ошибкой вместо честного «сервер не отвечает». Это найдено
                # прогоном стыка, а не чтением.
                got = fork_backend.run(
                    api["api"], out_dir=out / "render",
                    transport=fork_backend.HttpTransport())
                steps.append(_step("рендер", got["outcome"], got["note"],
                                   time.perf_counter() - t))
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            steps.append(_step("рендер", UNMEASURED,
                               f"{type(exc).__name__}: {str(exc)[:180]}",
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


def from_template(path: str | Path, out_dir: str | Path, *,
                  production: bool | None = None, **kw) -> dict:
    """Сквозной путь ОТ ОПИСАНИЯ ТЕМПЛЕЙТА — точка входа оператора.

    Это то, ради чего писался операторский слой: завести новый драйвинг,
    фотореференс и промт, не трогая ни строки кода и не входя в сессию
    разработки. Описание читается `fork_template.load`, и он же его судит —
    второй проверки здесь нет нарочно (Е1).

    ОПИСАНИЕ, НЕ ПРОШЕДШЕЕ ПРОВЕРКУ, ДО ПРОГОНА НЕ ДОПУСКАЕТСЯ, и это не
    вежливость. Дорогие шаги стоят минут на карте; негодный путь к драйвингу
    ловится за миллисекунду (П2). Поэтому провал описания возвращается ОТЧЁТОМ
    ИЗ ОДНОГО ШАГА, а не исключением: оператор должен увидеть тот же формат
    «пройдено N, провалено M, не смогли K», а не трассировку.

    `production=True` дополнительно требует, чтобы описание не было стендовым.
    Прогнать боевого клиента по испытательному описанию — ошибка, которую
    видно только по готовому ролику.
    """
    import time

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    try:
        desc = fork_template.load(path)
        if production:
            fork_template.assert_production(desc)
    except (OSError, ValueError, KeyError, fork_template.RetargetForbidden) as exc:
        return _report([_step("описание", FAIL, str(exc)[:400],
                              time.perf_counter() - t)], out)

    root = Path(desc.get("_root") or Path(path).parent)
    photo = root / desc["photo"]
    # ДРАЙВИНГ-ВИДЕОФАЙЛ РАСКОДИРУЕТСЯ ЗДЕСЬ, а не считается пустым списком.
    # Поле карточки называется «драйвинг», и оператор кладёт в него `.mp4` —
    # так и задумано. До 18.08 сбор шёл `glob`-ом ТОЛЬКО по каталогу, файл
    # давал ноль кадров молча, и путь собирал граф на 150 кадров при нуле
    # поданных. Раскодировщик закрывает это, и заодно приносит НАСТОЯЩУЮ
    # частоту драйвинга — без неё длина ролика считалась по нашей выходной
    # частоте, и драйвинг 24 к/с проигрывался бы на четверть быстрее.
    frames = kw.pop("driving_frames", None)
    driving_fps = None
    decoded_note = ""
    if frames is None:
        driving = root / desc["driving"]
        if driving.is_dir():
            frames = (sorted(driving.glob("*.png"))
                      + sorted(driving.glob("*.jpg")))
            decoded_note = f"кадры взяты из каталога {driving.name}"
        elif driving.is_file():
            meta = fork_video.probe(driving)
            # ПРИВОДИМ К НАШЕЙ ВЫХОДНОЙ ЧАСТОТЕ, а не берём как есть.
            # Раскодировщик это умеет и без просьбы не делает — а весь
            # остальной путь считает, что кадры драйвинга идут по 30 в
            # секунду: `seconds_for` делит на неё, `frames_for` умножает на
            # неё. Взяв 60 к/с как есть, мы получили бы вдвое больше кадров,
            # чем окон, и ролик в половину заказанной скорости. Драйвинг ниже
            # 30 отказывается ЗДЕСЬ же — вверх не приводим, интерполяции нет,
            # и то же самое требование уже стоит осью в проверке карточки.
            got = fork_video.frames(driving, out / "driving",
                                    fps=fork_comfy.WRAP_FPS)
            if got["outcome"] == FAIL:
                return _report([_step("описание", FAIL,
                                      f"драйвинг не раскодирован: {got['note']}",
                                      time.perf_counter() - t)], out)
            frames = sorted((out / "driving").glob("*.png"))
            # После приведения кадры идут по нашей частоте — значит и длину
            # надо считать по ней. Частота исходника остаётся в отчёте, но
            # арифметику больше не задаёт.
            driving_fps = fork_comfy.WRAP_FPS
            source_fps = meta.get("fps") if meta["outcome"] == PASS else None
            decoded_note = (f"драйвинг {source_fps or '?'} к/с -> "
                            f"{fork_comfy.WRAP_FPS} к/с: {got['note'][:140]}")
        else:
            frames = []
            decoded_note = f"драйвинг {desc['driving']!r} не найден"
    # ИСХОД БЕРЁТСЯ У САМОГО ПРОВЕРЯЛЬЩИКА, а не пишется литералом. Прежде
    # здесь стоял `PASS`, и вердикт `_check` — вместе с его числами
    # «проверено / нарушений / не смогли» — до отчёта не доезжал вовсе: ось,
    # которую описание не смогло проверить, читалась оператором как «годно».
    check = desc.get("_check") or {}
    steps = [_step("описание", check.get("outcome", UNMEASURED),
                   f"{Path(path).name}: {desc.get('name')!r}, драйвинг "
                   f"{desc['driving']}, кадров подано {len(frames)}. "
                   f"{decoded_note}. "
                   f"{check.get('note', 'проверяльщик описания не отчитался')}",
                   time.perf_counter() - t)]
    # ПЛЕЧО БЕРЁТСЯ ИЗ КАРТОЧКИ, если оператор его назвал. Иначе получилось
    # бы, что в описании стоит `narrow`, а маску расширяет роутер по своему
    # разумению, — и оператор об этом узнал бы по ролику. `auto` в карточке
    # означает «спроси роутера», и тогда `grow_px` остаётся None, а решает
    # шаг «корзина» внутри `run`.
    armed = fork_template.resolve_arm(desc, root=root)
    steps.append(_step("плечо", armed["outcome"], armed["note"], 0.0))
    # ПОЛЯ КАРТОЧКИ -> ПАРАМЕТРЫ ГРАФА. Имена сверены с сигнатурой
    # `fork_comfy.derive_wrapper`; чего в ней нет, сюда не попадает, иначе
    # вызов упал бы на неизвестном ключе вместо тихой потери значения.
    wrap = {k: desc[k] for k in ("prompt", "negative", "width", "height",
                                 "fps") if k in desc}
    inner = run(photo, frames, out,
                grow_px=armed["grow_px"] if desc.get("arm") != "auto" else None,
                seconds=desc.get("seconds"), driving_fps=driving_fps,
                wrap=wrap,
                props=(root / desc["props"]) if "props" in desc else None,
                **kw)
    return _report(steps + inner["steps"], out)


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
