"""Последняя ступень: апскейл x2 РОВНО ПОСЛЕ того, как гейт вынес вердикт.

ЗАЧЕМ ОНА ЕСТЬ. Генерация идёт на 512x768 — это потолок карты на 6 ГБ
(`animate.plan`), и для показа этого мало. x2 даёт 1024x1536, то есть кадр,
на который можно смотреть.

ГЛАВНОЕ ЗДЕСЬ — ПОРЯДОК, А НЕ КАЧЕСТВО

Апскейлер ВЫДУМЫВАЕТ детали. Это не дефект и не придирка к конкретной модели,
это его работа: из четырёх выходных пикселей на один входной три взяты не из
данных, а из того, что модель считает правдоподобным. Значит гейт, померивший
апскейленный кадр, померил выдумку апскейлера, а не то, что породила модель, —
и вердикт демо перестал быть про продукт. В `gpu_keyframes` (строка 89) это
записано одной фразой: «upscaling happens AFTER the gate — never before, or the
gate measures the upscaler's invention».

ЧТО ИМЕННО ВЫДУМЫВАЕТСЯ — ЧИСЛОМ, А НЕ СЛОВОМ, И ЧИСЛО ВЫШЛО НЕ ТО, КОТОРОЕ
ЖДАЛИ. Взят настоящий кадр `kit/driving/0000.jpg` (720x1278), из него вырезан
РОДНОЙ кроп 512x768 — без единого пересэмплирования, чтобы измерялась плёнка,
а не моё же сглаживание, — и посчитаны две полосы спектра плюс дисперсия
лапласиана. Обе доли считаются на своей сетке, поэтому сравнимы между
кадрами разного размера (`high_frequency_share`).

                          r>=0.5      r>=0.25     дисперсия
                       (выше Найк-   (верхняя    лапласиана
                       виста входа)   октава)       x1e4
    родной кроп          0.000553    0.006099      9.5237   (на своей сетке)
    Lanczos x2           0.000023    0.000457      1.0240
    Swin2SR x2           0.000049    0.001162      1.8397

Вторая и третья строки посчитаны на ОДНОЙ сетке 1024x1536 — различается только
апскейлер, поэтому сравнивать их можно. Первая приведена для контекста: она на
своей сетке 512x768, и переносить её число на удвоенный холст нельзя.

Читается это так, и не так, как ожидалось:

* **выше Найквиста исходника (полоса r>=0.5 на удвоенной сетке) обе строки
  практически нулевые** — 0.000023 и 0.000049 против 0.000553, которые
  исходник несёт в своей собственной верхней полосе. То есть
  `swin2SR-classical-sr-x2-64` НЕ насыпает частот в полосу, которой у входа не
  было вовсе. Ходовое «апскейлер галлюцинирует детали из ничего» на ЭТОМ
  чекпойнте замером не подтвердилось, и записать это обязательно:
  непроверенное ожидание, поданное как факт, — ровно то, чем этот проект уже
  обжигался;
* **в верхней октаве, которую исходник нёс сам (r>=0.25), Swin2SR даёт в
  2.54 раза больше энергии, чем Lanczos, при дисперсии лапласиана
  в 1.80 раза больше.** Обе величины — на ОДНОЙ сетке 1024x1536, где
  различается только апскейлер, поэтому сравнение честное;
* **Lanczos не добавляет ничего** ни в одной полосе — он и есть определение
  «крупнее, но не детальнее».

Воспроизводится командой (третья строка требует весов):

    python3 -c "from ball_reel.upscale import sharpness_row; \
        print(sharpness_row('ball_reel/kit/driving/0000.jpg'))"

ПОЧЕМУ ЭТО НЕ ОСЛАБЛЯЕТ ТРЕБОВАНИЕ О ПОРЯДКЕ, А УТОЧНЯЕТ ЕГО. Требование
никогда не держалось на величине выдумки — оно держится на том, что гейту
достались бы НЕ ТЕ ПИКСЕЛИ. И полоса, в которой Swin2SR переписывает картинку,
— это в точности та полоса, где живут структуры, которые меряют наши
измерители: кромка века и губ у ArcFace, контур конечности у DWPose, мелкая
фактура у метрик реализма. Переписать их в 2.54 раза и после этого
померить — значит померить апскейлер. Дисперсия лапласиана, выросшая в
1.80 раза, сдвинет любую метрику резкости просто по определению.

И обратная сторона того же замера, которую тоже надо произнести: раз выше
Найквиста ничего не появляется, то апскейл НЕ ЧИНИТ судимость лица. Лицо в
78 px после x2 станет 156 px по счёту пикселей, но новых различающих деталей
в нём не прибавится — их добавляет доводка (`refine`), и порядок «доводка,
потом апскейл» держится по той же причине, что и «гейт, потом апскейл».

МЕХАНИЗМ ЗАЩИТЫ ПОРЯДКА: ПЕЧАТЬ, ПРИБИТАЯ К ПИКСЕЛЯМ

Просьба в докстринге («сначала гейт, потом апскейл») не защищает ни от чего:
её читают после того, как порядок уже перепутан. Поэтому `upscale_frames`
требует ВТОРЫМ ОБЯЗАТЕЛЬНЫМ аргументом `GateSeal` — печать, которую выдаёт
`seal_verdict(frames, verdict, gate=...)` и которая хранит sha256 КАЖДОГО
судимого кадра. Апскейл сверяет дайджесты входа с печатью и отказывается
работать, если они не совпадают.

Отсюда прямое следствие: **апскейленных кадров нельзя получить раньше, чем
гейт вынес вердикт по этим самым пикселям.** Не «не рекомендуется», а нельзя:
печати неоткуда взяться, а без печати `upscale_frames` бросает `ValueError`.

Вторая половина замка — реестр произведённого. Всё, что модуль выдал, он
запоминает по дайджесту, и `seal_verdict` ОТКАЗЫВАЕТСЯ печатать кадры из этого
реестра. То есть «сначала апскейлить, потом судить и запечатать задним числом»
тоже не собирается.

ПОЧЕМУ ИМЕННО ТАК, А НЕ ПРОЩЕ. Разобраны четыре варианта, три отвергнуты:

* **абзац в докстринге** — отвергнут: не исполняется. Именно так этот проект
  уже потерял три соответствия «код против рунбука» за один день;
* **булев аргумент `gate_passed=True`** — отвергнут, и это главный из отказов:
  флаг сообщает НЕ ТО, ЧТО БЫЛО, а то, что вызывающий помнит о порядке. Он
  ровно так же надёжен, как память человека, который порядок и перепутал, —
  то есть переписывает ту же ошибку в аргумент и делает её похожей на
  проверку;
* **проверка размера входа (`<= 512x768`)** — отвергнута: ловит не ту величину.
  Кадр 512x768 может быть уже апскейленным из 256x384, а холст генерации
  завтра поменяется — и проверка начнёт врать в обе стороны;
* **печать, привязанная к СОДЕРЖИМОМУ** — принята. Дайджест нельзя предъявить,
  не имея на руках доапскейльных пикселей в момент вердикта; порядок вычислений
  становится физическим условием, а не соглашением.

ГРАНИЦА ЭТОГО МЕХАНИЗМА, И ЕЁ НАДО ПРОИЗНОСИТЬ. Печать останавливает ОШИБКУ,
а не злой умысел: кто угодно может вызвать `seal_verdict(frames, {"ok": True},
gate="я так решил")` и получить печать без единого измерения. Разница в том,
что это уже не перепутанный порядок, а написанная в коде неправда с именем
гейта — её видно на ревью, в отличие от строки `frames = upscale(frames)`,
поставленной на две функции выше по течению. Реестр произведённого живёт в
пределах процесса: кадры, записанные на диск и прочитанные заново, модулю
незнакомы. Это осознанная граница, а не недосмотр — второй апскейл при этом
всё равно не пройдёт, потому что печати на новые пиксели ни у кого нет.

ТРИ ИСХОДА, А НЕ ДВА

`upscale_frames` возвращает `outcome` из трёх значений, и они не сводимы друг
к другу:

* `"weights"` — отработал Swin2SR. Резкость выросла, детали ПОРОЖДЕНЫ
  (`invents=True`);
* `"fallback"` — отработал Lanczos. Пикселей больше, деталей не прибавилось
  (`invents=False`) — это измерено, см. таблицу выше;
* `"skipped"` — апскейл не выполнен, кадры возвращены как были
  (`invents=None`: выдумывать было нечему).

«Не смогли» здесь отдельный исход, а не «плохо»: демо с исходным разрешением —
это демо, а демо с молча подставленным Lanczos'ом вместо весов — это отчёт,
в котором написано не то, что произошло. Поэтому `invents` трёхзначен, а
булева «ок» в ответе нет вовсе.

ВЕСА И ЛИЦЕНЗИЯ — ПРОВЕРЕНО, А НЕ ПРИПОМНЕНО

`caidas/swin2SR-classical-sr-x2-64`. Как проверено (14.08.2026):

* репозиторий существует — `HfApi().model_info(...)` отвечает, среди файлов
  `model.safetensors` 48 460 660 байт и `config.json`;
* лицензия ДОСЛОВНО. В шапке карточки модели (`README.md`, YAML front matter)
  стоит `license: apache-2.0`, и тот же `apache-2.0` приходит в
  `model_info(...).card_data["license"]` и в теге `license:apache-2.0`. Это не
  тот случай, когда поля нет вовсе: у `guoyww/animatediff-*` и
  `openai/clip-*` на этом проекте поля `license` в карточке НЕ ОКАЗАЛОСЬ, и
  молчание — не разрешение. Здесь поле есть и оно Apache-2.0, продукт
  коммерческий, ступень проходит;
* множитель взят НЕ ИЗ ИМЕНИ репозитория, а из `config.json` чекпойнта:
  `upscale: 2`. Имя может соврать, конфиг — нет;
* класс `Swin2SRForImageSuperResolution` и процессор `Swin2SRImageProcessor`
  существуют в УСТАНОВЛЕННОМ transformers (5.15.0), а не «должны быть»;
* веса лежат в локальном кэше (`~/.cache/huggingface/hub/models--caidas--
  swin2SR-classical-sr-x2-64`, 47 МБ) и исполнялись — 12 091 571 параметр,
  прогон 512x768 на CPU занял примерно 13 минут.

ЛОВУШКА API, ПРОВЕРЕННАЯ ПО ИСХОДНИКУ УСТАНОВЛЕННОГО ПАКЕТА.
`Swin2SRImageProcessor.pad` считает добавку как
`(height // size_divisor + 1) * size_divisor - height`, то есть добавляет
ПОЛНЫЙ `size_divisor` даже когда сторона уже кратна ему. На входе 128x128
выход получается 272x272, а не 256x256. Кадр обязан обрезаться до
`(w*factor, h*factor)` — иначе на каждом кадре появляется лишняя полоса
симметричного отражения по правому и нижнему краю. Замерено прогоном.

ЧТО ЗДЕСЬ ЗАМЕР, А ЧТО РАСЧЁТ

Всё про видеопамять — РАСЧЁТ (карты в среде разработки нет), и `vram_delta`
повторяет эту пометку внутри своего ответа, чтобы её нельзя было потерять при
переносе числа в отчёт. Резкость, размеры и множитель — замеры и арифметика.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# ------------------------------------------------------------------ веса

#: Чекпойнт. ПРОВЕРЕН по HF API, а не по памяти: см. докстринг модуля.
SR_REPO = "caidas/swin2SR-classical-sr-x2-64"

#: Лицензия ДОСЛОВНО из шапки карточки модели. Здесь строкой, а не «Apache»
#: словами, потому что отличать «поле есть и в нём apache-2.0» от «поля нет
#: вовсе» на этом проекте пришлось уже дважды.
SR_LICENSE = "apache-2.0"

#: Размер `model.safetensors` в байтах. ИЗМЕРЕН: HfApi().model_info(SR_REPO,
#: files_metadata=True). Веса fp32; в fp16 будет вдвое меньше, но выигрыш
#: в 23 МБ не стоит потери точности на единственной ступени, где мы просим
#: модель придумать детали.
SR_WEIGHTS_BYTES = 48_460_660

#: Ширина эмбеддинга, глубина и окно внимания. ИЗМЕРЕНЫ по `config.json`
#: чекпойнта (`embed_dim`, `num_layers`, `window_size`), а не по статье:
#: в репозитории лежат четыре разных Swin2SR, и числа у них разные.
SR_EMBED_DIM = 180
SR_DEPTH = 6
SR_HEADS = 6
SR_WINDOW = 8

#: Кратность, до которой процессор дополняет вход, ПЛЮС его особенность:
#: добавка делается всегда, даже к уже кратной стороне. ИЗМЕРЕНО по исходнику
#: установленного transformers 5.15.0 (`image_processing_swin2sr.pad`).
SR_PAD_DIVISOR = 8

# -------------------------------------------------------------- множитель

#: КАНОНИЧЕСКИЙ множитель апскейла для всего пайплайна. ВЫБРАН, и опорных
#: фактов два, оба проверяемые:
#:  * 512x768 (потолок карты, `gpu_keyframes.WIDTH/HEIGHT`) x2 = 1024x1536 —
#:    это 2:3, то есть кадрировка не меняется, а видеоступень доводит до 9:16;
#:  * сам чекпойнт обучен ровно на x2: `config.json` -> `upscale: 2`. Просить
#:    у него x3 нельзя, а прогонять дважды — значит выдумывать поверх
#:    выдуманного.
#: `refine.UPSCALE_FACTOR` — второй способ узнать это же число; правится оно
#: там (импортом отсюда), а не копией здесь.
UPSCALE_FACTOR = 2

#: Сторона тайла на ВХОДЕ апскейлера. ВЫБРАНА, опорные факты:
#:  * память Swin2SR линейна по числу входных пикселей и не зависит от того,
#:    как они нарезаны, — значит тайл превращает «пик растёт с кадром» в «пик
#:    ограничен константой» (см. `vram_delta`);
#:  * 256 кратно и окну внимания 8, и обучающему патчу 64 (`image_size` в
#:    конфиге), то есть окна внутри тайла не разъезжаются с обучением.
SR_TILE = 256

#: Перекрытие тайлов на входе. ВЫБРАНО, опорный факт — `SR_WINDOW` = 8:
#: 16 = два полных окна внимания, поэтому пиксель шва ни в одном из двух
#: тайлов не оказывается на границе окна. Ноль здесь означает видимую сетку
#: на кадре, и это самая заметная глазом ошибка тайлинга.
SR_TILE_OVERLAP = 16

# -------------------------------------------------- измеренная резкость

#: ОПОРНЫЕ ЗАМЕРЫ, а не пороги: на них держится всё, что модуль утверждает про
#: «выдумывает / не выдумывает». ИЗМЕРЕНО на родном кропе 512x768 из
#: `kit/driving/0000.jpg` командой из докстринга модуля. `above_half` — доля
#: энергии выше половины Найквиста (для удвоенного кадра это полоса, которой у
#: входа не было вовсе), `above_quarter` — включая верхнюю октаву самого
#: исходника, `lapvar` — дисперсия лапласиана, умноженная на 1e4.
MEASURED_HF_SHARE = {
    "source": {"above_half": 0.000553, "above_quarter": 0.006099,
               "lapvar": 9.5237},
    "lanczos": {"above_half": 0.000023, "above_quarter": 0.000457,
                "lapvar": 1.0240},
    "swin2sr": {"above_half": 0.000049, "above_quarter": 0.001162,
                "lapvar": 1.8397},
}

#: Во сколько раз Swin2SR богаче Lanczos'а в верхней октаве и по лапласиану.
#: ИЗМЕРЕНЫ на ОДНОЙ сетке 1024x1536 — различается только апскейлер. Это и
#: есть та величина, ради которой гейт обязан отработать раньше.
MEASURED_INVENTION_RATIO = 2.54
MEASURED_LAPVAR_RATIO = 1.80

#: Граница полосы, в которой считается «выдумка»: 0.5 от Найквиста ВЫХОДА.
#: ВЫБРАНА, опорный факт арифметический: при увеличении вдвое всё, что выше
#: половины выходного Найквиста, лежит выше Найквиста ВХОДА — то есть в этой
#: полосе у входа не было информации вообще, любая энергия там порождена.
INVENTION_BAND = 0.5


# ------------------------------------------------------------- дайджесты

#: Дайджесты всего, что модуль когда-либо выдал в этом процессе. Вторая
#: половина замка порядка: запечатать произведённое апскейлером нельзя.
_PRODUCED: set[str] = set()


def frame_digest(frame) -> str:
    """sha256 по САМИМ ПИКСЕЛЯМ кадра, а не по имени файла и не по объекту.

    Принимает PIL.Image, массив numpy, байты или путь. Имя файла не годится
    принципиально: перезаписанный по тому же пути кадр — другие пиксели при
    том же имени, и печать, привязанная к имени, разрешила бы апскейлить то,
    чего гейт не видел.

    В дайджест входят форма и тип, а не только байты: массив 512x768 и его
    транспонированный близнец состоят из одних и тех же байтов.
    """
    h = hashlib.sha256()
    if isinstance(frame, (bytes, bytearray, memoryview)):
        h.update(b"raw:")
        h.update(bytes(frame))
        return h.hexdigest()
    tobytes = getattr(frame, "tobytes", None)
    size = getattr(frame, "size", None)
    mode = getattr(frame, "mode", None)
    if tobytes is not None and mode is not None and isinstance(size, tuple):
        h.update(f"pil:{mode}:{size[0]}x{size[1]}:".encode())
        h.update(frame.tobytes())
        return h.hexdigest()
    shape = getattr(frame, "shape", None)
    dtype = getattr(frame, "dtype", None)
    if tobytes is not None and shape is not None:
        h.update(f"arr:{dtype}:{tuple(shape)}:".encode())
        h.update(frame.tobytes())
        return h.hexdigest()
    try:
        with open(frame, "rb") as fh:
            h.update(b"file:")
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except (TypeError, OSError) as exc:
        raise TypeError(
            f"кадр {type(frame).__name__} не опознан как пиксели: ожидались "
            f"PIL.Image, массив, байты или путь к файлу ({exc})") from exc
    return h.hexdigest()


@dataclass(frozen=True)
class GateSeal:
    """Печать гейта: чем судили, что получилось и КАКИЕ ИМЕННО пиксели.

    Неизменяема намеренно: печать, у которой можно дописать дайджест, —
    это не печать. Сравнение идёт по множеству дайджестов, а не по порядку
    кадров: гейт мог судить их в другом порядке или подмножеством.
    """

    gate: str
    digests: frozenset
    verdict_outcome: object
    frames: int

    def covers(self, frames) -> tuple:
        """(накрывает ли печать эти кадры, чего не хватило).

        Второй элемент — список дайджестов, которых в печати нет. Пустой
        список означает «накрывает»; ошибку с ним читать можно, ошибку
        «не совпало» — нельзя.
        """
        missing = [d for d in (frame_digest(f) for f in frames)
                   if d not in self.digests]
        return (not missing, missing)


def _outcome_of(verdict) -> object:
    """Что гейт сказал, если он говорит на понятном языке. Иначе None."""
    if isinstance(verdict, dict):
        for key in ("outcome", "ok", "judgeable", "verdict"):
            if key in verdict:
                return verdict[key]
    return None


def seal_verdict(frames, verdict, *, gate: str) -> GateSeal:
    """Запечатать кадры, которые гейт УЖЕ ПОСУДИЛ. Без этого апскейла не будет.

    `gate` — имя измерителя (например
    `"identity_arcface.arcface_drift"`). Оно обязательно и не может быть
    пустым: печать без имени гейта ничем не отличается от отсутствия печати,
    а с именем в отчёте видно, чей вердикт относится к этим пикселям.

    `verdict` — ответ гейта. Любой непустой объект; из словаря вынимается
    `outcome`/`ok`/`judgeable` — не ради проверки, а чтобы в отчёте апскейла
    было видно, какой вердикт относится к этому кадру. `None` не принимается:
    «гейт не звали» и «гейт сказал» обязаны выглядеть по-разному.

    ОТКАЗ, РАДИ КОТОРОГО ФУНКЦИЯ И НАПИСАНА: кадр, произведённый этим
    модулем, запечатать нельзя. Иначе связка «апскейлить -> посудить
    апскейленное -> запечатать задним числом» была бы законной, а это ровно
    тот порядок, который здесь запрещён.
    """
    if not gate or not str(gate).strip():
        raise ValueError(
            "печать без имени гейта не выдаётся: непонятно, чей вердикт "
            "относится к этим кадрам. Передайте gate='<модуль>.<функция>'")
    if verdict is None:
        raise ValueError(
            f"гейт {gate} не вернул вердикта (None). «Гейта не звали» и «гейт "
            f"сказал» обязаны различаться, поэтому печать не выдаётся")
    frames = list(frames)
    if not frames:
        raise ValueError("печатать нечего: пустая последовательность кадров")
    digests = [frame_digest(f) for f in frames]
    produced = [d for d in digests if d in _PRODUCED]
    if produced:
        raise ValueError(
            f"{len(produced)} из {len(digests)} кадров произведены апскейлером "
            f"этого же процесса. Значит вердикт {gate} вынесен по выдуманным "
            f"деталям, а не по тому, что породила модель, — печать на такое не "
            f"выдаётся. Судить надо кадры ДО апскейла")
    return GateSeal(gate=str(gate), digests=frozenset(digests),
                    verdict_outcome=_outcome_of(verdict), frames=len(frames))


# -------------------------------------------------------------------- план


def _full_body_share() -> float:
    """Доля лица при полном росте — из `animate`, не из копии.

    Импорт, а не число: дублирование константы на этом проекте уже стоило
    1.7 ГБ, скачанных дважды, и трёх расхождений «код против рунбука» за день.
    """
    from .animate import FULL_BODY_FACE_SHARE

    return FULL_BODY_FACE_SHARE


def _judge_bar() -> int:
    """Бар судимости лица для КЛИПА — из гейта, не из копии."""
    from .identity_arcface import MIN_FACE_PX

    return MIN_FACE_PX


def upscale_plan(width: int, height: int, *, factor: int | None = None,
                 face_px: float | None = None,
                 share: float | None = None) -> dict:
    """Что даст апскейл: размер, память и во что превращается лицо. До GPU-минут.

    `face_px` — ИЗМЕРЕННЫЙ размер лица на кадрах первой ступени. `None`
    означает «не мерили»: тогда берётся типовая доля из `animate`, а
    `judgeable` возвращается `None`, а не `False`. Причина измеренная и
    записана в `refine.refine_plan`: типовая доля уже один раз обещала 146 px
    там, где по факту оказалось 59. Трёхзначность здесь не аккуратность, а
    единственный способ не выдать «не мерили» за «не проходит».

    ЧЕГО ЗДЕСЬ НЕТ. Ответа на вопрос «стоит ли делать доводку лица» — на него
    отвечает `refine.refine_plan`, и он же владеет прогнозом по лицу. Доля и
    бар сюда ИМПОРТИРУЮТСЯ из `animate` и `identity_arcface`, поэтому
    арифметика двух модулей сойтись обязана; на это есть тест, и он —
    единственная защита от того, чтобы два прогноза начали расходиться молча.
    """
    factor = UPSCALE_FACTOR if factor is None else factor
    if width <= 0 or height <= 0:
        raise ValueError(f"размер кадра обязан быть положительным: {width}x{height}")
    if factor < 1:
        raise ValueError(f"множитель апскейла обязан быть от 1: {factor}")

    bar = _judge_bar()
    measured = face_px is not None
    share = _full_body_share() if share is None else share
    face_in = float(face_px) if measured else share * height
    face_out = round(face_in * factor, 1)
    judgeable = (face_out >= bar) if measured else None

    mem = vram_delta(width, height, factor=factor)
    row = {"size_in": (width, height), "factor": factor,
           "size_out": (width * factor, height * factor),
           "pixels_in": width * height,
           "pixels_out": width * factor * height * factor,
           "face_px_in": round(face_in, 1), "face_px_out": face_out,
           "bar": bar, "measured": measured, "judgeable": judgeable,
           "share": share if not measured else None,
           "vram": mem}

    tail = (f"Память — РАСЧЁТ, не замер: +{mem['weights_gb']} ГБ весов и "
            f"~{mem['peak_gb_lower_bound']} ГБ активаций на тайл "
            f"{SR_TILE}x{SR_TILE} (НИЖНЯЯ ГРАНИЦА). Запасной путь Lanczos "
            f"не добавляет ни весов, ни активаций вовсе")
    if not measured:
        row["note"] = (
            f"{width}x{height} -> {width * factor}x{height * factor} (x{factor}). "
            f"Лицо НЕ ИЗМЕРЕНО: по типовой доле {share:g} это ~{face_in:.0f} px "
            f"и ~{face_out:.0f} px после апскейла при баре {bar}. Верить нельзя "
            f"— типовая доля уже обещала 146 px там, где было 59; померить на "
            f"готовых кадрах и позвать заново. {tail}")
    elif judgeable:
        row["note"] = (
            f"{width}x{height} -> {width * factor}x{height * factor} (x{factor}). "
            f"Лицо {face_in:.0f} -> {face_out:.0f} px при баре {bar}: гейту "
            f"было бы что судить. Но судить он обязан ДО этой ступени: в "
            f"верхней октаве после Swin2SR энергии в {MEASURED_INVENTION_RATIO}x "
            f"больше, чем после интерполяции (ИЗМЕРЕНО на одной сетке), и эта "
            f"разница нарисована апскейлером, а не порождена моделью. {tail}")
    else:
        row["note"] = (
            f"{width}x{height} -> {width * factor}x{height * factor} (x{factor}). "
            f"Лицо {face_in:.0f} -> {face_out:.0f} px, бар {bar} не берётся: "
            f"апскейл делает картинку крупнее, но судимость лица этим не "
            f"чинится — нужна доводка (`refine`) или более тесная кадрировка. "
            f"{tail}")
    return row


def vram_delta(width: int, height: int, *, factor: int | None = None,
               tile: int | None = None) -> dict:
    """Сколько ДОБАВЛЯЕТ апскейл к пику видеопамяти. РАСЧЁТ, не замер.

    ГЛАВНЫЙ ВЫВОД, ради которого функция написана: при тайловой нарезке пик
    НЕ ЗАВИСИТ от размера кадра. Swin2SR держит по вектору ширины
    `SR_EMBED_DIM` на КАЖДЫЙ входной пиксель (`patch_size: 1` в конфиге —
    токен на пиксель, а не на патч 16x16, как у обычного ViT), поэтому память
    линейна по площади входа; нарезка на тайлы превращает эту линейность из
    свойства кадра в свойство тайла, а площадь кадра уходит во ВРЕМЯ, то есть
    в число тайлов. Для карты на 6 ГБ это разница между «влезет» и «как
    повезёт».

    ЧТО В ЭТОМ ЧИСЛЕ НЕПРАВДА, если читать его как замер. Считаются две
    заведомо присутствующие величины: одна карта признаков на тайл
    (`tile_px * SR_EMBED_DIM * 4` байт, fp32) и одна матрица оконного внимания
    (`tile_px / window^2` окон на `SR_HEADS` голов по `window^2 x window^2`).
    Настоящий пик КРАТНО больше: живых карт признаков одновременно несколько
    (вход блока, остаток, выход), у Swin V2 добавляются буферы косинусного
    внимания и логит-масштаба, а аллокатор округляет. Множителя без замера на
    карте мы не знаем и выдумывать его не будем — поэтому здесь честно
    написано «НИЖНЯЯ ГРАНИЦА», а не «сколько нужно».

    Веса — единственное абсолютное число: `SR_WEIGHTS_BYTES` байт fp32,
    ИЗМЕРЕНО по HF API. Их немного (около 0.05 ГБ), но добавляются они
    поверх всего, что уже лежит на карте, — а на 6 ГБ после SD1.5, модуля
    движения и адаптера лица свободного места считаные сотни мегабайт.
    Поэтому правильный порядок на карте: освободить пайплайн, потом грузить
    апскейлер.
    """
    factor = UPSCALE_FACTOR if factor is None else factor
    tile = SR_TILE if tile is None else tile
    if width <= 0 or height <= 0 or tile <= 0:
        raise ValueError(f"размеры обязаны быть положительными: "
                         f"{width}x{height}, тайл {tile}")

    # Считается ПО ДОПОЛНЕННОМУ тайлу, а не по запрошенному: процессор
    # добавляет SR_PAD_DIVISOR к каждой стороне ВСЕГДА (см. докстринг модуля),
    # и модель работает на дополненных пикселях. Разница на тайле 256 — 6%
    # памяти, и терять её в расчёте, который и так нижняя граница, незачем.
    tile_px = ((min(tile, width) + SR_PAD_DIVISOR)
               * (min(tile, height) + SR_PAD_DIVISOR))
    feature_b = tile_px * SR_EMBED_DIM * 4
    windows = max(1, tile_px // (SR_WINDOW * SR_WINDOW))
    attention_b = windows * SR_HEADS * (SR_WINDOW ** 4) * 4
    out_b = width * factor * height * factor * 3 * 4
    weights_gb = round(SR_WEIGHTS_BYTES / 2 ** 30, 4)
    peak = round((feature_b + attention_b) / 2 ** 30, 4)
    tiles = len(tile_grid(width, height, tile=tile))
    return {
        "weights_gb": weights_gb,
        "feature_map_gb": round(feature_b / 2 ** 30, 4),
        "attention_gb": round(attention_b / 2 ** 30, 4),
        "peak_gb_lower_bound": peak,
        "output_buffer_gb": round(out_b / 2 ** 30, 4),
        "tiles": tiles,
        "measured": False,
        "note": (f"РАСЧЁТ, не замер (карты в среде разработки нет). Веса "
                 f"{weights_gb} ГБ ИЗМЕРЕНЫ по HF API; активации — НИЖНЯЯ "
                 f"ГРАНИЦА {peak} ГБ на тайл {tile}x{tile} и от размера кадра "
                 f"НЕ ЗАВИСЯТ: кадр {width}x{height} режется на {tiles} тайлов, "
                 f"то есть площадь уходит во время, а не в память. Настоящий "
                 f"пик кратно больше: в модели {SR_DEPTH} блоков, живых карт "
                 f"признаков одновременно несколько, и множитель без замера на "
                 f"карте неизвестен. Запасной путь Lanczos не требует ни "
                 f"весов, ни активаций"),
    }


def tile_grid(width: int, height: int, *, tile: int | None = None,
              overlap: int | None = None) -> list:
    """Нарезка входа на перекрывающиеся тайлы: список (x0, y0, x1, y1).

    Три свойства, которые обязаны держаться и на которые есть тесты:
    покрытие (каждый пиксель кадра попал хотя бы в один тайл), перекрытие
    (соседние тайлы делят полосу шириной `overlap`) и отсутствие огрызков
    (последний тайл прижимается к краю, а не режется в полосу шириной в
    несколько пикселей — на огрызке у модели нет контекста, и шов виден).
    """
    tile = SR_TILE if tile is None else tile
    overlap = SR_TILE_OVERLAP if overlap is None else overlap
    if width <= 0 or height <= 0:
        raise ValueError(f"размер кадра обязан быть положительным: {width}x{height}")
    if tile <= 0:
        raise ValueError(f"сторона тайла обязана быть положительной: {tile}")
    if overlap < 0 or overlap >= tile:
        raise ValueError(f"перекрытие {overlap} обязано быть от 0 и меньше "
                         f"стороны тайла {tile}")

    def starts(total):
        if total <= tile:
            return [0]
        step = tile - overlap
        out = list(range(0, total - tile + 1, step))
        if out[-1] + tile < total:
            out.append(total - tile)
        return out

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


# ------------------------------------------------------------- резкость


def high_frequency_share(gray, *, above: float | None = None) -> float:
    """Доля спектральной энергии выше `above` от Найквиста. Мера «выдумки».

    `gray` — двумерный массив яркости в [0, 1].

    Две обязательные подготовки, и обе поставлены после того, как метрика
    один раз соврала:

    * **вычитание среднего.** Без него в знаменателе сидит энергия постоянной
      составляющей, а она на порядки больше всей переменной части: белый шум,
      у которого выше половины Найквиста лежит 60% энергии по построению,
      показывал 0.078 — то есть метрика мерила яркость кадра, а не его спектр.
      Первая редакция этого модуля была именно такой, и поймал её тест на
      синтетике с известным ответом, а не глаз;
    * **окно Ханна.** Без него разрыв на границе кадра сам по себе даёт крест
      высоких частот через весь спектр, и метрика начинает мерить рамку.

    Радиус нормируется на диагональ, поэтому величина сравнима между кадрами
    РАЗНОГО размера — а сравнивать приходится именно их: вход 512x768 и выход
    1024x1536. Для увеличенного вдвое кадра полоса выше 0.5 — та самая, где у
    входа не было ничего (`INVENTION_BAND`).

    ПРОВЕРЯЕМОЕ СВОЙСТВО, а не только сравнение: у белого шума доля обязана
    сойтись с ДОЛЕЙ ПЛОЩАДИ спектра выше порога — для 0.5 это 1 - pi/8 ~ 0.61.
    Это и есть тест, который метрику сторожит: подгонять его нечем, ответ
    известен из геометрии.
    """
    import numpy as np

    above = INVENTION_BAND if above is None else above
    a = np.asarray(gray, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"ожидался двумерный массив яркости, дано {a.shape}")
    if min(a.shape) < 4:
        raise ValueError(f"кадр {a.shape} слишком мал для спектра")
    a = a - a.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    energy = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    fy = np.abs(np.fft.fftshift(np.fft.fftfreq(a.shape[0]))) / 0.5
    fx = np.abs(np.fft.fftshift(np.fft.fftfreq(a.shape[1]))) / 0.5
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / np.sqrt(2.0)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(energy[r >= above].sum() / total)


def _gray(image):
    """PIL или массив -> двумерная яркость в [0, 1]."""
    import numpy as np

    convert = getattr(image, "convert", None)
    if convert is not None:
        return np.asarray(convert("L"), dtype=np.float64) / 255.0
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a / 255.0 if a.max() > 1.0 else a


def sharpness_row(frame_path, *, factor: int | None = None,
                  size=(512, 768)) -> dict:
    """Три строки резкости на ОДНОМ настоящем кадре: исходник, Lanczos, веса.

    Именно эта функция стоит за таблицей в докстринге модуля, и она же —
    команда, которой числа пересчитываются. Если весов нет, третья строка
    приходит `None`, а не подменяется Lanczos'ом: «не смогли» — отдельный
    исход.

    Кадр берётся ЦЕНТРАЛЬНЫМ КРОПОМ, а не уменьшением, и это не мелочь:
    ИЗМЕРЕНО, что подготовка сдвигает измеряемую величину в 2.5 раза. Тот же
    кадр `kit/driving/0000.jpg` в верхней полосе даёт 0.000553 родными
    пикселями и 0.001401 после приведения к 512x768 — уменьшение 720x1278 в
    512x768 и меняет пропорции (1:1.78 против 1:1.5), и упаковывает настоящую
    фактуру ближе к Найквисту. То есть первая редакция этого замера мерила
    собственную подготовку наравне с кадром. Кроп не трогает ни одного пикселя.

    На ОТНОШЕНИИ Lanczos к Swin2SR подготовка почти не сказывается (оба
    апскейлера получают один и тот же вход), но абсолютные числа в таблице
    были бы числами подготовки, а не кадра. Уменьшение остаётся запасным путём
    для кадров МЕЛЬЧЕ `size` и помечено в ответе (`prepared`).
    """
    from PIL import Image

    factor = UPSCALE_FACTOR if factor is None else factor
    w, h = int(size[0]), int(size[1])
    with Image.open(frame_path) as raw:
        full = raw.convert("RGB")
        if full.width >= w and full.height >= h:
            x0, y0 = (full.width - w) // 2, (full.height - h) // 2
            src, prepared = full.crop((x0, y0, x0 + w, y0 + h)), "crop"
        else:
            src, prepared = full.resize((w, h), Image.LANCZOS), "resize"
    rows = {"frame": str(frame_path), "size": (w, h), "factor": factor,
            "prepared": prepared, "source": _sharpness(src),
            "lanczos": _sharpness(lanczos(src, factor=factor)),
            "swin2sr": None, "weights_error": None}
    try:
        up, _ = _swin2sr(src, factor=factor)
    except Exception as exc:                       # noqa: BLE001 — причина в отчёт
        rows["weights_error"] = f"{type(exc).__name__}: {exc}"
        return rows
    rows["swin2sr"] = _sharpness(up)
    return rows


def _sharpness(image) -> dict:
    """Три числа резкости одного кадра — ровно те, что в MEASURED_HF_SHARE."""
    import numpy as np

    g = _gray(image)
    lap = (-4.0 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return {"above_half": round(high_frequency_share(g, above=0.5), 6),
            "above_quarter": round(high_frequency_share(g, above=0.25), 6),
            "lapvar": round(float(lap[2:-2, 2:-2].var()) * 1e4, 4)}


# ---------------------------------------------------------------- ступень


def why_unavailable(repo: str | None = None) -> str | None:
    """Почему веса недоступны, или None если доступны. Причина, а не «нет».

    Различает то, что на этом проекте уже путали: пакета нет, весов нет в
    кэше и сети нет. Лечатся они по-разному, и «апскейл не поехал» без
    причины стоит часа.
    """
    repo = SR_REPO if repo is None else repo
    try:
        import torch                                        # noqa: F401
    except ImportError:
        return "torch не установлен — pip install -r requirements-gpu.txt"
    try:
        from transformers import Swin2SRForImageSuperResolution  # noqa: F401
    except ImportError:
        return ("transformers без Swin2SR — нужен пакет transformers "
                "(класс Swin2SRForImageSuperResolution)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
    except Exception as exc:                       # noqa: BLE001 — причина в отчёт
        return (f"весов {repo} нет в локальном кэше ({type(exc).__name__}); "
                f"скачать: huggingface-cli download {repo} "
                f"(лицензия {SR_LICENSE}, ~{SR_WEIGHTS_BYTES // 10 ** 6} МБ)")
    return None


#: Загруженный апскейлер, чтобы не читать веса на КАЖДЫЙ кадр. Клип — это 16
#: кадров, и загрузка на каждый означала бы шестнадцать чтений 48 МБ и
#: шестнадцать разборов конфига. Кэш процессный и снимается `free_upscaler`.
_LOADED: dict = {}


def _load_swin(device: str | None = None):
    """(процессор, модель), загруженные один раз на процесс."""
    from transformers import (Swin2SRForImageSuperResolution,
                              Swin2SRImageProcessor)

    key = device or "cpu"
    if key not in _LOADED:
        model = Swin2SRForImageSuperResolution.from_pretrained(SR_REPO).eval()
        if device:
            model = model.to(device)
        _LOADED[key] = (Swin2SRImageProcessor.from_pretrained(SR_REPO), model)
    return _LOADED[key]


def free_upscaler() -> None:
    """Снять апскейлер с устройства. На 6 ГБ это не гигиена, а условие работы.

    Ступень последняя, и после неё держать на карте ещё одну модель незачем;
    вызывающий, который гонит несколько клипов подряд, освобождает её сам.
    """
    _LOADED.clear()


def _swin2sr(image, *, factor: int, tile: int | None = None,
             overlap: int | None = None, device: str | None = None):
    """Swin2SR по тайлам. Возвращает (изображение, отчёт). Требует весов."""
    import torch

    proc, model = _load_swin(device)
    if model.config.upscale != factor:
        raise ValueError(
            f"чекпойнт {SR_REPO} обучен на x{model.config.upscale}, а просят "
            f"x{factor}. Множитель берётся из конфига весов, а не из имени "
            f"репозитория и не из желания вызывающего")

    def run(patch):
        import numpy as np
        from PIL import Image

        with torch.no_grad():
            batch = proc(patch, return_tensors="pt")
            got = model(**{k: v.to(model.device) if hasattr(v, "to") else v
                           for k, v in batch.items()})
        arr = got.reconstruction.squeeze(0).clamp(0, 1).float().cpu()
        arr = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        # Процессор дополняет вход ВСЕГДА, даже уже кратный SR_PAD_DIVISOR
        # (проверено по исходнику установленного transformers: добавка равна
        # `(h // d + 1) * d - h`). Без обрезки по правому и нижнему краю
        # остаётся полоса зеркального отражения шириной d * factor.
        return Image.fromarray(arr).crop(
            (0, 0, patch.width * factor, patch.height * factor))

    return _tiled(image, factor=factor, tile=tile, overlap=overlap, fn=run)


def _tiled(image, *, factor: int, fn, tile: int | None = None,
           overlap: int | None = None):
    """Применить `fn` к тайлам и собрать кадр без шва. Отчёт — вторым.

    Сборка идёт с ЛИНЕЙНЫМ весом от края тайла: в полосе перекрытия два
    соседа складываются с весами, дающими в сумме единицу. Проверяется это
    без весов вовсе — подстановкой точного увеличителя (`_nearest`): у него
    соседние тайлы обязаны совпадать в перекрытии ПОБИТОВО, а значит и
    собранный кадр обязан совпасть с несобранным. Шов, который «почти не
    виден», так не поймать; побитово — ловится.
    """
    import numpy as np
    from PIL import Image

    tile = SR_TILE if tile is None else tile
    overlap = SR_TILE_OVERLAP if overlap is None else overlap
    # Приводится к RGB на входе: чекпойнт трёхканальный (`num_channels: 3` в
    # конфиге), и одноканальный кадр он всё равно не примет — лучше это
    # случится здесь и явно, чем формой тензора внутри модели.
    image = image.convert("RGB")
    boxes = tile_grid(image.width, image.height, tile=tile, overlap=overlap)
    out_w, out_h = image.width * factor, image.height * factor
    acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    wsum = np.zeros((out_h, out_w, 1), dtype=np.float64)
    for x0, y0, x1, y1 in boxes:
        patch = image.crop((x0, y0, x1, y1))
        got = np.asarray(fn(patch), dtype=np.float64)
        ph, pw = got.shape[0], got.shape[1]
        w = _ramp(pw, ph, factor * overlap,
                  left=x0 > 0, top=y0 > 0,
                  right=x1 < image.width, bottom=y1 < image.height)
        acc[y0 * factor:y0 * factor + ph, x0 * factor:x0 * factor + pw] += got * w
        wsum[y0 * factor:y0 * factor + ph, x0 * factor:x0 * factor + pw] += w
    blended = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), acc)
    img = Image.fromarray(blended.round().clip(0, 255).astype("uint8"))
    return img, {"tiles": len(boxes), "tile": tile, "overlap": overlap,
                 "size_out": (out_w, out_h)}


def _ramp(w: int, h: int, band: int, *, left: bool, top: bool,
          right: bool, bottom: bool):
    """Веса тайла: линейный спад только на тех краях, где есть сосед."""
    import numpy as np

    def axis(n, lo, hi):
        v = np.ones(n, dtype=np.float64)
        k = min(band, n // 2)
        if k > 0:
            ramp = (np.arange(k) + 1.0) / (k + 1.0)
            if lo:
                v[:k] = ramp
            if hi:
                v[n - k:] = ramp[::-1]
        return v

    return (axis(h, top, bottom)[:, None] * axis(w, left, right)[None, :])[..., None]


def _nearest(patch, factor: int):
    """Точное увеличение повтором пикселя. Не апскейлер — измерительный эталон."""
    from PIL import Image

    return patch.resize((patch.width * factor, patch.height * factor),
                        Image.NEAREST)


def lanczos(image, *, factor: int | None = None):
    """Запасной путь. Ничего не выдумывает — ИЗМЕРЕНО, а не заявлено.

    На настоящем кадре (`MEASURED_HF_SHARE`): выше Найквиста входа Lanczos
    кладёт 0.000023 против 0.000049 у Swin2SR, а в верхней октаве — 0.000457
    против 0.001162, то есть в 2.54 раза меньше. В полосе, где у входа не было
    информации, у него её и на выходе практически нет. Ровно этим он
    безопаснее для гейта и ровно поэтому он не добавляет резкости: крупнее —
    да, детальнее — нет.

    Проверяемое свойство сильнее замера на одном кадре: на белом шуме, где
    выше половины Найквиста лежит 60% энергии, после Lanczos x2 остаётся
    меньше половины процента — интерполяция физически не может положить
    энергию выше Найквиста входа, и тест это сторожит.
    """
    from PIL import Image

    factor = UPSCALE_FACTOR if factor is None else factor
    return image.resize((image.width * factor, image.height * factor),
                        Image.LANCZOS)


def upscale_frames(frames, seal: GateSeal, *, factor: int | None = None,
                   method: str = "auto", tile: int | None = None,
                   overlap: int | None = None, device: str | None = None):
    """Апскейл кадров, УЖЕ посуженных гейтом. Возвращает (кадры, отчёт).

    `seal` обязателен и обязан накрывать ровно эти кадры — см. докстринг
    модуля про механизм. Порядок «сначала апскейл, потом гейт» здесь не
    отговаривается, а не собирается: печати на неотсуженные пиксели взять
    неоткуда.

    `method`:

    * `"auto"` — веса, если они есть; иначе Lanczos, и в отчёте написано,
      почему именно;
    * `"weights"` — только веса. Их отсутствие — ОШИБКА, а не тихий откат:
      демо, обещавшее Swin2SR и показавшее Lanczos, врёт о себе;
    * `"lanczos"` — только запасной путь, явным решением вызывающего;
    * `"none"` — не апскейлить. Исход `"skipped"`, кадры возвращаются как
      были: показать исходное разрешение честнее, чем подменить его молча.

    `outcome` в отчёте трёхзначен (`weights` / `fallback` / `skipped`), и
    булева «всё хорошо» рядом с ним нет намеренно: она бы схлопнула эти три
    исхода в два, а разница между «сделали весами» и «сделали запасным путём»
    — это и есть разница между «резче» и «просто крупнее».
    """
    factor = UPSCALE_FACTOR if factor is None else factor
    if not isinstance(seal, GateSeal):
        raise ValueError(
            "апскейл без печати гейта не выполняется: `seal` обязан быть "
            "GateSeal из seal_verdict(frames, verdict, gate=...). Печать "
            "доказывает, что ЭТИ пиксели уже посужены — апскейлер выдумывает "
            "детали, и гейт после него мерил бы выдумку, а не модель")
    frames = list(frames)
    if not frames:
        raise ValueError("апскейлить нечего: пустая последовательность кадров")
    ok, missing = seal.covers(frames)
    if not ok:
        raise ValueError(
            f"печать гейта {seal.gate} не накрывает {len(missing)} из "
            f"{len(frames)} кадров: судили не эти пиксели. Печать привязана к "
            f"СОДЕРЖИМОМУ, а не к именам файлов, поэтому подменённый или "
            f"пересохранённый кадр обязан здесь останавливаться")
    if method not in ("auto", "weights", "lanczos", "none"):
        raise ValueError(f"неизвестный метод апскейла: {method!r}")

    # `factor_asked` и `factor` разведены намеренно: попросили x2, а исход
    # «не выполнен» — это применённый множитель 1 при запрошенном 2, и в
    # отчёте обязано быть видно и то, и другое.
    base = {"gate": seal.gate, "verdict_outcome": seal.verdict_outcome,
            "frames": len(frames), "factor_asked": factor, "repo": SR_REPO,
            "license": SR_LICENSE, "method_asked": method}
    if method == "none" or factor == 1:
        why = ("не просили" if method == "none" else
               "множитель 1 — увеличивать не во что")
        return frames, {**base, "outcome": "skipped", "method": None,
                        "factor": 1, "invents": None,
                        "size_out": (frames[0].width, frames[0].height),
                        "note": (f"апскейл НЕ ВЫПОЛНЕН ({why}). Кадры отданы "
                                 f"как есть — это отдельный исход, а не "
                                 f"неудачный апскейл: показывать исходное "
                                 f"разрешение честно, подменять его молча — нет")}

    reason = why_unavailable() if method in ("auto", "weights") else "не просили"
    if method == "weights" and reason:
        raise ValueError(
            f"апскейл весами затребован явно, а весов нет: {reason}. Тихого "
            f"отката на Lanczos здесь нет намеренно — демо, обещавшее "
            f"Swin2SR и показавшее интерполяцию, врёт о себе")
    if method == "lanczos" or reason:
        out = [lanczos(f, factor=factor) for f in frames]
        note = (f"апскейл ЗАПАСНЫМ ПУТЁМ (Lanczos x{factor}). Деталей он не "
                f"добавляет — ИЗМЕРЕНО: в верхней октаве "
                f"{MEASURED_HF_SHARE['lanczos']['above_quarter']} против "
                f"{MEASURED_HF_SHARE['swin2sr']['above_quarter']} у Swin2SR на "
                f"той же сетке. Кадр стал крупнее, но не детальнее — и ровно "
                f"поэтому он безопаснее для любого измерителя")
        if reason and method != "lanczos":
            note += f". Веса не использованы: {reason}"
        rep = {**base, "outcome": "fallback", "method": "lanczos",
               "factor": factor, "invents": False, "weights_reason": reason,
               "note": note}
    else:
        out, tiled = [], None
        for f in frames:
            got, tiled = _swin2sr(f, factor=factor, tile=tile,
                                  overlap=overlap, device=device)
            out.append(got)
        rep = {**base, "outcome": "weights", "method": "swin2sr",
               "factor": factor, "invents": True,
               "tiles": (tiled or {}).get("tiles"),
               "note": (f"апскейл ВЕСАМИ ({SR_REPO}, {SR_LICENSE}) x{factor}. "
                        f"Детали ПОРОЖДЕНЫ: в верхней октаве энергии в "
                        f"{MEASURED_INVENTION_RATIO}x больше, чем после "
                        f"интерполяции, дисперсия лапласиана в "
                        f"{MEASURED_LAPVAR_RATIO}x больше (ИЗМЕРЕНО на одной "
                        f"сетке). Выше Найквиста исходника при этом пусто — "
                        f"апскейлер переписывает то, что было, а не сочиняет "
                        f"поверх. Вердикт относится к кадрам ДО этой ступени "
                        f"(печать {seal.gate}), а не к этим")}
    _PRODUCED.update(frame_digest(f) for f in out)
    rep["size_out"] = (out[0].width, out[0].height)
    return out, rep
