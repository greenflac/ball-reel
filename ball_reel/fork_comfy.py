"""Поток E форка: производство графа ComfyUI. Две части, и вторая — рабочая.

ЧИТАТЬ ОТСЮДА. 18.08.2026 решением владельца производство переехало со ШТАТНЫХ
нод на обёртку `kijai/ComfyUI-WanVideoWrapper`: цикл по окнам живёт ВНУТРИ её
сэмплера, и длина ролика становится числом `num_frames`, а не графом, который
надо строить руками. Рабочее производство — `derive_wrapper`, ЧАСТЬ II, в конце
файла; там же геометрия 480x832, 30 к/с, `pose_strength` и реестр имён,
доказанных скачанным исходником.

ЧАСТЬ I ниже оставлена целиком и намеренно. Она описывает, что делает ШТАТНАЯ
`WanAnimateToVideo` — включая опровержение §1a, которого больше нигде нет, — и
её приборы (`audit_lock`, `audit_weights`, `graph_weights`, `provenance_of`,
`render`) обслуживают обе части. Производством по умолчанию она быть перестала.

---

~~Граф ПРОИЗВОДИТСЯ из официального, а не сочиняется~~ — верно для части I.

ПОЧЕМУ ПРОИЗВОДИТСЯ. Сочинённый граф — это набор допущений о чужом контракте,
каждое из которых выяснится на арендованной машине по одному. Официальный
темплейт лежит в `workflows/upstream/` (MIT, Comfy Org, хэши записаны), и всё,
что нам нужно, — вырезать из него одну ветку и подставить свою.

---

ЧТО ВЫРЕЗАЕТСЯ И ПОЧЕМУ ДВЕ ПРИЧИНЫ, А НЕ ОДНА.

Первая: три вендорских набора кастомных нод тянут за собой лицензионную цепочку,
которую придётся проверять целиком до релиза.

Вторая, и она непреодолимая: среди них `PointsEditor`, и он **ждёт, что человек
нажмёт Shift и щёлкнет мышью по персонажу** — так задаются положительные точки
для SAM2. В продукте, который принимает фотографию и отдаёт ролик, человека с
мышью посередине быть не может. Это не «неудобно», это отменяет продукт.

Поэтому вся ветка препроцессинга — сегментация SAM2, DWPose, подгонка
разрешения — заменяется загрузкой ПОСЧИТАННЫХ НАМИ последовательностей: маски
из `fork_mask`, два канала условий из `fork_channels`.

---

ЧТО ПРОВЕРЯЕТСЯ ЗДЕСЬ И ЧТО НЕ ПРОВЕРЯЕТСЯ.

НЕПРОВЕРЕНО (Ц4): **граф не исполнялся.** ComfyUI в этой среде нет, и
поставить его — работа другой смены. Проверка тут СТРУКТУРНАЯ: какие типы нод
остались, куда идут связи, на месте ли обязательные входы `WanAnimateToVideo`.
Что произведённый граф действительно поедет, здесь не показано.

Ц10 СОБЛЮДЁН ТАК. Существование имени ноды доказывается ФАЙЛОМ ТЕМПЛЕЙТА: тип,
встречающийся в официальном графе, существует наверняка. Типы, которых там нет,
модуль вводить не отказывается — иначе замену нечем собрать, — но каждый такой
тип помечает и выносит в отчёт отдельным списком. Молча ввести имя, которого
может не быть, нельзя: на этом проекте уже писали код против выдуманного
репозитория с весами.

Второй способ доказать имя — ИСХОДНИК COMFYUI, скачанный командой. Он появился
17.08.2026: `ImageToMask` был единственным недоказанным именем, и вместо того
чтобы оставить пометку навсегда, имя проверено загрузкой `nodes_mask.py`. См.
`PROVEN_BY_SOURCE` — там URL, sha256 скачанного тела и номер строки.

---

ЧТО ДЕЛАЕТ `render` И ПОЧЕМУ ОН НЕ УМЕЕТ ВРАТЬ.

ComfyUI в этой среде нет и ставить его запрещено (§10 хэндофа). Значит кадры
взять неоткуда, и адаптер это ГОВОРИТ, а не изображает. Устройство — в докстринге
`render`; главное свойство: **мок физически не может вернуть `PASS`**, а кадры
мока помечены в пикселях, и настоящий бэкенд, вернувший помеченные кадры, ловится
как подлог. Проверка идёт в обе стороны, потому что мок, выдающий себя за прогон,
— ровно тот дефект, против которого написан весь этот проект.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

#: Официальный темплейт и его хэш. Хэш здесь, а не только в README, потому что
#: проверять его должен КОД перед производством, а не человек при чтении.
UPSTREAM = "workflows/upstream/video_wan2_2_14B_animate.json"
UPSTREAM_SHA256 = "06ad8b95e64215328a2a3d2f90495b5bab3e251175e4258d01b977f6dcdecb69"

#: Наборы кастомных нод, от которых форк отказывается. Имена — в точности те,
#: что стоят в `properties.cnr_id` узлов темплейта, в его же регистре.
#: Сравнение регистронезависимое: записка темплейта пишет `ComfyUI-KJNodes`, а
#: узлы — `comfyui-kjnodes`, и это разошлось бы молча.
CUSTOM_PACKS = ("comfyui_controlnet_aux", "comfyui-kjnodes",
                "comfyui-segment-anything-2")

#: Ключ, по которому узел признаётся своим или чужим.
PACK_KEY = "cnr_id"

#: Пак, который считается встроенным.
CORE_PACK = "comfy-core"

#: Узел, ради которого вырезание непреодолимо: ЖДЁТ ЧЕЛОВЕКА С МЫШЬЮ.
MANUAL_NODE = "PointsEditor"

#: Обязательные входы `WanAnimateToVideo` — контракт, ради которого всё
#: остальное и делается. Вычитаны из README темплейта, который, в свою очередь,
#: вычитал их из `links`.
ANIMATE_INPUTS = ("reference_image", "face_video", "pose_video",
                  "background_video", "character_mask")

#: Наши входы: что мы подставляем вместо вырезанной ветки. Ключ — вход
#: `WanAnimateToVideo`, значение — что туда поедет из нашего пайплайна.
OUR_SOURCES = {
    "face_video": "fork_channels.render_sequence -> face/",
    "pose_video": "fork_channels.render_sequence -> body/",
    "character_mask": "fork_mask.sequence (уже блокифицирована на 32)",
}

#: Лок-файл: воркфлоу, веса и происхождение каждого утверждения о контракте.
#: Лежит рядом с темплейтом, читается кодом, а не только человеком.
LOCKFILE = "workflows/fork_stack.lock.json"

#: Типы нод, чьё существование доказано НЕ темплейтом, а исходником ComfyUI,
#: скачанным командой. Ключ — тип, значение — чем именно доказан.
#:
#: ЗАЧЕМ ОТДЕЛЬНЫЙ РЕЕСТР, А НЕ ПРОСТО СНЯТАЯ ПОМЕТКА. Пометка «проверено»,
#: не называющая чем, через месяц неотличима от пометки «вроде помню». Здесь
#: лежит URL, sha256 ТЕЛА, которое я читал, номер строки и дата: любой может
#: скачать заново, сверить sha256 и увидеть, читал ли я тот же файл.
PROVEN_BY_SOURCE = {
    "ImageToMask": {
        "url": ("https://raw.githubusercontent.com/comfyanonymous/ComfyUI/"
                "master/comfy_extras/nodes_mask.py"),
        "body_sha256": ("f235d52703229b0bb7840cc48f0fffe6b00bf9398e119aa7"
                        "d7f2fc9f3bdaba12"),
        "line": 131,
        "checked": "2026-08-17",
        "quote": ('node_id="ImageToMask", inputs=[Image("image"), '
                  'Combo("channel", options=["red","green","blue","alpha"])], '
                  'outputs=[Mask]'),
        "note": ("вход `image`, комбо `channel` со значением `red` в списке, "
                 "выход MASK — ровно то, как мы его ставим"),
    },
    # ЧЕТВЁРТЫЙ СТОРОННИЙ ПАК, и он единственный, который форк впускает
    # обратно, — только потому, что владелец платит за него сознательно, а не
    # потому, что он приехал с темплейтом. Оба имени доказаны исходником, а не
    # памятью: у моделей примерно пятая часть предлагаемых имён не существует
    # (Ц10), и `CLIPLoaderGGUF` — ровно тот случай, где догадка «наверное, есть
    # парный загрузчик» была бы правдоподобной и непроверенной.
    "UnetLoaderGGUF": {
        "url": ("https://raw.githubusercontent.com/city96/ComfyUI-GGUF/"
                "main/nodes.py"),
        "body_sha256": ("16be3b08b13de6279fc432addc628320019fcb24963cbc6b5"
                        "2b248de8f06316e"),
        "line": 135,
        "checked": "2026-08-17",
        "quote": ('class UnetLoaderGGUF: INPUT_TYPES -> {"required": '
                  '{"unet_name": (unet_names,)}}, RETURN_TYPES = ("MODEL",); '
                  'в NODE_CLASS_MAPPINGS строка 322'),
        "note": ("единственный виджет — имя файла, выход MODEL: подставляется "
                 "туда же, где стоял UNETLoader"),
    },
    "CLIPLoaderGGUF": {
        "url": ("https://raw.githubusercontent.com/city96/ComfyUI-GGUF/"
                "main/nodes.py"),
        "body_sha256": ("16be3b08b13de6279fc432addc628320019fcb24963cbc6b5"
                        "2b248de8f06316e"),
        "line": 200,
        "checked": "2026-08-17",
        "quote": ('class CLIPLoaderGGUF: {"required": {"clip_name": '
                  '(s.get_filename_list(),), "type": '
                  'base["required"]["type"]}}; в NODE_CLASS_MAPPINGS 323'),
        "note": ("виджетов ДВА, а не три: имя файла и тип энкодера. Штатный "
                 "CLIPLoader несёт третьим `device`, и перенести его виджеты "
                 "как есть значило бы отдать ноде лишнее значение. Значение "
                 "`wan` берётся из списка штатного CLIPLoader (nodes.py, "
                 "тело sha256 38f918673c5300bc..., в списке есть)"),
    },
}

#: Пак, из которого приезжают загрузчики GGUF. Лицензия проверена ДО встраивания
#: (Ц5): `curl .../ComfyUI-GGUF/main/LICENSE` -> «Apache License Version 2.0»,
#: `pyproject.toml` -> `name = "comfyui-gguf"`, `version = "2.0.0"`, зависимости
#: `gguf>=0.13.0`, `sentencepiece`, `protobuf`. Ни `non-commercial`, ни
#: `research-only` в заголовке нет.
GGUF_PACK = "comfyui-gguf"

#: ---------------------------------------------------------------------------
#: §1a. СНЯТО: «вне маски пиксели копируются». Проверено мной независимо.
#:
#: ~~«background_video — подложка, поверх которой врисовывается персонаж, а вне
#: маски идёт нетронутая копия драйвинга»~~ — ЛОЖНО. Так написано даже в
#: вендорской записке, которую мы скачали вместе с темплейтом:
#: `workflows/upstream/WanAnimateToVideo.doc.md:27` говорит «Background video to
#: composite with generated content». Слово `composite` в НОДЕ не встречается ни
#: разу. Записку править нельзя — она первоисточник, лежит как скачана; поэтому
#: опровержение живёт здесь, рядом с кодом, который иначе унаследовал бы ошибку.
#:
#: Что в действительности (`comfy_extras/nodes_wan.py`, master, скачан
#: 17.08.2026, sha256 тела dcd8b81d1225d84e...; класс `WanAnimateToVideo` —
#: строки 1113..1252):
#:   1166  image = torch.ones((length,h,w,3)) * 0.5      — СЕРЫЙ ХОЛСТ
#:   1219  image[ref:bg.shape[0]] = background_video[ref:] — драйвинг В ХОЛСТ
#:   1240  concat_latent_image = cat(..., vae.encode(image))
#:   1245  conditioning_set_values(..., concat_latent_image, concat_mask)
#:   1248  latent = torch.zeros(...)                     — ВОЗВРАТ НУЛЕЙ
#:   noise_mask                                          — НЕ СТАВИТСЯ
#: Единственный `noise_mask` во всём файле — строка 1454, и она внутри ДРУГОГО
#: класса, `Wan22ImageToVideoLatent` (1414..1458). Проверено границами классов,
#: а не глазами: соседние классы начинаются на 1253, 1382, 1414, 1459.
#:
#: Следствие для этого модуля: `background_video` — это ОБУСЛОВЛИВАНИЕ.
#: Не подложка: весь кадр сэмплируется из шума и декодируется через VAE.
#: Ни одна строка здесь не смеет обещать сохранность пикселей вне маски.
NO_COMPOSITE = {
    "claim": "у WanAnimateToVideo нет noise_mask и нет композитинга",
    "source": ("https://raw.githubusercontent.com/comfyanonymous/ComfyUI/"
               "master/comfy_extras/nodes_wan.py"),
    "body_sha256": ("dcd8b81d1225d84e8b0c6743675d59772449899018e9f50bb2726"
                    "7dbb8962002"),
    "checked": "2026-08-17",
    "evidence": ("класс 1113..1252; серый холст 1166; драйвинг в холст 1219; "
                 "возврат torch.zeros 1248; единственный noise_mask 1454 "
                 "принадлежит Wan22ImageToVideoLatent (1414..1458); слова "
                 "`composite` в классе нет"),
    "refutes": ("workflows/upstream/WanAnimateToVideo.doc.md:27 — «Background "
                "video to composite with generated content». Записка вендора "
                "ошибается; править её нельзя, она первоисточник"),
}

#: Три исхода (Е1: те же три слова, что у остальных потоков).
from .fork_identity import FAIL, PASS, UNMEASURED  # noqa: E402

#: ---------------------------------------------------------------------------
#: ОТКУДА КАДРЫ. Вопрос другой, чем «годен ли результат», поэтому и слова другие.
#: Смешивать их нельзя: `PASS` про качество, `RAN` про происхождение, и отчёт,
#: где одно подменяет другое, — это и есть мок, выдающий себя за прогон.
RAN, MOCK, NOTHING = "прогнали", "мок", "не смогли"

#: Метка мока, вписываемая В САМИ ПИКСЕЛИ. Не поле в словаре: поле теряется при
#: первом же `result["frames"]`, а пиксели едут с кадрами куда угодно — в файл,
#: в другой процесс, в чужую функцию, которая про наш словарь не знает.
MOCK_MAGIC = b"MOCK:NOT-GENERATED"

#: Минимальная ширина кадра. Обоснование НЕ в метке: §5.4 хэндофа — маска
#: квантуется до блоков 32 px, и кадр уже одного блока для модели бессмыслен.
#: Побочно этого хватает и на метку (18 байт), но причина не в ней.
MIN_SIDE = 32

#: Кратности из §3.2 и из первоисточника `WanAnimateToVideo.doc.md:20`.
#: Проза §3.2 говорит «length кратна 4», первоисточник — «default: 77, step: 4»
#: при минимуме 1. ~~«длина кратна 4»~~ снято: 77 на 4 не делится, и буквальное
#: чтение прозы забраковало бы штатную геометрию стека. Верх за первоисточником.
SIDE_MULTIPLE = 16
LENGTH_STEP = 4
LENGTH_BASE = 1
FACE_SIDE = 512


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_upstream(path: str | Path | None = None, *,
                  check_hash: bool = True) -> dict:
    """Официальный темплейт, с проверкой хэша ДО разбора.

    Хэш сверяется первым (П2: дешёвая проверка раньше дорогой) и роняет
    загрузку: производить граф из файла, который кто-то подменил или обновил, и
    молчать об этом — значит унаследовать чужое изменение как своё.
    """
    p = Path(path) if path else _root() / UPSTREAM
    if not p.exists():
        raise FileNotFoundError(
            f"нет официального темплейта {p}. Он лежит в репозитории намеренно "
            f"(см. workflows/upstream/README.md): вторая сессия идёт на "
            f"арендованной машине, и зависеть от сети там нельзя.")
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if check_hash and got != UPSTREAM_SHA256:
        raise ValueError(
            f"темплейт {p.name} не тот, что записан: {got[:16]}... вместо "
            f"{UPSTREAM_SHA256[:16]}.... Если вендор обновил файл — это "
            f"НАХОДКА: перечитать, что изменилось, и обновить хэш осознанно.")
    return json.loads(raw.decode("utf-8"))


def pack_of(node: dict) -> str:
    """Из какого набора узел. Пустая строка — набор не указан (заметки)."""
    return str(node.get("properties", {}).get(PACK_KEY, "") or "")


def custom_nodes(graph: dict) -> list:
    """Узлы из трёх кастомных наборов. Список, а не флаг: важно, КАКИЕ."""
    out = []
    for n in graph.get("nodes", []):
        if pack_of(n).lower() in CUSTOM_PACKS:
            out.append({"id": n["id"], "type": n["type"],
                        "pack": pack_of(n)})
    return out


def known_types(graph: dict) -> set:
    """Типы нод, чьё существование ДОКАЗАНО файлом темплейта (Ц10).

    ~~«типы верхнего уровня»~~ — правка 18.08.2026, и она по делу, а не для
    красоты. Сборка графа обёртки объявила НЕДОКАЗАННЫМ штатный `CreateVideo`,
    который в темплейте есть — но лежит ВНУТРИ определения сабграфа, куда
    прежний разбор не заглядывал. Пометка «существование не подтверждено
    ничем» на ноде, стоящей в официальном файле трижды, — ложная тревога, а
    ложные тревоги снимают вместе с проверкой. Обход идёт по всем спискам
    `nodes`, на любой глубине: то же место уже ловило `WanAnimateToVideo`,
    которого на верхнем уровне тоже нет (см. `contract`).
    """
    found: set = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "nodes" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and isinstance(
                                item.get("type"), str):
                            found.add(item["type"])
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(graph)
    return found


def provenance_of(node_type: str, proven: set) -> str:
    """Чем доказано существование типа. Одна строка, и в ней ВСЕГДА источник.

    Три ответа, а не два (Р1): доказан темплейтом, доказан скачанным
    исходником, не доказан ничем. Третий — не «плохо», а «неизвестно», и он
    обязан доехать до отчёта отдельным списком.
    """
    if node_type in proven:
        return f"ДОКАЗАНО темплейтом: тип встречается в {UPSTREAM}"
    src = PROVEN_BY_SOURCE.get(node_type)
    if src:
        return (f"ДОКАЗАНО исходником: {src['url']}:{src['line']}, тело "
                f"sha256 {src['body_sha256'][:16]}..., скачано {src['checked']}")
    return ("НЕПРОВЕРЕНО: типа нет ни в темплейте, ни в реестре доказанных "
            "исходником — существование не подтверждено ничем")


def _links_of(graph: dict) -> list:
    return list(graph.get("links", []))


def _subgraph_feeding_animate(graph: dict) -> tuple:
    """Узел-сабграф, внутри которого лежит `WanAnimateToVideo`, и номера слотов.

    Возвращает `(id_узла, {имя_входа: номер_слота})`. Слоты берутся из
    ОБЪЯВЛЕНИЯ входов самого узла, а не из порядка связей: связи можно
    пересортировать, и номер, выведенный из их порядка, поедет молча.
    """
    for n in graph.get("nodes", []):
        names = [i.get("name") for i in n.get("inputs", [])]
        if all(req in names for req in ANIMATE_INPUTS):
            return n["id"], {name: i for i, name in enumerate(names)
                             if name in ANIMATE_INPUTS}
    return None, {}


def unfed_inputs(graph: dict) -> list:
    """Обязательные входы `WanAnimateToVideo`, в которые ничего не приходит.

    Проверка появилась после того, как аудит объявил ЧИСТЫМ граф, где
    вырезание сняло четыре питающие связи, а поставленные взамен загрузчики
    лежали неподключёнными. Кастомных нод в нём действительно не осталось —
    и поехать он не мог.
    """
    node_id, slots = _subgraph_feeding_animate(graph)
    if node_id is None:
        return []
    fed = {l[4] for l in _links_of(graph) if len(l) > 4 and l[3] == node_id}
    return sorted(name for name, slot in slots.items() if slot not in fed)


def derive(graph: dict | None = None, *,
           face_dir: str = "fork/face", pose_dir: str = "fork/body",
           mask_dir: str = "fork/mask",
           quant: str | None = None) -> dict:
    """Произвести наш граф: вырезать ветку препроцессинга, подставить свою.

    Возвращает `{"graph": ..., "removed": [...], "introduced": [...]}` — и
    список введённых типов возвращается ВСЕГДА, даже пустой. Отчёт, в котором
    поле появляется только когда есть что сказать, читается как «всё чисто»
    ровно тогда, когда сломался сбор.

    ~~«`quant` — РАЗВИЛКА §1, И ОНА НАРОЧНО НЕ РЕШЕНА ЗДЕСЬ; умолчание
    оставляет ступень такой, какая приехала с темплейтом»~~ — СНЯТО
    18.08.2026. Развилку владелец закрыл: Q4_K_M, и это же стоит в локе.
    Умолчание теперь `QUANT_DEFAULT`, и `audit_weights` на нём зелёный.
    `quant=QUANT_TEMPLATE` оставлен ровно затем, чтобы сторож имел вход, на
    котором обязан краснеть, — и чтобы прежнее состояние можно было
    воспроизвести одной строкой, а не археологией по истории.

    NB: часть II (`derive_wrapper`) ступени не имеет вовсе. Там имя файла
    диффузии приходит из лока и другого источника у него нет, так что ступень
    перестала быть выбором кода — она стала записью в локе.
    """
    # Умолчание разрешается ЗДЕСЬ, а не в сигнатуре: значение по умолчанию
    # связывается на импорте, и подмена константы модуля до него не доходит —
    # то есть мутация `QUANT_TEMPLATE` не покраснела бы ни в одном тесте. Эту
    # форму на проекте уже выгребали в семи местах (И7), и вносить её обратно
    # ради одной строки нельзя.
    quant = QUANT_DEFAULT if quant is None else quant
    if quant not in QUANTS:
        raise ValueError(f"ступень {quant!r} неизвестна, есть {QUANTS}")
    src = load_upstream() if graph is None else graph
    proven = known_types(src)
    out = copy.deepcopy(src)

    doomed = {n["id"] for n in custom_nodes(out)}
    removed = [{"id": n["id"], "type": n["type"], "pack": pack_of(n),
                "why": ("ждёт человека с мышью" if n["type"] == MANUAL_NODE
                        else "кастомный набор — лицензионная цепочка")}
               for n in out.get("nodes", []) if n["id"] in doomed]

    out["nodes"] = [n for n in out.get("nodes", []) if n["id"] not in doomed]

    # Связи, оба конца которых пережили вырезание. Оборванная связь в графе
    # Comfy — не безобидный мусор: загрузчик по ней ищет несуществующий узел.
    survivors = {n["id"] for n in out["nodes"]}
    out["links"] = [l for l in _links_of(out)
                    if len(l) > 3 and l[1] in survivors and l[3] in survivors]

    # НА МЕСТО ВЫРЕЗАННОГО — НАШИ ЗАГРУЗЧИКИ, И ОНИ ПОДКЛЮЧАЮТСЯ, А НЕ ЛЕЖАТ
    # РЯДОМ. Сначала здесь стояли три `LoadVideo` без единой связи: аудит
    # говорил «ЧИСТО», кастомных нод не осталось, оборванных связей нет — а
    # четыре входа `WanAnimateToVideo` не были подключены ни к чему. Найдено
    # разбором проводки сабграфа: вырезание сняло связи 704, 705, 706, 707.
    # Отчёт, зеленеющий на графе, который не может поехать, хуже отсутствия
    # отчёта, поэтому `audit` теперь проверяет и питание входов.
    subgraph, feeds = _subgraph_feeding_animate(src)
    introduced: list[dict] = []
    next_id = max([n["id"] for n in out["nodes"]] + [0]) + 1
    next_link = int(src.get("last_link_id", 0)) + 1

    def add(node_type: str, widgets: list, title: str, out_type: str,
            out_name: str, inputs: list | None = None) -> int:
        nonlocal next_id
        node = {"id": next_id, "type": node_type, "mode": 0,
                "inputs": inputs or [],
                "outputs": [{"name": out_name, "type": out_type, "links": []}],
                "properties": {"Node name for S&R": node_type,
                               PACK_KEY: CORE_PACK},
                "widgets_values": widgets, "title": title}
        out["nodes"].append(node)
        if node_type not in proven:
            introduced.append({"type": node_type, "id": next_id,
                               "provenance": provenance_of(node_type, proven)})
        next_id += 1
        return node["id"]

    def link(src_id: int, src_slot: int, dst_id: int, dst_slot: int, kind: str):
        nonlocal next_link
        out["links"].append([next_link, src_id, src_slot, dst_id, dst_slot,
                             kind])
        next_link += 1

    if subgraph is not None:
        for name, folder, kind in (("face", face_dir, "IMAGE"),
                                   ("pose", pose_dir, "IMAGE"),
                                   ("mask", mask_dir, "MASK")):
            target = {"face": "face_video", "pose": "pose_video",
                      "mask": "character_mask"}[name]
            slot = feeds.get(target)
            if slot is None:
                continue
            loader = add("LoadVideo", [f"{folder}.mp4", "image"],
                         f"наш {name} -> {target}", "VIDEO", "VIDEO")
            frames = add("GetVideoComponents", [],
                         f"кадры {name}", "IMAGE", "images",
                         inputs=[{"name": "video", "type": "VIDEO",
                                  "link": None}])
            link(loader, 0, frames, 0, "VIDEO")
            if kind == "MASK":
                # ЕДИНСТВЕННОЕ МЕСТО, ГДЕ ВВОДИТСЯ НЕДОКАЗАННОЕ ИМЯ. Вход
                # `character_mask` требует MASK, а наши маски приезжают
                # картинками; в темплейте преобразователя IMAGE->MASK нет,
                # потому что там маску отдавал вырезанный BlockifyMask.
                # ~~«Имя ImageToMask взято из встроенного набора Comfy и
                # ПРОВЕРЕНО НЕ БЫЛО — ComfyUI в этой среде нет»~~ — ЗАКРЫТО
                # 17.08.2026. Comfy ставить по-прежнему нельзя, но исходник
                # читается без установки: `curl raw.githubusercontent.com/
                # comfyanonymous/ComfyUI/master/comfy_extras/nodes_mask.py`
                # даёт `node_id="ImageToMask"` на строке 131, вход `image`,
                # комбо `channel` со значением `red`, выход MASK. Виджет ниже
                # ставит ровно `red`. Реестр — `PROVEN_BY_SOURCE`.
                # Пометка не стёрта, а перечёркнута: полгода этот тип ездил
                # недоказанным, и знать это следующей смене полезнее, чем
                # видеть чистый код.
                conv = add("ImageToMask", ["red"], "маска из кадров",
                           "MASK", "MASK",
                           inputs=[{"name": "image", "type": "IMAGE",
                                    "link": None}])
                link(frames, 0, conv, 0, "IMAGE")
                link(conv, 0, subgraph, slot, "MASK")
            else:
                link(frames, 0, subgraph, slot, "IMAGE")

        # `background_video` вырезанный DrawMaskOnImage брал из кадров
        # драйвинга. Кадры драйвинга в графе уже есть — GetVideoComponents,
        # который кормит звук, — и подключить надо их, а не заводить четвёртый
        # загрузчик того же файла.
        bg_slot = feeds.get("background_video")
        driving = next((n["id"] for n in out["nodes"]
                        if n["type"] == "GetVideoComponents"
                        and n["id"] not in {x["id"] for x in introduced}
                        and not n.get("title")), None)
        if bg_slot is not None and driving is not None:
            link(driving, 0, subgraph, bg_slot, "IMAGE")

    if quant == QUANT_GGUF:
        introduced.extend(_retarget_to_gguf(out, proven))

    out["last_node_id"] = next_id - 1
    out["last_link_id"] = next_link - 1

    out["extra"] = dict(out.get("extra", {}))
    out["extra"]["fork"] = {
        "derived_from": UPSTREAM,
        "derived_from_sha256": UPSTREAM_SHA256,
        "quant": quant,
        "removed": len(removed),
        "our_sources": OUR_SOURCES,
        "unrun": ("НЕПРОВЕРЕНО: граф не исполнялся, проверка структурная — "
                  "типы нод, связи, обязательные входы"),
        # Едет ВНУТРИ произведённого файла, потому что файл уедет на
        # арендованную машину без этого репозитория и без хэндофа, и там его
        # прочтёт человек, у которого из контекста только сам граф.
        "no_composite": NO_COMPOSITE,
    }
    return {"graph": out, "removed": removed, "introduced": introduced}


def audit(derived: dict, *, upstream: dict | None = None) -> dict:
    """Структурная проверка произведённого графа. Три исхода, числа в отчёте.

    Проверяется списком, а не глазами: «посмотрел и вроде нет» — это то же
    самое утверждение без прогона, против которого написан весь проект.
    """
    graph = derived["graph"] if "graph" in derived else derived
    src = load_upstream() if upstream is None else upstream

    left = custom_nodes(graph)
    dangling = []
    ids = {n["id"] for n in graph.get("nodes", [])}
    for l in _links_of(graph):
        if len(l) > 3 and (l[1] not in ids or l[3] not in ids):
            dangling.append(l[0])

    manual = [n for n in graph.get("nodes", []) if n["type"] == MANUAL_NODE]
    introduced = derived.get("introduced", []) if isinstance(derived, dict) else []
    unproven = [i for i in introduced if "НЕПРОВЕРЕНО" in i.get("provenance", "")]

    starving = unfed_inputs(graph)

    problems = []
    if left:
        problems.append(f"кастомных нод осталось {len(left)}: "
                        + ", ".join(f"{n['type']}#{n['id']}" for n in left))
    if manual:
        problems.append(f"{MANUAL_NODE} на месте — граф всё ещё ждёт мыши")
    if dangling:
        problems.append(f"оборванных связей {len(dangling)}")
    if starving:
        problems.append(f"входов без питания {len(starving)}: "
                        + ", ".join(starving))

    # ЧТО НАДО ПОСТАВИТЬ НА МАШИНЕ, кроме самого Comfy. Не нарушение и не
    # проблема — расход, и он обязан быть НАЗВАН. `custom_left` мерит другое:
    # осталось ли что-то из ТРЁХ ВЫРЕЗАННЫХ паков. После перевода на GGUF он
    # честно печатает «кастомных осталось 0», а в графе при этом стоят две ноды
    # четвёртого пака — и строка отчёта читалась бы как «сторонних наборов нет»
    # ровно тогда, когда их снова один. Поэтому число рядом.
    packs = sorted({pack_of(n) for n in graph.get("nodes", [])
                    if pack_of(n) and pack_of(n).lower() != CORE_PACK})
    before = len(src.get("nodes", []))
    after = len(graph.get("nodes", []))
    return {
        "outcome": FAIL if problems else PASS,
        "nodes_before": before, "nodes_after": after,
        "custom_left": left, "dangling": dangling, "unfed": starving,
        "unproven_types": unproven,
        "packs_required": packs,
        "problems": problems,
        "note": (f"нод было {before}, стало {after}; кастомных осталось "
                 f"{len(left)}, оборванных связей {len(dangling)}, входов без "
                 f"питания {len(starving)}, введённых недоказанных типов "
                 f"{len(unproven)}"
                 + (f" ({', '.join(i['type'] for i in unproven)})"
                    if unproven else "")
                 + f", сторонних паков к установке {len(packs)}"
                 + (f" ({', '.join(packs)})" if packs else "") + ". "
                 f"{'ЧИСТО' if not problems else '; '.join(problems)}. "
                 f"НЕПРОВЕРЕНО: граф не исполнялся."),
    }


def contract(graph: dict | None = None) -> dict:
    """На месте ли обязательные входы `WanAnimateToVideo`.

    Узел лежит ВНУТРИ сабграфа, поэтому в списке нод верхнего уровня его нет —
    там UUID сабграфа. Ищем по определениям сабграфов, а не по верхнему уровню:
    поиск только по верхнему дал бы уверенное «узла нет» на исправном файле,
    и это худший сорт ложной тревоги — он снимается отключением проверки.
    """
    src = load_upstream() if graph is None else graph
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "WanAnimateToVideo":
                found.extend(i.get("name", "") for i in node.get("inputs", []))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(src)
    if not found:
        return {"outcome": UNMEASURED, "missing": list(ANIMATE_INPUTS),
                "note": ("WanAnimateToVideo не найден ни на верхнем уровне, ни "
                         "в сабграфах: контракт проверить нечем. Это НЕ "
                         "«контракт нарушен».")}
    missing = [i for i in ANIMATE_INPUTS if i not in found]
    return {
        "outcome": FAIL if missing else PASS,
        "present": sorted(set(found) & set(ANIMATE_INPUTS)),
        "missing": missing,
        "note": (f"обязательных входов на месте "
                 f"{len(ANIMATE_INPUTS) - len(missing)} из "
                 f"{len(ANIMATE_INPUTS)}"
                 + (f"; нет: {', '.join(missing)}" if missing else "")),
    }


# ---------------------------------------------------------------------------
# АДАПТЕР ИСПОЛНЕНИЯ. Мока, который можно перепутать с прогоном, здесь нет.
# ---------------------------------------------------------------------------


def check_inputs(inputs: dict) -> list:
    """Структурные нарушения входов `render`. Список причин, а не флаг.

    ПОЧЕМУ ПРОВЕРЯЕТ АДАПТЕР, А НЕ COMFY. Каждое из этих нарушений Comfy тоже
    поймает — на арендованной машине, через девять секунд загрузки весов, по
    одному за перезапуск (П2). Здесь они ловятся за миллисекунду и все сразу.

    Числа — из §3.2 хэндофа и из первоисточника; расхождение прозы «кратна 4» с
    первоисточником «step: 4 от 1» разобрано у `LENGTH_STEP`.
    """
    bad: list[str] = []
    missing = [k for k in ANIMATE_INPUTS if k not in inputs]
    if missing:
        bad.append("нет входов: " + ", ".join(missing))

    w, h = inputs.get("width"), inputs.get("height")
    length = inputs.get("length")
    for name, val in (("width", w), ("height", h), ("length", length)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            bad.append(f"{name} не положительное целое: {val!r}")
    if bad:
        return bad

    for name, val in (("width", w), ("height", h)):
        if val % SIDE_MULTIPLE:
            bad.append(f"{name}={val} не кратно {SIDE_MULTIPLE}")
        if val < MIN_SIDE:
            bad.append(f"{name}={val} меньше блока маски {MIN_SIDE} px — "
                       f"расширение маски для модели не существует (§5.4)")
    if (length - LENGTH_BASE) % LENGTH_STEP:
        bad.append(f"length={length}: шаг {LENGTH_STEP} от {LENGTH_BASE} "
                   f"нарушен (77 годится, 76 — нет)")

    for name in ANIMATE_INPUTS:
        arr = inputs.get(name)
        if arr is None:
            continue
        a = np.asarray(arr)
        if name == "reference_image":
            # Одна фотография, а не последовательность: §1 продукта.
            if a.ndim != 4 or a.shape[0] != 1 or a.shape[-1] != 3:
                bad.append(f"{name}: ожидалась одна картинка (1,H,W,3), "
                           f"пришло {a.shape}")
            continue
        if a.shape[0] != length:
            bad.append(f"{name}: кадров {a.shape[0]}, а length={length}")
        if name == "character_mask":
            if a.ndim != 3:
                bad.append(f"{name}: ожидалась (N,H,W) без канала, "
                           f"пришло {a.shape}")
            elif a.shape[1:] != (h, w):
                bad.append(f"{name}: {a.shape[1:]} вместо ({h},{w})")
            continue
        if a.ndim != 4 or a.shape[-1] != 3:
            bad.append(f"{name}: ожидалась (N,H,W,3), пришло {a.shape}")
            continue
        if name == "face_video":
            # 1207 в nodes_wan.py: face_video безусловно масштабируется в
            # 512x512. Подать другое не ошибка Comfy — но тогда мы не знаем,
            # что именно доехало до модели, а канал лица и так спорный (§2b).
            if a.shape[1:3] != (FACE_SIDE, FACE_SIDE):
                bad.append(f"{name}: {a.shape[1]}x{a.shape[2]} вместо "
                           f"{FACE_SIDE}x{FACE_SIDE}")
        elif a.shape[1:3] != (h, w):
            bad.append(f"{name}: {a.shape[1]}x{a.shape[2]} вместо {h}x{w}")
    return bad


def mock_frames(length: int, height: int, width: int) -> np.ndarray:
    """Кадры от мока. Помечены В ПИКСЕЛЯХ, и это не украшение.

    Словарь с полем `source` теряется на первом же `result["frames"]`, а
    дальше массив уедет в файл, в метрику, в чужую функцию, которая про наш
    словарь не знает, — и там будет неотличим от порождённого. Метка едет
    вместе с пикселями и переживает всё, кроме умышленного стирания.

    Содержимое намеренно не похоже на кадр: диагональная лесенка. Мок,
    выглядящий правдоподобно, однажды пройдёт глазами (П3).
    """
    frames = np.zeros((length, height, width, 3), dtype=np.uint8)
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    for i in range(length):
        frames[i, :, :, 1] = ((ys + xs + i * 8) % 256).astype(np.uint8)
    magic = np.frombuffer(MOCK_MAGIC, dtype=np.uint8)
    frames[:, 0, :magic.size, 0] = magic
    return frames


def is_mock(frames) -> bool:
    """Помечены ли кадры как мок. Отдельная функция, потому что спрашивать это
    будет чужой код, у которого нашего словаря на руках нет."""
    if frames is None:
        return False
    a = np.asarray(frames)
    if a.ndim != 4 or a.shape[0] == 0 or a.shape[2] < len(MOCK_MAGIC):
        return False
    magic = np.frombuffer(MOCK_MAGIC, dtype=np.uint8)
    return bool(np.all(a[:, 0, :magic.size, 0].astype(np.uint8) == magic))


def render(workflow: dict, inputs: dict, *, backend=None,
           kind: str | None = None) -> dict:
    """`render(workflow, inputs) -> frames` — на моке, и он в этом сознаётся.

    `kind` выбирает НАБОР ПРОВЕРОК: `native` — часть I (штатные ноды),
    `wrapper` — часть II (обёртка). Второй адаптер здесь не заведён намеренно:
    развилка на два производства касается только того, чем проверяют граф и как
    зовётся длина, а три исхода происхождения, метка в пикселях и обратная
    проверка на подлог — одни и те же. Скопировать их значило бы завести второй
    способ узнать известное (Е1), и чинить потом пришлось бы в двух местах.

    ТРИ ИСХОДА ПРОИСХОЖДЕНИЯ (`source`), а не два:

    * `RAN` — кадры породил настоящий бэкенд;
    * `MOCK` — бэкенда нет, вход проверен, кадры выданы моком;
    * `NOTHING` — кадров нет вообще: вход или граф не прошли проверку, либо
      бэкенд упал.

    И ОТДЕЛЬНО ВЕРДИКТ (`outcome`) в словах остальных потоков:

    * `PASS` — только настоящий прогон. **Мок не может вернуть `PASS` ни при
      каком входе**, и это проверяется тестом, а не обещается комментарием;
    * `FAIL` — структурное нарушение: граф или вход не годятся;
    * `UNMEASURED` — проверить было нечем: мок, либо упавший бэкенд.

    ПОЧЕМУ ПЕРЕПУТАТЬ НЕЛЬЗЯ, тремя независимыми способами:

    1. мок никогда не выдаёт `PASS` — сравнение `outcome == PASS` безопасно;
    2. кадры мока помечены в пикселях (`is_mock`) — метка едет с массивом
       туда, куда словарь не доедет;
    3. **проверка в обратную сторону**: если бэкенд, объявленный настоящим,
       вернул помеченные кадры, это подлог, и он ловится (`FAIL`). Без этой
       третьей проверки достаточно было бы подсунуть мок как `backend=` — то
       есть весь механизм обходился бы одним аргументом.

    Когда появится карта, подменяется ТОЛЬКО `backend`: сигнатура, проверки
    входа и вердикты остаются, поэтому сквозной путь собирается уже сейчас.

    НЕПРОВЕРЕНО (Ц4): ветка `RAN` испытана лишь подставным бэкендом. На
    настоящем ComfyUI она не выполнялась ни разу — его в этой среде нет.
    """
    problems: list[str] = []
    kind = "native" if kind is None else kind
    if kind not in RENDER_KINDS:
        raise ValueError(f"набор проверок {kind!r} неизвестен, есть "
                         f"{tuple(RENDER_KINDS)}")
    rules = RENDER_KINDS[kind]

    graph = workflow.get("graph", workflow) if isinstance(workflow, dict) else {}
    report = rules["audit"](graph) if graph else {"outcome": UNMEASURED,
                                                  "problems": ["графа нет"]}
    if report["outcome"] != PASS:
        problems += [f"граф: {p}" for p in report.get("problems", [])] or \
            ["граф: проверить не удалось"]

    problems += [f"вход: {p}" for p in rules["check"](inputs)]

    if problems:
        return {
            "source": NOTHING, "outcome": FAIL, "frames": None,
            "problems": problems, "backend": None,
            "note": (f"{NOTHING}: нарушений {len(problems)} — "
                     + "; ".join(problems)
                     + ". Кадры не выданы даже поддельные: пустой результат "
                       "заметен, а правдоподобный мок на негодном входе — нет"),
        }

    length = inputs[rules["length_key"]]
    h, w = inputs["height"], inputs["width"]

    if backend is None:
        frames = mock_frames(length, h, w)
        return {
            "source": MOCK, "outcome": UNMEASURED, "frames": frames,
            "problems": [], "backend": None,
            "note": (f"{MOCK}: ГЕНЕРАЦИИ НЕ БЫЛО. ComfyUI в этой среде нет и "
                     f"ставить его запрещено (§10). Вход и граф проверены "
                     f"структурно и годятся; выдано {length} кадров "
                     f"{w}x{h} от мока, помеченных в пикселях. Вердикт "
                     f"{UNMEASURED}, а не {PASS}: судить не по чему"),
        }

    name = getattr(backend, "__name__", type(backend).__name__)
    try:
        frames = backend(graph, inputs)
    except Exception as exc:  # бэкенд упал — это НЕ «не годно» (Р1)
        return {
            "source": NOTHING, "outcome": UNMEASURED, "frames": None,
            "problems": [f"бэкенд {name} упал: {exc!r}"], "backend": name,
            "note": (f"{NOTHING}: бэкенд {name} упал ({exc!r}). Это "
                     f"{UNMEASURED}, а не {FAIL}: про сам граф мы так ничего "
                     f"и не узнали"),
        }

    a = np.asarray(frames) if frames is not None else None
    if a is None or a.ndim != 4 or a.shape[0] != length:
        got = "None" if a is None else str(a.shape)
        return {
            "source": NOTHING, "outcome": FAIL, "frames": None,
            "problems": [f"бэкенд {name} вернул {got}, ожидалось "
                         f"({length},{h},{w},3)"], "backend": name,
            "note": f"{NOTHING}: бэкенд {name} вернул не кадры — {got}",
        }
    if is_mock(a):
        # ТРЕТЬЯ ПРОВЕРКА, БЕЗ КОТОРОЙ ВСЁ ОСТАЛЬНОЕ ОБХОДИТСЯ ОДНИМ АРГУМЕНТОМ.
        return {
            "source": NOTHING, "outcome": FAIL, "frames": None,
            "problems": [f"бэкенд {name} вернул кадры с меткой мока"],
            "backend": name,
            "note": (f"{NOTHING}: бэкенд {name} объявлен настоящим, а кадры "
                     f"помечены как мок. Это подлог, и он важнее любого "
                     f"числа, которое по ним посчитали бы"),
        }
    return {
        "source": RAN, "outcome": PASS, "frames": a, "problems": [],
        "backend": name,
        "note": (f"{RAN}: бэкенд {name} породил {a.shape[0]} кадров "
                 f"{a.shape[2]}x{a.shape[1]}, метки мока нет"),
    }


def require_generated(result: dict) -> np.ndarray:
    """Кадры, только если их правда породили. Иначе исключение.

    Для кода, которому мок не годится, — отгрузка, приёмочные числа, всё, что
    попадёт в отчёт как измеренное. Проверять `result["frames"] is not None`
    там недостаточно: у мока он тоже не None, и в этом весь смысл.
    """
    if result.get("source") != RAN or result.get("outcome") != PASS:
        raise ValueError(
            f"кадров от настоящего прогона нет: source={result.get('source')}, "
            f"outcome={result.get('outcome')}. {result.get('note', '')}")
    if is_mock(result.get("frames")):
        raise ValueError("кадры помечены как мок, а source говорит "
                         f"{RAN} — доверяем свидетельству, не флагу (Е2)")
    return np.asarray(result["frames"])


# ---------------------------------------------------------------------------
# ЛОК-ФАЙЛ
# ---------------------------------------------------------------------------


def load_lock(path: str | Path | None = None) -> dict:
    """Лок-файл стека. Читается кодом, чтобы не разошёлся с ним молча."""
    p = Path(path) if path else _root() / LOCKFILE
    if not p.exists():
        raise FileNotFoundError(f"нет лок-файла {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def audit_lock(lock: dict | None = None) -> dict:
    """Полон ли лок-файл. Три исхода, и «пусто» — не «чисто».

    Дефект, ради которого написано (Р2): проверка, обошедшая ноль записей,
    возвращала бы «нарушений 0». Поэтому рядом с числом нарушений всегда
    печатается число ПРОВЕРЕННЫХ записей, и пустой файл — не успех.
    """
    lock = load_lock() if lock is None else lock
    weights = lock.get("weights", [])
    claims = lock.get("contract_claims", [])
    bad: list[str] = []

    for w in weights:
        who = w.get("role", "?")
        if w.get("measured") is True:
            for field in ("url", "bytes", "sha256", "measured_by",
                          "measured_at"):
                if not w.get(field):
                    bad.append(f"{who}: замеренный вес без поля {field}")
            if w.get("sha256") and len(w["sha256"]) != 64:
                bad.append(f"{who}: sha256 длиной {len(w['sha256'])}, не 64")
            if isinstance(w.get("bytes"), int) and w.get("gib"):
                if abs(w["bytes"] / 2 ** 30 - w["gib"]) > 5e-4:
                    bad.append(f"{who}: bytes и gib разошлись — "
                               f"{w['bytes'] / 2 ** 30:.4f} против {w['gib']}")
        elif "НЕПРОВЕРЕНО" not in str(w.get("provenance", "")):
            bad.append(f"{who}: не замерен и не помечен НЕПРОВЕРЕНО")

    for c in claims:
        for field in ("claim", "source", "checked"):
            if not c.get(field):
                bad.append(f"утверждение {c.get('claim', '?')[:40]!r} без "
                           f"поля {field}")

    if not weights or not claims:
        return {"outcome": UNMEASURED, "problems": bad,
                "weights_checked": len(weights), "claims_checked": len(claims),
                "note": (f"лок-файл пуст или неполон: весов {len(weights)}, "
                         f"утверждений {len(claims)}. Ноль нарушений при нуле "
                         f"проверенных записей — не успех (Р2)")}

    measured = [w for w in weights if w.get("measured") is True]
    return {
        "outcome": FAIL if bad else PASS,
        "problems": bad,
        "weights_checked": len(weights), "claims_checked": len(claims),
        "measured": len(measured),
        "total_bytes": sum(w["bytes"] for w in measured
                           if isinstance(w.get("bytes"), int)),
        "note": (f"проверено весов {len(weights)} (замерено {len(measured)}), "
                 f"утверждений {len(claims)}, нарушений {len(bad)}"
                 + ("" if not bad else ": " + "; ".join(bad))),
    }


#: Загрузчики весов: тип узла -> (имя виджета по документации, его индекс в
#: `widgets_values`). Индекс нужен потому, что граф в UI-формате хранит виджеты
#: СПИСКОМ без имён, и «первый виджет загрузчика — имя файла» — утверждение,
#: которое надо было проверить, а не предположить. Проверено по шаблону:
#: `CLIPLoader` там несёт `['umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan',
#: 'default']`, то есть имя файла действительно нулевое, а `wan` и `default` —
#: тип энкодера и устройство.
WEIGHT_WIDGET = {
    "UNETLoader": ("unet_name", 0),
    "CLIPLoader": ("clip_name", 0),
    "VAELoader": ("vae_name", 0),
    "CLIPVisionLoader": ("clip_name", 0),
    "LoraLoaderModelOnly": ("lora_name", 0),
    "UnetLoaderGGUF": ("unet_name", 0),
    "CLIPLoaderGGUF": ("clip_name", 0),
}

#: Ступени, которыми граф может грузить диффузию. Значение — как в лок-файле,
#: чтобы сравнение шло по одному слову, а не по двум похожим (Е1).
QUANT_TEMPLATE = "как в шаблоне"
QUANT_GGUF = "gguf"
QUANTS = (QUANT_TEMPLATE, QUANT_GGUF)

#: Ступень по умолчанию. ~~QUANT_TEMPLATE~~ снято 18.08.2026: развилка §1
#: ЗАКРЫТА решением владельца — Q4_K_M под карту A16, и то же самое записано в
#: лок-файле. Пока умолчанием стоял шаблонный fp8, `audit_weights` был красным
#: при закрытой развилке, то есть проверка отвечала на вопрос, которого больше
#: нет. Красное «решение не принято» и красное «решение не доехало до кода» —
#: разные вещи, и второе чинится, а не оставляется.
#: Само ИМЯ ФАЙЛА здесь по-прежнему не пишется: оно читается из лока (Е1).
QUANT_DEFAULT = QUANT_GGUF


#: Чем оканчивается имя файла весов. ВЫБРАНО мной из наблюдаемого: оба
#: расширения встречаются и в шаблоне, и в лок-файле, и ничего третьего стек не
#: грузит. Нужно затем, чтобы виджет, оказавшийся НЕ именем файла, был находкой,
#: а не тихо принятой строкой.
WEIGHT_SUFFIXES = (".safetensors", ".gguf")


def _widget_slots(spec) -> tuple:
    """Реестр `WEIGHT_WIDGET` позволяет и один слот, и несколько.

    Несколько появилось из-за `WanVideoLoraSelectMulti`: у него пять пар
    «файл, сила» в одном узле. Свести это к «один загрузчик — один файл» можно
    было бы, заведя пять записей реестра, — и тогда сверка с локом молчала бы
    про адаптеры, которых в графе два. Форма записи расширена, а не обойдена.
    """
    widget, index = spec
    return widget, (index,) if isinstance(index, int) else tuple(index)


def graph_weights(graph: dict) -> list:
    """Какие ФАЙЛЫ ВЕСОВ граф реально подаёт в загрузчики.

    Не «какие веса упомянуты в графе» — упомянуты они и в записках-заметках
    темплейта, там лежат ссылки на bf16 и на fp8 сразу, и разбор по тексту
    вернул бы оба. Здесь берутся только виджеты узлов-загрузчиков, то есть то,
    что Comfy пойдёт открывать на диске.
    """
    out = []
    for node in graph.get("nodes", []):
        spec = WEIGHT_WIDGET.get(str(node.get("type")))
        if spec is None:
            continue
        widget, indices = _widget_slots(spec)
        values = node.get("widgets_values") or []
        for index in indices:
            if index >= len(values) or not isinstance(values[index], str):
                continue
            if not values[index].endswith(WEIGHT_SUFFIXES):
                continue
            out.append({"id": node.get("id"), "type": node["type"],
                        "widget": widget, "index": index,
                        "file": values[index]})
    return sorted(out, key=lambda r: (r["type"], r["file"]))


def unparsed_loaders(graph: dict) -> list:
    """Загрузчики, у которых объявленный виджет НЕ похож на имя файла весов.

    НАЙДЕНО МУТАЦИЕЙ, А НЕ ЧТЕНИЕМ, и это единственная причина, по которой
    функция существует. Подмена индекса `WEIGHT_WIDGET["CLIPLoader"]` с 0 на 1
    выжила: разбор брал второй виджет, находил там `wan` — тип энкодера — и
    объявлял его ИМЕНЕМ ФАЙЛА ВЕСОВ. Расхождений по-прежнему выходило 4,
    разобранных по-прежнему 6, вердикт не менялся ни в одну сторону. То есть
    прибор мерил не то, а отчёт выглядел ровно как раньше.

    Тихо пропустить такой виджет — не лучше: это молчаливое «в графе нет
    загрузчика», хотя загрузчик есть и реестр указывает не туда. Поэтому третий
    список, который `audit_weights` обязан назвать.
    """
    out = []
    for node in graph.get("nodes", []):
        spec = WEIGHT_WIDGET.get(str(node.get("type")))
        if spec is None:
            continue
        widget, indices = _widget_slots(spec)
        values = node.get("widgets_values") or []
        for index in indices:
            got = values[index] if index < len(values) else None
            if isinstance(got, str) and got.endswith(WEIGHT_SUFFIXES):
                continue
            # ПУСТОЙ СЛОТ АДАПТЕРА — НЕ ПОЛОМКА. `WanVideoLoraSelectMulti`
            # держит пять пар и незанятые помечает строкой `none`
            # (доказано исходником, nodes_model_loading.py:507). Объявить их
            # «загрузчиком без имени файла» значило бы печатать четыре
            # нарушения на каждом исправном графе — а сторож, который всегда
            # красный, снимают целиком.
            if got == EMPTY_LORA_SLOT:
                continue
            out.append({"id": node.get("id"), "type": node["type"],
                        "widget": widget, "index": index, "got": got})
    return out


def _retarget_to_gguf(out: dict, proven: set) -> list:
    """Перевести диффузию и текстовый энкодер на то, что объявлено в локе.

    ИМЯ ФАЙЛА БЕРЁТСЯ ИЗ ЛОКА, А НЕ ПИШЕТСЯ ЗДЕСЬ СТРОКОЙ, и это существенно:
    строка, скопированная в код, — второй способ узнать известное, то есть
    ровно тот дефект (Е1), который эта функция и закрывает. Разъедься лок с
    кодом — и мы получили бы третье расхождение вместо снятого второго.

    Возвращает список введённых типов — с происхождением, как и всё остальное,
    что вводит `derive`. Пустой список означал бы, что в графе не нашлось того,
    что надо было переводить, и это НЕ успех: `audit_weights` останется
    красным, потому что лок по-прежнему не сойдётся с графом.
    """
    lock = load_lock()
    by_role = {w.get("role"): Path(w["path"]).name
               for w in lock.get("weights", []) if w.get("path")}
    plan = {
        "UNETLoader": ("UnetLoaderGGUF", "diffusion", 1),
        "CLIPLoader": ("CLIPLoaderGGUF", "text_encoder", 2),
    }
    introduced = []
    for node in out.get("nodes", []):
        spec = plan.get(str(node.get("type")))
        if spec is None:
            continue
        new_type, role, keep = spec
        name = by_role.get(role)
        if name is None:
            continue
        values = list(node.get("widgets_values") or [])
        # `keep` — сколько виджетов у НОВОЙ ноды, проверено по её INPUT_TYPES
        # (реестр PROVEN_BY_SOURCE). Штатный CLIPLoader несёт третьим `device`,
        # которого у GGUF-варианта нет: перенести виджеты как есть значило бы
        # отдать ноде лишнее значение, и Comfy прочёл бы его как тип.
        node["widgets_values"] = [name] + values[1:keep]
        node["type"] = new_type
        props = dict(node.get("properties") or {})
        props["Node name for S&R"] = new_type
        props[PACK_KEY] = GGUF_PACK
        node["properties"] = props
        node["title"] = f"{new_type}: {name}"
        introduced.append({"type": new_type, "id": node.get("id"),
                           "provenance": provenance_of(new_type, proven),
                           "pack": GGUF_PACK})
    return introduced


def audit_weights(derived: dict, *, lock: dict | None = None) -> dict:
    """Сходится ли граф с лок-файлом ПО ФАЙЛАМ. Дефект, ради которого написано.

    НАЙДЕНО ЗАМЕРОМ, А НЕ ЧТЕНИЕМ. `audit` проверял структуру, `audit_lock` —
    полноту лока, и оба возвращали «годно» на состоянии, где граф грузит
    `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` (17.138 ГиБ), а лок
    объявляет `Wan2.2-Animate-14B-Q3_K_M.gguf` (8.038 ГиБ). Ни одного файла
    общего по диффузии и энкодеру — и ни одной красной проверки. Это ровно
    дефект Е1: одно знание («какие веса нужны прогону») в двух местах, и они
    разъехались молча. Смена, приехавшая на карту, скачала бы по локу 15.338
    ГиБ и запустила граф, который просит другие файлы.

    Три исхода (Р1), и «граф без загрузчиков» — третий, а не первый: пустой
    разбор дал бы «расхождений 0», то есть ложное «годно» (Р2). Поэтому рядом
    с числом расхождений всегда стоит число разобранных загрузчиков.
    """
    graph = derived["graph"] if "graph" in derived else derived
    lock = load_lock() if lock is None else lock

    in_graph = graph_weights(graph)
    unparsed = unparsed_loaders(graph)
    declared = {Path(w["path"]).name: w.get("role", "?")
                for w in lock.get("weights", [])}

    if not in_graph or not declared:
        return {"outcome": UNMEASURED, "problems": [],
                "loaders_checked": len(in_graph),
                "unparsed": unparsed,
                "declared": len(declared),
                "note": (f"разбирать нечего: загрузчиков {len(in_graph)}, "
                         f"неразобранных {len(unparsed)}, записей в локе "
                         f"{len(declared)}. Ноль расхождений при нуле "
                         f"разобранного — не успех (Р2)")}

    bad: list[str] = []
    for row in unparsed:
        bad.append(f"у {row['type']}#{row['id']} виджет №{row['index']} несёт "
                   f"{row['got']!r}, а не имя файла весов — реестр "
                   f"WEIGHT_WIDGET указывает не туда, и прибор мерит не то")
    for row in in_graph:
        if row["file"] not in declared:
            bad.append(f"граф грузит {row['file']} узлом {row['id']} "
                       f"({row['type']}), а в локе такого файла нет")
    fed = {r["file"] for r in in_graph}
    for name, role in sorted(declared.items()):
        if name not in fed:
            bad.append(f"лок объявляет {name} (роль {role}), но ни один "
                       f"загрузчик графа его не просит")

    return {
        "outcome": FAIL if bad else PASS,
        "problems": bad,
        "loaders_checked": len(in_graph),
        "unparsed": unparsed,
        "declared": len(declared),
        "note": (f"разобрано загрузчиков {len(in_graph)}, неразобранных "
                 f"{len(unparsed)}, объявлено в локе "
                 f"{len(declared)}, расхождений {len(bad)}"
                 + ("" if not bad else ". " + "; ".join(bad))),
    }


def write(derived: dict, out_path: str | Path) -> Path:
    """Сохранить произведённый граф. Отдельной функцией: производство и запись
    на диск — разные решения, и склеенные они мешают тестировать первое."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    graph = derived["graph"] if "graph" in derived else derived
    p.write_text(json.dumps(graph, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


# ===========================================================================
# ЧАСТЬ II. ПРОИЗВОДСТВО ГРАФА НА ОБЁРТКЕ `kijai/ComfyUI-WanVideoWrapper`
# ===========================================================================
#
# ПОЧЕМУ ВСЁ ВЫШЕ ОСТАЛОСЬ ЛЕЖАТЬ. Решение владельца 18.08.2026: производство
# переезжает со ШТАТНЫХ нод на обёртку. Часть I не стёрта, а перечёркнута
# (Ц: устаревшее перечёркивается, а не стирается) — она остаётся единственным
# описанием того, ЧТО ИМЕННО делает штатная `WanAnimateToVideo`, и опровержение
# §1a живёт там же. Производством по умолчанию она больше не является.
#
# ~~«граф производится из `workflows/upstream/video_wan2_2_14B_animate.json`»~~
# — теперь только для части I. Часть II граф СОБИРАЕТ, а не вырезает: шаблона
# обёртки в репозитории нет, а боевой воркфлоу владельца (93 узла) лежит вне
# дерева и хэшем не закреплён. Значит доказывать имена нечем, кроме исходника,
# и каждое имя доказано командой — реестр `PROVEN_BY_SOURCE`, ниже.
#
# ЗАЧЕМ ПЕРЕХОД, одной строкой: цикл по окнам живёт ВНУТРИ сэмплера обёртки
# (`nodes_sampler.py:2194`, `# region wananimate loop`), длина — просто число
# `num_frames`. На штатных нодах этот цикл надо строить в графе руками, и у нас
# его не было. Плюс `pose_strength`, `colormatch` между окнами и `blockswap`
# идут готовыми ручками.
#
# ЧТО ЗДЕСЬ НЕ ПРОВЕРЕНО (Ц4): граф обёртки НЕ ИСПОЛНЯЛСЯ. ComfyUI в этой среде
# нет. Проверка структурная — типы нод, связи, питание обязательных входов,
# сверка загружаемых файлов с лок-файлом и с тем, какие форматы загрузчик умеет
# читать. Что произведённый граф поедет, здесь не показано.

#: Набор обёртки. Регистр — как в `properties.cnr_id` боевого воркфлоу
#: владельца (узел 22): `ComfyUI-WanVideoWrapper`. Сравнение везде ниже
#: регистронезависимое: на этом уже один раз разошлись имена паков.
WRAP_PACK = "ComfyUI-WanVideoWrapper"

#: Лицензия обёртки проверена ДО встраивания (Ц5), командой, а не по памяти:
#: `curl raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/LICENSE`
#: -> «Apache License Version 2.0»; `pyproject.toml` -> `version = "1.4.7"`,
#: зависимости `accelerate, diffusers, peft, ftfy, gguf, pyloudnorm`. Ни
#: `non-commercial`, ни `research-only` в заголовке нет.
WRAP_LICENSE = {
    "pack": WRAP_PACK,
    "license": "apache-2.0",
    "version": "1.4.7",
    "checked": "2026-08-18",
    "checked_by": ("curl https://raw.githubusercontent.com/kijai/"
                   "ComfyUI-WanVideoWrapper/main/LICENSE и .../pyproject.toml"),
}

#: Паки, которых в графе быть НЕ ДОЛЖНО, и причина у каждого своя. Список
#: закрытый и проверяется кодом: «мы же решили их не брать» — это утверждение
#: без прогона, ровно того сорта, против которого написан весь проект.
FORBIDDEN_PACKS = {
    "comfyui-kjnodes": ("GPL-3.0, решение владельца 18.08.2026 не брать; три "
                        "нужные ноды подставляем своими"),
    "comfyui_controlnet_aux": "LICENSE 404 (не объявлена) и больше не нужен",
    "comfyui-videohelpersuite": ("лицензия не проверена этой сменой; выход "
                                 "собирается штатными CreateVideo/SaveVideo"),
}

#: Реестр доказанных ИСХОДНИКОМ имён пополняется, а не заводится второй (Е1):
#: спрашивать «чем доказан тип» код обязан в одном месте, иначе появится тип,
#: доказанный во втором реестре и неизвестный первому.
#:
#: КАК СНЯТО. Три файла обёртки скачаны командой 2026-08-18:
#:   curl -sS https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/\
#:            main/{nodes.py,nodes_sampler.py,nodes_model_loading.py}
#: sha256 тел (`sha256sum`), они же стоят в каждой записи ниже:
#:   nodes.py               6b3bb6a619e5a928259e6e8400c73ffe7140f17439c5007dd8596b90055f0666
#:   nodes_sampler.py       374e552a2f96e10ddfcb37785718be0ea0dfa34a6551a5930e8963e3bc87630b
#:   nodes_model_loading.py 2297aed82d0908505b735e3248f98fc39f815119e4a32c57ee9cf59d4a8a5ba5
#: Каждое имя найдено ДВАЖДЫ: как `class X:` и как ключ в `NODE_CLASS_MAPPINGS`
#: того же файла. Одного класса мало: класс может быть не зарегистрирован, и
#: тогда имени в графе не существует ровно так же, как если бы его не было.
#:
#: ЧЕГО ТЕСТ ЗДЕСЬ НЕ МОЖЕТ, и это надо знать (Ц4). Сверить номера строк с
#: телами тест не вправе: в сеть он не ходит (Т4), а самих файлов в репозитории
#: нет. Проверено ОДНОРАЗОВОЙ КОМАНДОЙ, и первый прогон её нашёл дефект — семь
#: `mapping_line` были поставлены по догадке и не совпали. Повторить так:
#:
#:   for f in nodes.py nodes_sampler.py nodes_model_loading.py; do
#:     curl -sS -o "w_$f" "$WRAP_SRC$f"; sha256sum "w_$f"; done
#:   grep -n 'class WanVideoSampler:\|"WanVideoSampler":' w_nodes_sampler.py
#:
#: Тест сторожит лишь то, что доступно без сети: поля на месте, sha64, дата
#: форматом, номер регистрации ПОЗЖЕ номера класса.
_WRAP_SRC = ("https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/"
             "main/")
_SHA_NODES = "6b3bb6a619e5a928259e6e8400c73ffe7140f17439c5007dd8596b90055f0666"
_SHA_SAMPLER = ("374e552a2f96e10ddfcb37785718be0ea0dfa34a6551a5930e8963e3bc8"
                "7630b")
_SHA_LOADING = ("2297aed82d0908505b735e3248f98fc39f815119e4a32c57ee9cf59d4a8"
                "a5ba5")


def _wrap_proof(file: str, sha: str, line: int, mapping: int, quote: str,
                note: str) -> dict:
    """Одна запись реестра. Функцией, чтобы поля нельзя было забыть по одному."""
    return {"url": _WRAP_SRC + file, "body_sha256": sha, "line": line,
            "mapping_line": mapping, "checked": "2026-08-18",
            "quote": quote, "note": note}


PROVEN_BY_SOURCE.update({
    "WanVideoModelLoader": _wrap_proof(
        "nodes_model_loading.py", _SHA_LOADING, 1077, 2116,
        'class WanVideoModelLoader: "model": (get_filename_list("unet_gguf") '
        '+ get_filename_list("diffusion_models")), base_precision, '
        'quantization, load_device; RETURN_TYPES = ("WANVIDEOMODEL",)',
        "принимает .gguf НАПРЯМУЮ (строка 1131: `if model.endswith('.gguf')` "
        "-> quantization='gguf'). Значит ступень Q4_K_M грузится обёрткой без "
        "стороннего UnetLoaderGGUF — на один пак меньше, чем в части I"),
    "WanVideoVAELoader": _wrap_proof(
        "nodes_model_loading.py", _SHA_LOADING, 1877, 2117,
        'class WanVideoVAELoader: "model_name": (get_filename_list("vae"),), '
        'precision ["fp16","fp32","bf16"]; RETURN_TYPES = ("WANVAE",)',
        "грузит через load_torch_file — только safetensors, .gguf не читает"),
    "WanVideoBlockSwap": _wrap_proof(
        "nodes_model_loading.py", _SHA_LOADING, 294, 2126,
        'class WanVideoBlockSwap: "blocks_to_swap": ("INT", {"default": 20, '
        '"min": 0, "max": 48}); RETURN_TYPES = ("BLOCKSWAPARGS",)',
        "предел 48 — из самой ноды; у 14B блоков 40, и 38 владельца ниже "
        "предела обоих"),
    "WanVideoSetBlockSwap": _wrap_proof(
        "nodes.py", _SHA_NODES, 43, 2309,
        'class WanVideoSetBlockSwap: inputs model:WANVIDEOMODEL, '
        'block_swap_args:BLOCKSWAPARGS; RETURN_TYPES = ("WANVIDEOMODEL",)',
        "виджетов нет вовсе — всё приходит связями"),
    "WanVideoLoraSelectMulti": _wrap_proof(
        "nodes_model_loading.py", _SHA_LOADING, 502, 2125,
        'class WanVideoLoraSelectMulti: lora_files = ["none"] + '
        'get_filename_list("loras"); пять пар lora_i/strength_i, затем '
        'low_mem_load, merge_loras; RETURN_TYPES = ("WANVIDLORA",)',
        "пустой слот — строка 'none' (строка 507), а не пустая строка и не "
        "None: разбор весов обязан знать это слово, иначе объявит слот "
        "сломанным"),
    "WanVideoSetLoRAs": _wrap_proof(
        "nodes_model_loading.py", _SHA_LOADING, 729, 2120,
        'class WanVideoSetLoRAs: model:WANVIDEOMODEL, lora:WANVIDLORA; '
        'raise ValueError если merge_loras is True (строка 759)',
        "требует merge_loras=False в узле выбора. Для GGUF слияние всё равно "
        "невозможно, так что это единственный путь адаптеров на нашей ступени"),
    "WanVideoClipVisionEncode": _wrap_proof(
        "nodes.py", _SHA_NODES, 611, 2298,
        'class WanVideoClipVisionEncode: clip_vision:CLIP_VISION, '
        'image_1:IMAGE, strength_1, strength_2, crop, combine_embeds, '
        'force_offload; RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)',
        "штатный CLIPVisionLoader отдаёт CLIP_VISION — то есть половина "
        "тракта остаётся ядровой и доказывается темплейтом"),
    "WanVideoTextEncodeCached": _wrap_proof(
        "nodes.py", _SHA_NODES, 189, 2319,
        'class WanVideoTextEncodeCached: "model_name": '
        '(get_filename_list("text_encoders"),), precision, positive_prompt, '
        'negative_prompt, quantization, use_disk_cache, device',
        "В ГРАФ НЕ СТАВИТСЯ, и это находка: внутри зовёт "
        "LoadWanVideoT5TextEncoder (строка 1962), а тот читает load_torch_file "
        "и падает на 'Invalid T5 text encoder model' — .gguf он не понимает. "
        "Лок объявляет umt5-xxl-encoder-Q5_K_M.gguf, значит этот узел нам не "
        "подходит. Имя доказано и оставлено в реестре намеренно: следующая "
        "смена увидит, что узел рассматривали и почему отвергли"),
    "WanVideoTextEmbedBridge": _wrap_proof(
        "nodes.py", _SHA_NODES, 586, 2305,
        'class WanVideoTextEmbedBridge: positive:CONDITIONING, '
        'negative:CONDITIONING (optional); RETURN_TYPES = '
        '("WANVIDEOTEXTEMBEDS",)',
        "выход из положения выше: штатное CONDITIONING переводится в формат "
        "обёртки. Энкодер грузит CLIPLoaderGGUF, который .gguf читать умеет"),
    "WanVideoAnimateEmbeds": _wrap_proof(
        "nodes.py", _SHA_NODES, 1178, 2326,
        'class WanVideoAnimateEmbeds: обязательные width, height, num_frames, '
        'force_offload, frame_window_size, colormatch, pose_strength, '
        'face_strength; необязательные clip_embeds, ref_images, pose_images, '
        'face_images, bg_images, mask, start_ref_image, tiled_vae',
        "здесь живут обе ручки, которых не было в штатной ноде: pose_strength "
        "и colormatch между окнами"),
    "WanVideoSampler": _wrap_proof(
        "nodes_sampler.py", _SHA_SAMPLER, 35, 2872,
        'class WanVideoSampler: model, image_embeds, steps, cfg, shift, seed, '
        'force_offload, scheduler, riflex_freq_index; text_embeds '
        'необязательный; RETURN_TYPES = ("LATENT","LATENT")',
        "цикл по окнам внутри: `# region wananimate loop`, строка 2194 того "
        "же файла. Ради этого и переход"),
    "WanVideoDecode": _wrap_proof(
        "nodes.py", _SHA_NODES, 2081, 2295,
        'class WanVideoDecode: vae:WANVAE, samples:LATENT, enable_vae_tiling, '
        'tile_x, tile_y, tile_stride_x, tile_stride_y; RETURN_TYPES = '
        '("IMAGE",)',
        "VALIDATE_INPUTS требует tile_x > tile_stride_x и tile_y > "
        "tile_stride_y — виджеты ниже это соблюдают"),
})

#: Как обёртка ОБХОДИТСЯ С НАШИМИ КАРТИНКАМИ. Читано в теле
#: `WanVideoAnimateEmbeds.process`, там же где и всё остальное; номера строк —
#: в том же теле с sha256 `_SHA_NODES`. Это не украшение: каждое из трёх чисел
#: означает молчаливое преобразование, и все три уже стоили нам по дефекту.
WRAP_RESIZE_FACTS = {
    "ref_images": {
        "line": 1288,
        "what": ('common_upscale(..., W, H, "lanczos", "disabled")'),
        "means": ("crop=disabled — референс НЕ обрезается, а РАСТЯГИВАЕТСЯ до "
                  "кадра. Портретное фото 3:4 в кадре 480x832 поедет по "
                  "пропорциям молча"),
    },
    "pose_images": {
        "line": 1253,
        "what": 'common_upscale(..., W, H, "lanczos", "disabled")',
        "means": "то же растяжение; поза обязана приезжать уже в геометрии кадра",
    },
    "face_images": {
        "line": 1324,
        "what": 'common_upscale(..., 512, 512, "lanczos", "center")',
        "means": ("лицо приводится к 512 с ОБРЕЗКОЙ ПО ЦЕНТРУ — тот же "
                  "дефект, что у штатной ноды. Подать не 512 значит не знать, "
                  "что доехало до модели"),
    },
    "num_frames": {
        "line": 1230,
        "what": "num_frames = ((num_frames - 1) // 4) * 4 + 1",
        "means": ("длина ПРИЖИМАЕТСЯ к шагу 4 от 1 молча: 150 кадров "
                  "превращаются в 149, и никто об этом не скажет"),
    },
    "width_height": {
        "line": 1223,
        "what": "W = (width // 16) * 16",
        "means": ("виджет объявляет шаг 8, а тело округляет ВНИЗ до 16. "
                  "Значит кратность 16 — свойство кода, а не подсказки"),
    },
    "source": _WRAP_SRC + "nodes.py",
    "body_sha256": _SHA_NODES,
    "checked": "2026-08-18",
}

# ---------------------------------------------------------------------------
# ЧИСЛА. У каждого — происхождение (И4): ИЗМЕРЕНО / РАСЧЁТ / ВЫБРАНО.
# ---------------------------------------------------------------------------

#: ВЫБРАНО владельцем (§«Формат выдачи: решено»): 480x832 вертикально.
#: Вертикаль бесплатна — 480x832 стоит ровно столько же токенов, сколько
#: 832x480. Кратность 16 — свойство кода обёртки, см. WRAP_RESIZE_FACTS.
WRAP_WIDTH = 480
WRAP_HEIGHT = 832

#: ВЫБРАНО владельцем: выход 30 к/с. Совпадает с умолчанием препроцессинга
#: вендора (`process_pipepline.py:38`, fps=30). ~~16 к/с~~ снято: это было
#: умолчание виджета CreateVideo, принятое за свойство модели.
WRAP_FPS = 30

#: ИЗМЕРЕНО в боевом воркфлоу владельца (узел 62, `frame_window_size`), и там
#: же стоит умолчание самой ноды (nodes.py:1186). Два независимых свидетельства
#: одного числа.
WRAP_WINDOW = 77

#: ИЗМЕРЕНО в боевом воркфлоу владельца (узел 62). Умолчание ноды — 1.0;
#: владелец поднял до 1.1, и это вероятный источник хорошей синхронности.
#: Параметром, а не константой в графе: это рычаг, его будут крутить.
POSE_STRENGTH = 1.1
FACE_STRENGTH = 1.0

#: ИЗМЕРЕНО там же: между окнами цветовое согласование ВЫКЛЮЧЕНО.
COLORMATCH = "disabled"

#: ИЗМЕРЕНО в боевом воркфлоу владельца (узел 27): 3 шага, cfg 1, shift 5,
#: dpm++_sde. Три шага — это distill-адаптеры в цепочке, без них столько мало.
WRAP_STEPS = 3
WRAP_CFG = 1
WRAP_SHIFT = 5
WRAP_SCHEDULER = "dpm++_sde"

#: ВЫБРАНО: фиксированное зерно. У владельца стоит `randomize` — для съёмки
#: это удобно, для замеров смертельно: два прогона перестают быть сравнимыми.
WRAP_SEED = 0

#: ИЗМЕРЕНО в боевом воркфлоу владельца (узел 51): 38 блоков из 40 у 14B.
#: Предел самой ноды — 48 (nodes_model_loading.py:299).
BLOCKS_TO_SWAP = 38

#: Предел самой ноды: `"blocks_to_swap": ("INT", {"min": 0, "max": 48})`,
#: ДОКАЗАНО исходником (nodes_model_loading.py:299, тело `_SHA_LOADING`).
#: Число здесь затем, чтобы негодное значение поймалось при сборке за
#: миллисекунду, а не на карте после загрузки весов (П2).
BLOCKS_MAX = 48

#: ВЫБРАНО (§«Формат выдачи»): длина ролика 5-10 с.
SECONDS_MIN = 5.0
SECONDS_MAX = 10.0

#: Пустой слот адаптера — строка `none`. ДОКАЗАНО исходником
#: (nodes_model_loading.py:507): `lora_files = ["none"] + ...`.
EMPTY_LORA_SLOT = "none"

#: Обязательные входы узлов графа обёртки: что без чего не поедет. Это тот же
#: прибор, что `unfed_inputs` части I, переработанный под новый формат — там
#: обязательных входов было пять у одной ноды, здесь их больше и у семи.
#:
#: `bg_images` В СПИСОК НЕ ВХОДИТ НАМЕРЕННО: в боевом воркфлоу владельца он не
#: подключён, а §1a показал, что сохранности пикселей вне маски эта ветка всё
#: равно не даёт. Пустая строка тут была бы тихим «а вдруг надо».
WRAP_REQUIRED = {
    "WanVideoAnimateEmbeds": ("vae", "clip_embeds", "ref_images",
                              "pose_images", "face_images", "mask"),
    "WanVideoSampler": ("model", "image_embeds", "text_embeds"),
    "WanVideoDecode": ("vae", "samples"),
    "WanVideoTextEmbedBridge": ("positive", "negative"),
    "WanVideoClipVisionEncode": ("clip_vision", "image_1"),
    "WanVideoSetLoRAs": ("model", "lora"),
    "WanVideoSetBlockSwap": ("model", "block_swap_args"),
}

#: Какие расширения умеет читать каждый загрузчик. ДОКАЗАНО исходником, а не
#: предположено: см. `PROVEN_BY_SOURCE`. Прибор, который это сверяет, появился
#: не от хорошей жизни — лок объявляет энкодер в .gguf, а родной текстовый
#: загрузчик обёртки читает только safetensors, и без этой сверки граф уехал бы
#: на карту с узлом, который падает на загрузке.
LOADER_FORMATS = {
    "WanVideoModelLoader": (".safetensors", ".gguf"),
    "WanVideoVAELoader": (".safetensors",),
    "WanVideoTextEncodeCached": (".safetensors",),
    "WanVideoLoraSelectMulti": (".safetensors",),
    "CLIPVisionLoader": (".safetensors",),
    "CLIPLoaderGGUF": (".gguf",),
    "UnetLoaderGGUF": (".gguf",),
    "UNETLoader": (".safetensors",),
    "CLIPLoader": (".safetensors",),
    "VAELoader": (".safetensors",),
}

#: Как референс приводится к кадру. НЕ обрезкой: штатная нода режет `center`
#: (`nodes_wan.py:1159`) и срезает голову портретному фото, а обёртка вместо
#: обрезки РАСТЯГИВАЕТ (WRAP_RESIZE_FACTS["ref_images"]). Оба поведения нам не
#: годятся, поэтому референс приводится ДО графа, своей функцией
#: `pad_reference`, и в граф приезжает уже точно в геометрии кадра.
REFERENCE_FIT = {
    "mode": "pad",
    "align": "top",
    "by": "ball_reel.fork_comfy.pad_reference",
    "why": ("обрезка по центру срезает голову портретному фото (найдено "
            "17.08.2026), растяжение обёртки меняет пропорции лица. Поля "
            "добавляются снизу, кадр прижат к верху — как в боевом воркфлоу "
            "владельца (ImageResizeKJv2, pad_edge_pixel, выравнивание top)"),
    "not_kjnodes": ("узел ImageResizeKJv2 из KJNodes (GPL-3.0) не берётся — "
                    "решение владельца; то же делает наш numpy"),
}


def snap_frames(requested: int) -> int:
    """Сколько кадров ДЕЙСТВИТЕЛЬНО породит обёртка на запрошенных.

    Одно знание — одно место (Е1): формула не переписана здесь по памяти, а
    повторяет строку 1230 обёртки, и повторяет её ровно потому, что молчаливое
    прижатие надо уметь ПОСЧИТАТЬ до прогона, а не обнаружить по длине файла.
    """
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise TypeError(f"кадров ожидалось целое, пришло {requested!r}")
    if requested < LENGTH_BASE:
        raise ValueError(f"кадров {requested}, минимум {LENGTH_BASE}")
    return ((requested - LENGTH_BASE) // LENGTH_STEP) * LENGTH_STEP + LENGTH_BASE


def frames_for_seconds(seconds: float, *, fps: int | None = None) -> dict:
    """Длина ролика в кадрах, с честной разницей между «просили» и «выйдет».

    Умолчание частоты разрешается в теле, а не в сигнатуре: значение по
    умолчанию связывается на импорте, и подмена `WRAP_FPS` до него не дошла бы
    (И7, эту форму на проекте уже выгребали).
    """
    fps = WRAP_FPS if fps is None else fps
    if not SECONDS_MIN <= seconds <= SECONDS_MAX:
        raise ValueError(
            f"длина {seconds} с вне полосы {SECONDS_MIN}-{SECONDS_MAX} с "
            f"(решение владельца, §«Формат выдачи»)")
    requested = int(round(seconds * fps))
    frames = snap_frames(requested)
    return {
        "seconds_requested": seconds, "fps": fps,
        "frames_requested": requested, "frames": frames,
        "snapped_away": requested - frames,
        "seconds_actual": round(frames / fps, 4),
        "note": (f"{seconds} с при {fps} к/с = {requested} кадров, обёртка "
                 f"прижмёт к {frames} (шаг {LENGTH_STEP} от {LENGTH_BASE}); "
                 f"пропадёт молча кадров: {requested - frames}"),
    }


def window_plan(frames: int, *, window: int | None = None) -> dict:
    """Сколько окон сэмплер отработает и сколько кадров сгенерит впустую.

    РАСЧЁТ, а не замер: формула выведена из того, что окна смыкаются по одному
    кадру (`num_frames > frame_window_size` включает цикл, nodes.py:1232), и
    СВЕРЕНА с таблицей хэндофа, посчитанной независимо: 5 с -> 2 окна и 153
    кадра, 7 с -> 3 и 229, 10 с -> 4 и 305. Три совпадения — не доказательство
    исполнения, но расхождение бы здесь всплыло.
    """
    window = WRAP_WINDOW if window is None else window
    if frames < 1 or window < 1:
        raise ValueError(f"кадров {frames}, окно {window} — оба от 1")
    if frames <= window:
        return {"windows": 1, "generated": frames, "discarded": 0,
                "window": window,
                "note": f"{frames} кадров влезают в одно окно {window}"}
    step = window - 1
    windows = -(-(frames - 1) // step)
    generated = windows * step + 1
    return {"windows": windows, "generated": generated,
            "discarded": generated - frames, "window": window,
            "note": (f"{frames} кадров окнами по {window}: окон {windows}, "
                     f"сгенерится {generated}, выброшено "
                     f"{generated - frames}")}


def pad_reference(image, *, width: int | None = None, height: int | None = None,
                  align: str | None = None) -> tuple:
    """Референс в геометрию кадра ПОЛЯМИ, а не обрезкой. Возвращает (кадр, запись).

    ЧТО ЭТА ФУНКЦИЯ ОБЯЗАНА ГАРАНТИРОВАТЬ, и это проверяется тестом, а не
    обещается здесь: **ни одна строка и ни один столбец исходника не выпадают**.
    Дефект, ради которого написано, найден 17.08.2026: штатная нода приводит
    референс `crop="center"`, и у портретного фото во весь рост голова лежит
    выше вырезаемой полосы — она уезжает из картинки целиком и молча. Обёртка
    вместо обрезки растягивает (`nodes.py:1288`), что не лучше: пропорции лица
    едут, а планка личности меряет как раз лицо.

    Поля заполняются КРАЕВЫМ ПИКСЕЛЕМ (как `pad_edge_pixel` у владельца), а не
    чёрным: ровная чёрная полоса под ногами — сильный контур, и модель имеет
    полное право принять её за часть сцены.
    """
    width = WRAP_WIDTH if width is None else width
    height = WRAP_HEIGHT if height is None else height
    align = REFERENCE_FIT["align"] if align is None else align
    if align not in ("top", "center"):
        raise ValueError(f"выравнивание {align!r}: есть 'top' и 'center'")
    a = np.asarray(image)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"референс: ожидалась (H,W,3), пришло {a.shape}")
    src_h, src_w = a.shape[0], a.shape[1]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, min(width, int(round(src_w * scale))))
    new_h = max(1, min(height, int(round(src_h * scale))))
    ys = (np.arange(new_h) * (src_h / new_h)).astype(int).clip(0, src_h - 1)
    xs = (np.arange(new_w) * (src_w / new_w)).astype(int).clip(0, src_w - 1)
    scaled = a[ys][:, xs]

    pad_top = 0 if align == "top" else (height - new_h) // 2
    pad_bottom = height - new_h - pad_top
    pad_left = (width - new_w) // 2
    pad_right = width - new_w - pad_left
    out = np.pad(scaled, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                 mode="edge")
    record = dict(REFERENCE_FIT)
    record.update({
        "align": align, "source": (src_h, src_w), "scaled": (new_h, new_w),
        "target": (height, width),
        "pad_top": pad_top, "pad_bottom": pad_bottom,
        "pad_left": pad_left, "pad_right": pad_right,
        "rows_dropped": 0, "cols_dropped": 0,
        "note": (f"{src_w}x{src_h} -> {new_w}x{new_h}, поля сверху {pad_top}, "
                 f"снизу {pad_bottom}, слева {pad_left}, справа {pad_right}; "
                 f"выброшено строк 0, столбцов 0"),
    })
    return out[None, ...], record


class _Wire:
    """Сборщик графа в UI-формате Comfy. Связи кладутся ПО ИМЕНИ входа.

    По имени, а не по номеру слота, намеренно: номер, выведенный из порядка,
    поедет при первом же добавлении входа в чужую ноду, и поедет молча. Имя
    сверяется с объявлением узла, и несуществующее имя роняет сборку здесь, а
    не на арендованной машине.
    """

    def __init__(self, proven: set):
        self.proven = proven
        self.nodes: list = []
        self.links: list = []
        self.introduced: list = []
        self._id = 0
        self._link = 0
        self._by_id: dict = {}

    def node(self, node_type: str, *, pack: str, inputs=(), outputs=(),
             widgets=None, title: str = "") -> int:
        self._id += 1
        node = {
            "id": self._id, "type": node_type, "mode": 0, "order": self._id,
            "flags": {}, "pos": [0, 0], "size": [300, 100],
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
            "properties": {"Node name for S&R": node_type, PACK_KEY: pack},
            "widgets_values": list(widgets or []),
            "title": title or node_type,
        }
        self.nodes.append(node)
        self._by_id[self._id] = node
        if node_type not in self.proven:
            self.introduced.append(
                {"type": node_type, "id": self._id,
                 "provenance": provenance_of(node_type, self.proven),
                 "pack": pack})
        return self._id

    def link(self, src_id: int, out_name: str, dst_id: int, in_name: str):
        src, dst = self._by_id[src_id], self._by_id[dst_id]
        out_slot = next((i for i, o in enumerate(src["outputs"])
                         if o["name"] == out_name), None)
        in_slot = next((i for i, o in enumerate(dst["inputs"])
                        if o["name"] == in_name), None)
        if out_slot is None:
            raise KeyError(f"у {src['type']}#{src_id} нет выхода {out_name!r}")
        if in_slot is None:
            raise KeyError(f"у {dst['type']}#{dst_id} нет входа {in_name!r}")
        self._link += 1
        kind = dst["inputs"][in_slot]["type"]
        self.links.append([self._link, src_id, out_slot, dst_id, in_slot, kind])
        src["outputs"][out_slot]["links"].append(self._link)
        dst["inputs"][in_slot]["link"] = self._link


def _lock_files(lock: dict) -> dict:
    """Роль -> имя файла, из лок-файла. Строкой в код имена не переносятся (Е1)."""
    return {w.get("role"): Path(w["path"]).name
            for w in lock.get("weights", []) if w.get("path")}


def derive_wrapper(*, seconds: float | None = None,
                   pose_strength: float | None = None,
                   face_strength: float | None = None,
                   colormatch: str | None = None,
                   blocks_to_swap: int | None = None,
                   width: int | None = None, height: int | None = None,
                   fps: int | None = None,
                   prompt: str = "", negative: str = "",
                   reference: str = "fork/reference.png",
                   driving: str = "fork/driving.mp4",
                   face_dir: str = "fork/face", pose_dir: str = "fork/body",
                   mask_dir: str = "fork/mask",
                   lock: dict | None = None) -> dict:
    """Собрать граф на обёртке. Возвращает `{"graph","introduced","params"}`.

    ВСЕ УМОЛЧАНИЯ РАЗРЕШАЮТСЯ В ТЕЛЕ, а не в сигнатуре: значение по умолчанию
    связывается на импорте, и подмена константы модуля до вызова не доехала бы
    — то есть мутация не покраснела бы ни в одном тесте (И7).

    `pose_strength` вынесен параметром намеренно: это рычаг синхронности, у
    владельца он поднят до 1.1 против умолчания ноды 1.0, и крутить его будут.

    Имена файлов весов берутся ИЗ ЛОК-ФАЙЛА, а не пишутся здесь строками:
    второй способ узнать известное — дефект (Е1), и именно он уже стоил
    расхождения графа со стеком.
    """
    seconds = SECONDS_MIN if seconds is None else seconds
    pose_strength = POSE_STRENGTH if pose_strength is None else pose_strength
    face_strength = FACE_STRENGTH if face_strength is None else face_strength
    colormatch = COLORMATCH if colormatch is None else colormatch
    blocks_to_swap = BLOCKS_TO_SWAP if blocks_to_swap is None else blocks_to_swap
    width = WRAP_WIDTH if width is None else width
    height = WRAP_HEIGHT if height is None else height
    fps = WRAP_FPS if fps is None else fps
    lock = load_lock() if lock is None else lock

    # Дешёвые проверки раньше дорогих (П2): всё, что можно забраковать
    # арифметикой, бракуется до чтения лока и до сборки узлов.
    if not 0 <= blocks_to_swap <= BLOCKS_MAX:
        raise ValueError(
            f"blocks_to_swap={blocks_to_swap} вне 0..{BLOCKS_MAX} — предел "
            f"самой ноды (nodes_model_loading.py:299). У 14B блоков 40")
    for name, side in (("width", width), ("height", height)):
        if side % SIDE_MULTIPLE:
            raise ValueError(
                f"{name}={side} не кратно {SIDE_MULTIPLE}: обёртка округлит "
                f"вниз молча (nodes.py:1223), и геометрия выхода перестанет "
                f"совпадать с записанной")
        if side < MIN_SIDE:
            raise ValueError(f"{name}={side} меньше блока маски {MIN_SIDE} px")

    length = frames_for_seconds(seconds, fps=fps)
    plan = window_plan(length["frames"])
    files = _lock_files(lock)
    missing = [r for r in ("diffusion", "text_encoder", "vae", "clip_vision",
                           "lora_1", "lora_2") if r not in files]
    if missing:
        raise KeyError(f"в лок-файле нет ролей {missing} — граф собирать не из "
                       f"чего, и придумывать имена файлов здесь запрещено (Е1)")

    w = _Wire(known_types(load_upstream()))
    core, wrap = CORE_PACK, WRAP_PACK

    # --- наши последовательности: маска и два канала условий -----------------
    ref = w.node("LoadImage", pack=core, widgets=[reference, "image"],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                 title="референс, УЖЕ приведённый полями (см. extra.fork)")
    drive = w.node("LoadVideo", pack=core, widgets=[driving, "image"],
                   outputs=[("VIDEO", "VIDEO")], title="драйвинг")
    drive_c = w.node("GetVideoComponents", pack=core,
                     inputs=[("video", "VIDEO")],
                     outputs=[("images", "IMAGE"), ("audio", "AUDIO"),
                              ("fps", "FLOAT")], title="кадры и звук драйвинга")
    w.link(drive, "VIDEO", drive_c, "video")

    channels = {}
    for name, folder in (("pose", pose_dir), ("face", face_dir),
                         ("mask", mask_dir)):
        loader = w.node("LoadVideo", pack=core, widgets=[f"{folder}.mp4", "image"],
                        outputs=[("VIDEO", "VIDEO")], title=f"наш канал {name}")
        comps = w.node("GetVideoComponents", pack=core,
                       inputs=[("video", "VIDEO")],
                       outputs=[("images", "IMAGE"), ("audio", "AUDIO"),
                                ("fps", "FLOAT")], title=f"кадры {name}")
        w.link(loader, "VIDEO", comps, "video")
        channels[name] = (comps, "images")
    mask_conv = w.node("ImageToMask", pack=core, widgets=["red"],
                       inputs=[("image", "IMAGE")], outputs=[("MASK", "MASK")],
                       title="маска из кадров")
    w.link(channels["mask"][0], "images", mask_conv, "image")

    # --- модель, адаптеры, обмен блоков --------------------------------------
    model = w.node("WanVideoModelLoader", pack=wrap,
                   widgets=[files["diffusion"], "bf16", "disabled",
                            "offload_device", "sdpa", "default"],
                   inputs=[("compile_args", "WANCOMPILEARGS"),
                           ("block_swap_args", "BLOCKSWAPARGS"),
                           ("lora", "WANVIDLORA")],
                   outputs=[("model", "WANVIDEOMODEL")],
                   title=f"диффузия из лока: {files['diffusion']}")
    loras = w.node("WanVideoLoraSelectMulti", pack=wrap,
                   widgets=[files["lora_2"], 1.0, files["lora_1"], 1.0,
                            EMPTY_LORA_SLOT, 0.0, EMPTY_LORA_SLOT, 0.0,
                            EMPTY_LORA_SLOT, 0.0, False, False],
                   inputs=[("prev_lora", "WANVIDLORA")],
                   outputs=[("lora", "WANVIDLORA")], title="адаптеры из лока")
    set_loras = w.node("WanVideoSetLoRAs", pack=wrap,
                       inputs=[("model", "WANVIDEOMODEL"),
                               ("lora", "WANVIDLORA")],
                       outputs=[("model", "WANVIDEOMODEL")],
                       title="адаптеры без слияния (GGUF слить нельзя)")
    w.link(model, "model", set_loras, "model")
    w.link(loras, "lora", set_loras, "lora")

    swap_args = w.node("WanVideoBlockSwap", pack=wrap,
                       widgets=[blocks_to_swap, False, False, True, 0, 1, False],
                       outputs=[("block_swap_args", "BLOCKSWAPARGS")],
                       title=f"обмен {blocks_to_swap} блоков")
    set_swap = w.node("WanVideoSetBlockSwap", pack=wrap,
                      inputs=[("model", "WANVIDEOMODEL"),
                              ("block_swap_args", "BLOCKSWAPARGS")],
                      outputs=[("model", "WANVIDEOMODEL")], title="обмен блоков")
    w.link(set_loras, "model", set_swap, "model")
    w.link(swap_args, "block_swap_args", set_swap, "block_swap_args")

    # --- VAE, зрение, текст ---------------------------------------------------
    vae = w.node("WanVideoVAELoader", pack=wrap,
                 widgets=[files["vae"], "bf16", False],
                 outputs=[("vae", "WANVAE")], title=f"VAE: {files['vae']}")
    clipv_load = w.node("CLIPVisionLoader", pack=core,
                        widgets=[files["clip_vision"]],
                        outputs=[("CLIP_VISION", "CLIP_VISION")],
                        title="зрение (штатный загрузчик)")
    clipv = w.node("WanVideoClipVisionEncode", pack=wrap,
                   widgets=[1.0, 1.0, "center", "average", True, 0, 0.5],
                   inputs=[("clip_vision", "CLIP_VISION"), ("image_1", "IMAGE")],
                   outputs=[("image_embeds", "WANVIDIMAGE_CLIPEMBEDS")],
                   title="референс глазами CLIP")
    w.link(clipv_load, "CLIP_VISION", clipv, "clip_vision")
    w.link(ref, "IMAGE", clipv, "image_1")

    # ТЕКСТ ИДЁТ НЕ РОДНЫМ УЗЛОМ ОБЁРТКИ, И ЭТО ЗАМЕР, А НЕ ВКУС.
    # `WanVideoTextEncodeCached` зовёт `LoadWanVideoT5TextEncoder`
    # (nodes_model_loading.py:1962), а тот читает `load_torch_file` и роняет
    # «Invalid T5 text encoder model» на .gguf. Лок объявляет энкодер Q5_K_M в
    # .gguf — ради того он и выбран, он вдвое легче. Значит энкодер грузит
    # CLIPLoaderGGUF (city96, Apache-2.0, уже в реестре доказанных), а формат
    # обёртки даёт мост `WanVideoTextEmbedBridge`.
    enc = w.node("CLIPLoaderGGUF", pack=GGUF_PACK,
                 widgets=[files["text_encoder"], "wan"],
                 outputs=[("CLIP", "CLIP")],
                 title=f"энкодер из лока: {files['text_encoder']}")
    pos = w.node("CLIPTextEncode", pack=core, widgets=[prompt],
                 inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], title="промпт")
    neg = w.node("CLIPTextEncode", pack=core, widgets=[negative],
                 inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], title="негатив")
    w.link(enc, "CLIP", pos, "clip")
    w.link(enc, "CLIP", neg, "clip")
    bridge = w.node("WanVideoTextEmbedBridge", pack=wrap,
                    inputs=[("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING")],
                    outputs=[("text_embeds", "WANVIDEOTEXTEMBEDS")],
                    title="мост: штатное CONDITIONING -> формат обёртки")
    w.link(pos, "CONDITIONING", bridge, "positive")
    w.link(neg, "CONDITIONING", bridge, "negative")

    # --- условия Animate, сэмплер, декод -------------------------------------
    embeds = w.node("WanVideoAnimateEmbeds", pack=wrap,
                    widgets=[width, height, length["frames"], False,
                             WRAP_WINDOW, colormatch, pose_strength,
                             face_strength, False],
                    inputs=[("vae", "WANVAE"),
                            ("clip_embeds", "WANVIDIMAGE_CLIPEMBEDS"),
                            ("ref_images", "IMAGE"), ("pose_images", "IMAGE"),
                            ("face_images", "IMAGE"), ("bg_images", "IMAGE"),
                            ("mask", "MASK")],
                    outputs=[("image_embeds", "WANVIDIMAGE_EMBEDS")],
                    title=f"условия Animate {width}x{height}x{length['frames']}")
    w.link(vae, "vae", embeds, "vae")
    w.link(clipv, "image_embeds", embeds, "clip_embeds")
    w.link(ref, "IMAGE", embeds, "ref_images")
    w.link(channels["pose"][0], "images", embeds, "pose_images")
    w.link(channels["face"][0], "images", embeds, "face_images")
    w.link(mask_conv, "MASK", embeds, "mask")

    sampler = w.node("WanVideoSampler", pack=wrap,
                     # Порядок и типы сверены с объявлением ноды
                     # (`API_WIDGETS`, разбор исходника). ~~""~~ на месте
                     # `batched_cfg` заменено на False 18.08: виджет объявлен
                     # BOOLEAN, а пустая строка туда попала из шаблона и была
                     # НЕВИДИМА, пока граф жил в формате UI — там имён нет, и
                     # значение просто лежит в списке. Нашёл переводчик в API.
                     widgets=[WRAP_STEPS, WRAP_CFG, WRAP_SHIFT, WRAP_SEED,
                              "fixed", True, WRAP_SCHEDULER, 0, 1.0, False,
                              "comfy", 0, -1, False],
                     inputs=[("model", "WANVIDEOMODEL"),
                             ("image_embeds", "WANVIDIMAGE_EMBEDS"),
                             ("text_embeds", "WANVIDEOTEXTEMBEDS")],
                     outputs=[("samples", "LATENT"),
                              ("denoised_samples", "LATENT")],
                     title=(f"сэмплер: {WRAP_STEPS} шага, окно {WRAP_WINDOW} "
                            f"внутри"))
    w.link(set_swap, "model", sampler, "model")
    w.link(embeds, "image_embeds", sampler, "image_embeds")
    w.link(bridge, "text_embeds", sampler, "text_embeds")

    decode = w.node("WanVideoDecode", pack=wrap,
                    widgets=[False, 272, 272, 144, 128, "default"],
                    inputs=[("vae", "WANVAE"), ("samples", "LATENT")],
                    outputs=[("images", "IMAGE")], title="декод")
    w.link(vae, "vae", decode, "vae")
    w.link(sampler, "samples", decode, "samples")

    video = w.node("CreateVideo", pack=core, widgets=[fps],
                   inputs=[("images", "IMAGE"), ("audio", "AUDIO"),
                           ("fps", "FLOAT")],
                   outputs=[("VIDEO", "VIDEO")], title=f"сборка {fps} к/с")
    w.link(decode, "images", video, "images")
    w.link(drive_c, "audio", video, "audio")
    save = w.node("SaveVideo", pack=core, widgets=["video/fork", "auto", "auto"],
                  inputs=[("video", "VIDEO")], title="выход")
    w.link(video, "VIDEO", save, "video")

    params = {
        "width": width, "height": height, "fps": fps,
        "seconds": seconds, "length": length, "windows": plan,
        "pose_strength": pose_strength, "face_strength": face_strength,
        "colormatch": colormatch, "blocks_to_swap": blocks_to_swap,
        "steps": WRAP_STEPS, "cfg": WRAP_CFG, "shift": WRAP_SHIFT,
        "scheduler": WRAP_SCHEDULER, "seed": WRAP_SEED,
        "weights": files,
    }
    graph = {
        "id": "fork-wrapper", "revision": 0, "version": 0.4,
        "last_node_id": w._id, "last_link_id": w._link,
        "nodes": w.nodes, "links": w.links, "groups": [], "config": {},
        "extra": {"fork": {
            "built_on": WRAP_PACK,
            "license": WRAP_LICENSE,
            "params": params,
            "reference_fit": REFERENCE_FIT,
            "resize_facts": WRAP_RESIZE_FACTS,
            "no_composite": NO_COMPOSITE,
            "unrun": ("НЕПРОВЕРЕНО: граф не исполнялся, проверка структурная — "
                      "типы нод, связи, обязательные входы, файлы весов"),
            "deferred": [
                ("DEBT(2026-08-18): trim_to_audio у владельца стоит на "
                 "VHS_VideoCombine; штатный CreateVideo такого виджета не "
                 "несёт, а VHS в этой смене не взят (лицензия не проверена). "
                 "Обрезка по звуку не реализована."),
                ("DEBT(2026-08-18): блокификация маски 48 px из боевого "
                 "воркфлоу владельца против наших 32 — решение живёт в "
                 "fork_mask, не в этом модуле."),
            ],
        }},
    }
    return {"graph": graph, "introduced": w.introduced, "params": params}


#: Реестр загрузчиков пополняется, а не заводится второй (Е1). Индексы виджетов
#: взяты из ОБЪЯВЛЕНИЙ `INPUT_TYPES` скачанных исходников, а не из порядка,
#: который кажется естественным: у `WanVideoLoraSelectMulti` файлы стоят через
#: один (файл, сила, файл, сила, ...), и «первый виджет — имя файла» здесь
#: верно лишь для первого из пяти.
WEIGHT_WIDGET.update({
    "WanVideoModelLoader": ("model", 0),
    "WanVideoVAELoader": ("model_name", 0),
    "WanVideoTextEncodeCached": ("model_name", 0),
    "WanVideoLoraSelectMulti": ("lora_0..lora_4", (0, 2, 4, 6, 8)),
})


def wrap_unfed_inputs(graph: dict) -> tuple:
    """Обязательные входы узлов обёртки, в которые ничего не приходит.

    Возвращает `(сколько узлов проверено, список нарушений)` — ДВА числа, а не
    список. Ноль нарушений при нуле проверенных узлов не успех (Р2): именно так
    выглядел бы граф, в котором сборка не создала ни сэмплера, ни условий.

    Тот же прибор, что `unfed_inputs` части I, переработанный под новый формат:
    там пять обязательных входов у одной ноды внутри сабграфа, здесь таблица
    `WRAP_REQUIRED` — семь нод и семнадцать входов.
    """
    links = _links_of(graph)
    checked = 0
    bad: list[str] = []
    for node in graph.get("nodes", []):
        need = WRAP_REQUIRED.get(str(node.get("type")))
        if not need:
            continue
        checked += 1
        names = [i.get("name") for i in node.get("inputs", [])]
        slots = {name: i for i, name in enumerate(names)}
        fed = {l[4] for l in links if len(l) > 4 and l[3] == node.get("id")}
        for want in need:
            if want not in slots:
                bad.append(f"{node['type']}#{node.get('id')}: входа {want} нет "
                           f"в объявлении узла")
            elif slots[want] not in fed:
                bad.append(f"{node['type']}#{node.get('id')}.{want} без питания")
    return checked, sorted(bad)


def audit_loader_formats(graph: dict) -> dict:
    """Умеет ли загрузчик читать тот файл, который ему подан. Три исхода.

    ЗАЧЕМ ОТДЕЛЬНО ОТ `audit_weights`. Тот сверяет ИМЕНА с лок-файлом и на
    вопрос «а прочитается ли» не отвечает вовсе: имя может совпасть с локом
    идеально, а узел упадёт на первой же секунде, потому что читает только
    safetensors. Ровно это и вышло бы с текстовым энкодером: лок объявляет
    Q5_K_M в .gguf, а родной `WanVideoTextEncodeCached` обёртки читает его
    через `load_torch_file`. Проверка написана после того, как это нашлось
    чтением исходника, и она сторожит, чтобы находка не забылась.
    """
    rows = graph_weights(graph)
    known = [r for r in rows if r["type"] in LOADER_FORMATS]
    if not known:
        return {"outcome": UNMEASURED, "problems": [], "checked": 0,
                "skipped": len(rows) - len(known),
                "note": (f"разбирать нечего: загрузчиков с известным форматом "
                         f"0, файлов в графе {len(rows)}. Ноль конфликтов при "
                         f"нуле проверенного — не успех (Р2)")}
    bad = []
    for row in known:
        allowed = LOADER_FORMATS[row["type"]]
        if not row["file"].endswith(allowed):
            bad.append(f"{row['type']}#{row['id']} подан {row['file']}, а "
                       f"читать умеет только {', '.join(allowed)}")
    return {
        "outcome": FAIL if bad else PASS, "problems": bad,
        "checked": len(known), "skipped": len(rows) - len(known),
        "note": (f"проверено загрузчиков {len(known)}, форматов неизвестных "
                 f"{len(rows) - len(known)}, конфликтов {len(bad)}"
                 + ("" if not bad else ": " + "; ".join(bad))),
    }


def audit_wrapper(derived: dict, *, lock: dict | None = None) -> dict:
    """Структурная проверка графа обёртки. Три исхода, числа рядом с вердиктом.

    Это `audit` части I, переработанный под новый формат. Что осталось тем же:
    три исхода, список введённых недоказанных типов, числа в отчёте, вслух
    сказанное «граф не исполнялся». Что изменилось: вырезать больше нечего —
    граф собран, а не выкроен, — поэтому вместо «остались ли кастомные ноды»
    проверяется, что в нём НЕТ паков, которые владелец решил не брать, и что
    каждый пак, который всё-таки нужен, НАЗВАН вместе с лицензией.
    """
    graph = derived["graph"] if "graph" in derived else derived
    nodes = graph.get("nodes", [])
    ids = {n.get("id") for n in nodes}

    heart = [n for n in nodes if n.get("type") == "WanVideoAnimateEmbeds"]
    checked_nodes, starving = wrap_unfed_inputs(graph)
    if not heart or not checked_nodes:
        return {"outcome": UNMEASURED, "problems": [],
                "nodes": len(nodes), "checked_nodes": checked_nodes,
                "unfed": starving, "unproven_types": [], "packs_required": [],
                "forbidden": [],
                "note": (f"судить нечем: нод {len(nodes)}, узлов с "
                         f"обязательными входами {checked_nodes}, "
                         f"WanVideoAnimateEmbeds {len(heart)}. Ноль нарушений "
                         f"при нуле проверенного — не успех (Р2)")}

    dangling = [l[0] for l in _links_of(graph)
                if len(l) > 3 and (l[1] not in ids or l[3] not in ids)]
    introduced = derived.get("introduced", []) if isinstance(derived, dict) else []
    unproven = [i for i in introduced if "НЕПРОВЕРЕНО" in i.get("provenance", "")]
    packs = sorted({pack_of(n) for n in nodes
                    if pack_of(n) and pack_of(n).lower() != CORE_PACK})
    forbidden = sorted({p for p in packs if p.lower() in FORBIDDEN_PACKS})

    weights = audit_weights({"graph": graph}, lock=lock)
    formats = audit_loader_formats(graph)

    problems = []
    if dangling:
        problems.append(f"оборванных связей {len(dangling)}")
    if starving:
        problems.append(f"входов без питания {len(starving)}: "
                        + "; ".join(starving))
    if unproven:
        problems.append("введены недоказанные типы: "
                        + ", ".join(i["type"] for i in unproven))
    for p in forbidden:
        problems.append(f"пак {p} в графе, а решение владельца — "
                        f"{FORBIDDEN_PACKS[p.lower()]}")
    if weights["outcome"] == FAIL:
        problems.append("веса разошлись с локом: " + weights["note"])
    if formats["outcome"] == FAIL:
        problems.append("формат файла не по загрузчику: " + formats["note"])

    return {
        "outcome": FAIL if problems else PASS,
        "nodes": len(nodes), "links": len(_links_of(graph)),
        "checked_nodes": checked_nodes, "unfed": starving,
        "dangling": dangling, "unproven_types": unproven,
        "packs_required": packs, "forbidden": forbidden,
        "weights": weights, "formats": formats,
        "problems": problems,
        "note": (f"нод {len(nodes)}, связей {len(_links_of(graph))}; узлов с "
                 f"обязательными входами проверено {checked_nodes}, входов без "
                 f"питания {len(starving)}, оборванных связей {len(dangling)}, "
                 f"введённых недоказанных типов {len(unproven)}, сторонних "
                 f"паков {len(packs)}"
                 + (f" ({', '.join(packs)})" if packs else "")
                 + f", запрещённых паков {len(forbidden)}; "
                 f"весов разобрано {weights['loaders_checked']} из "
                 f"{weights['declared']} в локе, расхождений "
                 f"{len(weights['problems'])}; форматов проверено "
                 f"{formats['checked']}, конфликтов {len(formats['problems'])}. "
                 + ("ЧИСТО" if not problems else "; ".join(problems))
                 + ". НЕПРОВЕРЕНО: граф не исполнялся."),
    }


def check_wrapper_inputs(inputs: dict) -> list:
    """Структурные нарушения входов графа обёртки. Список причин, а не флаг.

    Отличие от `check_inputs` части I не в именах полей, а в том, ЧЕГО эта
    проверка боится. Штатная нода на негодном входе отказывала; обёртка
    МОЛЧА ПРИВОДИТ: длину прижимает к шагу 4 (nodes.py:1230), стороны
    округляет вниз до 16 (1223), лицо режет по центру до 512 (1324), референс
    и позу растягивает до кадра (1288, 1253). То есть негодный вход здесь не
    падает, а тихо становится другим входом, и разбираться в этом придётся по
    готовому ролику. Все пять чисел проверены чтением исходника — см.
    `WRAP_RESIZE_FACTS`.
    """
    bad: list[str] = []
    required = ("ref_images", "pose_images", "face_images", "mask")
    missing = [k for k in required if k not in inputs]
    if missing:
        bad.append("нет входов: " + ", ".join(missing))

    w, h = inputs.get("width"), inputs.get("height")
    n = inputs.get("num_frames")
    for name, val in (("width", w), ("height", h), ("num_frames", n)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            bad.append(f"{name} не положительное целое: {val!r}")
    if bad:
        return bad

    for name, val in (("width", w), ("height", h)):
        if val % SIDE_MULTIPLE:
            bad.append(f"{name}={val} не кратно {SIDE_MULTIPLE}: обёртка "
                       f"округлит вниз молча (nodes.py:1223)")
        if val < MIN_SIDE:
            bad.append(f"{name}={val} меньше блока маски {MIN_SIDE} px")
    snapped = snap_frames(n)
    if snapped != n:
        bad.append(f"num_frames={n}: обёртка прижмёт к {snapped} молча "
                   f"(шаг {LENGTH_STEP} от {LENGTH_BASE}, nodes.py:1230)")

    for name in required:
        arr = inputs.get(name)
        if arr is None:
            continue
        a = np.asarray(arr)
        if name == "ref_images":
            if a.ndim != 4 or a.shape[0] != 1 or a.shape[-1] != 3:
                bad.append(f"{name}: ожидалась одна картинка (1,H,W,3), "
                           f"пришло {a.shape}")
            elif a.shape[1:3] != (h, w):
                # ПОЧЕМУ ЭТО НАРУШЕНИЕ, А НЕ УДОБСТВО. Референс, поданный не в
                # геометрии кадра, обёртка растянет (1288) — пропорции лица
                # поедут, а личность мы меряем как раз по лицу. Приводить
                # обязан `pad_reference`, до графа и полями.
                bad.append(f"{name}: {a.shape[1]}x{a.shape[2]} вместо "
                           f"{h}x{w} — референс обязан приезжать приведённым "
                           f"полями (pad_reference), иначе обёртка растянет")
            continue
        if name == "mask":
            if a.ndim != 3:
                bad.append(f"{name}: ожидалась (N,H,W) без канала, "
                           f"пришло {a.shape}")
            elif a.shape[1:] != (h, w):
                bad.append(f"{name}: {a.shape[1:]} вместо ({h},{w})")
            if a.ndim >= 1 and a.shape[0] != n:
                bad.append(f"{name}: кадров {a.shape[0]}, а num_frames={n}")
            continue
        if a.ndim != 4 or a.shape[-1] != 3:
            bad.append(f"{name}: ожидалась (N,H,W,3), пришло {a.shape}")
            continue
        if a.shape[0] != n:
            bad.append(f"{name}: кадров {a.shape[0]}, а num_frames={n}")
        if name == "face_images":
            if a.shape[1:3] != (FACE_SIDE, FACE_SIDE):
                bad.append(f"{name}: {a.shape[1]}x{a.shape[2]} вместо "
                           f"{FACE_SIDE}x{FACE_SIDE} — обёртка дорежет по "
                           f"центру (nodes.py:1324)")
        elif a.shape[1:3] != (h, w):
            bad.append(f"{name}: {a.shape[1]}x{a.shape[2]} вместо {h}x{w}")
    return bad


#: Два производства — два набора проверок, и `render` выбирает НАБОР, а не
#: пишется дважды (Е1). Ключ едет параметром `kind`; умолчание — часть I,
#: чтобы уже написанное не поменяло поведения молча.
RENDER_KINDS = {
    "native": {"audit": lambda g: audit({"graph": g}),
               "check": check_inputs, "length_key": "length"},
    "wrapper": {"audit": lambda g: audit_wrapper({"graph": g}),
                "check": check_wrapper_inputs, "length_key": "num_frames"},
}


# ─────────────────────────────────────────────────────────────────────────────
# ФОРМАТ UI ПРОТИВ ФОРМАТА API: без этого перевода e2e не запускается
# ─────────────────────────────────────────────────────────────────────────────
#
# НАЙДЕНО 18.08.2026 прямым вызовом: `fork_backend.check_graph` на нашем графе
# отвечает «граф в формате UI (ключ `nodes` списком), а /prompt принимает
# формат API». Обе половины были зелёными по отдельности, а стыка не было — тот
# самый класс, ради которого в проекте заведён сторож достижимости.
#
# Формат UI — то, что сохраняет веб-интерфейс: список узлов, отдельный список
# связей, значения виджетов ПОЗИЦИОННЫМ списком БЕЗ ИМЁН. Формат API — то, что
# принимает `/prompt`: {"<id>": {"class_type": ..., "inputs": {имя: значение
# или [id_источника, слот]}}}. Перевод упирается ровно в одно: имена виджетов
# в UI не хранятся вовсе, их надо знать.
#
# ОТКУДА ВЗЯТЫ ИМЕНА. Не из головы и не из наблюдения за интерфейсом, а
# разбором `INPUT_TYPES`/`define_schema` в ИСХОДНИКАХ самих нод (правило Ц10:
# существование внешнего имени доказывается командой до того, как оно попало в
# код). Файлы скачаны и захешированы; два из трёх хешей СОШЛИСЬ с теми, что уже
# лежали в реестре доказательств этого модуля, — то есть разобран тот же файл,
# который читала предыдущая смена.

#: Источники имён виджетов: файл -> sha256 тела на 18.08.2026.
#: Проверяется командой `sha256sum` после `curl`; хеши `nodes.py` и
#: `nodes_model_loading.py` совпадают с `_SHA_NODES` и `_SHA_LOADING`.
WIDGET_SOURCES = {
    "kijai/ComfyUI-WanVideoWrapper@main:nodes.py":
        "6b3bb6a619e5a928259e6e8400c73ffe7140f17439c5007dd8596b90055f0666",
    "kijai/ComfyUI-WanVideoWrapper@main:nodes_model_loading.py":
        "2297aed82d0908505b735e3248f98fc39f815119e4a32c57ee9cf59d4a8a5ba5",
    "kijai/ComfyUI-WanVideoWrapper@main:nodes_sampler.py":
        "374e552a2f96e10ddfcb37785718be0ea0dfa34a6551a5930e8963e3bc87630b",
    "comfyanonymous/ComfyUI@master:nodes.py":
        "9f9162cc9ec180baad14c082fc61a781fed53baf325046d9853757e4ebd0c4e9",
    "comfyanonymous/ComfyUI@master:comfy_extras/nodes_video.py":
        "727653159de9a6e9d5b37b1101d3ece5df7dce8fc6a5a3a1f27e260afcfa6c14",
    "comfyanonymous/ComfyUI@master:comfy_extras/nodes_mask.py":
        "f235d52703229b0bb7840cc48f0fffe6b00bf9398e119aa7d7f2fc9f3bdaba12",
    "city96/ComfyUI-GGUF@main:nodes.py":
        "16be3b08b13de6279fc432addc628320019fcb24963cbc6b52b248de8f06316e",
}

#: Тип узла -> (имена виджетов В ПОРЯДКЕ ОБЪЯВЛЕНИЯ, сколько из них обязательны).
#: ИЗМЕРЕНО 18.08.2026 разбором исходников нод (см. `WIDGET_SOURCES`), а не
#: списано с интерфейса. Порядок существенен: в формате UI значения лежат
#: позиционно, и имя им даёт только он.
API_WIDGETS = {
    'CLIPLoaderGGUF': ((('clip_name', 'COMBO'), ('type', 'COMBO')), 2),
    # `clip` СНЯТ 18.08: он вход-связь, а не виджет. Разбор исходника принял
    # его за виджет (в новом API ноды объявлены `io.Clip.Input`, и мой парсер
    # свёл незнакомый вид к COMBO). Поймано сверкой с ЭТАЛОНОМ владельца —
    # `workflows/fork_widget_names.reference.json`, где фронтенд ComfyUI сам
    # называет виджеты: у `CLIPTextEncode` он знает ровно один, `text`.
    # В нашем графе `clip` всегда связан, поэтому дефект ничего не сломал; но
    # развяжи его кто-нибудь — и в него уехало бы значение промта.
    'CLIPTextEncode': ((('text', 'STRING'),), 1),
    'CLIPVisionLoader': ((('clip_name', 'COMBO'),), 1),
    'CreateVideo': ((('fps', 'FLOAT'), ('bit_depth', 'INT')), 1),
    'GetVideoComponents': ((), 0),
    'ImageToMask': ((('channel', 'COMBO'),), 1),
    'LoadImage': ((('image', 'COMBO'),), 1),
    'LoadVideo': ((('file', 'COMBO'),), 1),
    'SaveVideo': ((('filename_prefix', 'STRING'), ('format', 'COMBO'), ('codec', 'COMBO')), 3),
    'WanVideoAnimateEmbeds': ((('width', 'INT'), ('height', 'INT'), ('num_frames', 'INT'), ('force_offload', 'BOOLEAN'), ('frame_window_size', 'INT'), ('colormatch', 'COMBO'), ('pose_strength', 'FLOAT'), ('face_strength', 'FLOAT'), ('tiled_vae', 'BOOLEAN')), 8),
    'WanVideoBlockSwap': ((('blocks_to_swap', 'INT'), ('offload_img_emb', 'BOOLEAN'), ('offload_txt_emb', 'BOOLEAN'), ('use_non_blocking', 'BOOLEAN'), ('vace_blocks_to_swap', 'INT'), ('prefetch_blocks', 'INT'), ('block_swap_debug', 'BOOLEAN')), 3),
    'WanVideoClipVisionEncode': ((('strength_1', 'FLOAT'), ('strength_2', 'FLOAT'), ('crop', 'COMBO'), ('combine_embeds', 'COMBO'), ('force_offload', 'BOOLEAN'), ('tiles', 'INT'), ('ratio', 'FLOAT')), 5),
    'WanVideoDecode': ((('enable_vae_tiling', 'BOOLEAN'), ('tile_x', 'INT'), ('tile_y', 'INT'), ('tile_stride_x', 'INT'), ('tile_stride_y', 'INT'), ('normalization', 'COMBO')), 5),
    'WanVideoLoraSelectMulti': ((('lora_0', 'COMBO'), ('strength_0', 'FLOAT'), ('lora_1', 'COMBO'), ('strength_1', 'FLOAT'), ('lora_2', 'COMBO'), ('strength_2', 'FLOAT'), ('lora_3', 'COMBO'), ('strength_3', 'FLOAT'), ('lora_4', 'COMBO'), ('strength_4', 'FLOAT'), ('low_mem_load', 'BOOLEAN'), ('merge_loras', 'BOOLEAN')), 10),
    'WanVideoModelLoader': ((('model', 'COMBO'), ('base_precision', 'COMBO'), ('quantization', 'COMBO'), ('load_device', 'COMBO'), ('attention_mode', 'COMBO'), ('rms_norm_function', 'COMBO')), 4),
    'WanVideoSampler': ((('steps', 'INT'), ('cfg', 'FLOAT'), ('shift', 'FLOAT'), ('seed', 'INT'), ('force_offload', 'BOOLEAN'), ('scheduler', 'COMBO'), ('riflex_freq_index', 'INT'), ('denoise_strength', 'FLOAT'), ('batched_cfg', 'BOOLEAN'), ('rope_function', 'COMBO'), ('start_step', 'INT'), ('end_step', 'INT'), ('add_noise_to_samples', 'BOOLEAN')), 7),
    'WanVideoSetBlockSwap': ((), 0),
    'WanVideoSetLoRAs': ((), 0),
    'WanVideoTextEmbedBridge': ((), 0),
    'WanVideoVAELoader': ((('model_name', 'COMBO'), ('precision', 'COMBO'), ('use_cpu_cache', 'BOOLEAN'), ('verbose', 'BOOLEAN')), 1),
}

#: Каким типам объявленных виджетов какие значения годятся. Проверка дешёвая
#: и ловит целый класс: значение, попавшее не в тот виджет, чаще всего ещё и
#: не того типа. Найдено сразу же — `batched_cfg` (BOOLEAN) получал пустую
#: строку.
WIDGET_TYPE_OK = {
    "INT": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "FLOAT": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "BOOLEAN": lambda v: isinstance(v, bool),
    "STRING": lambda v: isinstance(v, str),
    "COMBO": lambda v: isinstance(v, (str, int, float, bool)),
}

#: Виджеты, которых В ОБЪЯВЛЕНИИ НОДЫ НЕТ, а в сохранённом UI-графе они есть.
#: Их дорисовывает фронтенд ComfyUI, и в формат API они не идут. Не выкинув их,
#: мы сдвинули бы ВСЕ последующие значения на одну позицию — то есть подали бы
#: `scheduler` туда, где ждут `force_offload`, и узнали бы об этом по ролику.
#:
#: 1. `control_after_generate` — приписывается фронтендом после целого виджета
#:    с именем `seed`. У нас всплыло на `WanVideoSampler`: 14 значений в графе
#:    против 13 объявленных, лишнее — строка "fixed" сразу после зерна.
#: 2. Кнопка загрузки у нод, которые принимают файл: `LoadImage`, `LoadVideo`.
#:    Там 2 значения против 1 объявленного.
#: Оба случая — известное поведение интерфейса, а не дефект нашего графа; это
#: проверено сверкой всех 27 узлов с объявлениями, расхождений больше нет.
UI_ONLY_AFTER_SEED = "control_after_generate"
SEED_WIDGETS = ("seed", "noise_seed")
UI_UPLOAD_NODES = ("LoadImage", "LoadVideo")


def api_widget_names(node: dict) -> dict:
    """Имена виджетов ЭТОГО узла с учётом того, что часть входов уже связана.

    Связанный вход виджетом быть перестаёт: значение придёт по связи, и в
    позиционном списке его нет. Считать иначе значит сдвинуть весь список.
    """
    spec = API_WIDGETS.get(node.get("type"))
    if spec is None:
        return {"outcome": UNMEASURED, "names": None, "required": 0,
                "note": (f"тип {node.get('type')!r} не разобран: имён его "
                         f"виджетов в реестре нет. Позиционная догадка здесь "
                         f"запрещена — она молча подаёт значение не в тот вход")}
    pairs, required = spec
    linked = {i.get("name") for i in node.get("inputs", [])
              if i.get("link") is not None}
    kept = [(n, t) for n, t in pairs if n not in linked]
    req = sum(1 for n, _ in pairs[:required] if n not in linked)
    return {"outcome": PASS, "names": [n for n, _ in kept],
            "types": [t for _, t in kept], "required": req, "note": ""}


def _strip_ui_widgets(node: dict, names: list, values: list) -> tuple:
    """Выкинуть дорисованное интерфейсом. Возвращает (значения, что выкинуто)."""
    values = list(values)
    dropped = []
    if node.get("type") in UI_UPLOAD_NODES and len(values) == len(names) + 1:
        dropped.append(f"кнопка загрузки {values.pop()!r}")
    for i, name in enumerate(names):
        if name in SEED_WIDGETS and len(values) > len(names) and i + 1 < len(values):
            dropped.append(f"{UI_ONLY_AFTER_SEED} {values.pop(i + 1)!r}")
    return values, dropped


def to_api(graph: dict) -> dict:
    """Перевести граф из формата UI в формат API. Три исхода, числа рядом.

    НЕ УГАДЫВАЕТ. Узел, для которого имён виджетов нет, или узел, у которого
    число значений не сходится с объявлением, — это «не смогли перевести», а
    не «переведём как получится». Позиционная догадка здесь стоит не ошибки
    сборки, а неверного ролика через двадцать минут счёта на карте.
    """
    graph = graph.get("graph", graph)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return {"outcome": UNMEASURED, "api": None, "problems": [],
                "checked": 0, "converted": 0,
                "note": ("это не граф формата UI: ключа `nodes` списком нет. "
                         "Ноль ошибок при нуле переведённого — не успех (Р2)")}

    by_link = {l[0]: l for l in _links_of(graph) if len(l) > 4}
    api, problems, unknown, dropped_all = {}, [], [], []
    for node in nodes:
        nid = str(node.get("id"))
        spec = api_widget_names(node)
        if spec["outcome"] != PASS:
            # ОТДЕЛЬНЫЙ СПИСОК, а не общий: «имени в реестре нет» — это
            # «не смогли перевести», а «значение не того типа» — «не годно».
            # Свёрнутые вместе, они дают вердикт «граф НЕ ГОДЕН» на добавлении
            # ноды из нового пака, и следующая смена идёт искать дефект в
            # графе, которого там нет. Правильный ответ — пополнить реестр.
            unknown.append(spec["note"])
            continue
        names, values = spec["names"], list(node.get("widgets_values") or [])
        values, dropped = _strip_ui_widgets(node, names, values)
        dropped_all.extend(dropped)
        if not spec["required"] <= len(values) <= len(names):
            problems.append(
                f"{node.get('type')}#{nid}: значений виджетов {len(values)}, а "
                f"объявлено {spec['required']}..{len(names)} ({', '.join(names)}"
                f"). Разложить позиционно нельзя — какое значение чьё, "
                f"неизвестно")
            continue
        for name, kind, value in zip(names, spec["types"], values):
            check = WIDGET_TYPE_OK.get(kind)
            if check is not None and not check(value):
                problems.append(
                    f"{node.get('type')}#{nid}: виджет {name} объявлен {kind}, "
                    f"а подано {value!r} ({type(value).__name__}). В формате UI "
                    f"это невидимо — имён там нет, и значение просто лежит "
                    f"в списке")
        inputs = dict(zip(names, values))
        for slot in node.get("inputs", []):
            link = by_link.get(slot.get("link"))
            if link is not None:
                inputs[slot["name"]] = [str(link[1]), link[2]]
        api[nid] = {"class_type": node.get("type"), "inputs": inputs,
                    "_meta": {"title": node.get("title") or node.get("type")}}

    outcome = (FAIL if problems else
               UNMEASURED if unknown or not api else PASS)
    return {
        "outcome": outcome,
        "api": api if outcome == PASS else None,
        "problems": problems, "unknown_types": unknown,
        "checked": len(nodes), "converted": len(api),
        "dropped_ui_widgets": dropped_all,
        "note": (f"узлов в графе {len(nodes)}, переведено {len(api)}, "
                 f"негодных {len(problems)}, неизвестных типов "
                 f"{len(unknown)}; выкинуто дорисованных интерфейсом "
                 f"виджетов {len(dropped_all)}"
                 + ("" if not problems else
                    ". НЕ ГОДЕН: " + "; ".join(problems[:3]))
                 + ("" if not unknown else
                    ". НЕ СМОГЛИ ПЕРЕВЕСТИ: " + "; ".join(unknown[:3]))
                 + ". НЕПРОВЕРЕНО: настоящий ComfyUI этот перевод не принимал"),
    }
