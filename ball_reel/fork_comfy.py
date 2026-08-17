"""Поток E форка: граф ПРОИЗВОДИТСЯ из официального, а не сочиняется.

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
    """Типы нод, чьё существование ДОКАЗАНО файлом темплейта (Ц10)."""
    return {n["type"] for n in graph.get("nodes", [])}


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

    `quant` — РАЗВИЛКА §1, И ОНА НАРОЧНО НЕ РЕШЕНА ЗДЕСЬ. Умолчание оставляет
    ступень такой, какая приехала с темплейтом (fp8, 17.138 ГиБ), и в этом
    состоянии `audit_weights` КРАСНЫЙ: лок объявляет Q3_K_M. Красное здесь —
    не недоделка, а единственный честный способ показать нерешённое: развилку
    выбирает владелец, а модуль, который выбрал бы за него молча, спрятал бы
    решение в умолчание. `quant=QUANT_GGUF` переводит граф на то, что в локе, и
    та же проверка становится зелёной — то есть решение стоит одного слова, а
    не переписывания графа.
    """
    # Умолчание разрешается ЗДЕСЬ, а не в сигнатуре: значение по умолчанию
    # связывается на импорте, и подмена константы модуля до него не доходит —
    # то есть мутация `QUANT_TEMPLATE` не покраснела бы ни в одном тесте. Эту
    # форму на проекте уже выгребали в семи местах (И7), и вносить её обратно
    # ради одной строки нельзя.
    quant = QUANT_TEMPLATE if quant is None else quant
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


def render(workflow: dict, inputs: dict, *, backend=None) -> dict:
    """`render(workflow, inputs) -> frames` — на моке, и он в этом сознаётся.

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

    graph = workflow.get("graph", workflow) if isinstance(workflow, dict) else {}
    report = audit({"graph": graph}) if graph else {"outcome": UNMEASURED,
                                                    "problems": ["графа нет"]}
    if report["outcome"] != PASS:
        problems += [f"граф: {p}" for p in report.get("problems", [])] or \
            ["граф: проверить не удалось"]

    problems += [f"вход: {p}" for p in check_inputs(inputs)]

    if problems:
        return {
            "source": NOTHING, "outcome": FAIL, "frames": None,
            "problems": problems, "backend": None,
            "note": (f"{NOTHING}: нарушений {len(problems)} — "
                     + "; ".join(problems)
                     + ". Кадры не выданы даже поддельные: пустой результат "
                       "заметен, а правдоподобный мок на негодном входе — нет"),
        }

    length, h, w = inputs["length"], inputs["height"], inputs["width"]

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


#: Чем оканчивается имя файла весов. ВЫБРАНО мной из наблюдаемого: оба
#: расширения встречаются и в шаблоне, и в лок-файле, и ничего третьего стек не
#: грузит. Нужно затем, чтобы виджет, оказавшийся НЕ именем файла, был находкой,
#: а не тихо принятой строкой.
WEIGHT_SUFFIXES = (".safetensors", ".gguf")


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
        widget, index = spec
        values = node.get("widgets_values") or []
        if index >= len(values) or not isinstance(values[index], str):
            continue
        if not values[index].endswith(WEIGHT_SUFFIXES):
            continue
        out.append({"id": node.get("id"), "type": node["type"],
                    "widget": widget, "file": values[index]})
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
        widget, index = spec
        values = node.get("widgets_values") or []
        got = values[index] if index < len(values) else None
        if isinstance(got, str) and got.endswith(WEIGHT_SUFFIXES):
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
