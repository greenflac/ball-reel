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
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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

#: Три исхода (Е1: те же три слова, что у остальных потоков).
from .fork_identity import FAIL, PASS, UNMEASURED  # noqa: E402


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
           mask_dir: str = "fork/mask") -> dict:
    """Произвести наш граф: вырезать ветку препроцессинга, подставить свою.

    Возвращает `{"graph": ..., "removed": [...], "introduced": [...]}` — и
    список введённых типов возвращается ВСЕГДА, даже пустой. Отчёт, в котором
    поле появляется только когда есть что сказать, читается как «всё чисто»
    ровно тогда, когда сломался сбор.
    """
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
            introduced.append({
                "type": node_type, "id": next_id,
                "provenance": ("НЕПРОВЕРЕНО: типа нет в темплейте, "
                               "существование не доказано файлом")})
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
                # Имя `ImageToMask` взято из встроенного набора Comfy и
                # ПРОВЕРЕНО НЕ БЫЛО — ComfyUI в этой среде нет. Поэтому оно
                # попадает в `introduced` с пометкой и выносится в отчёт, а не
                # ставится молча (Ц10).
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

    out["last_node_id"] = next_id - 1
    out["last_link_id"] = next_link - 1

    out["extra"] = dict(out.get("extra", {}))
    out["extra"]["fork"] = {
        "derived_from": UPSTREAM,
        "derived_from_sha256": UPSTREAM_SHA256,
        "removed": len(removed),
        "our_sources": OUR_SOURCES,
        "unrun": ("НЕПРОВЕРЕНО: граф не исполнялся, проверка структурная — "
                  "типы нод, связи, обязательные входы"),
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

    before = len(src.get("nodes", []))
    after = len(graph.get("nodes", []))
    return {
        "outcome": FAIL if problems else PASS,
        "nodes_before": before, "nodes_after": after,
        "custom_left": left, "dangling": dangling, "unfed": starving,
        "unproven_types": unproven,
        "problems": problems,
        "note": (f"нод было {before}, стало {after}; кастомных осталось "
                 f"{len(left)}, оборванных связей {len(dangling)}, входов без "
                 f"питания {len(starving)}, введённых недоказанных типов "
                 f"{len(unproven)}"
                 + (f" ({', '.join(i['type'] for i in unproven)})"
                    if unproven else "") + ". "
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


def write(derived: dict, out_path: str | Path) -> Path:
    """Сохранить произведённый граф. Отдельной функцией: производство и запись
    на диск — разные решения, и склеенные они мешают тестировать первое."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    graph = derived["graph"] if "graph" in derived else derived
    p.write_text(json.dumps(graph, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p
