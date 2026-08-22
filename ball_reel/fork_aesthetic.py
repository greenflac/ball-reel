"""ЭСТЕТИКА: шаг СОСТАВИТЕЛЯ шаблона. Промт плюс демо-личность -> эстетика.

РЕШЕНИЕ ВЛАДЕЛЬЦА 22.08, дословно: «пишем промт (у нас есть база этих самых
годных промтов, бери любой), в качестве инпут ставим ассет демо-человека в
любой одежде, на выходе получаем стилевой референс с нашей демо личностью,
именно этот стиль будем пробрасывать при генерации собранной рефки для клинг».
И там же: «я считаю надо назвать наш стилевой реф эстетикой».

## ДВА РАЗНЫХ ЧЕЛОВЕКА В ДВУХ РАЗНЫХ ШАГАХ

    СОСТАВИТЕЛЬ, один раз на шаблон:  промт + ДЕМО-личность -> эстетика
    КЛИЕНТ, каждый заказ:             фото клиента + эстетика -> рефка -> Kling

Это меняет природу стилевого референса. Раньше он был чужой картинкой, и
запрет `NO_LOOK_TRANSFER_CLAUSE` велел НЕ брать с него одежду и аксессуары.
Теперь эстетика — наш собственный кадр, и одежду с неё брать НАДО: она и есть
шаблон. А вот лицо с неё брать нельзя НИКОГДА, и это единственная ось, где
цена ошибки — чужой человек в ролике клиента.

## ЧТО ЗДЕСЬ ИЗМЕРИМО, А ЧТО НЕТ

ИЗМЕРИМО и меряется: осталась ли на эстетике ДЕМО-личность (ArcFace против
демо-ассета). Если не осталась — промт перерисовал человека, и «эстетика с
нашей демо личностью» не получилась, как бы красиво ни вышло.

НЕ ИЗМЕРИМО ничем, что у нас есть: попал ли кадр в эстетику, которую владелец
имел в виду. Это судит глаз составителя, и здесь так и написано.

## ПРОТИВОРЕЧИЕ, НЕ РАЗРЕШЁННОЕ МОЛЧА

Промт `y2k` называет «Adidas sneakers», `fisheye` — «Balenciaga trench», а
запрет проекта гласит «no brand names, no logos». Промты владельца НЕ
ПРАВЯТСЯ: это его материал. Запрет добавляется отдельной строкой и виден в
отчёте, а решение, что победит, принимает владелец — см. `brand_conflict`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .fork_identity import FAIL, PASS, SAME_PERSON_MAX, UNMEASURED

#: База эстетик лежит ДАННЫМИ, а не кодом: промты — материал владельца, и
#: правка промта не должна быть правкой модуля.
BASE_PATH = Path(__file__).resolve().parent.parent / "assets" / "fork_aesthetics.json"

#: ГЛАВНАЯ строка этого модуля. Промты владельца описывают ЧУЖУЮ внешность
#: («brunette hair», «Slavic woman with sleek platinum hair»), а личность
#: обязана прийти с картинки. Без явного разрешения конфликта модель выбирает
#: победителя сама и каждый раз по-разному.
#:
#: РАЗДЕЛ ПРОВЕДЁН ПО ИЗМЕРИМОСТИ: лицо и цвет волос — это то, что ArcFace
#: судит, поэтому они идут с картинки. Причёска, одежда, свет, оптика и сцена
#: приборами не судятся и потому отданы промту — там решает глаз составителя.
IDENTITY_CLAUSE = (
    "the person in the frame is the person from the input image: same face, "
    "same facial features, same skin tone and same hair colour; where the "
    "description above names a different appearance, the input image wins on "
    "identity and the description applies only to wardrobe, hairstyling, "
    "setting, lens, lighting, pose and mood"
)

# ---------------------------------------------------------------------------
# АНТРОПОМЕТРИЯ. Решение владельца 22.08: «антропометрию мы всю вырезаем»
# ---------------------------------------------------------------------------
#
# ПОЧЕМУ ЭТОГО НЕ РЕШИТЬ ОДНОЙ СТРОКОЙ В ПРОМТЕ. Строка «личность идёт с
# картинки» уже стояла и ПРОИГРАЛА: ИЗМЕРЕНО на шести эстетиках — те, где
# промт описывает лицо, ушли в среднюю полосу (y2k 0.3966, country 0.4399 при
# планке 0.35), а где не описывает — остались (icecream 0.1310, tomatoes
# 0.1458). Глазом на y2k видно то же: наша блондинка стала шатенкой, потому
# что «brunette hair» весит больше, чем «same hair colour».
#
# ДВА РАЗНЫХ РЕЗА, ПОТОМУ ЧТО АНТРОПОМЕТРИЯ СИДИТ ДВУМЯ РАЗНЫМИ СПОСОБАМИ:
#   оборотом целиком   «she has warm tanned skin with visible freckles»
#   одним словом внутри нужного оборота  «brunette hair styled in a messy bun»
# Резать всё оборотами значило бы унести причёску вместе с цветом волос.

#: ВЫБРАНО (кем: этот модуль; из чего: обороты шести промтов владельца).
#:
#: ОБРАЗЦЫ, А НЕ СЛОВА, и это ИСПРАВЛЕНИЕ ИЗМЕРЕННОЙ ОШИБКИ. Первая редакция
#: резала оборот по голому слову и унесла три невиновных:
#:   «one hand raised near her lips holding a lip gloss applicator» — поза,
#:      сердце эстетики y2k, унесена из-за слова «lips»
#:   «high contrast yet natural skin texture» — качество рендера, не человек
#:   «highly detailed textures of fabric skin and accessories» — то же
#: Голое слово «skin» встречается и в описании кожи, и в требовании к
#: текстуре. Различает их не слово, а оборот вокруг него.
ANTHROPOMETRY_CLAUSES = (
    r"\bhas\b[^,]*\bskin\b",                 # she has warm tanned skin
    r"\b\w+ skin with\b",                     # flawless skin with ...
    r"\bflawless skin\b",
    r"\bfreckles?\b",
    r"\bcomplexion\b",
    r"\b(green|blue|brown|hazel|grey|gray|dark|light|piercing) eyes\b",
    r"\bfacial features?\b",
    r"\bcheekbones?\b",
    r"\bjawline\b",
    r"\bbody type\b",
    r"\bphysique\b",
)

#: ВЫБРАНО: то, что уносится ПООДИНОЧКЕ, оставляя оборот на месте. Здесь
#: живут прилагательные: «extremely beautiful woman seated in a minimal
#: armchair placed in a vast Scottish landscape» — оборот несёт ВСЮ сцену, и
#: унести его целиком значило бы выбросить эстетику вместе с антропометрией.
#: Усилитель уносится вместе с прилагательным, иначе остаётся висеть
#: «extremely person».
ANTHROPOMETRY_WORDS = (
    r"\b(?:extremely|very|incredibly|stunningly|exceptionally)?\s*beautiful\b",
    r"\bsupermodel-level\b", r"\bsupermodel\b", r"\bbeauty\b",
    r"\b(?:extremely|very)?\s*(?:gorgeous|stunning|attractive|pretty)\b",
    r"\bbrunette\b", r"\bblonde?\b", r"\bplatinum\b", r"\bginger\b",
    r"\bauburn\b", r"\bredhead\b", r"\b(?:red|dark|fair)-haired\b",
    r"\btanned\b", r"\b(?:olive|pale|fair)-skinned\b",
    r"\bslavic\b", r"\bnordic\b", r"\bscandinavian\b", r"\basian\b",
    r"\bafrican\b", r"\blatina\b", r"\bcaucasian\b",
    r"\bslim\b", r"\bcurvy\b", r"\bpetite\b", r"\bathletic\b",
)

#: ВЫБРАНО: пол — тоже антропометрия. Клиентом может оказаться кто угодно, а
#: слово «woman» воюет с картинкой ровно так же, как «brunette».
#: Порядок значим: длинные формы раньше коротких, иначе «her» съест «hers».
GENDER_SWAPS = (
    ("women", "people"), ("woman", "person"), ("men", "people"),
    ("man", "person"), ("girl", "person"), ("boy", "person"),
    ("lady", "person"), ("female", "person"), ("male", "person"),
    ("herself", "themselves"), ("himself", "themselves"),
    ("hers", "theirs"), ("her", "their"), ("his", "their"),
    ("she", "they"), ("he", "they"),
)


def _clause_is_anthropometric(clause: str) -> str | None:
    """Образец, по которому оборот признан описанием человека, или None."""
    for pattern in ANTHROPOMETRY_CLAUSES:
        if re.search(pattern, clause, re.IGNORECASE):
            return pattern
    return None


def strip_anthropometry(prompt: str) -> dict:
    """Убрать из промта всё, что описывает ЧЕЛОВЕКА, оставив всё про КАДР.

    Возвращает не только новый текст, но и ЧТО ИМЕННО унесено: рез, который
    нельзя прочитать, неотличим от реза, которого не было.

    Три исхода: `не смогли`, если резать нечего; `годно` в остальных случаях,
    В ТОМ ЧИСЛЕ когда не унесено ничего — это не ошибка, а негативный контроль
    резака на чистом промте.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return {**tally(0, 0, 1), "prompt": None, "dropped": [], "words": [],
                "genders": [], "cut_share": None,
                "note": "промта нет: резать нечего"}

    kept, dropped = [], []
    for clause in prompt.split(","):
        hit = _clause_is_anthropometric(clause)
        if hit:
            dropped.append({"clause": clause.strip(), "pattern": hit})
        else:
            kept.append(clause)
    text = ",".join(kept)

    words = []
    for pattern in ANTHROPOMETRY_WORDS:
        text, n = re.subn(pattern + r"\s*", "", text, flags=re.IGNORECASE)
        if n:
            words.append({"pattern": pattern, "times": n})

    genders = []
    for src, dst in GENDER_SWAPS:
        text, n = re.subn(rf"\b{re.escape(src)}\b", dst, text,
                          flags=re.IGNORECASE)
        if n:
            genders.append({"from": src, "to": dst, "times": n})

    # СЛЕДЫ ОПЕРАЦИИ, а не часть промта. Каждый наблюдался на боевых промтах
    # владельца, и каждый модель читает как значащий: сдвоенный пробел и
    # висящая запятая — как паузу, «an person» и строчная буква после точки —
    # как небрежность, за которой она тянется в остальном кадре.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"(,\s*){2,}", ", ", text).strip().strip(",").strip()
    # Артикль после унесённого прилагательного: «an extremely beautiful woman»
    # -> «an person». Согласуем по первой букве следующего слова.
    text = re.sub(r"\ban\s+(?=[^aeiouAEIOU\s])", "a ", text)
    text = re.sub(r"\ba\s+(?=[aeiouAEIOU])", "an ", text)
    # Заглавная в начале предложения: «14mm lens. person with sleek hair».
    text = re.sub(r"(^|[.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)

    return {**tally(1, 0, 0), "prompt": text,
            "dropped": dropped, "words": words, "genders": genders,
            "cut_share": round(1 - len(text.split()) / len(prompt.split()), 4),
            "note": (f"оборотов унесено {len(dropped)}, слов тела "
                     f"{sum(w['times'] for w in words)}, замен пола "
                     f"{sum(g['times'] for g in genders)}; слов было "
                     f"{len(prompt.split())}, стало {len(text.split())}")}


#: Три исхода вместо двух живут и здесь: «эстетика не собралась» и «эстетика
#: плохая» — разные события, и путать их дорого.
PLAN_NOTE = ("план 9:16 на эстетике НЕ ТРЕБУЕТСЯ: план навязывается на "
             "собранной рефке клиента, а эстетика несёт вид, а не кадр")


def tally(checked: int, violations: int, unmeasured: int) -> dict:
    """Числа рядом с вердиктом (Р2)."""
    if checked == 0:
        outcome = UNMEASURED
    elif violations:
        outcome = FAIL
    elif unmeasured:
        outcome = UNMEASURED
    else:
        outcome = PASS
    return {"outcome": outcome, "checked": checked,
            "violations": violations, "unmeasured": unmeasured}


# ---------------------------------------------------------------------------
# База
# ---------------------------------------------------------------------------

def load_base(path=None) -> dict:
    """База эстетик с диска. Отсутствие файла — исключение, а не пустая база:
    молча пустая база выглядит как «эстетик нет», а это разные вещи."""
    p = Path(BASE_PATH if path is None else path)
    if not p.is_file():
        raise FileNotFoundError(f"базы эстетик нет: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    got = doc.get("aesthetics")
    if not isinstance(got, list) or not got:
        raise ValueError(f"в базе {p} нет ни одной эстетики")
    return doc


def ids(path=None) -> list:
    return [a["id"] for a in load_base(path)["aesthetics"]]


def load(aesthetic_id: str, path=None) -> dict:
    """Одна эстетика по имени. Неизвестное имя — исключение со списком того,
    что есть: молчаливое умолчание подсунуло бы не тот шаблон."""
    for a in load_base(path)["aesthetics"]:
        if a["id"] == aesthetic_id:
            return a
    raise KeyError(f"эстетики {aesthetic_id!r} нет; есть: "
                   f"{', '.join(ids(path))}")


def brand_conflict(aesthetic: dict) -> dict:
    """Называет ли промт владельца бренд. НЕ ПРАВИТ и НЕ РОНЯЕТ — сообщает.

    Три исхода честные: `не годно` было бы неправдой (промт рабочий), `годно`
    — тоже (запрет проекта нарушен). Поэтому «не смогли решить»: решает
    владелец, а прибор его об этом извещает.
    """
    text = str(aesthetic.get("prompt", ""))
    hits = [w for w in ("Adidas", "Balenciaga", "Nike", "Gucci", "Prada",
                        "Zara", "Levi's", "Chanel") if w.lower() in text.lower()]
    if not hits:
        return {**tally(1, 0, 0), "brands": [],
                "note": "брендов в промте не названо"}
    return {**tally(0, 0, 1), "brands": hits,
            "note": (f"промт называет бренды {hits}, а запрет проекта гласит "
                     f"«no brand names, no logos». Промт владельца НЕ ПРАВЛЕН; "
                     f"решает владелец")}


# ---------------------------------------------------------------------------
# Промт
# ---------------------------------------------------------------------------

def no_brands_clause() -> str:
    """Запрет надписей ОДНИМ источником на проект (Е1). Импорт ленивый."""
    from .fork_e2e import NO_BRANDS_CLAUSE                # noqa: PLC0415

    return NO_BRANDS_CLAUSE


def compose(aesthetic, *, with_ban: bool = True, cut_body: bool = True) -> dict:
    """Промт эстетики: материал владельца + разрешение конфликта личности.

    Порядок ВЫБРАН и не случаен: промт владельца идёт ПЕРВЫМ и целиком, потому
    что ведущие токены весят больше; наши строки идут после как оговорки к
    нему. Обратный порядок превратил бы служебную приписку в тему кадра.
    """
    if isinstance(aesthetic, str):
        aesthetic = load(aesthetic)
    if not isinstance(aesthetic, dict) or not aesthetic.get("prompt"):
        return {**tally(0, 0, 1), "prompt": None,
                "note": "эстетика без промта: собирать нечего"}
    # РЕЗ ИДЁТ ПЕРВЫМ, до всех наших приписок. Иначе резак прошёлся бы и по
    # IDENTITY_CLAUSE, где слова «same face» и «same skin tone» стоят намеренно
    # и обязаны выжить: это единственное место, которому антропометрия нужна.
    own = aesthetic["prompt"].strip()
    cut = strip_anthropometry(own) if cut_body else None
    body = cut["prompt"] if cut and cut["outcome"] == PASS else own

    parts = [body, IDENTITY_CLAUSE]
    if with_ban:
        parts.append(no_brands_clause())
    text = ". ".join(parts)
    how = ("промт владельца без антропометрии" if cut_body else
           "промт владельца ДОСЛОВНО (РЕЗ ОТКЛЮЧЁН ЯВНО)")
    return {**tally(1, 0, 0), "prompt": text,
            "id": aesthetic.get("id"), "kind": aesthetic.get("kind"),
            "words": len(text.split()), "cut": cut,
            "brand_conflict": brand_conflict(aesthetic),
            "note": (f"эстетика {aesthetic.get('id')}: слов {len(text.split())}, "
                     f"{how} + личность"
                     + ("" if with_ban else " (ЗАПРЕТ НАДПИСЕЙ ОТКЛЮЧЁН ЯВНО)")
                     + (f"; {cut['note']}" if cut else ""))}


# ---------------------------------------------------------------------------
# Приёмка эстетики
# ---------------------------------------------------------------------------

def accept(*, made, demo, distances=None) -> dict:
    """Осталась ли на эстетике ДЕМО-личность. Единственная измеримая ось.

    Лестница та же, что на всём проекте (Е1): 0.0652 тот же человек, планка
    0.35, 0.7137 другой, 1.0217 чужой. Средняя полоса здесь значит ровно то
    же, что везде: прибор не судья, судит глаз.

    ЧЕГО ЭТА ФУНКЦИЯ НЕ ДЕЛАЕТ: не судит, красиво ли и попало ли в задуманную
    эстетику. Прибора для этого нет, и выдумывать его вместо честного «судит
    составитель» было бы худшим из трёх исходов.
    """
    t0 = time.perf_counter()
    if distances is None:
        from . import fork_identity                       # noqa: PLC0415

        distances = fork_identity.distances
    try:
        d = distances([str(made)], str(demo))
    except Exception as exc:                              # noqa: BLE001
        return {**tally(0, 0, 1), "median": None,
                "seconds": round(time.perf_counter() - t0, 3),
                "note": f"прибор личности упал: {type(exc).__name__}: {exc}"}

    med = d.get("median")
    tail = (f"лестница: 0.0652 тот же, {SAME_PERSON_MAX} планка, 0.7137 другой, "
            f"1.0217 чужой")
    if d.get("outcome") == UNMEASURED or med is None:
        return {**tally(0, 0, 1), "median": med,
                "seconds": round(time.perf_counter() - t0, 3),
                "note": f"личность НЕ ИЗМЕРЕНА: {str(d.get('note'))[:200]}"}
    if med <= SAME_PERSON_MAX:
        return {**tally(1, 0, 0), "median": med,
                "seconds": round(time.perf_counter() - t0, 3),
                "note": (f"демо-личность на месте: медиана {med} при планке "
                         f"{SAME_PERSON_MAX} ({tail}). {PLAN_NOTE}. "
                         f"ПОПАДАНИЕ В ЭСТЕТИКУ СУДИТ СОСТАВИТЕЛЬ ГЛАЗАМИ — "
                         f"прибора для этого нет")}
    if med < 0.7137:
        return {**tally(0, 0, 1), "median": med,
                "seconds": round(time.perf_counter() - t0, 3),
                "note": (f"медиана {med} между планкой {SAME_PERSON_MAX} и "
                         f"ступенью «другой человек» 0.7137: лицо изменено или "
                         f"закрыто, ArcFace здесь НЕ СУДЬЯ, судит составитель "
                         f"({tail})")}
    return {**tally(1, 1, 0), "median": med,
            "seconds": round(time.perf_counter() - t0, 3),
            "note": (f"медиана {med} выше ступени «другой человек» 0.7137: "
                     f"промт ПЕРЕРИСОВАЛ человека, это не наша демо-личность "
                     f"({tail})")}


def render(report: dict) -> str:
    """Печать для человека."""
    return (f"ЭСТЕТИКА: {report['outcome']}  (проверено {report['checked']}, "
            f"нарушений {report['violations']}, не смогли "
            f"{report['unmeasured']})\n  {report.get('note', '')}")
