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


# ---------------------------------------------------------------------------
# ЭСТЕТИКИ ПО ПОЛУ. Решение владельца 22.08: «верно, эстетики по полу»
# ---------------------------------------------------------------------------
#
# ПОЧЕМУ. ИЗМЕРЕНО 22.08: гардероб переезжает с эстетики ВМЕСТЕ С ПОЛОМ.
# Клиент-мужчина, собранный с женской эстетикой y2k, получил женскую мини-юбку
# и блеск для губ. Ни один прибор этого не увидел — личность на месте (0.3727),
# утечки нет (до демо 0.9258), надписей нет, план цел. Все датчики зелёные, а
# показывать клиенту нельзя. Увидел только глаз.
#
# ПОЧЕМУ ПОЛ ОБЪЯВЛЯЕТСЯ, А НЕ ОПРЕДЕЛЯЕТСЯ. Классификатор пола — это ещё один
# прибор, который надо мерить, сторожить и защищать, и который будет ошибаться
# на живых клиентах. Владелец уже ввёл ровно это правило для драйвингов вручную
# («к мужским липсингам применяй только мужскую личность»), и здесь оно такое
# же: пол называет оператор, а машина СЛЕДИТ, чтобы пары не разъехались.
#
# ПРАВИЛО МАШИННОЕ, А НЕ ЗАПИСКА (Ц7): `pair_check` роняет несовпадение в «не
# годно» до всякой генерации. Записанное словами правило живёт до первой спешки.

#: ВЫБРАНО: две демо-личности проекта, обе в универсальном плане 9:16.
DEMOS = {
    "m": "assets/fork_plan_man_fullbody.png",
    "f": "assets/fork_plan_woman_fullbody.png",
}

GENDERS = tuple(DEMOS)

#: Куда складываются собранные эстетики. Пол — ЧАСТЬ ИМЕНИ ФАЙЛА, а не запись
#: в стороннем реестре: имя едет вместе с файлом и не может от него отстать.
AESTHETIC_DIR = Path("assets") / "aesthetics"


def demo_for(gender: str):
    """Демо-личность по полу. Неизвестный пол — исключение, а не умолчание:
    молчаливое умолчание подсунуло бы женскую эстетику мужчине, то есть ровно
    тот дефект, ради которого правило и заведено."""
    key = str(gender).strip().lower()
    if key not in DEMOS:
        raise KeyError(f"пол {gender!r} не из {GENDERS}")
    return DEMOS[key]


def gender_of(aesthetic) -> str:
    """Пол ШАБЛОНА. Он же пол эстетики: решение владельца 22.08 — «какая
    стилевая рефка по полу такой и пол шаблона».

    Читается ИЗ БАЗЫ, а не выводится из промта машиной: гардероб называет
    владелец, и «мини-юбка» против «tailored bottoms» — его выбор, а не вывод
    регулярного выражения. Отсутствие поля — исключение, а не умолчание:
    молчаливое умолчание подсунуло бы клиенту шаблон чужого пола.
    """
    if isinstance(aesthetic, str):
        aesthetic = load(aesthetic)
    got = (aesthetic or {}).get("demo")
    if str(got).strip().lower() not in DEMOS:
        raise KeyError(f"у эстетики {(aesthetic or {}).get('id')!r} не назван "
                       f"пол (поле «demo» из {GENDERS}), получено {got!r}")
    return str(got).strip().lower()


def aesthetic_file(aesthetic_id: str, gender: str | None = None, *,
                   root=None) -> Path:
    """Путь эстетики. Пол В ИМЕНИ ФАЙЛА, а не только в базе: имя едет вместе с
    файлом и не может от него отстать. По умолчанию берётся ИЗ БАЗЫ.
    """
    g = gender_of(aesthetic_id) if gender is None else str(gender).strip().lower()
    demo_for(g)                             # проверка пола до склейки имени
    base = AESTHETIC_DIR if root is None else Path(root)
    return base / f"{aesthetic_id}_{g}.png"


def pair_check(*, client_gender: str, aesthetic_gender: str) -> dict:
    """Совпадают ли пол клиента и пол эстетики. ГЕЙТ, а не совет.

    Три исхода: `годно` — совпали; `не годно` — разъехались, и это настоящий
    брак с ценой «мужчина в женской юбке»; `не смогли` — пол не назван, и это
    НЕ разрешение продолжать.
    """
    def known(value):
        return str(value).strip().lower() if str(value).strip().lower() in DEMOS else None

    c, a = known(client_gender), known(aesthetic_gender)
    if c is None or a is None:
        return {**tally(0, 0, 1), "client": c, "aesthetic": a,
                "note": (f"пол не назван или не из {GENDERS}: клиент "
                         f"{client_gender!r}, эстетика {aesthetic_gender!r}. "
                         f"Это НЕ разрешение продолжать")}
    if c != a:
        return {**tally(1, 1, 0), "client": c, "aesthetic": a,
                "note": (f"ПОЛ РАЗЪЕХАЛСЯ: клиент {c}, эстетика {a}. ИЗМЕРЕНО, "
                         f"чем это кончается: клиент-мужчина с женской "
                         f"эстетикой получил женскую мини-юбку, и ни один "
                         f"прибор этого не увидел")}
    return {**tally(1, 0, 0), "client": c, "aesthetic": a,
            "note": f"пол совпал: {c}"}


# ---------------------------------------------------------------------------
# СОБРАННАЯ РЕФКА: фото клиента + эстетика -> вход Kling
# ---------------------------------------------------------------------------
#
# РОЛЕВАЯ СТРОКА ЗДЕСЬ ОБРАТНА ТОЙ, ЧТО В СТЕНДЕ, и это не небрежность.
# `fork_e2e.NO_LOOK_TRANSFER_CLAUSE` запрещает брать со второй картинки одежду,
# оправу, причёску и позу — он писался, когда второй картинкой была ЧУЖАЯ
# фотография и всё это было заразой. Эстетика — наш собственный кадр, и всё
# перечисленное в ней и есть шаблон, за который платит клиент.
#
# ЧТО ОСТАЁТСЯ ЗАПРЕЩЁННЫМ НАВСЕГДА: лицо. Это единственная ось, где цена
# ошибки — чужой человек в ролике клиента, и единственная, которую мы умеем
# мерить с ДВУХ сторон сразу: против клиента и против демо.

#: Роли под эстетику. Порядок картинок тот же, что в стенде: первая —
#: личность, вторая — эстетика (ИЗМЕРЕНО: роли держатся, когда названы
#: позицией).
AESTHETIC_ROLE_CLAUSE = (
    "keep the FACE and identity of the person from the FIRST image completely "
    "unchanged — same face, same facial features, same skin tone, same hair "
    "colour, same body; take the wardrobe, styling, accessories, hairstyling, "
    "pose, framing, lens, lighting, colour grade and setting from the SECOND "
    "image"
)

#: Единственный запрет, переживший смену модели. Он ИМЕННО про лицо: всё
#: остальное со второй картинки теперь берётся намеренно.
NEVER_THE_FACE_CLAUSE = (
    "never copy the face, facial features or identity of the person in the "
    "SECOND image; that person is a wardrobe and styling reference only, and "
    "must not appear in the result"
)


def assemble_prompt(*, legacy: bool = False, card=None) -> str:
    """Промт сборки рефки. `legacy=True` даёт СТАРЫЕ строки стенда.

    Старый вариант оставлен нарочно и не как совместимость: он НЕГАТИВНЫЙ
    КОНТРОЛЬ новых строк. Если обе редакции дают один результат, значит роли
    вообще не работают, и «мы поменяли строку» ничего не значит.
    """
    if legacy:
        from .fork_e2e import (NO_LOOK_TRANSFER_CLAUSE,     # noqa: PLC0415
                               ROLE_CLAUSE)

        return f"{ROLE_CLAUSE}. {NO_LOOK_TRANSFER_CLAUSE}. {no_brands_clause()}"
    framing = framing_clause(card)
    tail = f" {framing}." if framing else ""
    return (f"{AESTHETIC_ROLE_CLAUSE}. {NEVER_THE_FACE_CLAUSE}.{tail} "
            f"{no_brands_clause()}")


def leak_verdict(*, made, client, demo, distances=None) -> dict:
    """ДВУСТОРОННИЙ замер: кто на собранной рефке — клиент или демо.

    Одностороннего замера здесь мало, и это главный урок проекта: мера
    похожести умеет сказать «похоже», но не умеет сказать «похоже на ЭТОГО, а
    не на ТОГО». Поэтому меряем оба расстояния и смотрим на РАЗНОСТЬ.

    Четыре исхода сворачиваются в три:
      клиент близко, демо далеко  -> годно
      демо ближе клиента          -> НЕ ГОДНО: личность протекла, это брак
                                     с ценой «чужой человек в ролике»
      оба далеко или оба близко   -> не смогли: прибор не различает, судит глаз
    """
    t0 = time.perf_counter()
    if distances is None:
        from . import fork_identity                       # noqa: PLC0415

        distances = fork_identity.distances
    out = {"seconds": None, "to_client": None, "to_demo": None, "gap": None}
    try:
        c = distances([str(made)], str(client))
        d = distances([str(made)], str(demo))
    except Exception as exc:                              # noqa: BLE001
        return {**tally(0, 0, 1), **out,
                "note": f"прибор упал: {type(exc).__name__}: {exc}"}
    to_client, to_demo = c.get("median"), d.get("median")
    out.update({"to_client": to_client, "to_demo": to_demo,
                "seconds": round(time.perf_counter() - t0, 3)})
    if to_client is None or to_demo is None:
        return {**tally(0, 0, 1), **out,
                "note": (f"одно из расстояний не снято: до клиента "
                         f"{to_client}, до демо {to_demo}")}
    gap = round(to_demo - to_client, 4)
    out["gap"] = gap
    tail = (f"до клиента {to_client}, до демо {to_demo}, разность {gap} "
            f"(планка «тот же человек» {SAME_PERSON_MAX})")
    if to_demo < to_client:
        return {**tally(1, 1, 0), **out,
                "note": f"ЛИЧНОСТЬ ПРОТЕКЛА: демо БЛИЖЕ клиента; {tail}"}
    if to_client <= SAME_PERSON_MAX < to_demo:
        return {**tally(1, 0, 0), **out,
                "note": f"клиент на месте, демо не протекла; {tail}"}
    return {**tally(0, 0, 1), **out,
            "note": (f"прибор не различает: ни одно расстояние не по разные "
                     f"стороны планки; {tail}. СУДИТ ОПЕРАТОР ГЛАЗАМИ")}


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
    """Какие марки названы в промте. СПРАВКА, а не гейт.

    БЫЛО: третий исход «решает владелец» — промты называли «Adidas» и
    «Balenciaga», а запрет проекта запрещал названия марок.
    СТАЛО: владелец решил 22.08 — «бренды пусть остаются, просто добавляем no
    logo во все промты стилей». Конфликта больше нет.

    ПОЧЕМУ ФУНКЦИЯ РАЗЖАЛОВАНА, А НЕ УДАЛЕНА. Список марок остаётся полезным
    составителю. Но исход теперь всегда `годно`: гейт, докладывающий о решённом
    как о нерешённом, — ложная тревога, и она обесценивает настоящие.
    """
    text = str(aesthetic.get("prompt", ""))
    hits = [w for w in ("Adidas", "Balenciaga", "Nike", "Gucci", "Prada",
                        "Zara", "Levi's", "Chanel") if w.lower() in text.lower()]
    if not hits:
        return {**tally(1, 0, 0), "brands": [],
                "note": "марок в промте не названо"}
    return {**tally(1, 0, 0), "brands": hits,
            "note": (f"промт называет марки {hits} — РАЗРЕШЕНО решением "
                     f"владельца 22.08; запрещён только нарисованный знак, и "
                     f"его отсутствие СУДИТ ГЛАЗ: прибора для надписей нет")}


# ---------------------------------------------------------------------------
# Промт
# ---------------------------------------------------------------------------

def no_brands_clause() -> str:
    """Запрет надписей ОДНИМ источником на проект (Е1). Импорт ленивый."""
    from .fork_e2e import NO_BRANDS_CLAUSE                # noqa: PLC0415

    return NO_BRANDS_CLAUSE


def framing_clause(card) -> str:
    """Строка кадрирования из КАРТОЧКИ ДРАЙВИНГА (Е1: она живёт в fork_plan).

    Импорт ленивый: стенд зовёт оба модуля, и связывание на импорте замкнуло бы
    круг.
    """
    from . import fork_plan                               # noqa: PLC0415

    return fork_plan.framing_clause(card)


def compose(aesthetic, *, with_ban: bool = True, cut_body: bool = True,
            card=None) -> dict:
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
    # КАДРИРОВАНИЕ ИДЁТ ПОСЛЕДНИМ И ГОВОРИТ, ЧТО ОНО ГЛАВНЕЕ. Композиция уже
    # описана в промте владельца — у y2k это широкий угол вплотную, — и наша
    # строка обязана его перебить, иначе она просто добавит противоречие.
    # Замыкающая позиция ВЫБРАНА: ведущие токены задают тему, замыкающие —
    # ограничение, а кадрирование здесь именно ограничение.
    framing = framing_clause(card)
    if framing:
        parts.append(framing)
    if with_ban:
        parts.append(no_brands_clause())
    text = ". ".join(parts)
    how = ("промт владельца без антропометрии" if cut_body else
           "промт владельца ДОСЛОВНО (РЕЗ ОТКЛЮЧЁН ЯВНО)")
    return {**tally(1, 0, 0), "prompt": text,
            "id": aesthetic.get("id"), "kind": aesthetic.get("kind"),
            "words": len(text.split()), "cut": cut, "framed": bool(framing),
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
