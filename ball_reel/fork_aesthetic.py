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


def compose(aesthetic, *, with_ban: bool = True) -> dict:
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
    parts = [aesthetic["prompt"].strip(), IDENTITY_CLAUSE]
    if with_ban:
        parts.append(no_brands_clause())
    text = ". ".join(parts)
    return {**tally(1, 0, 0), "prompt": text,
            "id": aesthetic.get("id"), "kind": aesthetic.get("kind"),
            "words": len(text.split()),
            "brand_conflict": brand_conflict(aesthetic),
            "note": (f"эстетика {aesthetic.get('id')}: слов {len(text.split())}, "
                     f"промт владельца дословно + личность"
                     + ("+ запрет надписей" if with_ban else
                        " (ЗАПРЕТ НАДПИСЕЙ ОТКЛЮЧЁН ЯВНО)"))}


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
