"""Адаптер стиля: фоторференс -> карточка -> рабочий промт по скелету.

ЗАЧЕМ. Стиль темплейта задаётся картинкой, а видеомодель принимает текст.
Между ними нужен переходник, и он обязан быть ВОСПРОИЗВОДИМЫМ: один и тот же
референс обязан давать один и тот же промт, иначе два прогона одного стиля
нельзя сравнить между собой, и «стиль поехал» неотличимо от «промт поехал».

ЧТО ЗДЕСЬ НЕ ДЕЛАЕТСЯ. Карточка стиля тут не вычисляется. Её считает
`creative_eval.style.style_card` из `greenflac/vertical-creative-eval`
(код MIT, словарь стилей перелицензирован автором под MIT — проверено до
встраивания, Ц5). Повторять её вычисление здесь значило бы завести второй
способ узнать известное (Е1). Модуль принимает ГОТОВУЮ карточку, а чтение
картинки вынесено в точку внедрения `reader` — чтобы тесты не зависели ни от
внешнего пакета, ни от диска (Т4).

ЧЕГО МЫ НЕ КОПИРУЕМ. В корпусе `references/` ТЕКСТА промтов нет — только
скелет (число слов, число клауз, флаги). Это сделано намеренно, из-за
происхождения галереи, и мы это уважаем: текст здесь ПОРОЖДАЕТСЯ из
собственного словаря, а не заимствуется. Из корпуса взята только ФОРМА.

ПРОДУКТОВОЕ ОГРАНИЧЕНИЕ, встроенное в модуль. Промт стиля описывает ТОЛЬКО
внешний вид: палитру, свет, насыщенность, фактуру. Он не описывает человека,
одежду, позу и действие — они приходят из фотографии клиента и из драйвинга.
Промт, который берётся описывать персонажа, конкурирует с фотографией за одну
и ту же величину, и разделить потом «модель не поняла стиль» и «промт спорил
с фото» будет нечем. `SUBJECT_WORDS` сторожит это, и нарушение — исход
`не годно`, а не молчаливая правка.

ТРИ ИСХОДА (Р1). `годно` — промт собран и лёг в измеренную полосу;
`не годно` — собран, но вышел из полосы или задел запретную зону;
`не смогли проверить` — карточка не прочиталась, и это НЕ «стиля нет».
"""

from __future__ import annotations

import re

from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# Форма промта. ВСЕ ЧИСЛА ЗДЕСЬ ИЗМЕРЕНЫ ПО КОРПУСУ, А НЕ ВЫБРАНЫ
# ---------------------------------------------------------------------------

#: ИЗМЕРЕНО: медиана длины промта по 522 карточкам `references/` проекта
#: vertical-creative-eval (команда — подсчёт `skeleton.words` по всем файлам,
#: 22.08.2026). Не «примерно два десятка слов», а снятое число.
WORDS_TARGET = 24

#: ИЗМЕРЕНО: 5-й и 95-й процентили того же распределения (мин 5, макс 189).
#: Полоса, а не точка: попадание в медиану слово в слово не является целью,
#: цель — не выпасть из того, как выглядят рабочие промты этого корпуса.
WORDS_MIN = 9
WORDS_MAX = 67

#: ИЗМЕРЕНО: медиана `skeleton.clauses` по тому же корпусу. Рядом — полоса
#: (5-й и 95-й процентили, мин 1, макс 16) и САМОЕ ЧАСТОЕ значение.
#:
#: ЗАЧЕМ ПОЛОСА, А НЕ ТОЧКА. Первая редакция этого модуля утверждала, что
#: собирает ровно 5 клауз, и тест это опроверг: собирается 7, потому что
#: запятые есть ВНУТРИ палитры и внутри фразы про фактуру, а фраза приходит из
#: словаря и нам не принадлежит. Подгонять текст под свою же цифру значило бы
#: чинить гипотезу; вместо этого снято распределение — и 7 оказалось самым
#: частым значением корпуса (86 карточек из 522).
CLAUSES_TARGET = 5
CLAUSES_MIN = 1
CLAUSES_MAX = 13
CLAUSES_MOST_COMMON = 7

#: ВЫБРАНО: слова, которых в промте стиля быть не должно. Список короткий
#: намеренно — он сторожит грубое нарушение («опиши человека»), а не редактирует
#: язык. Негативный контроль к нему — `test_a_clean_prompt_is_not_accused`.
SUBJECT_WORDS = (
    "person", "man", "woman", "girl", "boy", "face", "hair", "body",
    "wearing", "dress", "shirt", "pose", "posing", "dancing", "smiling",
)

# ---------------------------------------------------------------------------
# Словарь. ВЫБРАНО: собственные формулировки, не заимствованные из галереи
# ---------------------------------------------------------------------------

#: ВЫБРАНО: как тональность карточки звучит в промте. Три значения — ровно те,
#: что порождает `style_card` (light/mid/dark), четвёртого не бывает, и
#: неизвестное значение поэтому не подставляется молча, а роняет сборку.
VALUE_WORDS = {
    "light": "bright high-key lighting",
    "mid": "even balanced lighting",
    "dark": "low-key shadowed lighting",
}

#: ВЫБРАНО: как насыщенность звучит в промте. Значения те же три, что у
#: карточки (muted/moderate/saturated).
SATURATION_WORDS = {
    "muted": "desaturated restrained colour",
    "moderate": "natural colour balance",
    "saturated": "rich saturated colour",
}

#: ВЫБРАНО: замыкающая клауза. Она не описывает сцену — она закрепляет, что
#: речь про ФОТОГРАФИЧЕСКИЙ вид, а не про иллюстрацию: без неё модель вольна
#: прочитать палитру как указание рисовать.
CLOSING = "photographic look"

#: ВЫБРАНО: сколько цветов карточки попадает в промт. `style_card` отдаёт три;
#: берём все три, потому что палитра из одного цвета не отличает стиль от
#: освещения. Мутация в обе стороны — `test_the_palette_width_is_guarded`.
PALETTE_WIDTH = 3


def _words(text: str) -> int:
    """Слов в промте. Считается ОДНИМ способом на весь модуль (Е1)."""
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))


def _clauses(text: str) -> int:
    """Клауз в промте: куски, разделённые запятыми. Пустые не считаются."""
    return len([c for c in text.split(",") if c.strip()])


def subject_leak(text: str) -> list:
    """Какие запретные слова оказались в промте. Пустой список — чисто.

    Ищется по границе слова: `bodysuit` не является нарушением из-за `body`,
    и негативный контроль на это стоит в тестах.
    """
    low = text.lower()
    return [w for w in SUBJECT_WORDS
            if re.search(r"\b" + re.escape(w) + r"\b", low)]


def compose(card: dict) -> dict:
    """Карточка стиля -> промт. Три исхода, а не два.

    `card` — то, что отдаёт `creative_eval.style.style_card`:
    `{"colours": [...], "value_key": ..., "saturation": ..., "texture": ...}`.

    ОТКАЗ ВМЕСТО ДОГАДКИ. Нет поля, пустая палитра, незнакомая тональность —
    исход `не смогли проверить`, и в `note` сказано, чего именно не хватило.
    Подставлять умолчание нельзя: промт, собранный из половины карточки,
    выглядит рабочим и тихо меряет не тот стиль.
    """
    empty = {"outcome": UNMEASURED, "prompt": None, "words": 0, "clauses": 0,
             "leak": [], "card": card}
    if not isinstance(card, dict):
        return {**empty, "note": "карточка не словарь: стиль НЕ ПРОЧИТАН"}

    colours = card.get("colours") or []
    value = card.get("value_key")
    sat = card.get("saturation")
    tex = card.get("texture")

    missing = [name for name, v in (("colours", colours), ("value_key", value),
                                    ("saturation", sat), ("texture", tex))
               if not v]
    if missing:
        return {**empty, "note": ("в карточке нет полей " + ", ".join(missing) +
                                  ": стиль НЕ ПРОЧИТАН, это не «стиля нет»")}
    if value not in VALUE_WORDS:
        return {**empty, "note": (f"тональность {value!r} не из словаря "
                                  f"{sorted(VALUE_WORDS)}: НЕ ПРОЧИТАНА")}
    if sat not in SATURATION_WORDS:
        return {**empty, "note": (f"насыщенность {sat!r} не из словаря "
                                  f"{sorted(SATURATION_WORDS)}: НЕ ПРОЧИТАНА")}

    taken = list(colours)[:PALETTE_WIDTH]
    if len(taken) > 1:
        palette = "a palette of " + ", ".join(taken[:-1]) + " and " + taken[-1]
    else:
        palette = "a palette of " + taken[0]

    parts = [palette, VALUE_WORDS[value], SATURATION_WORDS[sat], tex, CLOSING]
    prompt = ", ".join(parts)

    w, c = _words(prompt), _clauses(prompt)
    leak = subject_leak(prompt)
    if leak:
        outcome = FAIL
        note = (f"промт задел запретную зону {leak}: стиль обязан описывать "
                f"вид, а не персонажа — персонаж приходит из фото и драйвинга")
    elif not (WORDS_MIN <= w <= WORDS_MAX):
        outcome = FAIL
        note = (f"слов {w}, полоса корпуса {WORDS_MIN}..{WORDS_MAX} "
                f"(медиана {WORDS_TARGET})")
    elif not (CLAUSES_MIN <= c <= CLAUSES_MAX):
        outcome = FAIL
        note = (f"клауз {c}, полоса корпуса {CLAUSES_MIN}..{CLAUSES_MAX} "
                f"(медиана {CLAUSES_TARGET})")
    else:
        outcome = PASS
        note = (f"слов {w} при медиане корпуса {WORDS_TARGET} "
                f"(полоса {WORDS_MIN}..{WORDS_MAX}), клауз {c} при медиане "
                f"{CLAUSES_TARGET} и полосе {CLAUSES_MIN}..{CLAUSES_MAX}; "
                f"запретных слов 0")
    return {"outcome": outcome, "prompt": prompt, "words": w, "clauses": c,
            "leak": leak, "card": card, "note": note}


def from_image(path, *, reader=None) -> dict:
    """Картинка -> промт. `reader` — точка внедрения (Т4).

    Умолчание читает `creative_eval.style.style_card`. Импорт отложен внутрь
    функции намеренно: без внешнего пакета модуль обязан импортироваться и
    тесты обязаны идти — иначе прибор нельзя проверить там, где его чинят.
    """
    if reader is None:
        def reader(p):
            # Импортируется ФУНКЦИЯ, а не модуль под именем `style`. Это не
            # косметика: в форке действует решение владельца «`style.py` не
            # используется ни исполнителем, ни прибором», и гейт
            # `test_style_is_not_imported_anywhere_in_the_fork` ловит имя
            # `style` в импортах. Здесь речь про ЧУЖОЙ `creative_eval.style`
            # из другого пакета, но различить их сканер не может и не должен:
            # запрет стоит на имени. Импортируем `style_card` напрямую.
            from creative_eval.style import style_card  # noqa: PLC0415
            return style_card(p)
    try:
        card = reader(str(path))
    except Exception as exc:                     # noqa: BLE001
        return {"outcome": UNMEASURED, "prompt": None, "words": 0,
                "clauses": 0, "leak": [], "card": None,
                "note": (f"карточку прочитать не удалось: "
                         f"{type(exc).__name__}: {exc}")}
    out = compose(card)
    out["source"] = str(path)
    return out


def differ(left: dict, right: dict) -> dict:
    """Дают ли две карточки РАЗНЫЕ промты. Негативный контроль адаптера (И5).

    Прибор, который на любой референс отвечает одним и тем же промтом, выглядит
    работающим ровно до первого сравнения стилей. Три исхода: `годно` — промты
    разошлись; `не годно` — совпали; `не смогли` — хотя бы один не собрался.
    """
    a, b = compose(left), compose(right)
    if a["outcome"] == UNMEASURED or b["outcome"] == UNMEASURED:
        return {"outcome": UNMEASURED, "same": None,
                "note": "хотя бы одна карточка не прочиталась: сравнивать нечего"}
    same = a["prompt"] == b["prompt"]
    return {"outcome": FAIL if same else PASS, "same": same,
            "left": a["prompt"], "right": b["prompt"],
            "note": ("промты СОВПАЛИ на разных карточках: адаптер не различает"
                     if same else "промты разошлись, адаптер различает")}


def report_text(out: dict) -> str:
    """Человеческий отчёт: вердикт, числа, промт. Числа рядом с вердиктом (Р2)."""
    head = f"[{out['outcome']:<18}] промт стиля"
    body = f"  {out['note']}"
    tail = f"  промт: {out['prompt']}" if out.get("prompt") else "  промта нет"
    return "\n".join([head, body, tail])
