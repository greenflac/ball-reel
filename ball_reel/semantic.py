"""Тот ли ОБЪЕКТ, та ли ОДЕЖДА, та ли СЦЕНА — соответствует ли кадр заказу.

Вторая половина той же дыры, что закрывает `action.py`. Живой прогон дал кадр
с лучшим за день дрейфом лица (0.21), на котором человек в БАЛЬНОМ ПЛАТЬЕ сидел
в позе с фотографии; все двенадцать осей `produce.CHECK_ORDER` его пропустили.
`action` ловит неверное ДВИЖЕНИЕ и в своём докстринге прямо признаётся, что не
видит ни объекта, ни одежды: «человек в бальном платье и человек в спортивной
форме, совершающие одно движение, для неё неразличимы». Этот модуль отвечает
ровно на тот вопрос, которого не задаёт ни одна геометрическая ось: то, что в
кадре, — это то, что заказывали?

Кадр, ради которого написан модуль, лежит в репозитории: `startopt/face_first_0.png`
(тот самый дрейф 0.2124 из `startopt/result.json`). Ось даёт на нём -0.1201 при
пороге 0.012, то есть «НЕ ТА ОДЕЖДА».

## Прибор и лицензия

CLIP, локально, без сети и без платных вызовов:

    laion/CLIP-ViT-B-32-laion2B-s34B-b79K      license: mit

Лицензия проверена вызовом, а не по памяти:

    python3 -c "from huggingface_hub import HfApi; \
      print(HfApi().model_info('laion/CLIP-ViT-B-32-laion2B-s34B-b79K').card_data['license'])"
    mit

Веса `openai/clip-vit-*` брать НЕЛЬЗЯ, и причина не в тексте лицензии, а в её
ОТСУТСТВИИ: в карточке `openai/clip-vit-base-patch32` поля `license` нет вовсе
(тот же вызов возвращает `None`, тег `license:*` отсутствует). Продукт
коммерческий, а веса без объявленной лицензии — это не «разрешено по
умолчанию», это неизвестность, которую нельзя подписать. У обеих LAION-моделей
(B/32 и H/14) стоит явный `mit`.

## Чем НЕЛЬЗЯ пользоваться этим модулем: ось личности на CLIP

Соблазн очевиден и измерен: на наших кропах лиц CLIP «разделяет» тех же и
разных людей (медиана 0.068 против 0.540). Верить этому числу нельзя. Два
наших человека отличаются волосами, возрастом и тоном кожи, а кропы — резкостью
(портрет 148 px против мелкого лица в динамике), поэтому CLIP мог развести
ВНЕШНОСТЬ И КАЧЕСТВО, а не личность. Трудный случай — два ПОХОЖИХ человека — у
нас настоящий: дрейф шлюза 0.345 при баре 0.35, то есть генератор делает
похожего. Пар «похожие, но разные» в данных нет, значит это НЕ ИЗМЕРЕНО.
Идентичность меряет ArcFace (`identity_arcface.py`), у которого на тех же
кропах медиана 0.358 против 0.991 при непересекающихся диапазонах.

## Что сравнивается и почему не абсолютный скор

Абсолютный CLIP-скор мерит не то, ради чего его берут, и вот замер. Один текст
(«a photo of a person wearing athletic sportswear, a sports bra and leggings»)
на кадрах с ВЕРНОЙ одеждой даёт медианы 0.308 (съёмка), 0.266 (генерация в
топе) и 0.182 (генерация в худи), на кадрах в ПЛАТЬЕ — 0.105. Порог вроде бы
есть; его нет:

    разброс ВНУТРИ «верного» класса   0.127 .. 0.329   ширина 0.20
    зазор между классами              0.114 .. 0.127   ширина 0.013
    покадровый шум абсолютного скора  sd 0.020 на 71 кадре одного видео

Зазор УЖЕ собственного шума прибора. Абсолютное число разделяет прежде всего
источник кадра — съёмку от генерации (0.308 против 0.182 при ОДИНАКОВО верной
одежде), — и одежда живёт внутри этой разницы, а не поверх неё.

Шкала появляется в сравнении: заказ против ВЗАИМОИСКЛЮЧАЮЩИХ альтернатив
по тому же слоту, в одной и той же рамке предложения (минимальная пара — все
слова, кроме одежды, совпадают, поэтому разность меряет одежду, а не кадрирование):

    a photo of a person IN ATHLETIC SPORTSWEAR on a large fitness ball in a room
    a photo of a person IN A PARTY DRESS       on a large fitness ball in a room

Обе стороны — НАБОРЫ формулировок, а не по одной фразе. Это не украшение:
с единственной формулировкой «in athletic sportswear» все 71 настоящий
driving-кадр проваливались, потому что их обходило «in jeans and a shirt»
(0.258 против 0.233), — а «in a sports bra and leggings» описывает ровно то,
что на них надето, и даёт 0.297. Одна фраза меряет удачность фразы, набор
меряет слот.

    маржа = max(скор по формулировкам заказа) - max(скор по альтернативам)

## Числа контрольных случаев (одежда)

Разметка сделана ГЛАЗАМИ по каждому кадру, а не тем же CLIP: иначе прибор
проверялся бы собственными показаниями. `python3 -m ball_reel.semantic --controls`

    набор                        что на кадрах        n  медиана   вердикт
    kit/driving                  съёмка, спортивное  71  +0.0330  matches
    wiretest/frames_00_loop      ГЕНЕРАЦИЯ, худи     21  +0.1080  matches
    startopt plain_1..role_2     ГЕНЕРАЦИЯ, топ       5  +0.0195  matches
    startopt face_first_*,plain_0 ГЕНЕРАЦИЯ, ПЛАТЬЕ   4  -0.1134  mismatched
    kit/face.jpg                 портрет              1  -0.1152  mismatched
    kit/conditions               скелет ControlNet   71  -0.0216  mismatched

Покадровый разброс внутри наборов (те же прогоны): съёмка sd 0.0123 при
[-0.0004, +0.0506], генерация в худи sd 0.0220 при [+0.0727, +0.1309],
генерация в топе sd 0.0100 при [+0.0046, +0.0304], платье sd 0.0135 при
[-0.1201, -0.0897].

ШУМ измерен там, где его видно: 71 кадр ОДНОГО видео с ОДНИМ набором текстов
дают sd 0.0123 покадрово; медиана по окну в 16 кадров (продуктовый клип) гуляет
от +0.0181 до +0.0446 при sd 0.0065. Разрыв между худшим верным окном (+0.0181)
и ближайшим неверным случаем (-0.0216, скелет) — 0.040, то есть вшестеро больше
шума окна. Порог существует. Для сравнения: у абсолютного скора тот же зазор
0.013 при шуме 0.020, то есть УЖЕ шума — там порога нет.

## Три исхода

`matches` / `mismatched` / `not_measurable`. Третий — не «плохо», а честный
отказ, и причин у него пять:

1. нет весов CLIP — ~600 МБ, и на машине демо их может не быть;
2. в заказе нет ни одного утверждения, либо у утверждения нет альтернатив:
   сравнивать не с чем, шкалы нет;
3. кадров не передано вовсе или ни один не открылся;
4. текстовая башня схлопнулась (см. ниже) — прибор сломан;
5. маржа лежит ВНУТРИ шума прибора.

Пятая — не увёртка: 0.012 это измеренный покадровый sd, и называть решением
разницу, которую прибор не отличает от собственного дрожания, значит
подделывать измерение.

Отдельно сторожится ловушка, на которой этот модуль сам чуть не построил
ложное измерение. `CLIPProcessor.from_pretrained` с весами в кэше, но БЕЗ
файлов токенизатора собирает токенизатор-пустышку: любой текст кодируется
одними и теми же id, все текстовые векторы совпадают до 1.0000, все маржи
выходят нулевыми, и ось радостно печатает «соответствует» на чём угодно.
Прогон, где это случилось, выглядел совершенно нормально — все 19 текстов
получили одинаковый скор 0.0519 до четвёртого знака. Поэтому перед каждым
вердиктом проверяется, что тексты вообще различимы (`TEXT_COLLAPSE_MAX`), и
при схлопывании выносится `not_measurable`, а не вердикт.

## Чего ось НЕ видит

* ЛИЧНОСТЬ. См. выше: на CLIP её строить нельзя, и этот модуль её не строит.
* ДВИЖЕНИЕ. Ось покадровая и не знает порядка кадров: перетасованный клип для
  неё тот же самый. Движение меряет `action.py`.
* ОБЪЕКТ И СЦЕНУ — ПОКА НЕТ ПОРОГА, и это измеренный отрицательный результат,
  а не недоделка. Утверждения собраны (`OBJECT_UNCALIBRATED`, `SCENE_UNCALIBRATED`),
  но в `DEFAULT_CLAIMS` не входят:
    - объект: на настоящих driving-кадрах, где мяч ЕСТЬ, медиана -0.0087
      (доля выигрышей 0.42), на генерации с мячом -0.0206, а на кадрах вовсе
      без мяча (портрет -0.0532, скелет -0.0622) — то же самое с точностью до
      шума. Распределения перекрываются, порога не существует; включить эту
      ось значило бы браковать верные кадры;
    - сцена: все наборы, включая портрет (+0.0237) и скелет (+0.0202), лежат
      ПОЛОЖИТЕЛЬНО. Отрицательного контроля нет вовсе — в репозитории нет ни
      одного кадра на улице или на пляже, — и число, у которого не наблюдалось
      ни одного «не того» случая, порогом называть нельзя.
  Это ровно тот случай, что был с метрикой примет: на семи парах распределения
  перекрылись, и отрицательный результат записан как результат.
* ЦВЕТ И ПОСТОЯНСТВО одежды между кадрами — это `garment.py`. Здесь вопрос
  «та ли одежда заказана», там «одна ли она на всём клипе».
* КАЧЕСТВО. Ось не отличает красивый кадр от уродливого, если одежда та.

## На ОДНОМ кадре ось часто молчит, и это правильно

Те же девять кадров `startopt`, судимые ПОШТУЧНО (`--frames`):

    4 кадра в платье        -0.1201 -0.1155 -0.1114 -0.0897   все mismatched
    3 кадра в топе          +0.0304 +0.0220 +0.0195           все matches
    2 кадра в топе          +0.0107 +0.0046                   not_measurable

Ни одного неверного вердикта в обе стороны, но на двух верных кадрах из пяти
одиночная маржа попадает В ПОЛОСУ ШУМА, и ось честно отвечает «не смогла».
Продуктовый клип — 16 кадров, и медиана по окну гуляет впятеро меньше
одиночного кадра (sd 0.0065 против 0.0123), поэтому на клипе это редкость.
Ось, поставленная на ОДИН старт-кадр, будет говорить «не смогла» заметно чаще,
чем на клипе, — это её свойство, а не поломка.

## Цена

Замерено: `python3 -m ball_reel.semantic --cost` (4 ядра CPU, load average 7.5
— на машине работали соседние процессы, поэтому числа скорее пессимистичны):

    загрузка модели      4.0 с    один раз на процесс
    кодирование текстов  0.1 с    один раз на процесс (15 фраз)
    16 кадров            0.6-1.7 с   0.04-0.11 с на кадр
    ИТОГО первый вызов   4.7-5.8 с,  последующие ~1 с на клип

Под тяжёлой нагрузкой (load average 17) те же 16 кадров стоили 8.2 с — цена
зависит от того, кто ещё живёт на машине, и меньше секунды на клип она не
бывает. Для сравнения: `motion_amount` и `pose_wander` считаются в numpy за
доли секунды, а один видео-вызов шлюза стоит ~18 картинок. Отсюда место оси:
САМЫМ ПОСЛЕДНИМ в `CHECK_ORDER` — дешёвое раньше дорогого, и до неё доходят
только клипы, прошедшие всё остальное.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Веса. LAION, license MIT (проверено `HfApi().model_info`). Не менять на
#: `openai/clip-vit-*`: там поля `license` нет вовсе, а продукт коммерческий.
MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

#: Файлы, без которых вердикт невозможен. Токенизатор в списке НЕ для полноты:
#: без него `CLIPProcessor` собирается молча и кодирует любой текст одинаково
#: (см. докстринг). Веса без токенизатора — это прибор со снятой шкалой.
REQUIRED_FILES = ("config.json", "preprocessor_config.json",
                  "model.safetensors", "tokenizer.json")

#: Порог маржи. ИЗМЕРЕН как покадровый шум: 71 кадр одного и того же видео с
#: одним и тем же набором текстов дают sd 0.0123 (медиана окна в 16 кадров —
#: sd 0.0065). Внутри этой полосы прибор не отличает заказ от альтернативы, и
#: вердикт не выносится вовсе.
#: Полоса лежит В РАЗРЫВЕ между измеренными случаями, и разрыв тоже измерен:
#:      худший ВЕРНЫЙ случай   +0.0181  (худшее окно из 56 на driving)
#:      ближайший НЕВЕРНЫЙ     -0.0216  (скелет ControlNet: одежды нет вовсе)
#:      кадр в платье          -0.0897 .. -0.1201
#: Ноль оказался почти серединой разрыва, и это не подгонка: ноль — это точка,
#: где CLIP МЕНЯЕТ ОТВЕТ на вопрос «во что человек одет».
MARGIN_MIN = 0.012

#: Сколько альтернатив нужно, чтобы у утверждения была шкала. Одна — минимум;
#: ноль означает «сравнивать не с чем», и это `not_measurable`, а не «прошло».
MIN_DECOYS = 1

#: Выше этого косинуса два РАЗНЫХ текста считаются схлопнувшимися, то есть
#: текстовая башня сломана. ИЗМЕРЕНО: со сломанным токенизатором любые два
#: текста дают 1.0000 (совпадение до седьмого знака); на исправном токенизаторе
#: самая близкая пара из 15 текстов утверждения об одежде даёт 0.9757
#: («in athletic sportswear» против «in gym clothes» — синонимы в одной и той
#: же рамке предложения, ближе них ничего не будет).
TEXT_COLLAPSE_MAX = 0.99

#: Рамка предложения, общая для заказа и альтернатив. Общая часть и делает
#: сравнение минимальной парой: разность меряет слот, а не кадрирование.
_FRAME = "a photo of a person {} on a large fitness ball in a room"


@dataclass(frozen=True)
class Claim:
    """Одно утверждение о кадре: слот, формулировки заказа и альтернативы.

    `expected` и `decoys` — НАБОРЫ полных предложений, а не по одному. Набор
    меряет слот, одна фраза меряет удачность фразы (измерено: см. докстринг
    модуля). `check` — имя, под которым провал попадёт в отчёт гейта;
    статистика «что ломается первым» строится по именам, а не по текстам.
    """

    name: str
    check: str
    expected: tuple[str, ...]
    decoys: tuple[str, ...]

    def usable(self) -> bool:
        return bool(self.expected) and len(self.decoys) >= MIN_DECOYS


#: ОТКАЛИБРОВАННОЕ утверждение — одно. Формулировки заказа покрывают то, как
#: спортивная одежда реально выглядит на наших кадрах (топ и легинсы у донора,
#: худи и шорты у генерации); альтернативы — то, во что генератор одевал
#: человека вместо заказанного, плюс соседние «не спортивные» слоты.
CLOTHING = Claim(
    name="одежда",
    check="semantic_clothing",
    expected=tuple(_FRAME.format(x) for x in (
        "in athletic sportswear",
        "in a sports bra and leggings",
        "in gym clothes",
        "in a hoodie and shorts",
        "in a tank top and shorts")),
    decoys=tuple(_FRAME.format(x) for x in (
        "in a party dress",
        "in a long evening gown",
        "in a frilly tulle dress",
        "in a tutu",
        "in a skirt",
        "in jeans and a shirt",
        "in a business suit",
        "in a swimsuit",
        "in pyjamas",
        "with no clothes on")))

#: НЕ ОТКАЛИБРОВАНО: на настоящих кадрах, где мяч есть, медиана -0.0087, на
#: кадрах без мяча -0.0532 и -0.0622 — распределения перекрываются, порога не
#: существует. Оставлено в модуле как записанный отрицательный результат и как
#: заготовка на случай, когда появятся кадры БЕЗ заказанного объекта.
OBJECT_UNCALIBRATED = Claim(
    name="объект (НЕ ОТКАЛИБРОВАНО)",
    check="semantic_object",
    expected=("a photo of a person exercising on a large inflatable fitness ball",
              "a photo of a person balancing on a big gym ball",
              "a photo of a person with a large exercise ball"),
    decoys=("a photo of a person sitting on a chair",
            "a photo of a person with a dog",
            "a photo of a person standing on the floor with no equipment",
            "a photo of a person on a bicycle",
            "a photo of a person lifting dumbbells",
            "a photo of a person on a treadmill",
            "a photo of a person on a yoga mat"))

#: НЕ ОТКАЛИБРОВАНО по другой причине: отрицательного контроля нет вовсе.
#: Просмотрены 172 кадра из survey, chain_frames, ref_frames, bench_seedream,
#: veoprobe, standin_out, frametest, recheck и newref — кадра на улице среди
#: них нет ни одного (лучший «уличный» скор за вычетом «в помещении» равен
#: +0.0004, то есть ноль). Все имеющиеся наборы, включая портрет и рисунок
#: скелета, дают ПЛЮС (+0.0202..+0.1049).
#: В репозитории нет ни одного кадра на улице или на пляже,
#: то есть ось ни разу не наблюдалась в состоянии «не та сцена».
SCENE_UNCALIBRATED = Claim(
    name="сцена (НЕ ОТКАЛИБРОВАНО)",
    check="semantic_scene",
    expected=("a photo of a person exercising indoors in a room",
              "a photo taken inside a home studio",
              "a photo taken indoors"),
    decoys=("a photo of a person on a sandy beach",
            "a photo of a person outdoors in a park",
            "a photo of a person on a city street",
            "a photo taken outdoors in nature",
            "a photo of a person in a swimming pool"))

#: Что реально идёт в гейт. Одно утверждение, у которого есть измеренный порог.
DEFAULT_CLAIMS: tuple[Claim, ...] = (CLOTHING,)


# --------------------------------------------------------------------------
# ЧИСТАЯ АРИФМЕТИКА. Ни весов, ни картинок — как в `action.compare_trajectories`
# и `pose`: порог, который нельзя проверить без 600 МБ весов, — это порог,
# который не проверяется вовсе.
# --------------------------------------------------------------------------

def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def judge_margins(margins: dict, *, margin_min: float = MARGIN_MIN) -> dict:
    """Свести покадровые маржи в вердикт по клипу.

    `margins` — {имя утверждения: список покадровых марж}. Пустой список
    означает «по этому утверждению не измерено ничего».

    Возвращает словарь с `verdict` ("matches" / "mismatched" /
    "not_measurable"), `on_brief` (True / False / **None**), `check` — именем
    первого несработавшего утверждения, `claims` — разбором по утверждениям и
    `note`.

    `on_brief` при "not_measurable" намеренно None, а не False: булев флаг
    рядом с трёхзначным вердиктом схлопывает исходы обратно в два, и читатель,
    взявший флаг вместо вердикта, получил бы «кадр не тот» там, где мы просто
    не смогли измерить. Это разные решения: первое бракует клип, второе требует
    разобраться, почему нечем судить. Проект ловил эту ошибку в позе, в
    жидкости, в предполёте и в оси движения.
    """
    per: dict = {}
    for name, vals in margins.items():
        vals = [float(v) for v in vals]
        if not vals:
            per[name] = {"verdict": "not_measurable", "median": None,
                         "frames": 0, "share": None}
            continue
        med = _median(vals)
        share = sum(1 for v in vals if v > 0) / len(vals)
        if med >= margin_min:
            v = "matches"
        elif med <= -margin_min:
            v = "mismatched"
        else:
            v = "not_measurable"
        per[name] = {"verdict": v, "median": round(med, 4), "frames": len(vals),
                     "share": round(share, 3),
                     "worst": round(min(vals), 4), "best": round(max(vals), 4)}

    if not per:
        return {"verdict": "not_measurable", "on_brief": None, "check": None,
                "claims": {}, "margin": None, "note": (
            "СЕМАНТИКА НЕ ИЗМЕРЕНА: не задано ни одного утверждения о заказе. "
            "Сравнивать кадр не с чем — это не «кадр верный».")}

    # ПОРЯДОК ВАЖЕН: сначала ищем провал, и только потом отсутствие измерения.
    # Обратный порядок закрывал бы кадр в платье исходом «не смогли», если бы
    # рядом оказалось неизмеримое утверждение, — то есть терял бы находку.
    bad = [(n, d) for n, d in per.items() if d["verdict"] == "mismatched"]
    if bad:
        n, d = bad[0]
        return {"verdict": "mismatched", "on_brief": False,
                "check": check_of(n), "claims": per,
                "margin": d["median"], "note": (
            f"КАДР НЕ ПО ЗАКАЗУ ({n}): маржа {d['median']:+.4f} при пороге "
            f"{margin_min} — описание заказа проигрывает альтернативе на "
            f"{d['frames']} кадр(ах), заказ выигрывает лишь на "
            f"{d['share']:.0%} из них. Это тот самый случай, что прошёл весь "
            f"гейт: человек в бальном платье в верной позе.")}
    unk = [(n, d) for n, d in per.items() if d["verdict"] == "not_measurable"]
    if unk:
        n, d = unk[0]
        med = d["median"]
        return {"verdict": "not_measurable", "on_brief": None,
                "check": None, "claims": per,
                "margin": med, "note": (
            f"СЕМАНТИКА НЕ ИЗМЕРЕНА ({n}): "
            + ("измеримых кадров нет вовсе."
               if med is None else
               f"маржа {med:+.4f} лежит внутри шума прибора ({margin_min}) — "
               f"прибор не отличает заказ от альтернативы, и называть это "
               f"вердиктом значит подделывать измерение."))}
    worst = min(per.items(), key=lambda kv: kv[1]["median"])
    return {"verdict": "matches", "on_brief": True, "check": None,
            "claims": per, "margin": worst[1]["median"], "note": (
        f"КАДР ПО ЗАКАЗУ: слабейшее утверждение «{worst[0]}» даёт "
        f"{worst[1]['median']:+.4f} при пороге {margin_min}; заказ выигрывает "
        f"на {worst[1]['share']:.0%} кадров.")}


def check_of(name: str) -> str:
    """Имя проверки для отчёта гейта по имени утверждения.

    Отдельно от текста причины намеренно, как в `produce.Attempt.check`: по
    именам строится статистика «что ломается первым», а текст причины меняется
    при первой же правке формулировки, и статистика по нему разъезжается молча.
    """
    for c in (CLOTHING, OBJECT_UNCALIBRATED, SCENE_UNCALIBRATED):
        if c.name == name:
            return c.check
    return "semantic"


def texts_collapsed(vectors, *, limit: float = TEXT_COLLAPSE_MAX) -> bool:
    """Схлопнулась ли текстовая башня: разные тексты дали один и тот же вектор.

    `vectors` — список НОРМИРОВАННЫХ векторов (списки чисел). Проверка стоит
    отдельной чистой функцией потому, что ловит она не гипотезу, а случившееся:
    без файлов токенизатора `CLIPProcessor` собирается молча, кодирует любой
    текст одинаково, и все маржи становятся нулевыми. Ось при этом печатает
    ровный правдоподобный отчёт.
    """
    n = len(vectors)
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if sum(a * b for a, b in zip(vectors[i], vectors[j])) > limit:
                return True
    return False


def margins_from_scores(expected: list, decoys: list) -> list:
    """Покадровая маржа из двух таблиц скоров: max(заказ) - max(альтернативы).

    `expected` и `decoys` — списки по кадрам, в каждом элементе список скоров
    по формулировкам. Вынесено отдельно, чтобы правило «максимум по набору»
    проверялось без весов.
    """
    if len(expected) != len(decoys):
        raise ValueError(
            f"margins_from_scores: {len(expected)} кадр(ов) со скорами заказа "
            f"против {len(decoys)} со скорами альтернатив — таблицы должны "
            f"описывать одни и те же кадры.")
    out = []
    for e, d in zip(expected, decoys):
        if not e or not d:
            raise ValueError(
                "margins_from_scores: у кадра пустая сторона сравнения. "
                "Утверждение без альтернатив не имеет шкалы — такой случай "
                "закрывается исходом not_measurable, а не нулевой маржой.")
        out.append(max(e) - max(d))
    return out


# --------------------------------------------------------------------------
# Половина с весами. Отсутствие весов — состояние, а не ошибка.
# --------------------------------------------------------------------------

def missing_files(model_id: str = MODEL_ID) -> list:
    """Каких файлов модели нет в локальном кэше. В сеть не ходит."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return list(REQUIRED_FILES)
    out = []
    for f in REQUIRED_FILES:
        try:
            got = try_to_load_from_cache(model_id, f)
        except Exception:
            got = None
        if not isinstance(got, str):
            out.append(f)
    return out


def available(model_id: str = MODEL_ID) -> bool:
    """Есть ли чем судить. Пустой ответ — не отказ, а исход `not_measurable`."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return not missing_files(model_id)


def why_unavailable(model_id: str = MODEL_ID) -> str:
    """Внятная причина вместо ImportError посреди прогона."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return ("нет пакета transformers — семантику судить нечем. "
                "pip install transformers torch")
    miss = missing_files(model_id)
    if not miss:
        return ""
    return (f"нет файлов модели {model_id}: {', '.join(miss)}. Скачать (~600 МБ):\n"
            f"  python3 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_id}')\"\n"
            f"Отдельно про токенизатор: одних весов НЕ ХВАТИТ. Без "
            f"tokenizer.json процессор собирается молча и кодирует любой текст "
            f"одинаково — ось начинает печатать «соответствует» на чём угодно.")


_LOADED: dict = {}


def _clip(model_id: str = MODEL_ID):
    """Модель и процессор, один раз на процесс: 600 МБ и разбор весов."""
    if model_id not in _LOADED:
        from transformers import CLIPModel, CLIPProcessor

        miss = missing_files(model_id)
        if miss:
            raise FileNotFoundError(why_unavailable(model_id))
        model = CLIPModel.from_pretrained(model_id).eval()
        _LOADED[model_id] = (model, CLIPProcessor.from_pretrained(model_id))
    return _LOADED[model_id]


def _encode_texts(texts: list, model_id: str = MODEL_ID):
    """Нормированные текстовые векторы. `pooler_output`, а не проекция поверх.

    В установленной здесь версии transformers `get_text_features` и
    `get_image_features` возвращают НЕ тензор, а объект с полями
    `last_hidden_state` и `pooler_output`; готовый 512-мерный эмбеддинг лежит
    в `pooler_output`, и применять `text_projection`/`visual_projection` поверх
    него не надо — размерности не сойдутся. Проверено вызовом.
    """
    import torch

    model, proc = _clip(model_id)
    with torch.no_grad():
        out = model.get_text_features(
            **proc(text=texts, return_tensors="pt", padding=True))
    v = out.pooler_output if hasattr(out, "pooler_output") else out
    return torch.nn.functional.normalize(v, dim=-1)


def _encode_images(paths: list, model_id: str = MODEL_ID, batch: int = 8):
    """Нормированные векторы кадров и список кадров, которые не открылись.

    Кадры открываются ДО загрузки модели намеренно: «кадр не читается» — это
    исход, для которого 600 МБ весов не нужны, и платить за них, чтобы узнать,
    что файла нет, незачем.
    """
    import torch
    from PIL import Image

    ok, bad, vecs = [], [], []
    images = []
    for p in paths:
        try:
            images.append((p, Image.open(p).convert("RGB")))
        except Exception as exc:
            bad.append((str(p), f"{type(exc).__name__}: {exc}"))
    if not images:
        return None, ok, bad
    model, proc = _clip(model_id)
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        with torch.no_grad():
            out = model.get_image_features(
                **proc(images=[im for _, im in chunk], return_tensors="pt"))
        v = out.pooler_output if hasattr(out, "pooler_output") else out
        vecs.append(torch.nn.functional.normalize(v, dim=-1))
        ok.extend(str(p) for p, _ in chunk)
    if not vecs:
        return None, ok, bad
    return torch.cat(vecs), ok, bad


def semantic_match(frames, claims=DEFAULT_CLAIMS, *, model_id: str = MODEL_ID,
                   margin_min: float = MARGIN_MIN) -> dict:
    """Соответствуют ли кадры заказу. Отсутствие весов даёт «не смогли».

    `frames` — пути к кадрам клипа (или к одному старт-кадру). `claims` —
    утверждения о заказе; по умолчанию одно откалиброванное, про одежду.
    """
    frames = [str(f) for f in frames]
    empty = {"verdict": "not_measurable", "on_brief": None, "check": None,
             "claims": {}, "margin": None, "frames": len(frames),
             "unreadable": []}
    if not frames:
        return {**empty, "note": (
            "СЕМАНТИКА НЕ ИЗМЕРЕНА: кадров не передано вовсе.")}
    usable = [c for c in claims if c.usable()]
    if not usable:
        return {**empty, "note": (
            "СЕМАНТИКА НЕ ИЗМЕРЕНА: ни у одного утверждения нет альтернатив "
            f"(нужно хотя бы {MIN_DECOYS}). Абсолютный скор CLIP шкалы не "
            "имеет: измерено, что зазор между верными и неверными кадрами по "
            "нему 0.013 при собственном шуме 0.020, то есть уже шума. "
            "Сравнивать не с чем.")}
    if not available(model_id):
        return {**empty, "note": (
            "СЕМАНТИКА НЕ ИЗМЕРЕНА: " + why_unavailable(model_id))}

    vecs, ok, bad = _encode_images(frames, model_id)
    if vecs is None:
        return {**empty, "unreadable": bad, "note": (
            f"СЕМАНТИКА НЕ ИЗМЕРЕНА: ни один из {len(frames)} кадров не "
            f"открылся ({bad[:2]}).")}

    margins: dict = {}
    for c in usable:
        te = _encode_texts(list(c.expected), model_id)
        td = _encode_texts(list(c.decoys), model_id)
        rows = [r.tolist() for r in te] + [r.tolist() for r in td]
        if texts_collapsed(rows):
            return {**empty, "unreadable": bad, "note": (
                "СЕМАНТИКА НЕ ИЗМЕРЕНА: текстовая башня схлопнулась — разные "
                "тексты дали один и тот же вектор. Почти наверняка нет файлов "
                "токенизатора рядом с весами; без них любой текст кодируется "
                "одинаково, все маржи выходят нулевыми, и ось печатает "
                "«соответствует» на чём угодно.\n" + why_unavailable(model_id))}
        se = (vecs @ te.T).tolist()
        sd = (vecs @ td.T).tolist()
        margins[c.name] = margins_from_scores(se, sd)

    res = judge_margins(margins, margin_min=margin_min)
    return {**res, "frames": len(ok), "unreadable": bad, "measured": ok}


def render(res: dict) -> str:
    """Одна читаемая строка отчёта."""
    v = {"matches": "КАДР ПО ЗАКАЗУ", "mismatched": "КАДР НЕ ПО ЗАКАЗУ",
         "not_measurable": "НЕ СМОГЛИ ИЗМЕРИТЬ"}[res["verdict"]]
    m = "-" if res.get("margin") is None else f"{res['margin']:+.4f}"
    return (f"{v}  маржа={m}  кадров={res.get('frames', 0)}"
            f"  проверка={res.get('check') or '-'}\n  {res.get('note', '')}")


# --------------------------------------------------------------------------
# Контрольные случаи на НАСТОЯЩИХ кадрах репозитория.
# Разметка сделана глазами, не CLIP: прибор не проверяется своими показаниями.
# --------------------------------------------------------------------------

#: (имя, пути относительно корня репозитория, ожидаемый вердикт).
#: `startopt/*` размечены поштучно; четыре кадра в платье — это тот самый брак,
#: что прошёл весь гейт (`startopt/result.json`: face_first_0 дрейф 0.2124).
CONTROL_SETS = (
    ("kit/driving — съёмка, спортивная одежда", "kit/driving/*.jpg", "matches"),
    ("wiretest — ГЕНЕРАЦИЯ, худи и шорты",
     "wiretest/frames_00_loop/*.png", "matches"),
    ("startopt — ГЕНЕРАЦИЯ, спортивный топ", (
        "experiments/startopt/plain_1.png", "experiments/startopt/plain_2.png", "experiments/startopt/role_0.png",
        "experiments/startopt/role_1.png", "experiments/startopt/role_2.png"), "matches"),
    ("startopt — ГЕНЕРАЦИЯ, БАЛЬНОЕ ПЛАТЬЕ", (
        "experiments/startopt/face_first_0.png", "experiments/startopt/face_first_1.png",
        "experiments/startopt/face_first_2.png", "experiments/startopt/plain_0.png"), "mismatched"),
    ("kit/face.jpg — портрет, одежды в кадре нет", ("kit/face.jpg",),
     "mismatched"),
    ("kit/conditions — скелет ControlNet", "kit/conditions/*.png",
     "mismatched"),
)


def _resolve(spec, root: Path) -> list:
    if isinstance(spec, str):
        return sorted(str(p) for p in root.glob(spec))
    return [str(root / s) for s in spec]


def controls(root: str | Path = ".", *, claim: Claim = CLOTHING) -> list:
    """Прогнать контрольные случаи. Возвращает список (имя, ожидание, результат)."""
    root = Path(root)
    out = []
    for name, spec, want in CONTROL_SETS:
        frames = [f for f in _resolve(spec, root) if Path(f).exists()]
        out.append((name, want, semantic_match(frames, (claim,))))
    return out


def cost(frames: list, *, model_id: str = MODEL_ID) -> dict:
    """Сколько стоит ось в секундах. Печатается, а не подразумевается."""
    import time

    frames = [str(f) for f in frames]
    if not available(model_id):
        return {"measured": False, "note": why_unavailable(model_id)}
    t0 = time.time()
    _clip(model_id)
    t_load = time.time() - t0
    t0 = time.time()
    _encode_texts(list(CLOTHING.expected) + list(CLOTHING.decoys), model_id)
    t_text = time.time() - t0
    t0 = time.time()
    _encode_images(frames, model_id)
    t_img = time.time() - t0
    return {"measured": True, "frames": len(frames),
            "load_s": round(t_load, 2), "text_s": round(t_text, 2),
            "images_s": round(t_img, 2),
            "per_frame_s": round(t_img / max(1, len(frames)), 3),
            "total_s": round(t_load + t_text + t_img, 2)}


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.semantic",
        description="тот ли объект, та ли одежда, та ли сцена")
    ap.add_argument("--controls", action="store_true",
                    help="контрольные случаи на настоящих кадрах репозитория")
    ap.add_argument("--cost", action="store_true",
                    help="замерить цену оси на 16 кадрах")
    ap.add_argument("--root", default=".", help="корень репозитория")
    ap.add_argument("--frames", nargs="*", default=None, help="кадры клипа")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="прогнать и неоткалиброванные утверждения (объект, сцена)")
    ap.add_argument("--attribution", metavar="КАТАЛОГ",
                    help="чья одежда на кадрах: заказа, фото личности или "
                         "driving-видео. Отвечает на вопрос, который «та ли "
                         "одежда» не задаёт вовсе")
    args = ap.parse_args(argv)

    if args.attribution:
        import glob as _glob
        import os as _os

        d = args.attribution
        frames = (sorted(_glob.glob(_os.path.join(d, "*.png")))
                  + sorted(_glob.glob(_os.path.join(d, "*.jpg"))))
        if not frames:
            print(f"в {d} нет ни png, ни jpg — приписывать одежду нечему")
            return 1
        got = attribution(frames, ordered=DEMO_ORDERED,
                          identity=DEMO_IDENTITY, driving=DEMO_DRIVING)
        print(f"кадров: {len(frames)}")
        for k, v in sorted(got["scores"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:16s} {v:+.4f}")
        print(f"\nисточник: {got['source'] or 'НЕ РАЗЛИЧИЛИ'}")
        print(f"  {got['note']}")
        # Код возврата различает три исхода, а не два: 0 — одежда пришла из
        # заказа, 1 — протекла из референса, 2 — не смогли определить.
        return 0 if got["source"] == SOURCE_ORDER else (
            2 if got["source"] is None else 1)

    if args.controls:
        claims = ((CLOTHING, OBJECT_UNCALIBRATED, SCENE_UNCALIBRATED)
                  if args.uncalibrated else (CLOTHING,))
        # Код возврата ненулевой при ЛЮБОМ расхождении — иначе таблицу,
        # которая разъехалась с кодом, увидит только тот, кто её прочтёт.
        # Для неоткалиброванных утверждений ожиданий нет, и они в счёт не идут.
        off = 0
        for c in claims:
            print(f"\n=== утверждение: {c.name}")
            print(f"{'случай':<42}{'маржа':>9}{'кадров':>8}  ждали / вышло")
            print("-" * 82)
            for name, want, r in controls(args.root, claim=c):
                m = "-" if r.get("margin") is None else f"{r['margin']:+.4f}"
                bad = r["verdict"] != want and c is CLOTHING
                off += bad
                print(f"{name:<42}{m:>9}{r.get('frames', 0):>8}  "
                      f"{want} / {r['verdict']}{'   <-- РАСХОЖДЕНИЕ' if bad else ''}")
        if off:
            print(f"\nРАСХОЖДЕНИЙ: {off}. Числа в докстринге получены этой же "
                  f"командой — значит разошлись документ и код.")
        return 1 if off else 0

    if args.cost:
        frames = args.frames or sorted(
            str(p) for p in Path(args.root, "kit/driving").glob("*.jpg"))[:16]
        c = cost(frames)
        if not c["measured"]:
            print("ЦЕНА НЕ ЗАМЕРЕНА:", c["note"])
            return 0
        print(f"кадров {c['frames']}: загрузка модели {c['load_s']} с, тексты "
              f"{c['text_s']} с, картинки {c['images_s']} с "
              f"({c['per_frame_s']} с/кадр), всего {c['total_s']} с")
        return 0

    if not args.frames:
        ap.error("нужны --frames (или --controls / --cost)")
    print(render(semantic_match(args.frames)))
    return 0




# --------------------------------------------------------------------------
# ОТКУДА ПРИШЛА ОДЕЖДА. Не «та ли она», а «из какого источника».

#: Три источника, между которыми различает `attribution`. Формулировки — в той
#: же рамке предложения, что и всё остальное: сравнение обязано быть
#: минимальной парой, иначе меряется кадрирование, а не слот.
#:
#: ЗАЧЕМ ОТДЕЛЬНО ОТ `semantic_match`. Тот отвечает «одежда та / не та» одним
#: числом с порогом. Этого мало, когда надо ПОКАЗАТЬ, что конвейер работает:
#: «не та» не говорит, протекла ли она из фото личности или из driving-видео, а
#: это разные поломки с разной починкой. Первая означает, что канал личности
#: тащит с собой лишнее; вторая — что ControlNet тащит не только позу.
#:
#: ПОЧЕМУ ЗДЕСЬ ПОЧТИ НЕ НУЖЕН ПОРОГ. Вопрос поставлен как выбор из трёх, а не
#: как «выше ли числа X»: побеждает ближайший источник. Порог нужен ровно для
#: одного — объявить ничью, когда два источника разошлись на пустяк.
#:
#: Для ничьей ЗАИМСТВОВАН `MARGIN_MIN`, и это надо назвать вслух: он измерен
#: для ДРУГОЙ величины (маржа «заказ против альтернатив» внутри одного слота),
#: а здесь сравниваются лучшие скоры трёх РАЗНЫХ наборов. Величины
#: однопорядковые, обе — разности косинусов в одной рамке предложения, поэтому
#: заимствование разумно; но замером для ЭТОГО применения оно не является, и
#: выдавать его за таковой нельзя.
#:
#: На практике запас велик и вопрос пока академический. ИЗМЕРЕНО на кадрах, где
#: ответ известен заранее:
#:     driving-кадры (12 шт) -> «driving-видео», отрыв 0.0911 (7.6 шума)
#:     kit/face.jpg          -> «фото личности», отрыв 0.1651 (13.8 шума)
#: Третьего положительного контроля — кадра, где одежда пришла ИЗ ЗАКАЗА, — в
#: репозитории нет и быть не может до первого прогона. Значит про третий исход
#: сказать нечего, и это НЕПРОВЕРЕНО, а не «работает».
#: Формулировки трёх источников ДЛЯ НАШЕГО ДЕМО. Живут рядом с функцией, а не
#: в рунбуке: описание, разошедшееся с командой, врёт молча. Подобраны так,
#: чтобы заказ был ТРЕТЬИМ значением по каждой оси — цвету, крою и месту, —
#: иначе исход «спортивная одежда» не отличить от утечки из driving-видео.
DEMO_ORDERED = ("in a bright red tank top and black shorts",
                "in a red sleeveless top and dark shorts")
DEMO_IDENTITY = ("in a frilly pink tulle dress", "in a tutu",
                 "in a party dress")
DEMO_DRIVING = ("in a navy sports bra and leggings",
                "in dark athletic sportswear")

SOURCE_ORDER = "заказ"
SOURCE_IDENTITY = "фото личности"
SOURCE_DRIVING = "driving-видео"


def attribution(frames, *, ordered: tuple, identity: tuple, driving: tuple,
                model_id: str = MODEL_ID, margin_min: float = MARGIN_MIN) -> dict:
    """Чья одежда на кадрах: заказа, фото личности или driving-видео.

    `ordered`, `identity`, `driving` — наборы формулировок БЕЗ рамки: рамку
    функция накладывает сама, чтобы три набора были заведомо сравнимы.

    Смысл этой функции — сделать прогон ПОКАЗАТЕЛЬНЫМ. Пока заказ описывает то
    же, что видно на driving-кадре, опыт неинтерпретируем: спортивная одежда на
    выходе может быть и исполнением заказа, и утечкой из видео, и различить их
    нечем. Стоит заказать ТРЕТЬЕ — и три исхода становятся различимы:

        победил заказ            -> конвейер работает как обещано;
        победило фото личности   -> канал личности тащит не только лицо;
        победило driving-видео   -> ControlNet тащит не только позу.

    Три исхода и здесь: победитель / ничья в пределах шума / не смогли.
    """
    import numpy as np

    empty = {"source": None, "margin": None, "scores": {}}
    frames = [str(f) for f in frames]
    if not frames:
        return {**empty, "note": "кадров нет: приписывать одежду нечему"}
    sets = {SOURCE_ORDER: tuple(ordered), SOURCE_IDENTITY: tuple(identity),
            SOURCE_DRIVING: tuple(driving)}
    blank = [k for k, v in sets.items() if not v]
    if blank:
        return {**empty, "note": (
            f"нечем описать источник(и): {', '.join(blank)}. Выбор из двух "
            f"вариантов не отличает утечку от исполнения заказа")}
    if not available(model_id):
        return {**empty, "note": "НЕ ИЗМЕРЕНО: " + why_unavailable(model_id)}

    vecs, ok, bad = _encode_images(frames, model_id)
    if vecs is None:
        return {**empty, "note": (
            f"НЕ ИЗМЕРЕНО: ни один из {len(frames)} кадров не открылся "
            f"({bad[:2]})")}

    texts = {k: [_FRAME.format(x) for x in v] for k, v in sets.items()}
    enc = {k: _encode_texts(v, model_id) for k, v in texts.items()}
    rows = [r.tolist() for v in enc.values() for r in v]
    if texts_collapsed(rows):
        return {**empty, "note": (
            "НЕ ИЗМЕРЕНО: текстовая башня схлопнулась — разные формулировки "
            "дали один вектор. Прибор слеп, числа ничего не значат.\n"
            + why_unavailable(model_id))}

    # По каждому источнику берётся ЛУЧШАЯ из его формулировок на кадр, затем
    # медиана по кадрам. Лучшая — потому что набор описывает ОДИН слот разными
    # словами, и слабая формулировка не должна топить источник; медиана — потому
    # что один неудачный кадр не должен решать за клип.
    per = {k: _median([max(row) for row in (vecs @ v.T).tolist()])
           for k, v in enc.items()}
    ranked = sorted(per.items(), key=lambda kv: -kv[1])
    (top, best), (_, second) = ranked[0], ranked[1]
    margin = round(best - second, 4)
    scores = {k: round(v, 4) for k, v in per.items()}
    if margin < margin_min:
        return {"source": None, "margin": margin, "scores": scores,
                "note": (f"источники разошлись на {margin:.4f} при шуме "
                         f"{margin_min}: РАЗЛИЧИТЬ НЕ СМОГЛИ. Это не «заказ "
                         f"исполнен», а «мы не знаем, чья это одежда»")}
    return {"source": top, "margin": margin, "scores": scores,
            "note": (f"одежда ближе всего к источнику «{top}», отрыв "
                     f"{margin:.4f} при шуме {margin_min}. Порядок: "
                     + ", ".join(f"{k} {v:+.4f}" for k, v in ranked))}

if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
