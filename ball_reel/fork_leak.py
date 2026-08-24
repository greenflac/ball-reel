"""Поток B форка: ОСЬ ПРОТЕЧКИ — что изменилось ВНЕ маски.

ОТКУДА БЕРЁТСЯ ЭТА ОСЬ. Маска очерчивает персонажа: внутри модель имеет право
на всё, снаружи обязана вернуть драйвинг. Значит мерить надо ГРАНИЦУ, а не
общее впечатление от ролика. Метрика «похоже ли на драйвинг» по всему кадру
ответит «похоже» и тогда, когда модель стёрла мяч: мяч занимает проценты
площади.

---

ЧТО СНАРУЖИ МАСКИ НА САМОМ ДЕЛЕ ПРОИСХОДИТ. Прочитано в исходнике, не
припомнено: `comfy_extras/nodes_wan.py`, класс `WanAnimateToVideo`.

    background_video   → серый холст  torch.ones((length,h,w,3)) * 0.5
    холст              → vae.encode   → concat_latent_image
    character_mask     →              → concat_mask
    возвращаемый латент =             torch.zeros(...)      ← ПУСТОЙ ШУМ
    noise_mask                        НЕ СТАВИТСЯ ВОВСЕ

Единственный `noise_mask` во всём файле принадлежит другой ноде
(`Wan22ImageToVideoLatent`). Слова `composite` нет ни в ноде, ни в официальном
графе. То есть у ноды нет даже механизма маскированного блендинга.

**Весь кадр сэмплируется из шума и декодируется через VAE.** Драйвинг входит
обусловливанием в стиле инпейнта, а не подложкой, куда врисовывают персонажа.
Вне маски идёт РЕКОНСТРУКЦИЯ под сильным обусловливанием, и тождества там нет
и быть не может. Единственное, что действительно не проходит через модель, —
звук.

---

~~ПРЕЖНЯЯ ПЛАНКА: КОДЕК, 0.004513 (среднее) и 0.071895 (локально), ffmpeg crf 18.~~
СНЯТА И УДАЛЕНА ИЗ КОДА вместе с функцией `noise_floor`, которая её мерила.
Чем закрыта: §1a хэндофа. Планка меряла перекодирование кадров, то есть
последний и самый дешёвый шаг тракта, — при том что весь кадр перед этим
прошёл VAE и шесть шагов сэмплера. Заниженная планка, лежащая в коде, будет
использована: расхождение 0.03 против 0.004513 читалось бы как «протечка 6.6x»,
хотя столько может стоить одна реконструкция без всякой протечки. Пометкой
такой дефект не лечится — только удалением, поэтому здесь остался лишь след.

ВЗАМЕН ДВА ЧИСЛА, И ПОРЯДОК ВАЖЕН:

  (а) `vae_roundtrip_floor` — `vae.decode(vae.encode(драйвинг))` без всякого
      сэмплирования. Неустранимый пол ПРЕДСТАВЛЕНИЯ: ниже него не опустится
      ничто, что вообще прошло через латент;
  (б) `full_loop_floor` — полный круг при `character_mask = 0` всюду. Модели
      нечего генерировать, она обусловлена драйвингом целиком, остаток есть
      НАСТОЯЩИЙ пол, включая сэмплер. Это и есть планка оси протечки.

Их РАЗНОСТЬ (`sampler_contribution`) — вклад сэмплера. Если полный пол намного
выше VAE-пола, реконструкция вне маски нестабильна сама по себе, и знать это
надо ДО того, как мерить нашу протечку.

Судить протечку разрешено ТОЛЬКО по (б). Это не вкус, а устройство кода:
`verdict` принимает планку с клеймом происхождения (`kind`) и отказывается
судить по любой другой, включая (а). Планку без клейма подать нельзя — именно
потому, что однажды подали кодековую.

---

ТРИ ИСХОДА (Р1), И ТРЕТИЙ ЗДЕСЬ НЕ ФОРМАЛЬНОСТЬ. «Планка не замерена» — сегодня
ОБЫЧНОЕ состояние оси, и сворачивать его в «годно» значит объявить
герметичность на основании отсутствия проверки.

НЕПРОВЕРЕНО (Ц4), наверх:
  * `full_loop_floor` НЕ ЗАМЕРЕН и в этой среде замерен быть не может: нужен
    исполняемый Comfy с 14B-весами и карта. Число появится только на A16.
    До него ось на живом выходе возвращает «не смогли измерить»;
  * настоящей протечки прибор не видел — он видел нарисованное пятно, и на нём
    краснеет.

ЗАМЕРЕНО (И4): VAE-круг снят — среднее **0.007308**, локально **0.338698**, на
пяти кадрах драйвинга при 240×424 (половина целевой геометрии: на 480×848
прогон убит OOM). Команда и все условия — в `VAE_ROUNDTRIP_NOTE` ниже.

И сразу главное следствие: снятая кодековая планка 0.004513 лежит **ниже**
неустранимого пола представления 0.007308. Она была занижена не на проценты —
она была меньше того, что тракт тратит ещё до всякого сэмплирования.
"""

from __future__ import annotations

from pathlib import Path

#: Три исхода. Берутся у потока A (Е1): одни и те же три слова в двух копиях
#: разъедутся и дадут два несравнимых отчёта.
from .fork_identity import FAIL, PASS, UNMEASURED

#: Клейма происхождения планки. Планка — это НЕ просто число: важно, каким
#: прогоном оно получено. Кодековая планка была числом без клейма, и её
#: подставили туда, где нужен был совсем другой прогон.
FLOOR_VAE = "vae_roundtrip"          # (а) пол представления
FLOOR_FULL_LOOP = "full_loop_zero_mask"  # (б) настоящий пол, судит протечку

#: Единственное клеймо, по которому разрешено выносить PASS/FAIL. Список, а не
#: одна строка, — чтобы расширение шло правкой ЗДЕСЬ, а не в теле `verdict`.
FLOOR_KINDS_THAT_MAY_JUDGE = (FLOOR_FULL_LOOP,)

#: Во сколько раз замеренная протечка должна превысить пол, чтобы её можно было
#: назвать протечкой. ВЫБРАНО 2.0, и вот из чего.
#:
#: Пол — это реконструкция того же драйвинга тем же трактом, и она шумит от
#: прогона к прогону (сэмплер стохастичен даже при cfg 1). Множитель 1.0
#: объявлял бы протечкой разницу двух прогонов одного и того же. Двойка —
#: наименьшее число, при котором «вдвое хуже, чем реконструкция без задачи»
#: уже не объяснить дрожанием тракта.
#:
#: ВЫБРАНО, не ИЗМЕРЕНО, и калибруется первым же живым полом: когда (б) будет
#: снят несколько раз, разброс самого пола и даст честный множитель.
LEAK_OVER_FLOOR = 2.0

#: Минимальная доля кадра ВНЕ маски, при которой измерение вообще осмысленно.
#: ВЫБРАНО: если маска накрыла 98% кадра, «вне маски» — это рамка в пару
#: пикселей, и среднее по ней пляшет сильнее измеряемого. Тогда честный ответ
#: «не смогли», а не «протечки нет».
MIN_OUTSIDE_SHARE = 0.02

#: Сколько кадров минимум нужно, чтобы полу можно было верить. ВЫБРАНО: одна
#: пара кадров даёт одну точку, а реконструкция стоит неодинаково на разных
#: кадрах — на статике дешевле, на движении дороже. И у Wan-VAE сжатие по
#: времени 4x, то есть на одном кадре временная часть кодека вообще не работает.
MIN_FLOOR_FRAMES = 3

#: Во сколько раз полный пол должен превысить VAE-пол, чтобы сказать вслух:
#: реконструкция вне маски нестабильна сама по себе. ВЫБРАНО 2.0 — то же
#: «вдвое», что и у порога протечки, и по той же причине: разница меньше двух
#: раз объяснима дрожанием одного прогона. Отдельная константа, а не то же
#: число повторно (Е1: это два РАЗНЫХ решения, которые вправе разъехаться).
SAMPLER_FLOOR_ALARM = 2.0

#: Веса VAE. Проверено командой (Ц10), не припомнено:
#:   curl -s https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
#: даёт vae/config.json и vae/diffusion_pytorch_model.safetensors, лицензия
#: apache-2.0 (Ц5 — коммерческого запрета нет). 507 591 892 байта (fp32; в
#: стеке стоит 0.24 ГБ — тот же VAE, но в половинной точности).
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

#: ИЗМЕРЕНО (И4). Пол (а), снятый прогоном; команда и условия — ниже, в
#: `VAE_ROUNDTRIP_NOTE`. Лежит здесь как СПРАВКА и как негативный контроль к
#: прибору, а НЕ как планка протечки: судить по нему `verdict` откажется по
#: клейму, потому что вклад сэмплера сюда не входит.
#:
#: ЧТО ВИДНО ИЗ ЭТИХ ДВУХ ЧИСЕЛ СРАЗУ. Среднее 0.007308 — это УЖЕ в 1.6 раза
#: больше прежней кодековой планки 0.004513, и это только представление, без
#: единого шага сэмплера. То есть снятая планка была занижена не на проценты:
#: она вся целиком лежала ниже неустранимого пола.
#:
#: И локальное 0.338698 против среднего 0.007308 — в 46 раз. Реконструкция
#: врёт не равномерно, а на контурах и мелкой фактуре. Отсюда и требование
#: судить максимум ОТДЕЛЬНОЙ планкой: по средней он объявил бы протечкой
#: каждый край предмета.
VAE_ROUNDTRIP_MEASURED: dict | None = {
    "kind": FLOOR_VAE,
    "floor": 0.007308,        # худшее среднее по кадру
    "floor_max": 0.338698,    # худший пиксель
    "floor_mean_of_frames": 0.006926,
    "frames": 5,
    "geometry": "240x424",
    "latent": (1, 16, 2, 53, 30),
    "dtype": "float32", "device": "cpu", "tiled_decode": True,
}
VAE_ROUNDTRIP_NOTE = (
    "ИЗМЕРЕНО 2026-08-17. Пять кадров demo/kit/driving (720x1278 → 240x424, "
    "LANCZOS), веса Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae (apache-2.0, "
    "507 591 892 Б), diffusers 0.39.0, torch 2.13.0, fp32, CPU, декод тайлами. "
    "Команда: python3 -u vaerun.py 5 240 424 tile — то есть "
    "fl.vae_roundtrip_floor(кадры, vae_dir=…, tiled_decode=True). "
    "Время: encode 109.3 с, decode 397.0 с на загруженной машине. "
    "Кадры до и после лежат рядом и ОТКРЫТЫ глазами (П3): src0.png / rt0.png — "
    "картинка та же, мягче зерно на стене и на волосах, геометрия и лицо на "
    "месте; то есть прибор мерит смягчение, а не сломанный декод. "
    "ЧЕГО В ЭТОМ ЧИСЛЕ НЕТ: целевой геометрии 480x848 — прогон на ней "
    "ОТРИЦАТЕЛЬНЫЙ (И6), процесс убит OOM-killer'ом на 5.6 ГБ RSS при 16 ГБ "
    "общей памяти и восьми чужих прогонах рядом (dmesg: oom-kill … "
    "task=python3, anon-rss:5599100kB). Половинная геометрия — вынужденная "
    "замена, и число с неё занижено ровно настолько, насколько мельче деталь.")


def _as_array(image):
    """Кадр -> массив float64 в 0..1, RGB. Одно место, где решается это."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float64")
    elif hasattr(image, "convert"):
        arr = np.asarray(image.convert("RGB"), dtype="float64")
    else:
        arr = np.asarray(image, dtype="float64")
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
    return arr / 255.0 if arr.max() > 1.0 else arr


def outside_divergence(output, driving, mask) -> dict:
    """Насколько выход разошёлся с драйвингом ТАМ, ГДЕ ЕГО НЕ ТРОГАЛИ.

    Возвращает и среднее, и максимум, и это не избыточность. Среднее ловит
    общий сдвиг (модель перекрасила фон), максимум — локальное пятно (модель
    стёрла мяч). Одно без другого слепо ровно к тому, что ловит второе: пятно
    в 0.3% площади не сдвинет среднее, а равномерный сдвиг не даст максимума.
    """
    import numpy as np

    a = _as_array(output)
    b = _as_array(driving)
    m = np.asarray(mask, dtype=bool)

    if a.shape != b.shape:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": 0.0,
                "note": f"кадры разного размера: {a.shape} и {b.shape}"}
    if m.shape != a.shape[:2]:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": 0.0,
                "note": f"маска {m.shape} не по кадру {a.shape[:2]}"}

    outside = ~m
    share = round(float(outside.mean()), 4)
    if share < MIN_OUTSIDE_SHARE:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": share,
                "note": (f"вне маски осталось {share:.1%} кадра (нужно хотя бы "
                         f"{MIN_OUTSIDE_SHARE:.0%}): мерить не на чем. Это НЕ "
                         f"«протечки нет».")}

    diff = np.abs(a - b).mean(axis=2)
    vals = diff[outside]
    return {"outcome": None, "outside_share": share,
            "mean": round(float(vals.mean()), 6),
            "max": round(float(vals.max()), 6),
            "pixels": int(vals.size),
            "note": (f"вне маски ({share:.1%} кадра, {vals.size} px): "
                     f"среднее {vals.mean():.6f}, максимум {vals.max():.6f}")}


# ─────────────────────────────────────────────────────────────────────────────
# ПОЛ (а): VAE-круг. Меряется здесь и сейчас, карта не нужна.
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_floor(sources, reconstructions, kind, extra) -> dict:
    """Общее место для обоих полов: попарное сравнение и сборка ответа.

    Один код на два пола не ради краткости, а ради Е1: если (а) и (б) считать
    двумя разными арифметиками, их РАЗНОСТЬ перестанет что-либо значить, а
    разность и есть то, ради чего заведены оба.
    """
    import numpy as np

    if len(sources) != len(reconstructions):
        return {"outcome": UNMEASURED, "kind": kind, "floor": None,
                "floor_max": None, "frames": len(sources),
                "note": (f"на входе {len(sources)} кадров, обратно пришло "
                         f"{len(reconstructions)}: сравнивать попарно нечего")}
    means, maxes = [], []
    for src, rec in zip(sources, reconstructions):
        a, b = _as_array(src), _as_array(rec)
        if a.shape != b.shape:
            return {"outcome": UNMEASURED, "kind": kind, "floor": None,
                    "floor_max": None, "frames": len(sources),
                    "note": (f"круг вернул кадр другого размера: {a.shape} "
                             f"против {b.shape}")}
        d = np.abs(a - b).mean(axis=2)
        means.append(float(d.mean()))
        maxes.append(float(d.max()))
    out = {"outcome": PASS, "kind": kind, "frames": len(sources),
           "floor": round(max(means), 6), "floor_max": round(max(maxes), 6),
           "floor_mean_of_frames": round(float(sum(means) / len(means)), 6)}
    out.update(extra)
    out["note"] = (f"пол «{kind}» ИЗМЕРЕН на {len(sources)} кадрах: среднее до "
                   f"{max(means):.6f}, локально до {max(maxes):.6f}. "
                   + str(extra.get("how", "")))
    return out


def vae_roundtrip_floor(frames, *, vae_dir=None, repo: str = VAE_REPO,
                        dtype: str = "float32", device: str = "cpu",
                        tiled_decode: bool = True, keep=None) -> dict:
    """ПОЛ (а): `vae.decode(vae.encode(драйвинг))`, без единого шага сэмплера.

    Это неустранимый пол ПРЕДСТАВЛЕНИЯ. Всё, что вообще проходит через латент,
    не может разойтись с исходником меньше, чем на эту величину, — а вне маски
    через латент проходит всё, кроме звука (§1a).

    Три исхода. Нет torch/diffusers/весов — `НЕ СМОГЛИ ИЗМЕРИТЬ`, и это НЕ
    повод взять число «по умолчанию»: число по умолчанию есть то же выдуманное
    число, только с алиби. Ровно так в код попала кодековая планка.

    `frames` — пути или массивы. Wan-VAE сжимает время в 4 раза, поэтому длина
    вида 4k+1 (5, 13, 17…) проходит круг без досбора хвоста.
    """
    try:
        import numpy as np
        import torch
        from diffusers import AutoencoderKLWan
    except Exception as exc:                      # pragma: no cover - среда
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(list(frames)),
                "note": (f"нет torch/diffusers ({exc}): VAE-круг замерить "
                         f"нечем. Планку по умолчанию не берём.")}

    srcs = [_as_array(f) for f in frames]
    if len(srcs) < MIN_FLOOR_FRAMES:
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": (f"кадров {len(srcs)}, нужно хотя бы "
                         f"{MIN_FLOOR_FRAMES}: у Wan-VAE сжатие по времени 4x, "
                         f"и на коротышке временная часть круга не работает")}
    shapes = {s.shape for s in srcs}
    if len(shapes) != 1:
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": f"кадры разного размера: {sorted(map(str, shapes))}"}

    src = vae_dir if vae_dir is not None else repo
    # Путь проверяется ДО загрузчика, и это не придирка (Т4): на несуществующей
    # папке diffusers молча уходит в сеть за одноимённым репозиторием. Тест,
    # который так падает, краснеет от чужой аварии и зеленеет от кэша.
    if vae_dir is not None and not Path(vae_dir).is_dir():
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": (f"папки с весами VAE нет: {vae_dir}. Это НЕ ноль "
                         f"протечки и не планка — это отсутствие замера.")}
    try:
        vae = AutoencoderKLWan.from_pretrained(
            str(src), torch_dtype=getattr(torch, dtype)).to(device).eval()
    except Exception as exc:
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": (f"веса VAE не поднялись из {src}: {exc}. Это НЕ ноль "
                         f"протечки и не планка — это отсутствие замера.")}

    x = np.stack(srcs).astype("float32")                    # T H W C
    t = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0)  # B C T H W
    t = (t * 2 - 1).to(device=device, dtype=getattr(torch, dtype))
    try:
        with torch.no_grad():
            # `.mode()`, а не `.sample()`: пол ПРЕДСТАВЛЕНИЯ не должен включать
            # ещё и разброс латентного распределения — иначе (а) частично
            # померяет то же, что (б), и их разность перестанет быть вкладом
            # сэмплера.
            lat = vae.encode(t).latent_dist.mode()
            # Декод тайлами — как в стеке (§3, «VAE, декод тайлами»). Мерить
            # надо тот тракт, который поедет, а не более щедрый: полный декод
            # 480×848 на CPU здесь просто не влез в память.
            if tiled_decode:
                vae.enable_tiling()
            rec = vae.decode(lat).sample
    except Exception as exc:
        return {"outcome": UNMEASURED, "kind": FLOOR_VAE, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": f"круг не прошёл ({exc}): замера нет"}

    out_np = ((rec.float().clamp(-1, 1) + 1) / 2)
    out_np = out_np.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    recs = [out_np[i] for i in range(min(len(srcs), out_np.shape[0]))]

    got = _pairwise_floor(
        srcs, recs, FLOOR_VAE,
        {"latent": tuple(int(v) for v in lat.shape), "dtype": dtype,
         "device": device, "weights": str(src), "tiled_decode": tiled_decode,
         "how": (f"vae.decode(vae.encode(x)) на {dtype}/{device}, латент "
                 f"{tuple(int(v) for v in lat.shape)}, декод "
                 f"{'тайлами' if tiled_decode else 'целиком'}")})
    if keep is not None and got["floor"] is not None:
        _dump_pair(keep, srcs[0], recs[0])         # П3: посмотреть глазами
        got["artifacts"] = str(keep)
    return got


def _dump_pair(where, src, rec) -> None:
    """П3: положить исходный и восстановленный кадр рядом, чтобы ОТКРЫТЬ.

    Одно число «пол такой-то» ничего не говорит о том, ЧТО именно VAE сделал с
    кадром: размыл кожу, съел зерно, сдвинул цвет, поплыл на контуре. Это
    видно глазами за секунду и не видно ни в одной из наших метрик.
    """
    import numpy as np
    from PIL import Image

    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    for name, arr in (("src.png", src), ("roundtrip.png", rec)):
        Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype("uint8")).save(where / name)


# ─────────────────────────────────────────────────────────────────────────────
# ПОЛ (б): полный круг при character_mask = 0. Только он судит протечку.
# ─────────────────────────────────────────────────────────────────────────────

def full_loop_floor(frames, render, *, mask_shape=None) -> dict:
    """ПОЛ (б): весь тракт при `character_mask = 0` ВСЮДУ.

    Маска нулевая — модели нечего перерисовывать, она обусловлена драйвингом
    целиком. Всё, чем выход всё-таки отличается от драйвинга, есть цена
    прохода через VAE И СЭМПЛЕР, то есть настоящий пол оси.

    `render(frames, mask) -> кадры` — исполнитель (поток E, на карте). Здесь
    он ПАРАМЕТР, а не импорт: иначе модуль нельзя ни протестировать без Comfy,
    ни отличить «пол не замерен» от «Comfy не поднялся».

    НЕПРОВЕРЕНО: на настоящем исполнителе не запускалось ни разу — нужны 14B
    весов и карта. Функция написана, число за ней.
    """
    import numpy as np

    srcs = [_as_array(f) for f in frames]
    if len(srcs) < MIN_FLOOR_FRAMES:
        return {"outcome": UNMEASURED, "kind": FLOOR_FULL_LOOP, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": (f"кадров {len(srcs)}, нужно хотя бы "
                         f"{MIN_FLOOR_FRAMES}")}
    h, w = mask_shape if mask_shape else srcs[0].shape[:2]
    zero_mask = np.zeros((len(srcs), h, w), dtype="float32")

    try:
        rendered = render(srcs, zero_mask)
    except Exception as exc:
        return {"outcome": UNMEASURED, "kind": FLOOR_FULL_LOOP, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": (f"исполнитель не отработал ({exc}): пола нет. Пустой "
                         f"маской пол не подменяется — без прогона он не "
                         f"существует.")}
    if rendered is None:
        return {"outcome": UNMEASURED, "kind": FLOOR_FULL_LOOP, "floor": None,
                "floor_max": None, "frames": len(srcs),
                "note": "исполнитель вернул None: пола нет"}

    recs = [_as_array(f) for f in rendered]
    return _pairwise_floor(
        srcs, recs, FLOOR_FULL_LOOP,
        {"how": ("полный круг при character_mask = 0 всюду: модели нечего "
                 "генерировать, остаток — цена VAE плюс сэмплера")})


def declared_full_loop_floor(*, floor: float, floor_max: float,
                             measured_by: str, frames: int) -> dict:
    """Пол (б), снятый ОТДЕЛЬНЫМ прогоном на карте и внесённый руками.

    Нужен затем, что прогон на A16 и разбор его чисел разнесены во времени.
    `measured_by` обязателен и не может быть пустым (И4): планка без указания,
    чем получена, — это ровно та кодековая планка, от которой мы избавились.
    """
    if not measured_by or not str(measured_by).strip():
        return {"outcome": UNMEASURED, "kind": FLOOR_FULL_LOOP, "floor": None,
                "floor_max": None, "frames": frames,
                "note": ("пол подан без указания, чем замерен: не принят. "
                         "Число без происхождения на этой оси уже стоило "
                         "снятой кодековой планки.")}
    return {"outcome": PASS, "kind": FLOOR_FULL_LOOP, "frames": frames,
            "floor": float(floor), "floor_max": float(floor_max),
            "measured_by": str(measured_by),
            "note": (f"пол «{FLOOR_FULL_LOOP}» внесён из прогона: среднее "
                     f"{float(floor):.6f}, локально {float(floor_max):.6f}, "
                     f"кадров {frames}. Замерено: {measured_by}")}


def sampler_contribution(vae_floor: dict, loop_floor: dict) -> dict:
    """РАЗНОСТЬ двух полов — вот ради чего их два, а не один.

    VAE-пол — цена представления, полный пол — цена представления ПЛЮС
    сэмплера. Если полный намного выше VAE-пола, реконструкция вне маски
    нестабильна сама по себе, и это надо знать ДО того, как судить протечку:
    иначе нестабильность тракта будет прочитана как порча реквизита.
    """
    a = (vae_floor or {}).get("floor")
    b = (loop_floor or {}).get("floor")
    if (vae_floor or {}).get("kind") != FLOOR_VAE:
        return {"outcome": UNMEASURED, "delta": None,
                "note": f"первым аргументом нужен пол «{FLOOR_VAE}»"}
    if (loop_floor or {}).get("kind") != FLOOR_FULL_LOOP:
        return {"outcome": UNMEASURED, "delta": None,
                "note": f"вторым аргументом нужен пол «{FLOOR_FULL_LOOP}»"}
    if a is None or b is None:
        missing = "VAE-круг" if a is None else "полный круг"
        return {"outcome": UNMEASURED, "delta": None,
                "note": (f"{missing} не замерен: вклад сэмплера — разность, а "
                         f"не одно из слагаемых")}
    delta = round(b - a, 6)
    ratio = round(b / a, 3) if a > 0 else None
    return {"outcome": PASS, "delta": delta, "ratio": ratio,
            "vae_floor": a, "loop_floor": b,
            "note": (f"VAE-пол {a:.6f}, полный пол {b:.6f}: вклад сэмплера "
                     f"{delta:+.6f}"
                     + (f" ({ratio}x)" if ratio is not None else "")
                     + (". Полный пол много выше VAE-пола — реконструкция вне "
                        "маски нестабильна сама по себе."
                        if ratio is not None and ratio > SAMPLER_FLOOR_ALARM
                        else ""))}


# ─────────────────────────────────────────────────────────────────────────────
# ВЕРДИКТ
# ─────────────────────────────────────────────────────────────────────────────

def verdict(divergence: dict, floor: dict, *,
            over: float = LEAK_OVER_FLOOR) -> dict:
    """Протечка или пол тракта. Три исхода, и «планки нет» — один из них.

    ПЕРВОЕ, что проверяется, — не число, а КЛЕЙМО планки. Судить разрешено
    только по полному кругу при нулевой маске: VAE-пол занижен на весь вклад
    сэмплера, а кодековая планка была занижена на весь тракт целиком. Оба
    занижения выглядят строгими (число маленькое) и оба пропускают настоящую
    протечку, потому что накрывают её собственным шумом.
    """
    floor = floor or {}
    kind = floor.get("kind")
    if kind not in FLOOR_KINDS_THAT_MAY_JUDGE:
        return {"outcome": UNMEASURED, "ratio": None, "spot_ratio": None,
                "floor_kind": kind,
                "note": (f"планка происхождения «{kind}» судить протечку не "
                         f"вправе: нужен «{FLOOR_FULL_LOOP}». VAE-пол занижен "
                         f"на вклад сэмплера, кодековая планка была занижена "
                         f"на весь тракт — обе объявили бы протечкой "
                         f"собственную реконструкцию.")}
    if divergence.get("mean") is None:
        return {"outcome": UNMEASURED, "ratio": None, "spot_ratio": None,
                "note": f"расхождение не измерено: {divergence.get('note', '')}"}
    if floor.get("floor") is None:
        return {"outcome": UNMEASURED, "ratio": None, "spot_ratio": None,
                "note": (f"пол не измерен: {floor.get('note', '')}. "
                         f"Расхождение {divergence['mean']:.6f} само по себе "
                         f"ничего не значит — неизвестно, сколько из него "
                         f"стоит реконструкция.")}
    base = floor["floor"]
    if base <= 0:
        return {"outcome": UNMEASURED, "ratio": None, "spot_ratio": None,
                "note": ("пол нулевой: тракт не тронул ни пикселя, чего при "
                         "сэмплировании из шума не бывает — прогон пола, "
                         "скорее всего, сравнивал файл сам с собой")}
    ratio = round(divergence["mean"] / base, 3)
    leaking = ratio > over

    # ЛОКАЛЬНЫЙ КАНАЛ СУДИТСЯ ОТДЕЛЬНО И СВОЕЙ ПЛАНКОЙ. Сначала вердикт стоял
    # на одном среднем, и максимум печатался без порога — то есть канал, ради
    # которого он и заведён (стёртый мяч занимает проценты площади), не мог
    # ничего решить. Что локальный пол ВСЕГДА много выше среднего — не
    # рассуждение, а замер: VAE-круг дал среднее 0.007308 при максимуме
    # 0.338698, в 46 раз. Судить максимум по средней планке значило бы
    # объявлять протечкой каждый край объекта.
    base_max = floor.get("floor_max")
    spot_ratio = None
    if base_max and divergence.get("max") is not None:
        spot_ratio = round(divergence["max"] / base_max, 3)
        if spot_ratio > over:
            leaking = True

    return {
        "outcome": FAIL if leaking else PASS,
        "ratio": ratio, "spot_ratio": spot_ratio,
        "floor": base, "floor_max": base_max, "floor_kind": kind,
        "mean": divergence["mean"], "max": divergence.get("max"),
        "note": (f"вне маски среднее {divergence['mean']:.6f} при поле "
                 f"{base:.6f} — {ratio}x"
                 + (f"; локально {divergence['max']:.6f} при поле "
                    f"{base_max:.6f} — {spot_ratio}x"
                    if spot_ratio is not None else
                    "; локальный пол не замерен, точечная протечка не "
                    "судится")
                 + f". Порог {over}x: "
                   f"{'ПРОТЕЧКА' if leaking else 'в пределах пола тракта'}."),
    }
