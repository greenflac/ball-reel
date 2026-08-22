"""Судит ЛЮБОЙ ролик нашими приборами: раскладка на кадры + ArcFace.

У приборов НЕТ командной строки — только вызов из Python. Это не оговорка, а
факт: ни `fork_identity`, ни `fork_leak` не имеют `main()` и не принимают
видео. Видео сначала раскладывается `fork_video.frames`.

## КАЛИБРОВОЧНАЯ ЛЕСТНИЦА, по которой читается число

    0.35     планка проекта: ближе — тот же человек
    0.7137   отбракованный референс против боевого — РАЗНЫЕ ЛЮДИ
    1.0755   актёр драйвинга против фото клиента, 0 из 6 кадров в баре

Годный ролик обязан лечь НИЖЕ 0.35. Около 1.0 значит, что модель нарисовала
актёра драйвинга, а не клиента.

ВОСПРОИЗВЕДЕНО 22.08.2026 в чистом контейнере: 0.7137 сошлось до четвёртого
знака (`assets/legacy_hero_NOT_FOR_PRODUCTION.png` против
`assets/photoref_profile.jpg`, лицо 127 px). То есть прибор здесь работает, и
это проверено с ОБЕИХ сторон (И5) — он умеет сказать «далеко» и умеет сказать
«близко»: та же фотография сама с собой дала 0.0.

## ПРО ПЛАНКУ РАЗМЕРА ЛИЦА — назвать вслух, а не замести

`MIN_FACE_PX = 100`, а лицо на нашем полном кадре 84-88 px. То есть на этом
плане прибор systematically ответит «не смогли измерить», и это ТРЕТИЙ ИСХОД,
а не ноль и не брак.

ИЗМЕРЕНО 22.08 на боевом референсе: та же фотография, уменьшенная до лица
86 px, дала расстояние до себя же **0.0102** — в 34 раза меньше планки 0.35.
Значит РАЗМЕР САМ ПО СЕБЕ судить не мешает; отсев по 100 px выбрасывает кадры,
которые прибор рассудил бы верно. ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ: он снят на
уменьшенной копии, где нет ни смаза, ни смены позы, ни генерации. На настоящих
кадрах в полосе 84-88 px мешать будут именно они, а не размер.

Поэтому меряем ОБА режима и печатаем оба, а не выбираем удобный.
"""

from __future__ import annotations

import json
from pathlib import Path

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"
BAR = 0.35


def _prober():
    """Настоящий ffprobe, если он есть; иначе заменитель на PyAV.

    Порядок именно такой: заменитель — запасной путь, а не умолчание. Кто
    снимал числа, едет в отчёт, потому что заменитель против настоящего
    ffprobe не сверялся (сверять не с чем — бинарника в среде нет).
    """
    import shutil
    if shutil.which("ffprobe"):
        return None, "ffprobe"
    from .fork_bench_prober import pyav_probe
    return pyav_probe, "заменитель PyAV (ffprobe в среде нет)"


def measure(video_path, frames_dir, anchor="assets/photoref_profile.jpg",
            *, bars=(None, 100)) -> dict:
    """Раскладывает ролик и судит каждый кадр против фото клиента."""
    from ball_reel import fork_video, fork_identity

    if not Path(video_path).exists():
        return {"outcome": UNMEASURED,
                "note": f"ролика {video_path} нет: судить нечего"}
    if not Path(anchor).exists():
        return {"outcome": UNMEASURED,
                "note": f"якоря {anchor} нет: мерить не от чего"}

    prober, prober_name = _prober()
    got = fork_video.frames(str(video_path), str(frames_dir), overwrite=True,
                            prober=prober)
    paths = got.get("paths") or []
    if not paths:
        return {"outcome": UNMEASURED, "decode": got, "prober": prober_name,
                "note": f"раскладка не дала кадров: {got.get('note')}"}

    modes = {}
    for bar in bars:
        rep = fork_identity.distances(paths, anchor, min_face_px=bar)
        unmeasured = len(rep["no_face"]) + len(rep["too_small"])
        modes[str(bar)] = {
            "median": rep["median"], "min": rep["min"], "max": rep["max"],
            "inside": rep["inside"], "judged": rep["judged"],
            "total": rep["total"],
            "not_measured": unmeasured,
            "no_face": len(rep["no_face"]), "too_small": len(rep["too_small"]),
            "face_px_min": min(rep["face_px"].values()) if rep["face_px"] else None,
            "face_px_max": max(rep["face_px"].values()) if rep["face_px"] else None,
            "outcome": rep["outcome"], "note": rep["note"],
        }
    return {"outcome": modes[str(bars[0])]["outcome"],
            "frames_decoded": len(paths), "modes": modes,
            "prober": prober_name,
            "note": (f"разложено {len(paths)} кадров прибором «{prober_name}», "
                     f"режимов отсева {len(modes)}")}


def render(rep: dict) -> str:
    """Числа рядом с вердиктом: проверено N, в баре M, не смогли K."""
    out = [f"исход: {rep['outcome']}", rep.get("note", "")]
    for bar, m in (rep.get("modes") or {}).items():
        out.append(
            f"\n--- отсев по размеру лица: {'выключен' if bar == 'None' else bar + 'px'} ---\n"
            f"  медиана {m['median']} при планке {BAR} (мин {m['min']}, макс {m['max']})\n"
            f"  проверено {m['judged']} из {m['total']}, в баре {m['inside']}, "
            f"НЕ СМОГЛИ {m['not_measured']} "
            f"(без лица {m['no_face']}, мельче планки {m['too_small']})\n"
            f"  лицо {m['face_px_min']}..{m['face_px_max']} px\n"
            f"  вердикт прибора: {m['outcome']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "work/fal_out.mp4"
    d = sys.argv[2] if len(sys.argv) > 2 else "work/fal_frames"
    r = measure(v, d)
    print(render(r))
    Path("work").mkdir(exist_ok=True)
    Path(f"work/measure_{Path(v).stem}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
