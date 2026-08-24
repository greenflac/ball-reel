"""Готовит ОДИН И ТОТ ЖЕ кусок драйвинга для всех сравниваемых сервисов.

Почему не командой из мануала. Команда
    ffmpeg -ss 2.708 -i D -t 4.208 -c copy -an work/bench_driving.mp4
ЗАМЕРЕНА 22.08.2026 на подставном ролике той же геометрии (720x1278, 24 к/с,
362 кадра, GOP 250, scenecut выключен) и даёт **103 кадра, 65..167, 4.29 с**
вместо заказанных 101 кадра 65..165. Перебор 2 кадра — это +2% длины и +2%
счёта у обоих вендоров, и он МОЛЧАЛИВЫЙ.

Замеренная таблица (`work/readidx.py`, индекс кадра закодирован блоками):

    -ss до -i, -c copy, -t 4.208     103 кадра   65..167   4.29 с
    -ss до -i, -frames:v 101         101 кадр    65..165   4.21 с   <-- берём
    select=between(n,65,165)         101 кадр    65..165   4.17 с

ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ: он снят на СИНТЕТИЧЕСКОМ ролике, потому что
боевого `assets/driving_yogaball.mp4` в репозитории нет (см. README). Раскладка
ключевых кадров боевого файла может отличаться, поэтому проверка ниже
(`verify`) обязательна и на боевом входе тоже.

Второй замеренный дефект той же команды: если точка реза за концом ролика,
ffmpeg НЕ падает, а молча отдаёт файл целиком (проверено на 1.33-секундном
`demo/clip.mp4`: на выходе те же 672 097 Б, что на входе). Поэтому `verify`
ниже — не украшение, а единственное, что отличает рез от копии.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

#: ВЫБРАНО (мануал §2): кадры куска, включительно с обоих концов.
FIRST_FRAME = 65
LAST_FRAME = 165
#: РАСЧЁТ: 165 - 65 + 1.
WANT_FRAMES = LAST_FRAME - FIRST_FRAME + 1
#: ИЗМЕРЕНО приборами проекта на боевом драйвинге (assets/README.md).
SOURCE_FPS = 24

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"


def ffmpeg_exe() -> str | None:
    """ffmpeg из PATH, иначе из imageio-ffmpeg. None — если нет ни того ни другого."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def cut_argv(src, dst, exe: str) -> list:
    """Точный рез: старт по времени, длина ПО КАДРАМ, а не по секундам.

    `-frames:v` считает кадры, поэтому округление секунд на длину не влияет —
    ровно этим он и отличается от `-t`, давшего 103 кадра вместо 101.
    """
    start = FIRST_FRAME / SOURCE_FPS
    return [exe, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.6f}", "-i", str(src),
            "-frames:v", str(WANT_FRAMES),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an", str(dst)]


def count_frames(path, exe: str) -> int | None:
    """Сколько кадров реально ДЕКОДИРУЕТСЯ. None — если посчитать не вышло."""
    # ДЕКОДИРУЕМ полностью и намеренно: с `-c copy` ffmpeg не печатает
    # `frame=` вовсе (замерено 22.08), и счётчик молча возвращал None.
    out = subprocess.run(
        [exe, "-hide_banner", "-i", str(path), "-map", "0:v:0",
         "-f", "null", "-"],
        capture_output=True, text=True)
    for line in reversed(out.stderr.splitlines()):
        if "frame=" in line:
            try:
                return int(line.split("frame=")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def verify(path, exe: str) -> dict:
    """Три исхода, а не два: рез мог не состояться, и это не «не годно»."""
    p = Path(path)
    if not p.exists():
        return {"outcome": FAIL, "frames": None,
                "note": f"файла {p} нет: рез не состоялся"}
    got = count_frames(p, exe)
    if got is None:
        return {"outcome": UNMEASURED, "frames": None,
                "note": f"{p.name} существует ({p.stat().st_size} Б), но кадры "
                        f"посчитать не вышло — судить не по чему"}
    if got != WANT_FRAMES:
        return {"outcome": FAIL, "frames": got,
                "note": f"{p.name}: кадров {got}, заказано {WANT_FRAMES}. "
                        f"Расхождение {got - WANT_FRAMES:+d} кадра — вход у "
                        f"сервисов будет НЕ тот, что у нас"}
    return {"outcome": PASS, "frames": got,
            "note": f"{p.name}: ровно {got} кадров, "
                    f"{got / SOURCE_FPS:.3f} с при {SOURCE_FPS} к/с"}


def prepare(src, dst) -> dict:
    """Режет и ТУТ ЖЕ проверяет. Без проверки рез неотличим от копии."""
    exe = ffmpeg_exe()
    if exe is None:
        return {"outcome": UNMEASURED, "frames": None,
                "note": "ffmpeg не найден ни в PATH, ни через imageio-ffmpeg: "
                        "резать нечем"}
    if not Path(src).exists():
        return {"outcome": UNMEASURED, "frames": None,
                "note": f"исходного драйвинга {src} нет на диске — резать нечего. "
                        f"Это НЕ «рез плохой», это «резать не из чего»"}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(cut_argv(src, dst, exe), capture_output=True, text=True)
    if run.returncode != 0:
        return {"outcome": FAIL, "frames": None,
                "note": f"ffmpeg вернул {run.returncode}: "
                        f"{run.stderr.strip()[:300]}"}
    return verify(dst, exe)


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "assets/driving_yogaball.mp4"
    d = sys.argv[2] if len(sys.argv) > 2 else "work/bench_driving.mp4"
    rep = prepare(s, d)
    print(f"исход: {rep['outcome']}")
    print(f"кадров: {rep['frames']} (заказано {WANT_FRAMES})")
    print(rep["note"])
    raise SystemExit(0 if rep["outcome"] == PASS else 1)
