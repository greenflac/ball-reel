"""Заменитель `ffprobe` на PyAV — для сред, где бинарника ffprobe нет.

ЗАЧЕМ. `fork_video.read_probe` зовёт `ffprobe`, а в этой песочнице его нет:
`imageio-ffmpeg` кладёт только `ffmpeg`, а готовые сборки с `ffprobe` лежат на
хостах, закрытых сетевой политикой (`github.com/.../raw/...` и
`johnvansickle.com` — оба ЗАМЕРЕНЫ, оба 403/недоступны).

ПОЧЕМУ ЭТО НЕ ПРАВКА ЧУЖОГО МОДУЛЯ (Ц2). `fork_video.frames` принимает
`prober=` как ОБЪЯВЛЕННУЮ точку внедрения («ТОЧКА ВНЕДРЕНИЯ: тест подменяет
целиком»). Мы пользуемся ею, а не правим `ball_reel/fork_video.py`.

КОНТРАКТ соблюдается ровно тот же: возвращаем
`{"ran","code","out","err","why"}`, где `out` — JSON в форме ffprobe, который
разбирает `fork_video.parse_probe`.

ЧЕГО ЭТОТ ЗАМЕНИТЕЛЬ НЕ ДАЁТ: он не проверялся против настоящего `ffprobe` на
одном и том же файле — сравнить не с чем, бинарника нет. Поэтому числа,
снятые через него, помечаются в отчёте как снятые ЗАМЕНИТЕЛЕМ, а не ffprobe.
"""

from __future__ import annotations

import json


def pyav_probe(path) -> dict:
    """Метаданные через PyAV в форме ответа ffprobe."""
    try:
        import av
    except ImportError:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": "PyAV не установлен: спросить нечем (pip install av)"}
    try:
        with av.open(str(path)) as container:
            vs = [s for s in container.streams if s.type == "video"]
            audio = any(s.type == "audio" for s in container.streams)
            if not vs:
                payload = {"streams": ([{"codec_type": "audio"}] if audio else []),
                           "format": {}}
                return {"ran": True, "code": 0,
                        "out": json.dumps(payload), "err": "", "why": ""}
            v = vs[0]
            avg = v.average_rate
            dur = None
            if v.duration is not None and v.time_base:
                dur = float(v.duration * v.time_base)
            elif container.duration:
                dur = container.duration / 1_000_000
            nb = v.frames or 0
            stream = {
                "codec_type": "video",
                "codec_name": getattr(v.codec_context, "name", None),
                "width": v.codec_context.width,
                "height": v.codec_context.height,
                "avg_frame_rate": f"{avg.numerator}/{avg.denominator}" if avg else "0/0",
                "r_frame_rate": f"{avg.numerator}/{avg.denominator}" if avg else "0/0",
            }
            # nb_frames пишем ТОЛЬКО когда контейнер его действительно знает:
            # иначе parse_probe обязан честно оценить его как «длительность x
            # частота» и пометить это в frames_from. Подсунуть сюда ноль
            # значило бы выдать оценку за замер.
            if nb > 0:
                stream["nb_frames"] = str(nb)
            if dur is not None:
                stream["duration"] = str(dur)
            streams = [stream] + ([{"codec_type": "audio"}] if audio else [])
            payload = {"streams": streams,
                       "format": {"duration": str(dur) if dur is not None else None}}
            return {"ran": True, "code": 0,
                    "out": json.dumps(payload), "err": "", "why": ""}
    except Exception as exc:                        # noqa: BLE001
        return {"ran": True, "code": 1, "out": "", "err": str(exc)[:300],
                "why": ""}
