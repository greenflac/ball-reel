# `fork_bench` — приборы сравнения с рынком

Результаты и вердикты — `docs/FORK_BENCHMARK_RESULTS.md`. Методология —
`docs/FORK_BENCHMARK_MANUAL.md`. Здесь только как запускать.

**Существующие модули `ball_reel/*` не правятся.** Заменитель `ffprobe`
подаётся через объявленную точку внедрения `prober=`, а не правкой модуля (Ц2).

## Ключи

Только из окружения. В код, в лог, в карточку и в имя файла они не попадают
никогда: ключ, попавший в историю git, отзывается у вендора, а не убирается
коммитом.

    export FAL_KEY=...
    export KLING_KEY=...
    export KLING_BASE=...   # база API Kling, из кабинета
    export KLING_PATH=...   # путь метода Motion Control, из документации

`KLING_BASE`/`KLING_PATH` обязательны намеренно: непроверенное внешнее имя в
код не вписывается (Ц10), поэтому скрипт скорее откажется работать, чем
угадает путь и молча отправит запрос не туда.

## Порядок

```bash
pip install fal-client insightface onnxruntime numpy opencv-python-headless av

# 1. один и тот же вход для всех, с проверкой ПОСЛЕ реза
python3 -m fork_bench.fork_bench_input assets/driving_yogaball.mp4 work/bench_driving.mp4

# 2. эталон на той же модели
python3 -m fork_bench.fork_bench_fal 480p
python3 -m fork_bench.fork_bench_fal 720p

# 3. планка тир-1
python3 -m fork_bench.fork_bench_kling

# 4. судим ЛЮБОЙ полученный ролик нашими приборами
python3 -m fork_bench.fork_bench_measure work/fal_out.mp4 work/fal_frames
```

Журналы прогонов ложатся в `work/` (он в `.gitignore`): `bench_fal_journal.jsonl`,
`bench_kling_journal.jsonl`, `measure_*.json`. Пишутся ДО просмотра ролика и
при неуспехе тоже — отрицательный результат с числом и условиями это тоже
запись (И6).

## Три исхода, а не два

Каждый прибор возвращает `годно` / `не годно` / **`не смогли проверить`**.
Третий не сворачивается ни в первый, ни во второй: «домен закрыт», «файла
нет» и «лицо мельче планки» — это НЕ «плохой ролик».

Рядом с вердиктом всегда числа: `проверено N`, `в баре M`, `не смогли K`.
