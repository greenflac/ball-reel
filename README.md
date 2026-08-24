# Lip-sync templates

The user uploads an ordinary selfie and picks a template. Out comes a vertical
video with sound, and the person in the frame is them.

![Client selfie, driving footage, finished video](docs/img/showcase_icecream.gif)

<sub>Left to right: the client's selfie · the bought driving footage · the
finished video. One template ("Ice cream"), 10 seconds, nothing sped up. The
same clip **with sound**: [showcase_icecream.mp4](docs/video/showcase_icecream.mp4).</sub>

---

This is two things, not one.

**A generation pipeline** — eight stages from selfie to finished video. Seven
are free and stand before the paid one, so a defect is caught on a still for a
fraction of a cent instead of on a video for a dollar.

**An instrument for assembling templates** — the author writes a prompt, feeds
in a demo identity, and gets an aesthetic ready to sell. A new template takes a
run, not a sprint.

![Six templates of the family](docs/img/family.png)

<sub>Six templates of the first batch. One identity, six aesthetics, four
different drivings.</sub>

**Presentation:** [English (PDF)](docs/deck/lipsync_en.pdf) ·
[по-русски (PDF)](docs/deck/lipsync_ru.pdf)

---

## Numbers

| | |
|---|---|
| `$0.70` | a 10-second video; the `$0.07/s` rate confirmed by four balance measurements |
| `720×1280` | exactly 9:16 out of the model, 0% of the area lost to cropping in assembly |
| `300 frames` | exactly 10.0 seconds on all six, with no spread |
| `0 cuts` | edit seams in frame, on all six |

---

## How it works

```
1  driving intake        seams, scene length, window        free
2  aesthetic             prompt + demo identity             cents
3  aesthetic acceptance  demo identity in place             free
4  reference assembly    client photo + aesthetic           cents
5  reference acceptance  leak, plan, composition            free
6  video generation      Kling Motion Control               $0.07/s
7  assembly              9:16, sound, length                free
8  video acceptance      by eye                             free
```

Every check answers with one of **three** outcomes — `pass`, `fail`,
`could not measure` — and the third collapses into neither of the first two.
Numbers always stand next to the verdict: checked N, violations M, could not
measure K.

Zero violations out of zero checks is not a success, and the code prints that
literally.

## How to integrate

The pipeline's logic depends on neither the backend nor the model.

Every outward call is a parameter, not a hard reference: the model, the
storage and the queue are all swappable. That is why the tests run the whole
path without touching the network.

The acceptance layer does not know who generated its input. It can be placed in
front of any existing pipeline as a separate quality control.

Changing the video model is one call. Kling's limits (a three-second minimum,
unstable frame-by-frame output) are properties of the model, not of the
approach: they lift by moving to another model over the API, or to open source
on your own GPUs.

## Driving requirements

The only input that is bought rather than drawn. That is why its requirements
live in a section of their own — this is a purchasing checklist, not a setting.

- **Face larger than 100 px.** Below that there is nothing to measure the
  identity with, and the model transfers it uncertainly. Measured: drivings
  with a face of 34–56 px produced a drifting face; those with 85–139 px held
  it.
- **One scene, zero edit seams.** A seam inside the window produces a change of
  plan in the finished video.
- **Scene longer than the product length.** The window is cut from the middle
  of the longest scene with equal margins.
- **The person does not leave the frame.** The driving's composition is carried
  into the aesthetic prompt: the model lays the pose from the video.

Lip-sync is the heaviest material, and heaviest of all is the kind where the
character **turns around**: on the turn the face leaves the frame entirely.

## Tests

```
python -m unittest discover -s ball_reel/tests -p "test_fork_*.py"
```

More than four hundred guards. Each guards a defect that was actually found,
not a line of code, and its docstring says which one. Thresholds are mutated in
both directions: if a bar guards nothing, the test will not go red.

## Licence

The sources are open to read and to audit, but this is **not** open source:
using, copying and embedding require an agreement. Details in
[LICENSE](LICENSE).

Separately: some of the measuring components used during development are
licensed for non-commercial use only. This is accounted for and described in
the documents.

---
---

# Липсинк-шаблоны

Пользователь грузит обычное селфи и выбирает шаблон. На выходе — вертикальный
ролик со звуком, где в кадре он сам.

![Селфи клиента, драйвинг, готовый ролик](docs/img/showcase_icecream.gif)

<sub>Слева направо: селфи клиента · купленный драйвинг · готовый ролик. Один
шаблон («Мороженое»), 10 секунд, ничего не ускорено. Тот же ролик **со
звуком**: [showcase_icecream.mp4](docs/video/showcase_icecream.mp4).</sub>

---

Это два предмета, а не один.

**Пайплайн генерации** — восемь ступеней от селфи до готового ролика. Семь
бесплатны и стоят до платной, чтобы брак ловился на картинке за доли цента, а
не на видео за доллар.

**Прибор для сборки шаблонов** — составитель пишет промт, подаёт демо-личность
и получает эстетику, готовую к продаже. Новый шаблон рождается за прогон, а не
за спринт.

![Шесть шаблонов семейства](docs/img/family.png)

<sub>Шесть шаблонов первого батча. Одна личность, шесть эстетик, четыре разных
драйвинга.</sub>

**Презентация:** [по-русски (PDF)](docs/deck/lipsync_ru.pdf) ·
[English (PDF)](docs/deck/lipsync_en.pdf)

---

## Числа

| | |
|---|---|
| `$0.70` | ролик 10 секунд; ставка `$0.07/с` подтверждена четырьмя замерами счёта |
| `720×1280` | ровно 9:16 на выходе модели, обрезка в сборке — 0% площади |
| `300 кадров` | ровно 10.0 секунды у всех шести, без разброса |
| `0 резов` | монтажных швов в кадре, у всех шести |

---

## Как устроено

```
1  приём драйвинга        склейки, длина сцены, окно        бесплатно
2  эстетика               промт + демо-личность              центы
3  приёмка эстетики       демо-личность на месте             бесплатно
4  сборка рефки           фото клиента + эстетика            центы
5  приёмка рефки          утечка, план, композиция           бесплатно
6  генерация видео        Kling Motion Control               $0.07/с
7  сборка                 9:16, звук, длина                  бесплатно
8  приёмка ролика         глазами                            бесплатно
```

Каждая проверка отвечает одним из **трёх** исходов — `годно`, `не годно`,
`не смогли проверить` — и третий не сворачивается ни в первый, ни во второй.
Рядом с вердиктом всегда числа: проверено N, нарушений M, не смогли K.

Ноль нарушений при нуле проверок — не успех, и код это печатает буквально.

## Как встроить

Логика конвейера не зависит ни от бэкенда, ни от модели.

Каждый вызов наружу — параметр, а не жёсткая ссылка: подменяются модель,
хранилище, очередь. Поэтому тесты прогоняют весь путь, не касаясь сети.

Слой приёмки не знает, кто сгенерировал вход. Его можно поставить перед любым
существующим пайплайном как отдельный контроль качества.

Смена видеомодели — это один вызов. Ограничения Kling (минимум три секунды,
нестабильная покадровая выдача) — свойства модели, а не подхода: снимаются
переходом на другую по API или на опенсорс на своих GPU.

## Требования к драйвингу

Единственный вход, который не рисуется, а покупается. Поэтому требования к нему
вынесены отдельно — это чек-лист закупки, а не настройка.

- **Лицо крупнее 100 px.** Ниже этого личность нечем измерять, и модель
  переносит её неуверенно. Измерено: драйвинги с лицом 34–56 px дали плывущее
  лицо, с 85–139 px — удержали.
- **Одна сцена, ноль монтажных склеек.** Шов внутри окна даёт смену плана в
  готовом ролике.
- **Сцена длиннее продуктовой длины.** Окно режется из середины самой длинной
  сцены равными полями.
- **Человек не уходит за край.** Композиция драйвинга переносится в промт
  эстетики: модель кладёт позу с видео.

Липсинк — самый тяжёлый материал, и особенно тот, где персонаж
**разворачивается**: на развороте лицо уходит из кадра целиком.

## Тесты

```
python -m unittest discover -s ball_reel/tests -p "test_fork_*.py"
```

Больше четырёхсот сторожей. Каждый сторожит найденный дефект, а не строчку
кода, и говорит в докстринге, какой именно. Пороги мутируются в обе стороны:
если планка ничего не сторожит, тест не покраснеет.

## Лицензия

Исходники открыты для чтения и проверки, но это **не** open source:
использовать, копировать и встраивать нельзя без договорённости. Подробности —
[LICENSE](LICENSE).

Отдельно: часть измерительных компонентов, использованных при разработке,
лицензирована только для некоммерческого применения. Это учтено и описано в
документах.
