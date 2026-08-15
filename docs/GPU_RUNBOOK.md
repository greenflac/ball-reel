# Запуск на арендованном GPU — без лишних оплаченных минут

Правило всего документа: **всё, что можно сделать без GPU, делается до аренды.**
На карте выполняется только диффузия. Остальное — извлечение, ретаргетинг,
рендер условий, весь гейт — это CPU, и оно уже работает.

Счётчик тикает с момента запуска инстанса, поэтому самый дорогой ресурс тут не
VRAM, а **время до первой ошибки**. Всё ниже устроено так, чтобы ошибка
случалась как можно раньше и дешевле.

---

## 0. Дома, до аренды (0 ₽)

```bash
cd ball_reel
python3 -m unittest discover -s ball_reel/tests -p 'test_*.py'   # ожидается OK
```

Собрать пакет субъекта и условия локально — это самая большая экономия, потому
что переносит с GPU на свою машину всё, кроме генерации:

```bash
python3 - <<'EOF'
from ball_reel.intake import report
from ball_reel.metrics import body_metrics
from ball_reel.driving import extract, _sample_frames
from ball_reel.skeleton import render_sequence

print(report(["face.jpg"]))                     # 1. годится ли фото ВООБЩЕ

spec = extract("driving.mp4", fps=12, start=8, length=8, work_dir="frames")
print(spec.motion, spec.validate())             # 2. что за движение и чего в нём нет

import glob
prop = body_metrics("face.jpg")["proportions"]  # 3. пропорции клиента (3D)
m = render_sequence(sorted(glob.glob("frames/*.png")), "conditions",
                    proportions=prop, framing="waist_up")
print("условий:", len(m["conditions"]), "coverage:", m["coverage"],
      "лицо:", m["face_px"], "судимо:", m["identity_judgeable"])
print(*m["warnings"], sep="\n")
EOF

tar czf payload.tar.gz conditions/ face.jpg     # это поедет на VPS
```

**`framing` — новый и самый дешёвый рычаг, решается здесь, а не на карте.**
Умолчание `full_body` на потолке 512x768 даёт лицо **~72 px** при пороге ArcFace
100 px: гейт вернёт не «похож» и не «не похож», а «нечем судить». `waist_up` на
том же холсте даёт **~145 px** — судимо. Числа не выдуманы, они печатаются
`gpu_keyframes.identity_verifiable(768)` и получены из долей, замеренных живьём
(0.094 против 0.19 высоты кадра). На реальном ките доля вышла ещё ниже типовой —
0.074, потому что широкая стойка растягивает габарит фигуры; поэтому
планировать надо по манифесту, а не по таблице.
Манифест несёт `face_share_by_framing` — ОБЕ кадрировки сразу,
чтобы выбор делался до генерации и по числу; сигнатура
`render_sequence(frames, out_dir, *, proportions=None, width=512, height=768,
source=None, framing="full_body")`.

Проверить глазами **два** условия-скелета (первый и средний). Кривой скелет
кондиционирует хуже, чем никакой, а увидеть это на арендованной машине — значит
заплатить за просмотр.

Не арендовать, если:

* `intake` сказал `cannot identity` — генерация не спасёт, нужно другое фото;
* `coverage` условий заметно ниже 1.0 — в этих кадрах генератор не ограничен;
* `joint_coverage` низкое или `partial_frames` велико — кадр со скелетом ещё не
  значит скелет ЦЕЛИКОМ: невидимый локоть выбрасывает вместе с собой предплечье,
  и рука в этом кадре ничем не ограничена. Поймано глазами там, где `coverage`
  показывал 1.0, а у фигуры не было руки;
* `identity_judgeable` — `False`: клип будет сгенерирован и не проверен;
* в `spec.validate()` есть «motion track has holes».

---

## 0b. Своя карта вместо аренды — предпочтительнее

Если есть локальная NVIDIA (например ноутбучная RTX 3050, 4 ГБ) — брать её.
Снимается главное ограничение всего документа: **счётчик не тикает**. Вся
дисциплина «упасть как можно раньше» писалась против арендной минуты; на своей
машине можно спокойно итерировать, а веса скачиваются один раз навсегда.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

4096 MiB → план на 4 ГБ ниже без изменений. 6144 MiB → можно поднять кадр.

Две оговорки по ноутбуку: под нагрузкой он режет частоты, поэтому кейфрейм
пойдёт медленнее расчётных 15–40 с; и под Windows брать **WSL2 с CUDA** —
`insightface` и `onnxruntime` там ставятся без плясок, в отличие от нативной.

Дальше всё то же, но одной командой:

```bash
tar xzf payload.tar.gz

# ЦЕЛЕВОЙ ПУТЬ (умолчание): AnimateDiff + ControlNet + FaceID, ни одного
# сетевого вызова в генерации. Кадры считаются СОВМЕСТНО одним проходом.
python3 -m ball_reel.run_local --face face.jpg --conditions conditions \
  --prompt "a woman exercising on a fitness ball in a bright home studio, \
            black sports top and black leggings, natural light, photographic"

# ЗАПАСНОЙ ПУТЬ: кейфреймы поодиночке + сшивка видеомоделью через шлюз.
# Проверен живьём, поэтому оставлен. Единственный платный шаг во всём файле.
python3 -m ball_reel.run_local --engine chain \
  --face face.jpg --conditions conditions \
  --prompt "…тот же промт…" --video-model wan-fast --smoke   # дым: ОДИН кейфрейм
```

**Два движка, один гейт**, и это надо знать до запуска. `--engine` по умолчанию
`animatediff`; раньше в этом документе стояли команды без флага с описанием
шлюзового поведения («кейфреймы → сшивка»), то есть текст описывал не ту
команду, которую печатал.

Различие видно и в `--smoke`: на `chain` он рисует ОДИН кейфрейм и печатает
дрифт лица и позу (секунды против минут полного прогона); на `animatediff`
отдельного дешёвого кадра нет в принципе — кадры считаются совместно, — и
`--smoke` останавливается после сборки весов и печати того, какие LoRA реально
прицеплены.

Судятся оба одними и теми же мерами (`smoke_verdict`, `garment_drift`,
`garment_fit`, `clip_expression`, `clip_verdict`). Это не экономия кода: числа
двух путей сравнимы ровно постольку, поскольку сняты одним прибором.

`run_local` печатает измерение на каждом шаге и останавливается на первом
провале, так что разбираться приходится с одной причиной, а не с кучей
наполовину сделанных артефактов. Одежду задаёт промт; если она поплывёт между
узлами, он это скажет — но ~~предложит `--garment-ref`~~ **нет: `--garment-ref`
теперь отказывает сразу, первым же шагом, до предполёта.** Раньше он делал вид,
что работает: дописывал в промт «одежда как на референсе», а сам файл никуда не
передавался. Единственный адаптер занят лицом; второй референс требует второго
IP-Adapter'а и на железе не проверялся.

---

## 1. Выбор инстанса (если карты своей нет)

| VRAM | что доступно | вывод |
|---|---|---|
| < 3.5 ГБ | ничего из этого | не брать |
| **4 ГБ** | кейфреймы SD1.5 + ControlNet | рабочий вариант, дальше по тексту |
| 8–12 ГБ | + AnimateDiff, непрерывный контроль | брать, если разница в цене мала |
| 24 ГБ+ | Wan VACE, поза + объекты | только если нужна полная совместимость |

Смотреть не только на карту: **диск от 30 ГБ** (веса ~7 ГБ + кэш HF) и
нормальный канал — скачивание весов это оплаченное время.

---

## 2. Первые пять минут на машине

Сразу, одной командой — проверка карты И параллельная качка весов в фоне.
Качать в фоне, потому что это самая долгая часть, и она не должна ждать setup.

```bash
nvidia-smi                       # если пусто — гасить инстанс, это не GPU-машина

python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # индекс под свою CUDA/Python
pip install -r requirements-gpu.txt

# веса — в фоне, пока идёт остальная настройка
export HF_HOME=/workspace/hf
python3 -c "
from huggingface_hub import snapshot_download as d
d('runwayml/stable-diffusion-v1-5', allow_patterns=['*.json','*fp16.safetensors','*.txt'])
d('lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.json','*.safetensors'])
d('h94/IP-Adapter-FaceID', allow_patterns=['ip-adapter-faceid_sd15.bin',
                                           'ip-adapter-faceid_sd15_lora.safetensors'])
" &

mkdir -p ~/.mediapipe && curl -sSL -o ~/.mediapipe/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
tar xzf payload.tar.gz
```

~~`d('h94/IP-Adapter', allow_patterns=['models/ip-adapter-faceid_sd15.bin'])`~~ —
**здесь стояло именно это, и такого файла в том репозитории нет.** FaceID лежит
ОТДЕЛЬНО от обычных IP-Adapter'ов: в `h94/IP-Adapter` файлов со словом `faceid`
нет вообще, а в `h94/IP-Adapter-FaceID` они лежат в КОРНЕ, без подпапки
`models`. Скачивание молча вернуло бы пустой набор, а первый вызов на
арендованной карте упал бы `EntryNotFoundError` — после ~7 ГБ уже скачанных
весов SD1.5 и ControlNet. В коде исправлено (`gpu_keyframes.IP_ADAPTER_REPO`,
`animate.IP_ADAPTER_REPO`, `live_gen`), здесь — с этой правкой.

Оговорка о происхождении факта: содержимое обоих репозиториев проверено по HF
API тогда, когда правился код, и записано комментарием у константы. **Из среды,
где правился этот документ, повторно проверить нельзя — доступа к
`huggingface.co` нет.** Первый же запуск на карте это подтвердит или опровергнет
за секунды; если опровергнет — дописать сюда.

Вторым файлом тянется LoRA: FaceID — это адаптер **плюс** LoRA
(`gpu_keyframes.IP_ADAPTER_LORA`), без неё лицо обусловлено наполовину.

**Про установку — почему теперь `-r requirements-gpu.txt`, а не список пакетов.**
Здесь был свой список, и он разошёлся с файлом зависимостей по двум пунктам
сразу: индекс `cu121` вместо `cu126` и отсутствие `peft`. Второе критично —
без `peft` не работает `load_lora_weights`, то есть та самая LoRA FaceID, о
которой абзацем выше. В самом `requirements-gpu.txt` это записано комментарием у
строки.

Расхождение при этом никуда не делось, просто теперь оно в одном месте, а не в
трёх: **`gpu_keyframes.requirements()` по-прежнему печатает `cu121` и не
упоминает `peft`.** Это код, и правится он не отсюда; при следующей правке
модуля привести к `requirements-gpu.txt`. `onnxruntime-gpu` из старого списка
тоже ушёл сознательно: insightface здесь распознаёт лицо на десятке картинок за
прогон, на CPU это доли секунды, а gpu-сборка под Windows тянет возню с cuDNN.

Проверить, что torch действительно видит карту, **до** всего остального:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`False` здесь — это неверная сборка torch под драйвер. Чинить сразу, дальше
всё равно ничего не поедет.

---

## 3. Предполёт (одна команда, единицы секунд)

```bash
python3 -m ball_reel.preflight_gpu --conditions conditions --face face.jpg --vram 4
```

Здесь стояло «~2 минуты». Реальные цены замерены и записаны в самом модуле
(`preflight_gpu.COSTS`): вся миллисекундная группа — единицы миллисекунд,
условия на 71 png — 275 мс, импорт `torch` — 1.2–2.4 с (дороже всего, что выше,
вместе), лицо — 5.2 с. Итого без сети около десяти секунд; дольше только строка
`шлюз`, у которой таймаут до 30 с. Числа в этой таблице не украшение: порядок
проверок обязан следовать из них, и при расхождении правится порядок, а не
таблица.

Проверки сгруппированы **по цене**, и группа выполняется ЦЕЛИКОМ, а решение
принимается после неё (`preflight_gpu.build_checks`). Останавливаться на первом
же отказе внутри миллисекундной группы значило бы показать одну беду из пяти и
заставить перезапускать предполёт пять раз подряд; дорогое при этом не
начинается, пока дешёвое красное.

| группа | проверки |
|---|---|
| миллисекунды | пакеты, диск, ffmpeg, веса, модель позы, веса лица, сегментация, DWPose |
| доли секунды | драйвер (`nvidia-smi`), условия, driving-кадры |
| секунды | torch, драйвер/сборка, vram, onnxruntime |
| секунды дорогие | лицо |
| сеть | шлюз (отпадает при `--skip-gateway`) |

Аргументы: `--conditions`, `--face`, `--out` (место проверяется на ТОМ диске,
куда лягут кадры), `--vram` (по нему считается план, а по плану судится размер
условий), `--dwpose` (условия будут сниматься здесь же — тогда веса DWPose
обязательны, а не опциональны), `--skip-gateway`.

Исходов три, а не два: `PASS` / `FAIL` / **`НЕПРОВЕРЕНО`**. Последнее — не «в
порядке». Отдельно ловится случай, когда сломалась сама проверка: он так и
печатается — «САМА ПРОВЕРКА сломалась, это дефект предполёта, а не машины».

**Пока предполёт не даст OK, генерацию не запускать** — каждая последующая
ошибка стоит дороже.

(Раньше здесь было «карта, VRAM, torch, веса, условия, лицо, свободный диск» —
и порядок неверный, и половина проверок пропущена. Список выше снят с кода
2026-08-14; модуль в тот момент активно переписывался, так что при расхождении
читать `preflight_gpu.build_checks`, а не этот абзац.)

---

## 4. Дым: ОДИН кейфрейм

```bash
python3 - <<'EOF'
from ball_reel.gpu_keyframes import plan, render_keyframes
import glob
cond = sorted(glob.glob("conditions/*.png"))[:1]
r = render_keyframes(cond, "face.jpg",
                     "a woman exercising on a fitness ball in a bright studio, "
                     "sportswear, natural light, photographic",
                     "kf_smoke", cfg=plan(vram_gb=4.0),
                     negative="blurry, deformed, extra limbs, watermark, text")
print(r["count"], r["keyframes"])
EOF
```

Сразу измерить, а не смотреть глазами:

```bash
python3 - <<'EOF'
from ball_reel.identity_arcface import arcface_drift, START_MIN_FACE_PX
from ball_reel.pose import pose_delta, landmarks
import glob, json, pathlib
kf = sorted(glob.glob("kf_smoke/*.png"))[0]
cond = pathlib.Path(sorted(glob.glob("conditions/*.png"))[0])
# Позу сверяем с DRIVING-КАДРОМ, а не с условием: условие — это палки на
# чёрном фоне, детектор поз на нём не находит ничего, и сравнение молча
# превращается в None. Карту "условие -> driving-кадр" пишет render_sequence.
driving = json.loads(pathlib.Path("conditions/manifest.json").read_text())["driving_frames"][cond.stem]
print("identity:", arcface_drift([kf], "face.jpg", min_face_px=START_MIN_FACE_PX)["median"])
print("pose vs driving:", pose_delta(landmarks(driving), landmarks(kf)))
EOF
```

Целевые значения на этом шаге:

| метрика | норма | откуда бар | если хуже |
|---|---|---|---|
| identity median | ≤ 0.35 | `identity_arcface.SAME_PERSON_MAX` | поднять `ip_adapter_scale` до 0.85 |
| pose mean | ≤ 0.15 | `pose.SAME_POSE_MAX` | поднять `controlnet_scale` до 1.2 |
| pose worst joint | ≤ 0.40 | `pose.WORST_JOINT_MAX` | то же |
| время кадра | 15–40 c | РАСЧЁТ, не замер | если минуты — offload свопит, снизить до 448×640 |

Здесь стояло «pose mean ≤ 0.25». Это бар шлюзового кейфрейма
(`chain.KEYFRAME_POSE_MAX = 0.25`), а не старт-кадра: старт судится строго,
потому что ещё ничто не двигалось, и всё сверх 0.15 — это уже переосмысление
позы генератором, а не движение субъекта. Ровно эту же подмену чинили в
`run_local.smoke_verdict`, где два числа стояли литералами в теле `main` и один
из них молча пропускал расхождение, которое шлюзовой путь забраковал бы.

**Не запускать полный прогон, пока дым не в норме.** Один кадр стоит секунды,
пять неудачных — минуты, а отладка «почему всё не то» после полного прогона —
десятки минут.

---

## 5. Полный прогон

```bash
python3 - <<'EOF'
import glob, json
from ball_reel.gpu_keyframes import plan, render_keyframes
from ball_reel.chain import generate_chain, Keyframe
from ball_reel import pollinations

cond = sorted(glob.glob("conditions/*.png"))[::6]        # каждый 6-й: 12fps -> 2 узла/с
r = render_keyframes(cond, "face.jpg", "…тот же промт…", "kf", cfg=plan(vram_gb=4.0))

kfs = []
for i, p in enumerate(r["keyframes"]):
    k = Keyframe(index=i, t=float(i), driving_frame=cond[i], rendered=p)
    k.url = pollinations.upload(p); k.accepted = True
    kfs.append(k)

res = generate_chain(kfs, "…тот же промт…", "chain_out",
                     model="wan-fast", seconds_per_segment=2, loop=True)
print(res.note, res.clip_path)
json.dump(res.to_dict(), open("chain_out/report.json","w"), indent=2)
EOF
```

Затем гейт — он уже готов и ничего не знает про то, какой стек рисовал:

```bash
python3 - <<'EOF'
from ball_reel import pollinations
from ball_reel.identity_arcface import arcface_drift
from ball_reel.motion import loop_seam, motion_quality
from ball_reel.pose import limb_consistency
fs = pollinations.extract_frames("chain_out/chain.mp4", "chain_out/frames", fps=6)
print("identity:", arcface_drift(fs, "face.jpg")["note"])
print("loop    :", loop_seam(fs)["note"])
print("motion  :", motion_quality(fs)["note"])
print("anatomy :", limb_consistency(fs)["note"])
EOF
```

---

## 6. Сколько это стоит

| статья | оценка |
|---|---|
| установка + веса | 15–30 мин оплаченного времени, **один раз** |
| кейфрейм | 15–40 c на 4 ГБ |
| 16 кейфреймов | ~10 мин |
| интерполяция (`wan-fast`, 16 отрезков × 2 c) | **0.32 pollen** |
| тот же прогон на `seedance-2.0` | 5.76 pollen (в 18 раз) |

Отладку гонять на `wan-fast`, на `seedance-2.0` переходить только когда
связка сошлась. Цифры времени — оценка, не замер: модуль ни разу не исполнялся.

---

## 7. Типовые отказы

| симптом | причина | что делать |
|---|---|---|
| `CUDA out of memory` | offload включён после slicing | порядок: slicing → tiling → offload |
| то же на первом кадре | карта меньше заявленной | 448×640, либо `steps` до 18 |
| лицо чужое | IP-Adapter не загрузился | модуль бросает явную ошибку — читать её, не игнорировать |
| поза не совпала | `controlnet_scale` низкий | 1.0 → 1.2; если не помогло, проверить, что условия не растянуты |
| тело донора, не клиента | условия без ретаргетинга | `render_sequence(..., proportions=...)` |
| клип не зациклился | модель без `end_frame` | только `wan-fast`, `veo`, `wan-pro`, `seedance-2.0` |
| гейт «not verifiable» | лицо в выдаче < 100 px | ближе кадр, либо выше разрешение — **не** апскейл перед гейтом |

---

## 8. Перед выключением

Артефакты живут на инстансе и умрут вместе с ним:

```bash
tar czf result.tar.gz chain_out/ kf/ *.json
# скачать к себе, ТОЛЬКО потом гасить
```

Забрать обязательно: `chain_out/chain.mp4`, `report.json`, и замеры гейта —
без них прогон не воспроизводим и как результат не считается.

---

## Что тут непроверено

`gpu_keyframes` написан по документированному API diffusers и **ни разу не
исполнялся** — в среде разработки не было GPU. Оценки памяти и времени
рассчитаны, а не замерены. Первый запуск и есть его тест; расхождения ожидаемы,
и их надо дописать сюда, как дописывались расхождения по Pollinations в
`POLLINATIONS_CONTRACT.md`.

Всё остальное — `intake`, `metrics`, `driving`, `pose`, `motion`, `skeleton`,
гейт — проверено на живых данных и офлайн-тестами: **801 тест, 56 мутаций**
(`python3 -m ball_reel.codeaudit`, прогон 2026-08-14). Здесь стояло «152
офлайн-теста» — число из более ранней сессии. Считать его надо прогоном, а не
чтением: набор растёт, документ — нет.

С тех пор появился второй, ЦЕЛЕВОЙ локальный движок — `animate.py` (AnimateDiff
+ ControlNet + FaceID, один проход, без сети) и `run_local.py` как единая
команда над обоими. У него та же оговорка, что у `gpu_keyframes`: карты в среде
разработки не было, числа памяти и времени — расчёт. Проверяется он первым
запуском, и расхождения дописывать сюда.
