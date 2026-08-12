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
                    proportions=prop)
print("условий:", len(m["conditions"]), "coverage:", m["coverage"], m["warnings"])
EOF

tar czf payload.tar.gz conditions/ face.jpg     # это поедет на VPS
```

Проверить глазами **два** условия-скелета (первый и средний). Кривой скелет
кондиционирует хуже, чем никакой, а увидеть это на арендованной машине — значит
заплатить за просмотр.

Не арендовать, если:

* `intake` сказал `cannot identity` — генерация не спасёт, нужно другое фото;
* `coverage` условий заметно ниже 1.0 — в этих кадрах генератор не ограничен;
* в `spec.validate()` есть «motion track has holes».

---

## 1. Выбор инстанса

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
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors \
            insightface onnxruntime-gpu mediapipe pillow numpy requests

# веса — в фоне, пока идёт остальная настройка
export HF_HOME=/workspace/hf
python3 -c "
from huggingface_hub import snapshot_download as d
d('runwayml/stable-diffusion-v1-5', allow_patterns=['*.json','*fp16.safetensors','*.txt'])
d('lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.json','*.safetensors'])
d('h94/IP-Adapter', allow_patterns=['models/ip-adapter-faceid_sd15.bin'])
" &

mkdir -p ~/.mediapipe && curl -sSL -o ~/.mediapipe/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
tar xzf payload.tar.gz
```

Проверить, что torch действительно видит карту, **до** всего остального:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`False` здесь — это неверная сборка torch под драйвер. Чинить сразу, дальше
всё равно ничего не поедет.

---

## 3. Предполёт (одна команда, ~2 минуты)

```bash
python3 -m ball_reel.preflight_gpu --conditions conditions --face face.jpg
```

Печатает построчно и валится на первой же проблеме: карта, VRAM, torch, веса,
условия, лицо, свободный диск. **Пока он не даст OK, генерацию не запускать** —
каждая последующая ошибка стоит дороже.

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
import glob
kf = sorted(glob.glob("kf_smoke/*.png"))[0]
cond = sorted(glob.glob("conditions/*.png"))[0]
print("identity:", arcface_drift([kf], "face.jpg", min_face_px=START_MIN_FACE_PX)["median"])
print("pose vs условие:", pose_delta(landmarks(cond), landmarks(kf)))
EOF
```

Целевые значения на этом шаге:

| метрика | норма | если хуже |
|---|---|---|
| identity median | ≤ 0.35 | поднять `ip_adapter_scale` до 0.85 |
| pose mean | ≤ 0.25 | поднять `controlnet_scale` до 1.2 |
| pose worst joint | ≤ 0.40 | то же |
| время кадра | 15–40 c | если минуты — offload свопит, снизить до 448×640 |

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
гейт — проверено на живых данных и 108 офлайн-тестах.
