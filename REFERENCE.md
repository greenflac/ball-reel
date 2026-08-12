# ball_reel — справочная памятка

Собрано чтением кода (`ball_reel/ball_reel/*.py`) и документов `README.md`,
`POLLINATIONS_CONTRACT.md`, `MVP_RUNBOOK.md`, `GPU_BRANCH.md`, `GPU_RUNBOOK.md`.
Ничего не додумано: где в коде нет — написано «нет».
Версия пакета: `__version__ = "0.1.0"`. Составлено 2026-08-12 по состоянию
рабочего дерева (часть изменений на тот момент была не закоммичена).

Суть проекта в одну фразу (из докстрингов): фото лица + текстовый бриф →
короткий вертикальный клип «человек прыгает на фитболе» → локальный гейт
(ArcFace / поза / анатомия / луп / движение) → ретрай, пока не сойдётся.
Вся генерация — HTTP к одному шлюзу Pollinations; все проверки — локальные.

---

## 1. Карта модулей

Все файлы в `ball_reel/ball_reel/`. «GPU» = нужна карта, «сеть» = HTTP-вызовы,
«веса» = скачиваемые файлы моделей.

| модуль | за что отвечает | зависит от | GPU / сеть / веса |
|---|---|---|---|
| `__init__.py` | Докстринг пакета + `__version__`. Кода нет. | — | нет |
| `__main__.py` | Точка входа офлайн-реплея: `run()` + `render()`, флаг `--live`. | `pipeline` | нет |
| `brief.py` | Датакласс `Brief` (лицо, subject, оверлей, 1080x1920, 4 c, 9:16), `DEMO_BRIEF`, детерминированный `seed_for` (md5). | stdlib | нет |
| `pipeline.py` | Оркестрация офлайн-прогона: 3 стратегии → старт-кадр → «видео» → критик → ранжирование; печать таблицы. | `brief`, `critic`, `gen` | нет (при `live=True` — сеть/веса) |
| `gen.py` | Шлюз-абстракция трёх стадий (start_frame / video / voice): офлайн из фикстур, live делегирует в `live_gen`. Здесь же `STRATEGIES` (3 стратегии). | `brief`, `live_gen` (лениво) | live: GPU+сеть |
| `critic.py` | Порог `Bar` (3 отдельных гейта) и `score`/`rank`. Разделяет «вычисленное» и «мнение». | `identity` | нет |
| `identity.py` | Офлайн-прокси идентичности: dhash-совпадение с референсом + `motion_presence`. Плюс шов `arcface_drift`. | Pillow | нет |
| `identity_arcface.py` | Настоящий инструмент идентичности: ArcFace-эмбеддинги, косинусная дистанция, медиана/p90/coverage; сюда же `face_attributes` (пол/возраст). | insightface, onnxruntime, numpy, Pillow | сеть+веса (`buffalo_l` ~281–300 МБ) |
| `pose.py` | Поза телом: 33 точки MediaPipe, нормировка по торсу, `pose_drift`, `limb_consistency`, 3D world-пропорции. | mediapipe, numpy, Pillow | веса `pose_landmarker_lite.task` (5.5 МБ) |
| `motion.py` | Качество движения: шов лупа, разрывы/морфинг, поиск точки обрезки, `trim_to_loop` (ffmpeg). Плюс тексты промтов `PHYSICAL_MOTION`, `LOOP_MOTION`. | numpy, Pillow, ffmpeg | нет |
| `metrics.py` | Что даёт ОДНО фото: `face_metrics` (identity/geometry/estimated), `body_metrics` (3D-пропорции), `subject_package` + `missing`. | `identity_arcface`, `pose`, mediapipe | веса (buffalo_l + face_landmarker) |
| `intake.py` | Пригодность фото по капабилити (identity / face_geometry / expression / build) с указанием, что прислать вместо. Локально и бесплатно. | `metrics` | веса (через metrics) |
| `driving.py` | Драйвинг-видео → `DrivingSpec`: покадровый трек позы и мимики, амплитуда, каденс, скорость + `validate()`. | ffmpeg, mediapipe, numpy, `pose` | веса (pose + face_landmarker) |
| `skeleton.py` | Трек позы → COCO-18 OpenPose-скелеты (условия для ControlNet) с обязательным ретаргетингом под пропорции клиента. | `pose`, mediapipe, Pillow, numpy | веса (pose) |
| `subject.py` | Спека человека помимо лица (пол/возраст/телосложение/одежда/поза) + правило «текст для перерисовываемого, референс-картинка для остального»; выбор модели старт-кадра. | `identity_arcface` (для `from_photo`) | веса (только `from_photo`) |
| `router.py` | Выбор стека (pollinations / gpu) по требуемым капабилити и измеренным входам; блокеры до траты денег; оценка стоимости в pollen. | `chain` (список моделей) | нет |
| `chain.py` | Цепочка пиннинга движения: выбор кейфреймов по экстремумам, рендер кейфрейма в позе донора с проверкой, сборка сегментов `start|end` через ffmpeg. | `pollinations`, `pose`, `identity*` | сеть (платно), ffmpeg |
| `pollinations.py` | Клиент единственного шлюза: upload, image, images_edit, compose, video, video_loop, extract_frames, chat/judge, tts. | requests, ffmpeg | сеть (платно) |
| `produce.py` | Боевой пайплайн: старт-кадр → ранний отсев → upload → видео → гейт → ретрай; отчёт `produce_report.json`. CLI. | `pollinations`, `motion`, `pose`, `subject`, `identity*` | сеть (платно) |
| `mvp.py` | Урезанный прогон: кадры (свои или сгенерированные) + РЕАЛЬНЫЙ VLM-судья → тот же критик; пишет `mvp_report.md`. | `pollinations`, `critic`, `gen` | сеть (платно) |
| `doctor.py` | Предполёт по эндпоинтам Pollinations в порядке возрастания стоимости; видео — только с `--video`. | `pollinations`, requests | сеть (дёшево; `--video` платно) |
| `codeaudit.py` | Аудит самих тестов: прогон, покрытие несущих модулей и МУТАЦИОННАЯ проверка — каждый порог по очереди ломается в копии исходника, тесты обязаны покраснеть. Ничего не генерирует, в сеть не ходит. | unittest, coverage | нет |
| `preflight_gpu.py` | Предполёт на арендованной GPU-машине: torch/CUDA, VRAM, диск, кэш весов, модель позы, условия, лицо, живость ключа. Ничего не генерирует. Единственный модуль с русскими сообщениями. | torch, `pose`, `intake`, `pollinations` | GPU (проверяет), сеть (только `/account/key`) |
| `gpu_keyframes.py` | Кейфреймы на 4 ГБ VRAM: SD1.5 + ControlNet OpenPose + IP-Adapter FaceID. `plan()` инспектируется на CPU, `render_keyframes()` — только на карте. | torch, diffusers, Pillow | GPU + веса (~7 ГБ) |
| `live_gen.py` | Альтернатива шлюзу на своём железе: SDXL + IP-Adapter FaceID для старт-кадра, произвольный video-API, TTS+липсинк. Всё за env-переменными. | torch, diffusers, insightface, requests | GPU + сеть + веса |
| `tools/make_fixtures.py` | Рисует синтетические офлайн-фикстуры Pillow'ом (3 стратегии = 3 режима отказа) и хэнд-написанные вердикты судьи. | Pillow | нет |
| `tests/` (11 файлов: `test_arcface_math`, `test_chain`, `test_condition_render`, `test_driving`, `test_identity_gate`, `test_intake`, `test_motion_subject`, `test_pose`, `test_router`, `test_skeleton`, `test_verdict`) | Арифметика и вердикты без моделей и сети: синтетические скелеты, эмбеддинги, кадры; модельные швы заглушены. **152 теста, проходят за 0.3 c**. | unittest | нет |

---

## 2. Точки входа

Всё, что запускается как `python3 -m ball_reel.<X>` (у всех есть `if __name__ == "__main__"`).

### `python3 -m ball_reel` (`__main__.py`)
Офлайн-реплей на фикстурах: 3 стратегии → критик → ранжированная таблица.
Argparse НЕТ, аргумент проверяется как `"--live" in argv`.

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--live` | флаг | выкл | передаёт `live=True` в `pipeline.run` → `Gateway` пойдёт в `live_gen` (нужны GPU/веса/`VIDEO_API_URL`) |

Деньги: **нет** (без `--live`). Без фикстур печатает подсказку и возвращает 2.

### `python3 -m ball_reel.tools.make_fixtures`
Аргументов нет. Рисует `fixtures/{face_ref.png,start/,video/,judge/,index.json}`. Деньги: нет.

### `python3 -m ball_reel.doctor`
Проверка эндпоинтов по возрастанию цены: models (free) → image → images_edit → upload → tts → video.
Argparse НЕТ, проверяется `"--video" in argv`.

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--video` | флаг | выкл | добавляет один короткий клип — **тратит баланс** |

Env: `BALL_REEL_DOCTOR_OUT` (`doctor_out`), `BALL_REEL_VIDEO_MODEL` (`seedance-2.0`), `BALL_REEL_VIDEO_DURATION` (`4`).
Деньги: image/edit/upload/tts — «дёшево», video — платно. Возврат 0/1, 2 если нет ключа.

### `python3 -m ball_reel.mvp`
Прогон критика на реальных кадрах с реальным VLM-судьёй. Argparse НЕТ, флаги по вхождению в argv.

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--generate` | флаг | выкл | сгенерировать по 4 кадра на стратегию в `mvp_input/<sid>/NN.png` и выйти — **платно** |
| `--arcface` | флаг | выкл | идентичность настоящим ArcFace вместо dhash-прокси |
| `--no-judge` | флаг | выкл | не звать VLM, взять синтетический вердикт из фикстур |

Env: `POLLINATIONS_IMAGE_MODEL` (`flux`), `POLLINATIONS_JUDGE_MODEL` (`openai`).
Деньги: да (генерация кадров и/или судья), без `--generate --no-judge` — только судья. Пишет `ball_reel/mvp_report.md`.

### `python3 -m ball_reel.produce` — основной боевой вход (argparse есть)

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--face` | str, **обязателен** | — | фото лица |
| `--attempts` | int | 4 | число попыток ретрай-цикла |
| `--start-model` | str | `kontext` | модель старт-кадра (перекрывается `subject.start_model()`, если есть body/pose-референс) |
| `--video-model` | str | `seedance-2.0` | модель image-to-video; в help: seedance иногда отказывает, `wan/veo/happyhorse-1.1` приняли те же входы |
| `--out` | str | `produce_out` | каталог вывода |
| `--no-loop` | флаг | выкл | не просить и не гейтить бесшовный луп |
| `--body-ref` | str | `""` | фото, откуда взять ТЕЛОСЛОЖЕНИЕ и одежду (переключает на мульти-референс модель) |
| `--pose-ref` | str | `""` | фото, откуда взять ПОЗУ (лицо всё равно только из `--face`) |
| `--gender --age --build --hair --outfit --footwear --posture` | str | `""` | текстовые поля `Subject` |
| `--from-photo` | флаг | выкл | дозаполнить пол/возраст локальным оценщиком по самому фото |

Деньги: **да** — каждая попытка это kontext-стилл (~7 c) + upload + видео (seedance 4 c ≈ 0.72 pollen).
Код возврата 0, если `passed`, иначе 1. Пишет `<out>/produce_report.json`.

### `python3 -m ball_reel.preflight_gpu` (argparse есть)

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--conditions` | str | `conditions` | каталог с png-условиями (проверяются наличие, непустота, одинаковый размер) |
| `--face` | str | `face.jpg` | фото, прогоняется через `intake.inspect` |
| `--skip-gateway` | флаг | выкл | не проверять живость ключа Pollinations |

Деньги: нет (единственный сетевой вызов — `GET /account/key`). Падает на первой же проблеме, код 1.

### `python3 -m ball_reel.codeaudit` (argparse есть)

| аргумент | тип | по умолчанию | смысл |
|---|---|---|---|
| `--quick` | флаг | выкл | только тесты и покрытие, без мутаций |

Что делает: прогоняет `unittest discover`; если красно — останавливается (аудит на красных тестах бессмысленен);
считает покрытие по `CORE_MODULES` с порогом 70%; затем по очереди портит 15 порогов в КОПИИ пакета
(`identity_arcface`, `pose`, `motion`, `intake`, `chain`, `router`) и требует, чтобы тесты покраснели.
Выжившая мутация = порог, который никто не сторожит. Деньги: **нет**, сети нет.

Модулей с `def main`, но без `-m`-запуска, нет. У `pipeline.py` `main` отсутствует — он вызывается из `__main__.py`.

---

## 3. Ключевые функции по слоям

### 3.1 Извлечение (что дают вход и драйвинг-видео)

**`driving.py`**
- `extract(video_path, *, fps=12, start=0.0, length=None, work_dir=None, face_model=None, keep_track=True) -> DrivingSpec` — сэмплит кадры ffmpeg'ом, считает позу и блендшейпы, заполняет `motion`/`expression`/`pose_coverage`/`track`/`warnings`.
- `DrivingSpec.validate() -> list[str]` — проблемы (кадров <6, `pose_coverage<0.8`, нет каденса/амплитуды, `expression.coverage<0.5`), возвращает, а не бросает.
- `DrivingSpec.to_dict(with_track=True)` / `save(path, with_track=True)`.
- `to_prompt(spec) -> str` — лоссовая словесная форма («about 0.6 bounces per second…»); неизмеренное просто опускается.
- `MotionSummary`: `amplitude`, `cadence_hz`, `velocity`, `peak_velocity` (95-й перцентиль, не max), `pose_range`.
- Приватные, но нужны по рунбуку: `_sample_frames(...)->list[str]`, `_segment_scale(torsos)`, `_cadence(series, fps)`, `_hip_raw(points)`.
- **`survey()` упомянут в докстринге `extract`, но в коде его НЕТ.**

**`metrics.py`**
- `face_metrics(photo, *, face_model=None) -> dict` — ключи: `photo`, `identity{embedding_dim, face_px, detector_score}`, `geometry{face_width_to_height, eye_spacing_to_face_width, left_eye_width_to_face_width, mouth_width_to_face_width, nose_to_chin_over_face_height, eye_to_mouth_over_face_height}`, `estimated{sex, age}`, `expression{count, top, all}`, `notes[]`. Лицо кропится по bbox ArcFace и увеличивается до 512 px, иначе меш не находит лицо в полноростовом кадре.
- `body_metrics(photo) -> dict` — `proportions` (3D, в торсах), `projected` (2D, только про этот кадр), `turned`, `reliable`, `notes[]`.
- `subject_package(photo, **kw) -> {"face", "body", "missing"}` — `missing` перечисляет, чего фото не задаёт.

**`pose.py`** (измерительная часть)
- `landmarks(path) -> {name: (x, y, visibility)} | None` — 12 точек тела в нормированных координатах кадра.
- `world_landmarks(path) -> {name: (x, y, z, visibility)} | None` — метрические 3D относительно центра бёдер.
- `world_proportions(path) -> dict | None` — длины сегментов / ширина плеч / бёдер / `shoulder_to_hip` / `leg_length` в торсах + `torso_metres`.
- `pose_delta(a, b) -> {mean, worst, worst_joint, joints} | None`; `pose_distance(a, b) -> float | None` (только `mean`).
- `_normalise(points)` — центрирование на бёдрах и деление на торс (то, что делает «снято ближе» ≠ «другая поза»).

**`skeleton.py`**
- `pose_points(path) -> dict | None` — COCO-18 + синтезированные `neck`, `hip_c` + служебный `__size__` (исходный размер кадра, чтобы не растянуть фигуру при рисовании).
- `retarget(points, proportions, *, min_visibility=0.5) -> dict` — направления от донора, длины от клиента, обход от бёдер наружу.
- `draw(points, out_path, *, width=512, height=768, min_visibility=0.5, line_width=4) -> str` — OpenPose-скелет на чёрном фоне цветами конвенции.
- `render_sequence(frames, out_dir, *, proportions=None, width=512, height=768, from_mediapipe=True) -> {conditions, missing_frames, coverage, size, retargeted, source, warnings}`.

### 3.2 Решение (что можно обещать и куда идти)

**`intake.py`**
- `inspect(photo) -> Intake` — поля `supports[]`, `assumed[]`, `blocked{cap: как починить}`, `warnings[]`, `measurements{face_px, detector_score, expression_count, build_source, shoulder_to_hip}`; методы `can(cap)`, `usable` (== есть identity), `render()`.
- `report(photos: list) -> str` — какое фото для какой капабилити лучше; измеренное бьёт предположенное.

**`router.py`**
- `choose(required=("identity",), *, subject=None, spec=None, seconds=4, video_model="wan-fast", loop=True) -> Decision` — `Decision{stack, reasons[], blockers[], warnings[], estimated_pollen}`, `.runnable` (== нет блокеров), `.render()`. Неизвестная капабилити → `ValueError`.
- Логика: `pose_trajectory`/`object_fidelity` → `gpu`; иначе `pollinations` + смета; `loop=True` с моделью вне `END_FRAME_MODELS` → блокер; нет лица в `subject` → блокер; лицо <100 px → предупреждение; `pose_trajectory` без спеки или при `pose_coverage<0.9` → блокер.
- `_pollen(model, seconds)` — прайс: `wan-fast 0.01`, `seedance-pro 0.025`, `veo 0.08`, `wan/wan-pro 0.1`, `grok-imagine-video-1.5 0.14`, `seedance-2.0 0.18` за секунду видео.

**`subject.py`**
- `Subject(gender, age, build, hair, skin, outfit, footwear, posture, extra, body_ref, pose_ref)` — пустое поле значит «не задано» и в промт не попадает.
- `.to_prompt() -> str` — фиксированный порядок, чтобы два прогона отличались только там, где отличается субъект.
- `.specified -> tuple[str,...]`, `.reference_roles -> ((path|"__face__", role), ...)` (лицо всегда первое), `.reference_clause() -> str` («Use the FIRST image for…»), `.start_model(default="kontext") -> str`.
- `Subject.from_photo(face_photo, **overrides)` — заполняет только пол и возрастную полосу локальным оценщиком; `overrides` побеждают.
- `UNSPECIFIED = Subject()`.

**`brief.py`**: `Brief.seed_for(*parts) -> int` (md5, детерминированно), `resolve_face_ref(brief, root) -> str`.

### 3.3 Генерация

**`pollinations.py`** (всё возвращает путь к файлу, если не сказано иное)
- `upload(path) -> str` — публичный media-URL (нужен, потому что видео-эндпоинт фетчит картинку со своей стороны).
- `image(prompt, out_path, *, model="flux", seed=0, width=1080, height=1920, image_url=None) -> str`.
- `images_edit(prompt, ref_path, out_path, *, model="kontext", width=1080, height=1920) -> str` — мультипарт, локальный файл, без media-хоста; предпочтительный путь для старт-кадра.
- `compose(prompt, image_urls, out_path, *, model="nanobanana", width=768, height=1024, seed=0) -> str` — 2+ референса, роли по позиции в промте; <2 URL → `ValueError`.
- `video(prompt, out_mp4, *, model="seedance-2.0", image_url=None|str|list, duration=4, aspect_ratio="9:16", audio=False, resolution=None, seed=None) -> str` — список из двух URL = start|end кадры; после вызова заполняет `LAST_VIDEO_USAGE` из заголовков.
- `video_loop(prompt, out_mp4, start_url, **kwargs)` — один и тот же кадр как оба ключевых.
- `extract_frames(mp4, out_dir, *, fps=6) -> list[str]` (ffmpeg).
- `chat(messages, *, model="openai", temperature=0.0) -> str`; `judge_frame(frame, *, model="claude") -> dict` (ключи `first_frame_hook, trend_fit, composition, brand_safety, transcribed_text, opinion`); `opinion_of(frame, *, model="claude") -> float`.
- `tts(text, out_path, *, voice="nova", model="eleven-multilingual-v2") -> str`.

**`chain.py`**
- `pick_keyframe_times(track, count=5) -> list[int]` — индексы по локальным экстремумам высоты бедра (равномерная сетка только добивает нехватку); первый кадр всегда внутри, чтобы цепочка замкнулась.
- `render_keyframes(driving_frames, indices, face_photo, *, out_dir, body_ref="", subject_text="", scene="", model="nanobanana", attempts=2) -> list[Keyframe]` — рендерит человека в позе донора и СРАЗУ проверяет позу и идентичность; принятые кейфреймы заливает и получает `url`.
- `generate_chain(keyframes, prompt, out_dir, *, model="wan-fast", seconds_per_segment=2, loop=True, aspect_ratio="9:16") -> ChainResult{keyframes, segments, clip_path, note}` — по клипу на пару соседних узлов, склейка ffmpeg с ПЕРЕКОДИРОВАНИЕМ; модель вне `END_FRAME_MODELS` → `ValueError`.

**`produce.py`**
- `produce(face_photo, brief=DEMO_BRIEF, *, strategy=None, subject=UNSPECIFIED, loop=True, attempts=4, max_identity_drift=None, min_motion=0.02, start_model="kontext", video_model="seedance-2.0", out_dir="produce_out") -> Result` — `Result{face_photo, passed, clip_path, clip_frames, start_frame, attempts[], note}`; при неудаче возвращает лучшую попытку с `passed=False`.
- `Attempt`: `n, strategy_id, start_frame, video_frames, worst_identity_drift` (на деле МЕДИАНА), `motion, passed, reason, clip_path, identity{}, loop_ratio, seamless, worst_jump, pose_distance, limb_wobble`.
- `render(res) -> str` — таблица попыток; `_identity_summary(drift)`, `_short(e)` (распознаёт `content_policy_violation`).

**`gpu_keyframes.py`**
- `plan(vram_gb=4.0, keyframes=5) -> GPUPlan` — конфиг, инспектируемый БЕЗ карты; при `vram_gb>=8` поднимает до 640x960 и убирает часть экономий; всегда добавляет note «UNVERIFIED».
- `render_keyframes(condition_images, face_photo, prompt, out_dir, *, cfg=None, negative="", seed=0) -> {keyframes[], config, count, source="gpu"}` — тот же формат манифеста, что у `chain.render_keyframes`. Порядок обязателен: slicing → tiling → offload. Если IP-Adapter не загрузился — `RuntimeError` (иначе получишь чужое лицо в правильной позе).
- `requirements() -> str` — pip-список для 4 ГБ.

**`gen.py`** (офлайн-шлюз): `Gateway(root, live=False)`, `.start_frame(brief, strategy) -> Frame`, `.video(brief, strategy, start) -> list[str]`, `.voice(brief, start, script_ru="") -> str|None` (офлайн всегда `None`).
**`live_gen.py`**: `start_frame_live(prompt, face_ref, seed, width, height, out_path, *, base_model="stabilityai/stable-diffusion-xl-base-1.0", lora_path=None, lora_scale=0.8, ip_adapter_scale=0.6) -> str`; `video_live(start_frame, prompt, out_dir, *, duration_s=5, fps=24) -> list[str]`; `voice_live(start_frame, script_ru, out_path) -> str`.
**`mvp.py`**: `prompts_for(brief) -> {sid: prompt}`, `generate(brief, *, n=4)`, `run(brief, *, use_arcface=False, real_judge=True, bar=DEFAULT_BAR) -> {ranked, accepted, considered, judged_by, identity}`, `render(result) -> str`.

### 3.4 Проверка и гейт

**`identity_arcface.py`**
- `cosine_distance(a, b) -> float` — `1 - cos`, чистый numpy, юнит-тестируется без весов.
- `face_detail(path) -> {embedding, face_px, det_score, bbox, sex?, age?} | None` — самое крупное лицо.
- `face_embedding(path)`, `face_attributes(path) -> {sex, age, face_px} | None`.
- `arcface_drift(frame_paths, reference_path, *, min_face_px=100) -> dict` — ключи: `per_frame`, `face_px`, `worst=(имя, значение)`, `drifted[]`, `readable`, `judgeable`, `too_small[]`, `no_face[]`, `median`, `p90`, `coverage`, `note`. Кадры с мелким лицом не считаются максимальным дрифтом, а уходят в `too_small` — «не проверяемо» ≠ «другой человек».
- `_quantile(sorted_vals, q)` — линейная интерполяция, без scipy.

**`identity.py`** (офлайн-прокси)
- `identity_drift(frame_paths, reference_path) -> {per_frame, worst, drifted, readable, note}` — `1 - совпадение dhash`; note всегда называет прокси.
- `motion_presence(frame_paths) -> {motion, moving, note}` — средняя межкадровая разница яркости 64x114, нормированная к 0..1. Только присутствие движения, не правдоподобие.
- `arcface_drift(frame_paths, reference_path, **kwargs)` — тонкий шов в `identity_arcface`, при отсутствии зависимостей падает, а не откатывается на прокси.

**`motion.py`**
- `loop_seam(frames) -> {ratio, seam, typical_step, seamless, note}` — стык первый↔последний, делённый на МЕДИАННЫЙ межкадровый шаг (<3 кадров → `ratio=None`).
- `motion_quality(frames) -> {worst_jump, jumps[], moving, smooth, activity, note}`.
- `best_loop_cut(frames, *, min_keep=0.5) -> {cut_at, ratio, seamless, kept_fraction, note}` — где обрезать, ноль генерации.
- `trim_to_loop(mp4_path, frames, out_mp4, *, fps) -> {…cut, out, duration}` — фактическая обрезка ffmpeg.

**`pose.py`** (гейт)
- `pose_drift(frame_paths, reference_path, *, max_pose_distance=0.15, max_worst_joint=0.40) -> {per_frame, median, worst_joint, worst, coverage, measured, frames, held, note}` — держится, только если И медиана, И худший сустав в пределах.
- `limb_consistency(frame_paths, *, max_wobble=0.25) -> {wobble{}, worst, anatomical, measured, unstable[], note}` — коэффициент вариации длины конечности в торсах.

**`critic.py`**
- `score(root, strategy_id, strategy_label, video_frames, reference_frame, bar=DEFAULT_BAR, *, drift_fn=None, max_drift=None, opinion_override=None) -> Scored{strategy_id, strategy_label, worst_drift, motion, opinion, opinion_synthetic, frames, accepted, reason}` — порядок причин: идентичность → движение → мнение.
**`produce.py`** (сам вердикт вынесен из цикла генерации отдельно)
- `verdict(*, drift, motion, quality, seam, limbs, wander, bar, min_motion, loop) -> (passed, score, reason)` — чистая функция, тестируемая без единого платного вызова. Сначала «нечего было судить» (`median is None` или `coverage < MIN_COVERAGE`) → `(False, 1.0, "not verifiable")`, затем по порядку: медиана дрифта, p90, движение, плавность, анатомия, уход позы, луп. Возвращается ПЕРВАЯ несработавшая проверка.

- `rank(scored) -> list[Scored]` — принятые вперёд, затем меньший дрифт, больше движения, выше мнение.
- `_opinion(root, strategy_id)` — читает `fixtures/judge/<sid>.json`, по умолчанию `synthetic=True`.

---

## 4. Все пороги и константы

Колонка «откуда» переносит комментарий рядом с константой в коде.

| константа | модуль | значение | что означает | откалибровано / выбрано |
|---|---|---|---|---|
| `Bar.max_identity_drift` | `critic` | 0.20 | потолок дрифта на шкале ПРОКСИ | откалибровано на фикстурах: согласованный клип ~0.03, меняющаяся композиция ~0.28 → 0.20 между ними |
| `Bar.min_motion` | `critic` | 0.02 | пол присутствия движения | выбрано (пол против замороженного клипа) |
| `Bar.min_opinion` | `critic` | 0.60 | грубое «в целом достаточно хорошо» | выбрано, спорный дефолт (сказано прямо) |
| `_HASH_SIDE` | `identity` | 16 | сторона квадрата dhash → `2*16*16` бит | конструкция из eval-репозитория |
| `_SAME_LOOK_AT` | `identity` | 0.90 | согласие хэшей, выше которого «тот же вид» | ВЫБРАНО, не измерено; порог только для текста отчёта |
| `SAME_PERSON_MAX` | `identity_arcface` | 0.35 | косинусная дистанция на СУДИМЫХ кадрах | откалибровано на живых клипах 2026-08-12: истинная идентичность держалась 0.18–0.22, 0.35 даёт запас |
| `HARD_DRIFT_MAX` | `identity_arcface` | 0.6 | потолок по `p90` — клип «уезжает» в другого человека | откалибровано (см. клип A: p90 0.513 проходит) |
| `MIN_FACE_PX` | `identity_arcface` | 100 | минимальный размер лица (короткая сторона bbox) для доверия эмбеддингу на КАДРЕ ВИДЕО | измерено на двух живых клипах: 111–114 px → полоса 0.18–0.22; 64–86 px → никогда ниже 0.36 внутри одного дубля. ArcFace берёт кроп 112x112 |
| `START_MIN_FACE_PX` | `identity_arcface` | 70 | тот же пол, но для РЕЗКОГО СТИЛЛА | измерено: старт-кадры 72/78/91/98 px → 0.253/0.179/0.139/0.130, и именно кадр на 91 px дал полностью прошедший клип. Порог 100 завернул бы его (пойманный баг) |
| `MIN_COVERAGE` | `identity_arcface` | 0.5 | доля судимых кадров, ниже которой вердикт NOT VERIFIABLE | выбрано как защита от «слишком мелкое лицо проходит по умолчанию» |
| `MIN_VISIBILITY` | `pose` | 0.5 | ниже — точка это догадка MediaPipe, а не наблюдение | выбрано |
| `SAME_POSE_MAX` | `pose` | 0.15 | старт-кадр против позы-референса, строгий | шкала откалибрована живьём: 0.00 кадр сам с собой, 0.04–0.25 та же посадка, 0.05–0.30 разброс внутри клипа, 0.67 реально другая поза |
| `POSE_WANDER_MAX` | `pose` | 0.45 | кадры видео против референса, свободный | выбран так, чтобы пропускать 0.30 честного движения и ловить 0.67 |
| `WORST_JOINT_MAX` | `pose` | 0.40 | худший одиночный сустав | измерено по клипам: veo 0.13 / wan 0.33 / wan-fast 0.41 / happyhorse 0.49. **Порядок надёжен, точка отсечки — на одном референсе и одном промте, перепроверять** |
| `LIMB_WOBBLE_MAX` | `pose` | 0.25 | допустимая вариация длины конечности | часть вариации физична (ракурсное укорочение); на четырёх живых клипах максимум был 0.08–0.12 |
| `DEFAULT_MODEL` / `MODEL_ENV` | `pose` | `~/.mediapipe/pose_landmarker_lite.task` / `BALL_REEL_POSE_MODEL` | где лежат веса позы (5.5 МБ) | — |
| `SEAMLESS_MAX` | `motion` | 0.30 | стык лупа как доля типичного шага | измерено на seedance-2.0: старт-кадр обоими ключевыми → 0.09, только стартовый → 0.61. (В `POLLINATIONS_CONTRACT.md` та же пара записана как 0.03 / 0.54 — числа из разных прогонов, порог один) |
| `JUMP_MAX` | `motion` | 4.0 | шаг во столько раз больше медианного = телепорт/морфинг | выбрано |
| `STILL_MIN` | `motion` | 0.15 | ниже — ничего не происходит (используется как `STILL_MIN/10` в `motion_quality`) | выбрано |
| `END_FRAME_MODELS` | `chain` | `("wan-fast","veo","wan-pro","seedance-2.0")` | модели, реально принимающие второй кейфрейм | сверено живьём с `video_capabilities`: заявленное совпало с замерами (wan 1.71 и happyhorse 1.61 — `end_frame` НЕ заявляют) |
| `DEFAULT_KEYFRAMES` | `chain` | 5 | узлов на цепочку (верх, низ, верх + возврат) | выбрано |
| `KEYFRAME_POSE_MAX` | `chain` | 0.25 | насколько кейфрейм может промахнуться мимо позы донора | та же шкала торсов, что у `SAME_POSE_MAX`; на живом прогоне узлы дали 0.5914 и 0.6516 и были отклонены |
| `FRAME_FPS` | `produce` | 6 | частота извлечения кадров для ВСЕХ гейтов | одна константа, потому что обрезчик лупа переводит индекс кадра во время |
| `START_MODEL` / `VIDEO_MODEL` | `produce` | `kontext` / `seedance-2.0` | стек по умолчанию | из вакансии |
| `FRAMING` | `produce` | «от макушки до мяча, тело заполняет кадр…» | обязательная приписка к промту | A/B живьём на kontext: 1/6 высоты → 81 px / 0.175; «по колено» → 85 px / 0.168; **выбранная** → 98 px / 0.130; «по пояс» → 151 px / 0.053, но мяч уходит; крупный план → 320 px / 0.100, мяч уходит. Абстрактные доли кадра модель игнорирует |
| `VIDEO_FRAMING_STEPS` | `produce` | 4 варианта, последний — пустая строка | понижение формулировки на каждую неудачную попытку | реакция на живой 422 `content_policy_violation` от Seedance: триггер — фраза, повтор того же промта сожжёт попытки |
| `attempts` (дефолт) | `produce` | 4 | попыток ретрай-цикла | выбрано |
| `SPEC_FPS` | `driving` | 12 | частота сэмплирования драйвинг-видео | по Найквисту для нескольких подскоков в секунду |
| `MIN_EXPRESSION_FACE_PX` | `driving` | 100 | ниже блендшейпы недостоверны | та же шкала, что у порога идентичности (константа объявлена, в коде `extract` не используется) |
| пороги `validate()` | `driving` | frames<6, `pose_coverage<0.8`, `expression.coverage<0.5` | что делает спеку непригодной | выбрано |
| `MIN_SOURCE_FACE_PX` | `intake` | 100 | ниже — ВЫХОД тоже нельзя будет проверить | ссылается на калибровку `identity_arcface.MIN_FACE_PX` |
| `MIN_MESH_FACE_PX` | `intake` | 60 | ниже — 478-точечный меш не читает лицо даже после кропа | выбрано |
| `MIN_DETECTOR_SCORE` | `intake` | 0.6 | ниже — «лицо» может им не быть | выбрано |
| `ASSUMED_PROPORTIONS` | `metrics` | `shoulder_width 0.64`, `hip_width 0.43`, `shoulder_to_hip 1.49`, плечо→локоть 0.62, локоть→запястье 0.55, бедро→колено 0.92, колено→стопа 0.86, `leg_length 1.78` | подставные пропорции, когда тела на фото нет | сверено с двумя измеренными в 3D людьми (ширина плеч 0.638 и 0.643; плечи/бёдра 1.58 и 1.48). Явный СТЕНД-ИН, всегда помечается `assumed` |
| `SINGLE_REF_MODEL` / `MULTI_REF_MODEL` | `subject` | `kontext` / `nanobanana` | модель старт-кадра | измерено: nanobanana на 3 референсах удержала лицо (дрифт 0.176) и взяла телосложение со второй картинки; seedream5 принимает 14 референсов, но потерял лицо (0.752). Ёмкость ≠ верность |
| `CAPABILITIES` | `router` | identity, build, pose_trajectory, loop, audio, object_fidelity | словарь требований | — |
| `API_CANNOT` | `router` | `("pose_trajectory","object_fidelity")` | чего публичный шлюз не даёт ни за какие деньги | измерено: поза референс-картинкой дала 0.59–0.65 при пороге 0.25; словарь возможностей видео целиком = start_frame/end_frame/audio_output |
| прайс `_pollen` | `router` | см. §3.2 | pollen за секунду видео | из живого `GET /video/models` |
| `BASE_MODEL`, `CONTROLNET_OPENPOSE`, `IP_ADAPTER_*` | `gpu_keyframes` | `runwayml/stable-diffusion-v1-5`, `lllyasviel/control_v11p_sd15_openpose`, `h94/IP-Adapter` + `ip-adapter-faceid_sd15.bin` | стек на 4 ГБ | SD1.5, а не SDXL: один UNet SDXL в fp16 ~5 ГБ и не влезает |
| `WIDTH, HEIGHT` | `gpu_keyframes` | 512, 768 | максимальный 2:3 кадр в бюджете памяти | расчёт: UNet 1.7 + ControlNet 0.7 + VAE 0.2 + IP-Adapter 0.6 + латенты 0.4 ≈ 3.6 ГБ |
| `GPUPlan` дефолты | `gpu_keyframes` | steps 24, guidance 6.0, `controlnet_scale` 1.0, `ip_adapter_scale` 0.7, fp16, `estimated_vram_gb` 3.6 | конфиг генерации | РАСЧЁТ, не замер (модуль не исполнялся) |
| пороги `plan()` | `gpu_keyframes` | <3.5 ГБ → предупреждение и 448x640; ≥8 ГБ → 640x960 и меньше экономий | ветвление по карте | расчёт |
| `MIN_VRAM_GB` / `MIN_DISK_GB` | `preflight_gpu` | 3.5 / 15 | пороги предполёта (веса ~7 ГБ + кэш и выдача) | согласованы с `gpu_keyframes`; в `GPU_RUNBOOK.md` рекомендуется диск от 30 ГБ |
| порог весов | `preflight_gpu` | <3 ГБ в `HF_HOME` = качка не закончилась | — | выбрано |
| `line_width`, `margin` | `skeleton` | 4 px, 0.35 | толщина линий скелета, запас кропа вокруг фигуры | выбрано |
| `CORE_MODULES` / `MIN_CORE_COVERAGE` | `codeaudit` | 9 модулей / 70% | покрытие требуется только с тех, кто несёт вердикты; с CLI и сетевых обёрток — нет | выбрано |
| `MUTATIONS` | `codeaudit` | 15 штук | список порогов, снятие которых обязано ронять тесты | список = пороги из §4, на которых стоят вердикты |
| дефолты `live_gen` | `live_gen` | SDXL base, `lora_scale` 0.8, `ip_adapter_scale` 0.6, 30 шагов, guidance 5.0 | самохостовый старт-кадр | выбрано, никогда не исполнялось |

---

## 5. Переменные окружения

Все найденные `os.environ.get` (плюс обязательные через `_require_env`).

| имя | модуль | по умолчанию | зачем |
|---|---|---|---|
| `POLLINATIONS_API_KEY` | `pollinations._key` | **нет; `RuntimeError`** | Bearer-ключ `sk_...` для всех вызовов шлюза |
| `POLLINATIONS_BASE` | `pollinations._base` | `https://gen.pollinations.ai` | базовый хост генерации |
| `POLLINATIONS_MEDIA` | `pollinations._media` | `https://media.pollinations.ai` | ОТДЕЛЬНЫЙ хост загрузки; видео-эндпоинт фетчит старт-кадр оттуда серверной стороной |
| `POLLINATIONS_IMAGE_MODEL` | `mvp` | `flux` | модель для `mvp --generate` |
| `POLLINATIONS_JUDGE_MODEL` | `mvp` | `openai` | VLM-судья для оси мнения (у `pollinations.judge_frame` свой дефолт `claude`) |
| `BALL_REEL_DOCTOR_OUT` | `doctor` | `doctor_out` | каталог артефактов предполёта |
| `BALL_REEL_VIDEO_MODEL` | `doctor` | `seedance-2.0` | чем гонять `--video` (для отладки ставить `wan-fast`, в 18 раз дешевле) |
| `BALL_REEL_VIDEO_DURATION` | `doctor` | `4` | длительность тестового клипа в секундах |
| `BALL_REEL_POSE_MODEL` | `pose` (`MODEL_ENV`) | `~/.mediapipe/pose_landmarker_lite.task` | веса MediaPipe Pose; отсутствие → явный `RuntimeError` с командой скачивания |
| `BALL_REEL_FACE_MODEL` | `driving`, `metrics` | `~/.mediapipe/face_landmarker.task` | веса face-меша; без него мимика и геометрия лица просто отсутствуют (и это сказано в warnings), а не подменяются нейтральными |
| `CHARACTER_LORA` | `gen` | не задана → `None` | путь к LoRA персонажа для live-старт-кадра |
| `HF_HOME` | `preflight_gpu` | `~/.cache/huggingface` | где искать скачанные веса при проверке |
| `VIDEO_API_URL`, `VIDEO_API_KEY` | `live_gen` | **нет; `RuntimeError`** | самохостовый image-to-video провайдер |
| `TTS_API_URL`, `TTS_API_KEY` | `live_gen` | **нет; `RuntimeError`** | русский TTS |
| `LIPSYNC_API_URL`, `LIPSYNC_API_KEY` | `live_gen` | **нет; `RuntimeError`** | аудио-driven липсинк |

Прокси-переменные (`HTTPS_PROXY`, `REQUESTS_CA_BUNDLE`) кодом не читаются — их использует `requests` сам; в `POLLINATIONS_CONTRACT.md` отмечено, что они уже выставлены в среде.

---

## 6. Внешние зависимости и веса

**Офлайн-демо:** только `Pillow`.

**Живой путь (`requirements-live.txt` + `MVP_RUNBOOK.md`):**
```
pip install Pillow requests insightface onnxruntime numpy
apt-get install -y ffmpeg          # извлечение кадров и склейка
pip install mediapipe              # поза, мимика, скелеты (в requirements-live не перечислен)
```
`requirements-live.txt` дополнительно перечисляет `torch, diffusers, transformers, accelerate` — они нужны только для `live_gen`/`gpu_keyframes`, не для `produce`.

**Скачиваемые веса:**

| что | откуда / куда | размер | кто требует |
|---|---|---|---|
| InsightFace `buffalo_l` (detection + recognition + genderage) | тянется при первом вызове из релиза InsightFace, кэш insightface | ~281 МБ по контракту, «~300 МБ» в докстринге | `identity_arcface`, `metrics`, `subject.from_photo`, `live_gen` |
| `pose_landmarker_lite.task` | `storage.googleapis.com/mediapipe-models/...` → `~/.mediapipe/` | 5.5 МБ | `pose`, `skeleton`, `driving`, `metrics.body_metrics` |
| `face_landmarker.task` | MediaPipe, вручную → `~/.mediapipe/` | размер в коде не указан | `driving` (блендшейпы), `metrics.face_metrics` (геометрия и мимика) |
| `runwayml/stable-diffusion-v1-5` + `lllyasviel/control_v11p_sd15_openpose` + `h94/IP-Adapter` (`ip-adapter-faceid_sd15.bin`) | HuggingFace → `HF_HOME` | суммарно ~7 ГБ (по `GPU_RUNBOOK.md`), диск от 15 ГБ по предполёту / от 30 ГБ по рунбуку | `gpu_keyframes` |
| SDXL base + `ip-adapter-faceid_sdxl.bin` | HuggingFace | размер не указан | `live_gen.start_frame_live` |

**GPU-установка (из `gpu_keyframes.requirements()` и `GPU_RUNBOOK.md`):**
`torch` с индексом cu121, `diffusers>=0.27`, `transformers`, `accelerate`, `safetensors`, `insightface`, `onnxruntime-gpu`, `mediapipe`, `pillow`, `numpy`, `requests`.

**Проверенные версии (из контракта):** `insightface==1.0.1`, `onnxruntime==1.28.0` встают через pip, CPU-провайдера хватает — 8 кадров эмбеддятся за ~16 c.

**Сетевые требования:** egress должен пускать И `gen.pollinations.ai`, И `media.pollinations.ai` — иначе image-to-video ломается конкретно на фетче старт-кадра.

---

## 7. Состояние MVP

### 7.1 Проверено живьём (есть числа и/или артефакты в репозитории)

- **Весь боевой путь `produce`**: `/v1/images/edits` (kontext, ~7 c, 1024x1024) → `media/upload` (~1.5 c) → `/video/{prompt}` seedance-2.0 (200, сырой mp4, ~97 c, 720x1280 h264, 4.04 c, 0.72 pollen) → ffmpeg → ArcFace + motion. Отмечено как «весь проверен живьём» в `POLLINATIONS_CONTRACT.md`.
- **Клиент `pollinations`**: `upload`, `image`, `images_edit`, `compose`, `video` помечены `[verified live]` прямо в докстрингах; `tts` (русский, mp3, ~2.3 c) — тоже.
- **Калибровка ArcFace** на двух живых клипах (клип A: медиана 0.220, p90 0.513, покрытие 100% → PASS; клип B: покрытие 0% → NOT VERIFIABLE) и на четырёх живых старт-кадрах (72/78/91/98 px).
- **Гейт позы и анатомии**: поза старт-кадра у всех четырёх моделей 0.018–0.043, худший сустав по клипам veo 0.13 / wan 0.33 / wan-fast 0.41 / happyhorse 0.49; `limb_consistency` 0.08–0.12.
- **Луп**: `end_frame` у seedance-2.0 (0.03) и veo (0.17) работает, wan/happyhorse игнорируют (1.71 / 1.61); аварийный `trim_to_loop` поднял wan до 0.41 (сохранено 67% клипа) и happyhorse до 0.75 — планку 0.30 не берёт.
- **Модерация**: 422 `content_policy_violation` у seedance воспроизведена; изолировано, что триггерить может и картинка, и формулировка; смена модели — рабочий обход.
- **Ограничение kontext**: `width/height/size` игнорируются, всегда 1024x1024 — вертикальный старт-кадр от kontext получить нельзя, 9:16 делает видео-модель через `aspectRatio`. При этом `produce` продолжает передавать `width/height` (безвредно, но бессмысленно).
- **`chain.render_keyframes` исполнялся живьём и НЕ сошёлся**: `chain_out/keyframes.json` — два узла отвалились по 400 от провайдера картинок, два отклонены гейтом позы (0.5914 и 0.6516 при пороге 0.25), идентичность 0.5547 / 0.3886. `generate_chain` на этих данных отработать не мог (нужно ≥2 принятых узла).
- **`produce` на своём фото** (`wiretest/produce_report.json`): `passed=False`, одна попытка, «identity not verifiable: 0% кадров с лицом ≥100 px», при этом луп получился (ratio 0.15, seamless) и движение 0.0669. То есть отказ честный, а не молчаливый.
- **Офлайн-тесты**: `python3 -m unittest discover -s ball_reel/tests` → **148 тестов, OK** (проверено при составлении этой памятки).
- **Мутационный аудит** (`codeaudit`) уже дал результат: было обнаружено, что снятый `MIN_COVERAGE` не ронял ни одного теста, из-за чего вердикт `produce` вынесли в отдельную чистую функцию `verdict()`. Отмечено там же: подмена константы через `setattr` не работает для порогов, использованных как значения по умолчанию у аргументов (проверено на `LIMB_WOBBLE_MAX`), поэтому мутация правит ИСХОДНИК копии пакета.
- **`intake`, `metrics`, `driving`, `pose`, `motion`, `skeleton`, гейт** — `GPU_RUNBOOK.md` называет их проверенными на живых данных.

### 7.2 Написано, но ни разу не исполнялось

- **`gpu_keyframes.py`** — прямо в докстринге: «written but NOT executed — there is no GPU in the environment it was authored in»; `plan()` всегда добавляет note «UNVERIFIED: … expect the memory numbers to move». Оценки памяти и времени в `GPU_RUNBOOK.md` — расчёт, не замер.
- **`live_gen.py`** — «NOT RUN in this repo»; все три функции написаны против документированных API и требуют GPU/весов/ключей провайдеров.
- **`identity_arcface`, модельная часть** — в докстринге «NOT RUN in the offline package» (арифметика косинуса юнит-тестируется отдельно). Фактически при этом на бенче она отрабатывала — калибровка в контракте получена именно ею; читать пометку как «не в офлайн-пакете».
- **`preflight_gpu.py`** — по коду это чистая проверка, следов запуска на карте нет.
- **Судья (`pollinations.judge_frame` / `opinion_of`)** — в контракте помечен `[из доков, эндпоинт проверен на моделях]`, то есть проверено наличие моделей, а не сам вызов оценки.
- **`gen.Gateway` в режиме `--live`** — в репозитории нет ни артефактов, ни упоминаний живого запуска.

### 7.3 Не реализовано вообще

- **Липсинк.** Есть только шов в `live_gen.voice_live` за env-переменными; в каталоге Pollinations липсинка нет. TTS готов, драйв губ — нет.
- **`driving.survey()`** — упомянут в докстринге `extract` как способ выбрать сегмент, но функции в коде НЕТ.
- **`gen.LiveNotWired`** — класс объявлен и задокументирован, но нигде не бросается. `README` утверждает, что `--live` бросает `LiveNotWired`; фактически `Gateway` уходит в `live_gen`, который падает `RuntimeError` на отсутствующих зависимостях или env-переменных.
- **DWPose как источник условий.** `skeleton.render_sequence` умеет только MediaPipe и сам предупреждает, что кондиционер и верификатор совпали, а значит гейт подтверждает собственные ошибки экстрактора. Поле `source` умеет значение `dwpose`, кода за ним нет.
- **Апскейл после гейта** — описан как правило в `GPU_BRANCH.md`, кода нет.
- **Оверлейный текст.** `Brief.overlay_text` заполнен в `DEMO_BRIEF`, но ни один модуль его не рендерит.
- **Мнение офлайн** — синтетические вердикты из `fixtures/judge/*.json`, всегда помечены `synthetic`.

### 7.4 Известные незакрытые проблемы

Из раздела «Чего эта ветка НЕ решает» (`GPU_BRANCH.md`):
1. **Объекты.** Совместимость по мячу (тот ли мяч, там ли, того ли размера) не меряется ни в текущем гейте, ни в GPU-ветке. Нужен детектор объекта и своя метрика.
2. **Разделение камеры и субъекта.** Амплитуда и каденс считаются в предположении статичной камеры; на произвольном драйвинг-видео это систематическая ошибка. То же честно повторено в докстринге `driving.py` и всегда попадает в `spec.warnings`.
3. **Ракурс лица на входе.** 3D спасло телосложение, но не идентичность: на развёрнутом фото `id_drift` вышел 0.55 при визуально том же человеке.

Плюс из кода и контракта:
4. **Между кейфреймами траектория остаётся догадкой модели** (`chain.py`, «HONEST LIMIT»): пиннинг только в узлах, плотность узлов — регулятор точности и цены.
5. **Точка отсечки `WORST_JOINT_MAX=0.40`** получена на одном референсе и одном промте — порядок моделей надёжен, сама планка нет.
6. **Кондиционер = верификатор** (MediaPipe в обеих ролях), пока не появится DWPose — гейт слабее, чем выглядит.
7. **Луп у моделей без `end_frame`** обрезкой до 0.30 не доводится (лучшее 0.41).
8. **Мелкое лицо в выдаче** — самый частый живой отказ: вердикт «not verifiable», лечится более тесным кадрированием, НЕ апскейлом перед гейтом.
9. **`/account/usage` и `/account/balance` отдают 403** — баланс не прочитать, предполётом может быть только `GET /account/key`.
10. **`supported_endpoints` в `/v1/models` врут** для видео-моделей; верить `/openapi.json` и `/video/models` (а `video_capabilities` — можно).
