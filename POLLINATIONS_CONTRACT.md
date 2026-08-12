# Pollinations — проверенный контракт API (для живого прогона)

Всё ниже помечено `[проверено live]` (реальный ответ от `gen.pollinations.ai`
с ключом `sk_...` из окружения, 2026-08-12) либо `[из доков]` /
`[проверить в новой сессии]`. Считай непроверенное непроверенным.

## Базовое

- Base: `https://gen.pollinations.ai` — `[проверено live]` (пускается прокси).
- Media: `https://media.pollinations.ai` — **отдельный хост.** Был заблокирован
  egress-прокси (403 CONNECT); после вайтлиста `*.pollinations.ai`
  **`upload` работает** — `doctor` даёт 6/6. `[проверено live 2026-08-12]`
- **`GET /openapi.json` (200) — машиночитаемый контракт всего API.** Читай его
  вместо угадывания: параметры, диапазоны, коды ошибок, тип ответа. Это самый
  дешёвый способ проверить форму эндпоинта, не тратя баланс. `[проверено live]`
  Полный список путей включает `/video/{prompt}`, `/video/models`, `/upload`,
  `/media/{id}`, `/v1/images/edits`. **`/v1/videos/edits` НЕ существует** —
  мультипарт-файла для видео нет, значит media-хост для видео обязателен.
- Auth: `Authorization: Bearer sk_...` **или** `?key=sk_...`. Листинг моделей —
  без ключа. `[проверено live]`
- Ключ в окружении: `POLLINATIONS_API_KEY` (`sk_...`, 35 симв). `[проверено live]`
- ⚠️ Ключ **не имеет** права `account:usage` → `/account/usage` и
  `/account/balance` отдают 403. Баланс не прочитать, но **генерация работает**.
  Не пытайся читать баланс как предполёт — проверяй генерацией. `[проверено live]`
- ✅ Зато `GET /account/key` → **200** и работает без `account:usage`:
  `{"valid":true,"type":"secret","name":"demo_instories","permissions":
  {"models":null,"account":null},"pollenBudget":null,...}`. Это единственный
  бесплатный предполёт «ключ жив и не истёк». `[проверено live]`
- Прокси: `HTTPS_PROXY` уже выставлен, CA-бандл `/root/.ccr/ca-bundle.crt`,
  `REQUESTS_CA_BUNDLE` тоже. `requests` ходит сам. При 403 от прокси на хост —
  **не обходить**, это политика (см. `/root/.ccr/README.md`).

## Модели (нужный стек присутствует) `[проверено live]`

`GET /v1/models` → 200, 214 моделей. Из капабилити-метаданных:

| модель | input | output | endpoints |
|---|---|---|---|
| `flux` | text | image | `/image/{prompt}`, `/v1/images/generations`, `/v1/images/edits` |
| `kontext` | **text, image** | image | те же (умеет image-to-image) |
| `seedance-2.0` | **text, image** | **video, audio** | (video через `/video/{prompt}`) |
| `seedance-pro` | text, image | video | |
| `veo` | text, image | video | |
| `eleven-multilingual-v2` | text | audio | `/audio/{text}`, `/v1/audio/speech` |

Вывод: `kontext` реально берёт лицо как вход (image-to-image), `seedance-2.0`
реально делает image-to-video. Это ровно стек из вакансии.

## Картинка — text→image и image→image

### text→image `[проверено live]`
```
GET /image/{prompt_urlencoded}?model=flux&width=512&height=512&seed=7
```
→ 200, `content-type: image/jpeg`, тело — байты картинки (синхронно, ~2.6 c).
Параметры (из доков): `model, width, height, seed, quality, image, transparent`.
⚠️ **`nologo` — не существует**, убран из клиента. `[проверено: не в доках]`

### image→image, вариант A — по URL `[проверено live: поведение]`
```
GET /image/{prompt}?model=kontext&image={REF_URL}&width=&height=&seed=
```
Сервер Pollinations **сам скачивает** `REF_URL` со своей стороны. Если URL
недоступен извне — 400 `failed_to_download_image`. В прошлой сессии подсунул
самоссылку на `gen.pollinations.ai/image/...` → она отдала `HTTP 522` при
серверном фетче. **Вывод: `image=` требует реально фетчабельный публичный URL**
(правильный источник — `media.pollinations.ai/{id}` после `upload`).

### image→image, вариант B — мультипарт, БЕЗ media-хоста `[проверено live]` ✅
```
POST /v1/images/edits
  form: model=kontext, prompt=..., size=720x1280
  file: image=@face.jpg
```
→ 200 JSON `{created, data:[{b64_json, revised_prompt}], usage}`, ~7 c.
Картинка в `data[0].b64_json` (base64). **Это предпочтительный путь для
старт-кадра**: лицо грузится прямо в `gen.pollinations.ai` (разрешённый хост),
media-хост не нужен.

⚠️ **`size`/`width`/`height` у `kontext` ИГНОРИРУЮТСЯ — всегда 1024x1024.**
`[проверено live]` Просил `size=720x1280` через `/v1/images/edits` → отдал
1024x1024; просил `width=720&height=1280` через `/image/{prompt}?model=kontext`
→ тоже 1024x1024. (У `flux` через `/image/` `width/height` работают: 512x512
пришло как просили.) Вывод: **вертикальный старт-кадр от kontext не получить**;
9:16 делает уже видео-модель через `aspectRatio` (seedance отдал ровно
720x1280 из квадратного старт-кадра, кадрирование адекватное).

## Видео — image→video `[проверено live]` ✅
```
GET /video/{prompt}?model=seedance-2.0&image={MEDIA_URL}&duration=4&aspectRatio=9:16&audio=false
```
→ **200, `content-type: video/mp4`, тело — сырые байты mp4 синхронно.**
Никакого JSON-джоба и поллинга нет (в `/openapi.json` у 200 объявлен ровно один
тип — `video/mp4: binary`). Ветка поллинга в `video()` не нужна; guard оставлен
как растяжка на случай, если какая-то модель однажды ответит job'ом.

Факты первого живого прогона (seedance-2.0, duration=4, 9:16, старт-кадр по
media-URL): **97 c**, 1 612 283 байт, h264 **720x1280**, 24 fps, 97 кадров,
длительность 4.04 c. `image=` **принял** `https://media.pollinations.ai/{id}`
(сервер фетчит его со своей стороны — ради этого и нужен был вайтлист).
Дешёвая прогонка на `wan-fast` (duration=2): 25 c, 195 КБ, тоже mp4-байты.

### Параметры (из `/openapi.json`, `[проверено live]`)
- `image` — **обычный query-параметр, строка**. Несколько URL разделяются
  `|` или `,`; **`image[0]` = старт-кадр, `image[1]` = конечный кадр** (это
  нотация из доков про элементы списка, а НЕ имена параметров — слать надо
  `image=<url1>|<url2>`). Ошибки: `failed_to_download_image`,
  `invalid_image_url`, `image_too_large`, `unsupported_image_media_type`.
- ⚠️ `duration` — **валидируется по модели, 400 при выходе за диапазон**:
  `seedance-2.0` **4–15 c** (двойка из прошлого плана НЕ прошла бы),
  `seedance-pro` 2–10, `wan` 2–15, `veo` 4/6/8, `nova-reel` 6–120 (кратно 6).
- `aspectRatio` — `16:9` / `9:16`. `width/height` у видео-моделей только задают
  соотношение; тир разрешения — отдельный `resolution` (480p/720p/1080p) и
  только у `veo`, `wan-pro`, `p-video`, `seedance-pro`.
- `audio` — bool, у моделей с `audio_output`. `seed` поддерживает seedance.

### Биллинг — в ЗАГОЛОВКАХ ответа, не в теле `[проверено live]`
Тело — сырой mp4, поэтому `usage` приходит хедерами:
`x-usage-completion-video-seconds`, `x-model-used`, `x-request-id`, `x-cache`.
Клиент складывает их в `pollinations.LAST_VIDEO_USAGE`.

### Цены (`GET /video/models`, валюта `pollen`, за секунду видео)
| модель | pollen/с | 4-c клип | capabilities |
|---|---|---|---|
| `wan-fast` | **0.01** | 0.04 | start+end frame, 480p, silent — **для отладки труб** |
| `seedance-pro` | 0.025 | 0.10 | start frame |
| `veo` | 0.08 | 0.32 | start+end, audio |
| `seedance-2.0` | **0.18** | **0.72** | start+end frame, audio, 720p |
| `grok-imagine-video-1.5` | 0.14 | 0.56 | start frame, audio |
Все видео-модели `paid_only`. **Дисциплина: гоняй трубу на `wan-fast` (в 18 раз
дешевле), а `seedance-2.0` — только на боевой прогон.** В `doctor` модель и
длительность переопределяются: `BALL_REEL_VIDEO_MODEL`, `BALL_REEL_VIDEO_DURATION`.

⚠️ `supported_endpoints` в `/v1/models` у видео-моделей **врут** — там перечислены
только image-эндпоинты, `/video/{prompt}` не упомянут, хотя работает. Не верь
этому полю; верь `/openapi.json` и `/video/models`.

## Звук — TTS `[проверено live]` ✅
```
GET /audio/{text_urlencoded}?voice=nova&model=eleven-multilingual-v2
```
→ 200, `content-type: audio/mpeg`, байты mp3, ~2.3 c. Русский проходит.
Липсинк — отдельная модель, аудио-driven (в этом каталоге липсинка нет; голос
готов, драйв губ — следующий шов).

## Media upload `[проверено live]` ✅
```
POST https://media.pollinations.ai/upload   (Bearer)
  multipart:  -F file=@face.jpg
```
→ 200 `application/json`, **реальные поля ответа**:
```json
{"id":"87d2b912-ebb6-406a-9fc0-40e6a9e48c34",
 "url":"https://media.pollinations.ai/87d2b912-ebb6-406a-9fc0-40e6a9e48c34",
 "contentType":"image/jpeg","size":122539}
```
т.е. `id` + готовый `url` + `contentType` + `size` (~1.5 c на 130 КБ).
`GET {url}` → 200, отдаёт байты с исходным content-type, **публично** — именно
поэтому его берёт серверный фетч видео-эндпоинта. Лимит 100 МБ, жизнь 30 дней.
NB: `/upload` и `/media/{id}` есть и на `gen.pollinations.ai` (см. `/openapi.json`)
— если media-хост опять закроют, это запасной путь `[не проверено]`.

## Судья (vision) `[из доков, эндпоинт проверен на моделях]`
`POST /v1/chat/completions` с `image_url` (data URI) в контенте, модели
`claude` / `openai` / `gemini` (input включает image). Для оценки кадра.

## Итог: боевой путь produce — весь проверен живьём ✅

1. `POST /v1/images/edits` (мультипарт, `kontext`, локальное лицо) → старт-кадр.
   **Проверено.** ~7 c, 1024x1024 (размер не регулируется).
2. `POST media.pollinations.ai/upload` → `url`. **Проверено.** ~1.5 c.
3. `GET /video/{prompt}?model=seedance-2.0&image=<media_url>&duration=4` → mp4.
   **Проверено.** ~97 c, 720x1280, 4.04 c, 0.72 pollen.
4. `ffmpeg` → кадры → ArcFace (insightface `buffalo_l`) + motion → ретрай-луп.
   **Проверено:** `insightface==1.0.1` + `onnxruntime==1.28.0` встают через pip,
   модельный пак (~281 МБ) качается с GitHub при первом вызове, CPU-провайдер
   хватает: 8 кадров эмбеддятся за ~16 c.
5. (опц.) TTS русский. **Проверено.**

### Калибровка порога ArcFace на живых кадрах `[проверено live]`
Первый живой клип (тот же человек по всему клипу), 8 кадров против
референс-портрета, косинусная дистанция:
`0.176, 0.177, 0.205, 0.219, 0.362, 0.714, 0.426, 0.221`.
Спокойные фронтальные кадры лежат **0.18–0.22**, а всплески 0.36/0.43/0.71 — это
кадры в верхней точке прыжка: лицо мелкое, повёрнутое и смазанное. То есть
`SAME_PERSON_MAX = 0.35` **разделяет не «того/не того», а «видно/не видно лицо»**
на быстром движении. Практический вывод для гейта: судить по устойчивой массе
кадров (медиана/квантиль), а не по худшему кадру, иначе любой честный прыжок
падает в NOT PASSING из-за одного смазанного кадра. Порог по худшему кадру
уместнее ~0.45–0.5, либо мерить только кадры с достаточным размером лица.
