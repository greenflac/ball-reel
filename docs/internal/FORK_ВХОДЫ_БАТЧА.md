# Входы презентационного батча: что породили и чем это проверено

Смена приёма входов, 2026-08-22, ветка `fork/template-mvp`.
Ничего не коммитил и не пушил. Задание: довести входы витрины до
5 драйвингов / 5 стилей / 2 личностей. Драйвинги — не моя часть.

---

## 0. НЕПРОВЕРЕНО — наверх (Ц4)

Это чужие утверждения и внешние факты, которые я НЕ измерял:

* **НЕПРОВЕРЕНО.** Что порождённые входы дадут годный видеовыход. Ни один
  прогон стенда (`fork_run`, шлюз, ArcFace по кадрам) на этих файлах не
  запускался. Здесь измерен только **приём входа**, то есть «прибор возьмёт
  этот файл», а не «из этого файла выйдет хорошее видео».
* **НЕПРОВЕРЕНО.** Что стилевой референс без людей действительно снимает
  протечку одежды и аксессуаров. Цифра 0.3928 при планке 0.35 (очки со
  стилевого референса закрыли лицо клиента) взята из брифа задания, своего
  замера у меня нет. Отсюда правило «без людей в кадре» применено как
  ПРЕДОСТОРОЖНОСТЬ, а не как проверенная поправка.
* **НЕПРОВЕРЕНО.** Что `nanobanana-2` — лучший выбор из 74 моделей эндпоинта.
  Модель задана заданием; сравнения моделей я не делал.
* **ИЗМЕРЕНО, но важно понимать шкалу.** `creative_eval.style.similarity` — это
  грубый цвето-фактурный прокси на Pillow, а НЕ эмбеддинг-расстояние
  (CLIP/DINO). Оговорка — `creative_eval.style.PROXY_CAVEAT`, она едет вместе
  с каждым числом ниже.

**Долг владельцу:** драйвингов по-прежнему **3 из 5**
(`driving_arms.mp4`, `driving_selfie.mp4`, `driving_yogaball.mp4`).
Породить их нельзя — их даёт владелец.

---

## 1. Чем порождали

Один канал, одна модель, ни одной тихой замены:

```
эндпоинт  GET https://gen.pollinations.ai/image/{prompt}?model=&width=&height=&seed=
клиент    ball_reel.pollinations.image()
модель    nanobanana-2
ключ      POLLINATIONS_API_KEY (окружение)
fal.ai    НЕ ТРОГАЛСЯ ни разу
```

**Существование модели доказано командой до кода (Ц10):**

```
$ GET /image/models  ->  200, 74 модели
   nanobanana-2-lite, nanobanana-2, nanobanana-pro, nanobanana, vendouple/nano-banana-pro
```

`nanobanana-2` в списке есть. Отказов модели не было: все 9 вызовов вернули
200 и байты картинки. Если бы модель отказала — это был бы исход
«не смогли», и другой моделью я бы её молча не подменял.

**ИЗМЕРЕНО, побочный факт про эндпоинт.** Просил `960x1280`, вернулось
`896x1200`; просил `1024x1024`, вернулось `1024x1024`. То есть модель
округляет запрошенный размер к своей сетке. Это совпадает с уже записанным
в `docs/internal/POLLINATIONS_CONTRACT.md` наблюдением про подмену размера у
image-эндпоинта. На приём это не влияет (карточка читается, лицо мерится),
но планировать точный размер по запросу нельзя.

---

## 2. А. Мужская личность — `assets/fork_ref_man.png`

### Что сдано

| файл | кадрирование | лицо, px | вердикт приёма |
|---|---|---|---|
| `assets/fork_ref_man.png` | по грудь, анфас | **288** | годно (3/0/0) |
| `assets/fork_ref_man_waistup.png` | поясной, анфас | **196** | годно (3/0/0) |
| `assets/fork_ref_gym.png` (женская, эталон пары) | по грудь | **231** | годно (3/0/0) |

### Промт (модель `nanobanana-2`, seed 41, 1024x1024)

```
chest-up portrait photograph of a man in his early thirties, short dark brown
hair, clean shaven, facing the camera straight on, the head fills a large part
of the frame, tight framing from the top of the head down to the chest, neutral
friendly expression, even soft frontal studio lighting, plain light grey
seamless background, wearing a plain solid dark grey crew neck t-shirt with
absolutely no text no logo no print no pattern, no glasses, no sunglasses, no
hat, no cap, no headwear, no jewellery, exactly one person alone in the frame,
sharp focus on the face, photorealistic, 85mm lens
```

Поясной вариант (seed 11) отличается первой строкой: `waist-up portrait
photograph ...` плюс `head and shoulders large in the frame`.

### Приёмка: `ball_reel.fork_intake.photo_intake`

```
ПРИЁМ: фото клиента — fork_ref_man.png
  ВЕРДИКТ: годно  (проверено 3, нарушений 0, не смогли 0)
  face_found: годно — лиц найдено 1
  face_size:  годно — самое крупное лицо 288 px при планке 100 px
  one_person: годно — людей на кадре 1, ждали 1

ПРИЁМ: фото клиента — fork_ref_gym.png            (эталон пары, перемерен сейчас)
  ВЕРДИКТ: годно  (проверено 3, нарушений 0, не смогли 0)
  face_size:  годно — самое крупное лицо 231 px при планке 100 px

ПРИЁМ: фото клиента — fork_ref_man_waistup.png
  ВЕРДИКТ: годно  (проверено 3, нарушений 0, не смогли 0)
  face_size:  годно — самое крупное лицо 196 px при планке 100 px
```

Планка `MIN_FACE_PX = 100` приходит из `identity_arcface` импортом, не копией.

### Почему сдано ДВА файла, а не один — и это решение владельца, не моё

Задание требует одновременно (а) **поясной** портрет и (б) лицо, **сопоставимое
с 231 px** женского эталона. На квадрате 1024 эти два требования тянут в разные
стороны, и это ИЗМЕРЕНО, а не рассуждение:

```
поясной кадр (seed 11)   -> 196 px   на 15% МЕНЬШЕ женского
по грудь    (seed 41)    -> 288 px   на 25% БОЛЬШЕ женского
голова крупно (seed 42)  -> 362 px   на 57% больше, кадр уже не портрет пары
```

Третий, самый крупный, отброшен: 362 px — это уже другой тип кадра.
Из двух оставшихся под именем `fork_ref_man.png` лежит **кадр по грудь
(288 px)**, потому что сам женский эталон `fork_ref_gym.png` снят ИМЕННО по
грудь, а не по пояс — то есть буква задания («поясной») расходится с реальным
эталоном пары. Поясной вариант не выброшен, а сохранён рядом файлом
`fork_ref_man_waistup.png`: обе версии проходят приём, выбор между «строго по
букве задания» и «строго в пару к женскому кадру» — за владельцем, и
переключается переименованием одного файла.

### Глазами (П3)

Открывал оба файла инструментом `Read`.
`fork_ref_man.png`: мужчина около 30–35, тёмные короткие волосы, гладко выбрит,
анфас, взгляд в камеру, лёгкая улыбка; свет ровный фронтальный, теней на лице
нет; фон — гладкий светло-серый, пустой; футболка однотонная тёмно-серая,
**надписей, принтов и логотипов нет**; **очков нет, головного убора нет,
украшений нет**; в кадре **один человек**, других лиц и отражений нет.
То есть все запреты задания выполнены наблюдаемо, а не только по промту.

### Отдельно про женский эталон — находка, не претензия

**ИЗМЕРЕНО глазами** на `assets/fork_ref_gym.png`: на майке читается надпись
`NYC MARATHON`, а в зеркале на заднем плане видны ещё **три-четыре человека**.
Детектор при этом вернул `лиц найдено 1` — то есть ось `one_person` эти лица в
зеркале не поймала (они мельче порога детектора). На вердикт приёма это не
влияет, но два факта стоит знать до батча: (1) решение владельца «брендов не
генерируем» на существующем женском файле уже нарушено надписью, и мой мужской
файл ей НЕ симметричен; (2) `one_person: годно` здесь означает «одно лицо
достаточного размера», а не «в кадре один человек».

---

## 3. Б. Четыре стилевых референса

Все четыре — `nanobanana-2`, запрос `960x1280`, факт `896x1200`.
Целились так, чтобы покрыть словарь карточки, и попадание проверялось
карточкой, а не глазом.

| файл | seed | цель по словарю | карточка ФАКТ | цель взята? |
|---|---|---|---|---|
| `styleref_nightneon.png` | 21 | dark + muted | dark, muted, visible grain | да |
| `styleref_whitehall.png` | 22 | light + muted | light, muted, smooth | да |
| `styleref_filmgrain.png` | 34 | mid + moderate + visible grain | mid, moderate, visible grain | да, со второй попытки |
| `styleref_sunsetdune.png` | 24 | пятый, заведомо иной | mid, saturated, smooth | да, отличается палитрой |
| `styleref_bluesky.png` (был) | — | — | mid, saturated, smooth | — |

### Промты

```
nightneon (seed 21)
empty night city alley after rain, no people, no person, nobody in frame, low
key lighting, deep black shadows fill most of the frame, distant dim neon
signage, desaturated muted colours, faded washed out teal and dull grey, wet
asphalt reflections, moody dark photograph, background plate only

whitehall (seed 22)
empty high key minimalist interior room, no people, no person, nobody in frame,
bright soft diffused daylight from a large window, pale off white and bone
walls, very light airy overexposed feel, desaturated muted pastel palette,
smooth clean surfaces, bleached minimalism, architectural background plate only

filmgrain (seed 34, ВТОРАЯ попытка — см. ниже)
vintage grainy 35mm film photograph of an empty dusty country road with
weathered wooden fence, no people, no person, nobody in frame, faded washed out
warm sepia and dull sage tones, gentle low saturation, strong visible film grain
and dust texture, soft hazy afternoon light, balanced mid tones, background
plate only

sunsetdune (seed 24)
empty desert sand dunes at sunset, no people, no person, nobody in frame, vivid
intensely saturated orange crimson and magenta sky, strong warm colour, smooth
clean gradients, sharp dune ridge lines, glossy high contrast travel photograph,
background plate only
```

Во всех четырёх стоит тройное `no people, no person, nobody in frame` и
`background plate only` — стилевой референс задуман как источник света, цвета и
фона, и людей в кадре у всех четырёх нет (проверено глазами, §3.4).

### 3.0. Отрицательный результат, записан с числом (И6)

Первый заход на `filmgrain` (seed 23, «warm 35mm ... kodak portra warm amber and
olive tones, moderate saturation») дал карточку
`mid / **saturated** / visible grain` — то есть цель `moderate` НЕ взята.
ИЗМЕРЕНО: средняя насыщенность HSV на миниатюре **159.9** при границе
`_SATURATION_CUTS = (0.18, 0.45)`, то есть `saturated` начинается со 114.75.
Слово `moderate` в промте до пикселей не доехало.

Что сработало: явно выцветшая плёнка вместо тёплой насыщенной. Две пробы:

```
seed 33  overcast pine forest, faded olive/grey  -> HSV 68.7, яркость 101.7 -> mid/moderate/grain
seed 34  dusty road, sepia/sage, hazy            -> HSV 88.1, яркость 141.8 -> mid/moderate/grain
```

Взят seed 34: его палитра `brown, camel, beige` ни одним словом не пересекается
с существующим `bluesky` (`sky blue, chocolate, blue`), а у seed 33 первое слово
палитры — `chocolate`, то есть совпало бы с `bluesky`. Забракованный seed 23 в
`assets/` не лежит.

### 3.1. Карточка читается — `creative_eval.style.style_card`

`python3 -c "import creative_eval"` — импортируется, пакет виден среде
(`/home/user/greenflac/vertical-creative-eval/creative_eval/__init__.py`).

```
styleref_bluesky.png    {"colours": ["sky blue","chocolate","blue"],   "value_key":"mid",   "saturation":"saturated", "texture":"clean flat surfaces with smooth, untextured colour"}
styleref_filmgrain.png  {"colours": ["brown","camel","beige"],         "value_key":"mid",   "saturation":"moderate",  "texture":"visible grain and tactile surface texture"}
styleref_nightneon.png  {"colours": ["black","charcoal","slate grey"], "value_key":"dark",  "saturation":"muted",     "texture":"visible grain and tactile surface texture"}
styleref_sunsetdune.png {"colours": ["orange","black","rust"],         "value_key":"mid",   "saturation":"saturated", "texture":"clean flat surfaces with smooth, untextured colour"}
styleref_whitehall.png  {"colours": ["silver","light grey","stone"],   "value_key":"light", "saturation":"muted",     "texture":"clean flat surfaces with smooth, untextured colour"}
```

Все три оси словаря покрыты: тональность `dark/mid/light` — есть все три;
насыщенность `muted/moderate/saturated` — есть все три; фактура
`smooth/visible grain` — обе.

### 3.2. Приём стилевого референса — `fork_intake.style_intake`

Все пять: **годно, проверено 4, нарушений 0, не смогли 0** (4 поля карточки из 4).

```
styleref_bluesky.png    годно (4/0/0)
styleref_filmgrain.png  годно (4/0/0)
styleref_nightneon.png  годно (4/0/0)
styleref_sunsetdune.png годно (4/0/0)
styleref_whitehall.png  годно (4/0/0)
```

### 3.3. ГЛАВНОЕ: негативный контроль различимости (И5)

#### а) `ball_reel.fork_style_prompt.differ`, все 10 пар

```
bluesky    x filmgrain   -> годно  same=False
bluesky    x nightneon   -> годно  same=False
bluesky    x sunsetdune  -> годно  same=False
bluesky    x whitehall   -> годно  same=False
filmgrain  x nightneon   -> годно  same=False
filmgrain  x sunsetdune  -> годно  same=False
filmgrain  x whitehall   -> годно  same=False
nightneon  x sunsetdune  -> годно  same=False
nightneon  x whitehall   -> годно  same=False
sunsetdune x whitehall   -> годно  same=False

пар проверено 10, различились 10, совпали 0, не смогли 0
```

Собранные промты (все пять разные во всех трёх переменных частях):

```
bluesky    a palette of sky blue, chocolate and blue, even balanced lighting, rich saturated colour, clean flat surfaces with smooth, untextured colour, photographic look
filmgrain  a palette of brown, camel and beige, even balanced lighting, natural colour balance, visible grain and tactile surface texture, photographic look
nightneon  a palette of black, charcoal and slate grey, low-key shadowed lighting, desaturated restrained colour, visible grain and tactile surface texture, photographic look
sunsetdune a palette of orange, black and rust, even balanced lighting, rich saturated colour, clean flat surfaces with smooth, untextured colour, photographic look
whitehall  a palette of silver, light grey and stone, bright high-key lighting, desaturated restrained colour, clean flat surfaces with smooth, untextured colour, photographic look
```

#### б) У прибора есть вход, где он ОБЯЗАН сказать «нет» (И5)

Десять «годно» подряд ничего не стоят, пока не показано, что прибор умеет
краснеть. Прогнал каждую карточку саму с собой:

```
bluesky    x self -> не годно  same=True
filmgrain  x self -> не годно  same=True
nightneon  x self -> не годно  same=True
sunsetdune x self -> не годно  same=True
whitehall  x self -> не годно  same=True
```

Прибор шевелится в обе стороны: на пяти одинаковых входах — «не годно»
(«промты СОВПАЛИ на разных карточках: адаптер не различает»), на десяти
разных — «годно». Значит десять «годно» выше — измерение, а не заглушка.

#### в) Матрица `creative_eval.style.similarity`, 5x5

Оговорка едет с числами: это грубый цвето-фактурный прокси на Pillow, не
CLIP/DINO (`creative_eval.style.PROXY_CAVEAT`).

```
             bluesky filmgrain nightneon sunsetdune whitehall
bluesky       1.0000    0.7474    0.3038     0.4952    0.2485
filmgrain     0.7474    1.0000    0.3331     0.5597    0.3171
nightneon     0.3038    0.3331    1.0000     0.5109    0.1846
sunsetdune    0.4952    0.5597    0.5109     1.0000    0.3158
whitehall     0.2485    0.3171    0.1846     0.3158    1.0000
```

* **Диагональ ровно 1.0000 во всех пяти клетках** — как и требуется.
* Матрица симметрична (проверено по числам).
* Минимум вне диагонали **0.1846** (`nightneon` x `whitehall`) — тёмный и
  светлый края словаря, самая дальняя пара, и это ожидаемо.
* Максимум вне диагонали **0.7474** (`bluesky` x `filmgrain`).

**Про 0.7474 — не прячу, объясняю и указываю границу.** Это самая близкая пара
по прокси, и по карточке она при этом различается по ДВУМ осям из трёх
(`saturated` против `moderate`, `smooth` против `visible grain`) и не имеет ни
одного общего слова в палитре. Обе картинки — `mid` по тональности, и прокси
на 60% состоит из цветовой гистограммы, поэтому две «средние по яркости» сцены с
большим светлым небом садятся близко. Переделывать я эту пару НЕ стал, и вот
критерий: `differ` — прибор, которым стиль реально доезжает до модели, — на этой
паре даёт разные промты; `similarity` — прокси, который сам про себя пишет, что
не знает, ЧТО изображено. Если владелец считает планкой именно прокси,
переделывать надо `filmgrain` в сторону холодной или тёмной сцены.

### 3.4. ГЛАЗАМИ (П3)

Контактный лист собран на PIL (пять кадров 360x480 в ряд, подписаны) и открыт
инструментом `Read`:
`.../scratchpad/contact_styles.png`, 1800x508.
В `assets/` не кладу — это рабочий артефакт, а не вход батча.

Что видно, слева направо:

1. **bluesky** (существующий) — женщина в бежевом платье на фоне плоского
   ярко-синего неба, жёсткий солнечный свет, тени резкие. **На ней крупные
   зеркальные солнечные очки во весь глаз.**
2. **filmgrain** — пустая грунтовая дорога с деревянным забором, сухая трава,
   мягкое пасмурно-тёплое небо, всё выцветшее в бежево-оливковое, по кадру
   ровное зерно. Людей нет.
3. **nightneon** — ночной переулок после дождя, почти чёрный кадр, мокрый
   асфальт с редкими бликами, вдали неоновая вывеска `BAR` / `OPEN 24H`.
   Людей нет.
4. **sunsetdune** — песчаные дюны на закате, небо оранжево-пурпурное, солнце в
   кадре, гребень дюны режет диагональю. Людей нет.
5. **whitehall** — пустой белый интерьер со сплошным остеклением слева, всё
   выбелено, теней почти нет, поверхности гладкие. Людей нет.

**Подтверждаю: пять стилей различны не только по числам.** Различия видны
одновременно по свету (жёсткое солнце / пасмурная дымка / ночь / закат /
рассеянный день), по палитре (синий / бежевый / чёрный / оранжевый / белый) и
по фактуре (гладкая / зернистая). Ни один не спутать с другим на расстоянии
вытянутой руки.

**И ещё одно, что видно только глазами.** Единственный кадр с человеком —
существующий `bluesky`, и на этом человеке ровно те самые солнечные очки, из-за
которых, по брифу, ArcFace однажды дал 0.3928 при планке 0.35. Четыре моих
референса людей не содержат. То есть риск протечки одежды и аксессуаров в
витрине сосредоточен ровно в одном файле — старом, не моём, и решение по нему
за владельцем.

---

## 4. Свод: три исхода по каждому входу

| вход | прибор | проверено | нарушений | не смогли | вердикт |
|---|---|---|---|---|---|
| `fork_ref_man.png` | `photo_intake` | 3 | 0 | 0 | **годно** |
| `fork_ref_man_waistup.png` | `photo_intake` | 3 | 0 | 0 | **годно** |
| `styleref_nightneon.png` | `style_intake` | 4 | 0 | 0 | **годно** |
| `styleref_whitehall.png` | `style_intake` | 4 | 0 | 0 | **годно** |
| `styleref_filmgrain.png` | `style_intake` | 4 | 0 | 0 | **годно** |
| `styleref_sunsetdune.png` | `style_intake` | 4 | 0 | 0 | **годно** |
| различимость 5 стилей | `differ`, 10 пар | 10 | 0 | 0 | **годно** |
| различимость, негативный контроль | `differ`, 5 пар «сама с собой» | 5 | 5 | 0 | **годно** (обязан был покраснеть — покраснел) |
| годность видеовыхода | — | 0 | 0 | — | **не смогли проверить**: прогона стенда не было |

Комплектность витрины после смены:

```
драйвинги    3 из 5   долг владельца, породить нельзя
стили        5 из 5   ЗАКРЫТО
личности     2 из 2   ЗАКРЫТО (женская была, мужская добавлена)
```

## 5. Что осталось владельцу решить

1. Два драйвинга.
2. `fork_ref_man.png` (288 px, по грудь) или `fork_ref_man_waistup.png`
   (196 px, поясной) — какой из двух остаётся в батче.
3. Очки на `styleref_bluesky.png` и надпись `NYC MARATHON` на
   `fork_ref_gym.png` — оба существующих файла, оба под правилом «брендов и
   аксессуаров не тащим», оба не мои.
4. Считать ли `bluesky` x `filmgrain` = 0.7474 по прокси поводом переделать
   `filmgrain`.
