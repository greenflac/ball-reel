"""Настоящий бэкенд запуска графа в ComfyUI: подать, дождаться, забрать файл.

ЗАЧЕМ. Дыра №2 списка дыр хэндофа: `fork_comfy.render` — мок, и он в этом
честно сознаётся (`RAN`/`MOCK`/`NOTHING`), но исполнять граф нечем. Пока
исполнять нечем, арендованная карта простаивает, пока человек кликает мышью в
веб-интерфейсе. Здесь — HTTP-клиент к ComfyUI: `POST /prompt`, опрос
`GET /history/<prompt_id>` и `GET /queue`, забор файлов через `GET /view`.

ЧТО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ И ЧЕГО НЕ ГОВОРИТ (Ц4, наверх файла).

* **ComfyUI в этой среде нет, и ни один вызов не сделан к настоящему ComfyUI.**
  Весь набор проверок говорит с локальным `http.server`, поднятым в тесте и
  отвечающим телами, списанными с документации ComfyUI. Значит доказано, что
  клиент правильно разбирает ФОРМЫ ответов, которые ему подали, а не что
  настоящий ComfyUI отвечает именно так. Первый прогон на карте имеет право
  опровергнуть любое имя поля отсюда — они собраны в константы наверху ровно
  для того, чтобы правка была однострочной и видимой в диффе.
* **Ни одного кадра не порождено.** Модуль не судит содержимое выхода: он
  сообщает, что файл есть, сколько в нём байт и куда положен. Судят оси
  приёмки — `fork_identity`, `fork_leak`, `fork_seam`.
* **Таймаут ВЫБРАН, а не ИЗМЕРЕН** (И4). Ни одного замера длительности шага
  сэмплера на 14B у проекта нет; см. `SECONDS_PER_STEP_PER_WINDOW`.

ТРИ ИСХОДА, И ТРЕТИЙ НЕ СВОРАЧИВАЕТСЯ НИ В ОДИН ИЗ ДВУХ (Р1). Это главное
свойство модуля, и различаются исходы по ПРИЧИНЕ, а не по удобству вызывающего:

    сервер не отвечает (соединение не встало)      не смогли проверить
    сервер ответил 400 на графе                    не годно   — граф негодный
    задача принята, нода упала с ошибкой           не годно   — с текстом ноды
    задача принята и висит дольше таймаута         не смогли проверить
    задача выполнена, выходных файлов нет          не годно
    задача выполнена, файлы есть и забраны         годно
    файлы есть, но класть некуда — путь занят      не смогли проверить

Последняя строка — единственная спорная, и решение записано здесь: рендер
состоялся, но продукт команды («забрать файл») не получен, а чужой выход не
перетирается. Объявить это успехом значило бы соврать про доставку; объявить
провалом — соврать про граф, который отработал. Поэтому третий исход, и в
`note` печатаются имена файлов, оставшихся на сервере, чтобы забрать их руками.

РЯДОМ С ВЕРДИКТОМ ВСЕГДА ЧИСЛА (Р2, Е3): сколько секунд ждали, сколько опросов
сделали, сколько узлов было в графе, сколько файлов на выходе, сколько байт.
Ноль нарушений при нуле опросов — не успех, и напечатать это нечем, кроме
самих чисел.

ГДЕ ТОЧКА ВНЕДРЕНИЯ ТРАНСПОРТА (Т4). `run` принимает `transport`, `now`,
`sleep` и `log`. Набор проверок поднимает НАСТОЯЩИЙ `http.server` на
`127.0.0.1` со свободным портом и говорит с ним штатным `HttpTransport` — то
есть HTTP разбирается по-настоящему, а в сеть не ходит никто. Часы подменяются
отдельно, иначе проверка таймаута стоила бы двадцати минут.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .fork_comfy import (SECONDS_MAX, WRAP_FPS, WRAP_STEPS, WRAP_WINDOW,
                         NOTHING, RAN)
from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# АДРЕС
# ---------------------------------------------------------------------------

#: Имя переменной окружения с адресом ComfyUI. Отдельной константой, потому что
#: спрашивать её будут из точки входа, из теста и из чужого скрипта — три места,
#: и разъехавшийся строковый литерал на этом проекте уже стоил 1.7 ГБ (Е1).
ENV_ADDRESS = "BALL_REEL_COMFY"

#: Умолчание — штатный адрес ComfyUI. ВЫБРАНО не нами: это умолчание самого
#: ComfyUI (`--listen 127.0.0.1 --port 8188`). Локальный адрес, а не `0.0.0.0`:
#: клиент обязан по умолчанию говорить с картой, стоящей рядом, а не наружу.
DEFAULT_ADDRESS = "127.0.0.1:8188"

#: Таймаут ОДНОГО HTTP-вызова (не всей задачи). ВЫБРАНО этой сменой из того,
#: что `/prompt` и `/history` отвечают из памяти процесса; секунды здесь — это
#: «процесс жив, но занят GIL во время загрузки весов», а не «считает».
CONNECT_TIMEOUT_S = 20.0

# ---------------------------------------------------------------------------
# ПУТИ И ИМЕНА ПОЛЕЙ ПРОТОКОЛА
#
# Все до одного — константы-решения: каждое имя ниже это утверждение о чужом
# контракте, проверенное ТОЛЬКО набором проверок, который сам их и подаёт
# (см. Ц4 наверху). Держать их литералами по коду значило бы, что первый же
# прогон на карте чинится поиском по файлу вместо одной строки.
# ---------------------------------------------------------------------------

PROMPT_PATH = "/prompt"
HISTORY_PATH = "/history/"
QUEUE_PATH = "/queue"
VIEW_PATH = "/view"

#: Тело `POST /prompt`.
FIELD_PROMPT = "prompt"
FIELD_CLIENT_ID = "client_id"

#: Ответ `POST /prompt`: идентификатор задачи и разбор ошибок валидации.
FIELD_PROMPT_ID = "prompt_id"
FIELD_ERROR = "error"
FIELD_NODE_ERRORS = "node_errors"

#: Ответ `GET /history/<id>`.
FIELD_STATUS = "status"
FIELD_OUTPUTS = "outputs"
FIELD_COMPLETED = "completed"
FIELD_MESSAGES = "messages"
MSG_EXECUTION_ERROR = "execution_error"

#: Описание выходного файла внутри `outputs`.
FIELD_FILENAME = "filename"
FIELD_SUBFOLDER = "subfolder"
FIELD_TYPE = "type"

#: Ответ `GET /queue`: два списка, и позиция в записи — второй элемент.
FIELD_QUEUE_RUNNING = "queue_running"
FIELD_QUEUE_PENDING = "queue_pending"
QUEUE_ID_SLOT = 1

#: Коды состояния. 400 — «граф не прошёл валидацию», то есть НАШ дефект, и это
#: единственный код, который сворачивается в «не годно». Всё остальное, чем бы
#: оно ни было, — «не смогли»: про граф оно не говорит ничего.
HTTP_OK = 200
HTTP_BAD_GRAPH = 400

#: Сколько подряд неудачных опросов истории терпим, прежде чем объявить
#: «сервер перестал отвечать». ВЫБРАНО этой сменой: один сбой посреди
#: двадцатиминутного рендера не должен стоить рендера, а три подряд — это уже
#: не рябь. Не ИЗМЕРЕНО: частота сбоев ComfyUI под нагрузкой неизвестна.
HISTORY_RETRIES = 3

# ---------------------------------------------------------------------------
# ТАЙМАУТ ЗАДАЧИ — КОНСТАНТА-РЕШЕНИЕ, И ОНА РАЗЛОЖЕНА НА СЛАГАЕМЫЕ (И4)
#
# Числа стека берутся ИМПОРТОМ из `fork_comfy` (Е1): окно 77 кадров, 3 шага,
# 30 к/с, ролик 5–10 с. Копировать их сюда значило бы завести второй способ
# узнать известное — а именно так на этом проекте уже разъезжались числа.
# ---------------------------------------------------------------------------

#: ВЫБРАНО этой сменой (18.08.2026), НЕ ИЗМЕРЕНО. Основание — единственное,
#: и оно слабое: замера длительности шага сэмплера Wan2.2-Animate-14B у проекта
#: нет ни одного (см. «ЧТО ЖДЁТ КАРТЫ» хэндофа, пункт 7 — `SPEC_TFLOPS = None`).
#: Число взято как заведомо щедрая верхняя оценка одного шага на окне
#: 480x832x77 при переносе блоков (`BLOCKS_TO_SWAP = 38`), потому что цена
#: переноса тоже НЕ ЗАМЕРЕНА. Стоп-условие (Ц9): первый же прогон на карте
#: печатает `waited_s`, и если он расходится с расчётом больше чем вдвое —
#: число заменяется на ИЗМЕРЕНО и этот комментарий переписывается.
SECONDS_PER_STEP_PER_WINDOW = 45.0

#: ВЫБРАНО: загрузка весов с диска в память и на карту до первого шага.
#: Основание — размер, замеренный локом: 15.3 ГиБ на пути GGUF и 26.8 ГиБ на
#: пути fp8. Не ИЗМЕРЕНО: скорость диска арендованной машины неизвестна.
LOAD_SECONDS = 420.0

#: ВЫБРАНО: множитель запаса поверх расчёта. Асимметрия здесь та же, что у
#: полосы роутера: зря подождать лишние минуты дёшево, зря убить готовый рендер
#: за минуту до конца — дорого. Единица означала бы «расчёт точен», а он ВЫБРАН.
TIMEOUT_SLACK = 2.0

#: Пауза между опросами истории. ВЫБРАНО: рендер меряется минутами, поэтому
#: разрешение в 2 с ничего не теряет, а меньше делать нечего — каждый опрос
#: печатает строку, и 20 минут по 0.1 с это 12000 строк, то есть молчание
#: другим способом.
POLL_INTERVAL_S = 2.0


def windows_for(num_frames: int, *, window: int | None = None) -> int:
    """Сколько окон сэмплера потребует ролик такой длины.

    Отдельной функцией, а не строкой внутри расчёта таймаута (Т5): именно здесь
    живёт единственное содержательное допущение — окна идут ПОДРЯД. У обёртки
    соседние окна делят кадры движения, значит настоящее число окон не меньше
    этого. Ошибка в безопасную сторону, и она названа.
    """
    window = WRAP_WINDOW if window is None else window
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) \
            or num_frames <= 0:
        raise ValueError(f"кадров должно быть положительное целое, "
                         f"пришло {num_frames!r}")
    return -(-num_frames // window)


def timeout_for(num_frames: int, *, steps: int | None = None,
                per_step: float | None = None, load: float | None = None,
                slack: float | None = None) -> float:
    """Сколько секунд ждать задачу такой длины. РАСЧЁТ из ВЫБРАННЫХ слагаемых.

    Формула названа целиком, чтобы её можно было оспорить числом, а не мнением:

        таймаут = загрузка + окна * шаги * секунды_на_шаг * запас

    Для целевых 300 кадров (10 с при 30 к/с): окон 4, шагов 3, значит
    420 + 4*3*45*2 = 1500 с, то есть 25 минут.
    """
    steps = WRAP_STEPS if steps is None else steps
    per_step = SECONDS_PER_STEP_PER_WINDOW if per_step is None else per_step
    load = LOAD_SECONDS if load is None else load
    slack = TIMEOUT_SLACK if slack is None else slack
    return load + windows_for(num_frames) * steps * per_step * slack


#: Умолчание таймаута — по самому длинному ролику продукта (10 с при 30 к/с).
DEFAULT_TIMEOUT_S = timeout_for(int(WRAP_FPS * SECONDS_MAX))


# ---------------------------------------------------------------------------
# ТРАНСПОРТ
# ---------------------------------------------------------------------------


class Reply:
    """Ответ транспорта. Три поля, потому что исходов три.

    `status is None` означает РОВНО одно: ответа не было — соединение не встало,
    оборвалось, не уложилось в `CONNECT_TIMEOUT_S`. Свернуть это в код 500
    значило бы объявить «сервер сказал» там, где сервер молчал.
    """

    __slots__ = ("status", "body", "error")

    def __init__(self, status: int | None, body: bytes = b"",
                 error: str | None = None):
        self.status = status
        self.body = body
        self.error = error

    def json(self):
        """Тело как JSON или `None`, если оно не разбирается."""
        try:
            return json.loads(self.body.decode("utf-8"))
        except Exception:
            return None

    def __repr__(self) -> str:
        return (f"Reply(status={self.status}, {len(self.body)} байт, "
                f"error={self.error!r})")


class HttpTransport:
    """Штатный транспорт на `urllib`. Наружу не ходит — адрес задаёт вызывающий.

    Отдельным классом, а не тремя вызовами `urlopen` по коду, потому что это и
    есть точка внедрения: набор проверок подставляет сюда либо этот же класс,
    смотрящий на локальный `http.server`, либо подставной объект.
    """

    def __init__(self, *, timeout: float = CONNECT_TIMEOUT_S):
        self.timeout = timeout

    def _do(self, req: urllib.request.Request) -> Reply:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return Reply(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            # Сервер ОТВЕТИЛ, просто плохо: код есть, тело есть, и в нём у
            # ComfyUI лежит разбор ошибки валидации. Терять его нельзя.
            try:
                body = exc.read()
            except Exception:
                body = b""
            return Reply(exc.code, body)
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            return Reply(None, b"", error=repr(exc))

    def get(self, url: str) -> Reply:
        return self._do(urllib.request.Request(url, method="GET"))

    def post(self, url: str, payload: dict) -> Reply:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        return self._do(req)


def address(env: dict | None = None) -> str:
    """Адрес ComfyUI: из окружения, иначе умолчание.

    `env` параметром, а не чтением `os.environ` внутри (Т5): иначе проверить
    обе ветки можно было бы только правкой окружения процесса.
    """
    env = os.environ if env is None else env
    return str(env.get(ENV_ADDRESS) or DEFAULT_ADDRESS).strip().rstrip("/")


def base_url(addr: str | None = None, *, env: dict | None = None) -> str:
    """Полный корень URL. Схему дописываем сами, если её не задали."""
    addr = address(env) if addr is None else addr.strip().rstrip("/")
    if "://" in addr:
        return addr
    return f"http://{addr}"


# ---------------------------------------------------------------------------
# ГРАФ: ФОРМАТ API, А НЕ ФОРМАТ UI
# ---------------------------------------------------------------------------


def check_graph(graph) -> list:
    """Нарушения графа ДО отправки. Список причин, а не флаг.

    Дефект, ради которого написано: `fork_comfy.derive_wrapper` производит граф
    в формате UI (`nodes` списком, `links` отдельно), а `/prompt` принимает
    формат API (`{id: {class_type, inputs}}`). Подать первый вместо второго —
    самая дешёвая и самая вероятная ошибка этого стыка, и без этой проверки она
    приезжала бы как 400 без объяснения, за один сетевой круг вместо нуля (П2).
    """
    bad: list[str] = []
    if not isinstance(graph, dict):
        return [f"граф должен быть словарём, пришло {type(graph).__name__}"]
    if not graph:
        return ["граф пуст: узлов 0"]
    if "nodes" in graph and isinstance(graph.get("nodes"), list):
        return ["граф в формате UI (ключ `nodes` списком), а /prompt принимает "
                "формат API: {id: {class_type, inputs}}. Это не отказ сервера, "
                "это неверный формат на нашей стороне"]
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            bad.append(f"узел {node_id}: не словарь, а {type(node).__name__}")
            continue
        if not node.get("class_type"):
            bad.append(f"узел {node_id}: нет class_type")
        if not isinstance(node.get("inputs"), dict):
            bad.append(f"узел {node_id}: inputs не словарь")
    return bad


def graph_nodes(graph) -> int:
    """Сколько узлов в графе. Ноль — тоже число, и его печатают (Р2)."""
    return len(graph) if isinstance(graph, dict) else 0


# ---------------------------------------------------------------------------
# РАЗБОР ОТВЕТОВ — ЧИСТЫМИ ФУНКЦИЯМИ (Т5)
# ---------------------------------------------------------------------------


def classify_submit(reply: Reply) -> dict:
    """Что означает ответ на `POST /prompt`. Три исхода.

    Вынесено из `submit` отдельной функцией именно потому, что здесь и живёт
    развилка, ради которой написан модуль: 400 это «граф негодный» (не годно),
    молчание это «мы ничего не узнали» (не смогли), и путать их нельзя.
    """
    if reply.status is None:
        return {"outcome": UNMEASURED, "prompt_id": None,
                "problems": [f"сервер не ответил: {reply.error}"],
                "note": (f"{UNMEASURED}: ComfyUI не отвечает ({reply.error}). "
                         f"Про граф это не говорит ничего — он не доехал")}
    if reply.status == HTTP_BAD_GRAPH:
        obj = reply.json() or {}
        why = obj.get(FIELD_ERROR) or {}
        detail = why.get("message") if isinstance(why, dict) else str(why)
        nodes = obj.get(FIELD_NODE_ERRORS) or {}
        said = detail or "без объяснения"
        return {"outcome": FAIL, "prompt_id": None,
                "problems": [f"{HTTP_BAD_GRAPH} на графе: {said}"]
                            + [f"узел {k}: {v}" for k, v in sorted(nodes.items())],
                "note": (f"{FAIL}: сервер отверг граф кодом {HTTP_BAD_GRAPH} — "
                         f"{said}; узлов с ошибками {len(nodes)}")}
    if reply.status != HTTP_OK:
        return {"outcome": UNMEASURED, "prompt_id": None,
                "problems": [f"код {reply.status} на {PROMPT_PATH}"],
                "note": (f"{UNMEASURED}: сервер ответил {reply.status}. Это не "
                         f"{HTTP_BAD_GRAPH}, то есть про сам граф не сказано "
                         f"ничего")}
    obj = reply.json()
    if not isinstance(obj, dict) or not obj.get(FIELD_PROMPT_ID):
        return {"outcome": UNMEASURED, "prompt_id": None,
                "problems": [f"в ответе {HTTP_OK} нет поля {FIELD_PROMPT_ID}"],
                "note": (f"{UNMEASURED}: задача вроде принята, но её нечем "
                         f"опрашивать — поля {FIELD_PROMPT_ID} в ответе нет")}
    return {"outcome": PASS, "prompt_id": str(obj[FIELD_PROMPT_ID]),
            "problems": [],
            "note": f"задача принята, {FIELD_PROMPT_ID}={obj[FIELD_PROMPT_ID]}"}


def output_files(entry: dict) -> list:
    """Выходные файлы задачи из записи истории.

    Имена типов выходов (`images`, `gifs`, `videos`, ...) намеренно НЕ
    перечисляются списком: у каждой ноды-сохранялки он свой, и закрытый список
    молча терял бы файл, добавленный чужой нодой. Признак файла — наличие поля
    `filename`, и это наблюдение, а не соглашение (Е2).
    """
    def _by_number(pair):
        # Порядок узлов ЧИСЛОВОЙ, а не строковый: `sorted` по строкам ставит
        # узел «12» перед узлом «9», и файлы поехали бы в отчёт в порядке,
        # который человек читает как ошибку графа. Поймано собственным тестом.
        node_id = pair[0]
        return (0, int(node_id)) if str(node_id).isdigit() else (1, str(node_id))

    files: list[dict] = []
    for node_id, out in sorted((entry.get(FIELD_OUTPUTS) or {}).items(),
                               key=_by_number):
        if not isinstance(out, dict):
            continue
        for kind, items in sorted(out.items()):
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get(FIELD_FILENAME):
                    files.append({
                        "node": str(node_id), "kind": str(kind),
                        FIELD_FILENAME: str(item[FIELD_FILENAME]),
                        FIELD_SUBFOLDER: str(item.get(FIELD_SUBFOLDER, "")),
                        FIELD_TYPE: str(item.get(FIELD_TYPE, "output")),
                    })
    return files


def node_error(entry: dict) -> str | None:
    """Текст ошибки ноды, если задача упала. Иначе `None`.

    Текст возвращается ЦЕЛИКОМ и печатается в вердикт: «задача упала» без имени
    ноды и сообщения — это приглашение открыть веб-интерфейс руками, то есть
    ровно то, ради отмены чего написан модуль.
    """
    status = entry.get(FIELD_STATUS) or {}
    for msg in status.get(FIELD_MESSAGES) or []:
        if not (isinstance(msg, (list, tuple)) and len(msg) >= 2):
            continue
        if msg[0] != MSG_EXECUTION_ERROR:
            continue
        data = msg[1] if isinstance(msg[1], dict) else {}
        node = data.get("node_type") or data.get("node_id") or "?"
        text = data.get("exception_message") or data.get("exception_type") or ""
        return f"{node}: {text}".strip(": ")
    return None


def is_completed(entry: dict) -> bool:
    """Отработала ли задача. Флаг читается из `status`, а не из наличия
    `outputs`: выполненная задача без выходов — отдельный исход (не годно), и
    вывести его можно только различая «не доделала» и «доделала пусто»."""
    status = entry.get(FIELD_STATUS)
    return bool(isinstance(status, dict) and status.get(FIELD_COMPLETED))


def classify_history(entry: dict | None) -> dict | None:
    """Что означает запись истории. `None` = ещё рано, продолжаем ждать.

    Четвёртое значение (`None`) отделено от трёх вердиктов намеренно: «задача
    ещё идёт» — это не исход, а состояние, и сворачивание его в «не смогли»
    убило бы каждый рендер на первом же опросе.
    """
    if entry is None:
        return None
    err = node_error(entry)
    if err:
        return {"outcome": FAIL, "files": [],
                "note": f"{FAIL}: нода упала — {err}"}
    if not is_completed(entry):
        return None
    files = output_files(entry)
    if not files:
        return {"outcome": FAIL, "files": [],
                "note": (f"{FAIL}: задача отработала и не оставила ни одного "
                         f"файла. Граф исполнился, сохранять в нём нечему — "
                         f"это негодный граф, а не сбой")}
    return {"outcome": PASS, "files": files,
            "note": f"задача отработала, файлов {len(files)}"}


def queue_where(queue: dict | None, prompt_id: str) -> str:
    """Где задача: исполняется, в очереди N-й, или её в очереди нет.

    Третье состояние («нет в очереди») не сворачивается в «исполняется»:
    так выглядит и уже доделанная задача, и задача, потерянная перезапуском
    сервера, — и в опросе истории эти два различатся, а здесь нет.
    """
    if not isinstance(queue, dict):
        return "очередь не прочитана"
    for slot, name in ((FIELD_QUEUE_RUNNING, "исполняется"),
                       (FIELD_QUEUE_PENDING, "в очереди")):
        items = queue.get(slot) or []
        for pos, item in enumerate(items):
            if (isinstance(item, (list, tuple)) and len(item) > QUEUE_ID_SLOT
                    and str(item[QUEUE_ID_SLOT]) == str(prompt_id)):
                return name if name == "исполняется" else f"{name}, впереди {pos}"
    return "в очереди нет"


# ---------------------------------------------------------------------------
# ВЫЗОВЫ
# ---------------------------------------------------------------------------


def submit(graph: dict, *, transport, url: str, client_id: str) -> dict:
    """`POST /prompt`. Проверка формата графа — ДО сетевого круга (П2)."""
    bad = check_graph(graph)
    if bad:
        return {"outcome": FAIL, "prompt_id": None, "problems": bad,
                "note": (f"{FAIL}: граф не отправлен, нарушений {len(bad)} — "
                         + "; ".join(bad))}
    reply = transport.post(url + PROMPT_PATH,
                           {FIELD_PROMPT: graph, FIELD_CLIENT_ID: client_id})
    return classify_submit(reply)


def _get_json(transport, url: str):
    reply = transport.get(url)
    if reply.status is None:
        return None, f"нет ответа: {reply.error}"
    if reply.status != HTTP_OK:
        return None, f"код {reply.status}"
    obj = reply.json()
    if obj is None:
        return None, "тело не разбирается как JSON"
    return obj, None


def wait(prompt_id: str, *, transport, url: str,
         timeout_s: float = DEFAULT_TIMEOUT_S,
         poll_s: float = POLL_INTERVAL_S, now=time.monotonic,
         sleep=time.sleep, log=None) -> dict:
    """Опрос истории до исхода или до таймаута. Печатает КАЖДУЮ попытку.

    Молчащее ожидание в двадцать минут неотличимо от зависания — поэтому у
    каждого опроса печатается его номер, прошедшее время и ГДЕ задача:
    в очереди, исполняется или её в очереди уже нет. Это не украшение: «висит
    в очереди за чужой задачей» и «считает» лечатся по-разному.
    """
    log = (lambda s: None) if log is None else log
    started = now()
    polls = 0
    misses = 0
    where = "не спрошено"
    while True:
        polls += 1
        entry_obj, err = _get_json(transport, f"{url}{HISTORY_PATH}{prompt_id}")
        waited = now() - started
        if err is not None:
            misses += 1
            log(f"опрос {polls}: история недоступна ({err}), подряд неудач "
                f"{misses} из {HISTORY_RETRIES}, прошло {waited:.1f} с")
            if misses >= HISTORY_RETRIES:
                return {"outcome": UNMEASURED, "files": [], "polls": polls,
                        "waited_s": waited, "where": where,
                        "problems": [f"история недоступна {misses} раза подряд: "
                                     f"{err}"],
                        "note": (f"{UNMEASURED}: сервер перестал отвечать после "
                                 f"{polls} опросов за {waited:.1f} с. Задача "
                                 f"{prompt_id} могла и досчитать — мы этого не "
                                 f"узнали")}
        else:
            misses = 0
            entry = entry_obj.get(str(prompt_id)) if isinstance(entry_obj, dict) \
                else None
            verdict = classify_history(entry)
            if verdict is not None:
                verdict.update({"polls": polls, "waited_s": waited,
                                "where": "отработала", "problems":
                                    [] if verdict["outcome"] == PASS
                                    else [verdict["note"]]})
                log(f"опрос {polls}: {verdict['note']}, прошло {waited:.1f} с")
                return verdict
            queue, _ = _get_json(transport, url + QUEUE_PATH)
            where = queue_where(queue, prompt_id)
            log(f"опрос {polls}: {where}, прошло {waited:.1f} с из "
                f"{timeout_s:.0f} с")

        if now() - started >= timeout_s:
            waited = now() - started
            return {"outcome": UNMEASURED, "files": [], "polls": polls,
                    "waited_s": waited, "where": where,
                    "problems": [f"таймаут {timeout_s:.0f} с"],
                    "note": (f"{UNMEASURED}: задача {prompt_id} не отработала "
                             f"за {waited:.1f} с ({polls} опросов), последнее "
                             f"известное состояние — «{where}». Она НЕ отменена "
                             f"и может досчитать: это «не смогли дождаться», а "
                             f"не «не годно»")}
        sleep(poll_s)


def fetch(file_spec: dict, *, transport, url: str) -> tuple:
    """Забрать один файл через `GET /view`. Возвращает `(байты, причина)`."""
    query = urllib.parse.urlencode({
        FIELD_FILENAME: file_spec[FIELD_FILENAME],
        FIELD_SUBFOLDER: file_spec.get(FIELD_SUBFOLDER, ""),
        FIELD_TYPE: file_spec.get(FIELD_TYPE, "output"),
    })
    reply = transport.get(f"{url}{VIEW_PATH}?{query}")
    if reply.status is None:
        return None, f"{file_spec[FIELD_FILENAME]}: нет ответа ({reply.error})"
    if reply.status != HTTP_OK:
        return None, f"{file_spec[FIELD_FILENAME]}: код {reply.status}"
    if not reply.body:
        return None, f"{file_spec[FIELD_FILENAME]}: пустое тело"
    return reply.body, None


def save(out_dir, name: str, data: bytes) -> dict:
    """Положить файл, НЕ перетирая чужое. Три исхода, и второй — не ошибка.

    * файла не было — записан;
    * файл есть и байты совпадают — «уже лежит», повторный запуск идемпотентен
      и не пишет второй раз;
    * файл есть и байты ДРУГИЕ — не тронут. Это чужой выход: перетереть его
      значило бы уничтожить единственный экземпляр чужого рендера ради нашего,
      причём молча.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if target.exists():
        if target.read_bytes() == data:
            return {"outcome": PASS, "path": target, "bytes": len(data),
                    "note": f"{name}: уже лежит, байты совпадают"}
        return {"outcome": UNMEASURED, "path": None, "bytes": len(data),
                "note": (f"{name}: путь занят ДРУГИМ файлом "
                         f"({target.stat().st_size} байт против {len(data)}), "
                         f"не перетираю")}
    target.write_bytes(data)
    return {"outcome": PASS, "path": target, "bytes": len(data),
            "note": f"{name}: записан, {len(data)} байт"}


def run(graph: dict, *, out_dir, transport=None, url: str | None = None,
        env: dict | None = None, client_id: str | None = None,
        timeout_s: float | None = None, poll_s: float = POLL_INTERVAL_S,
        now=time.monotonic, sleep=time.sleep, log=None) -> dict:
    """Одна команда: подать граф, дождаться, забрать файлы. Вердикт и ЧИСЛА.

    Возвращает словарь той же формы, что `fork_comfy.render`: `outcome`,
    `source`, `problems`, `note` — плюс числа `waited_s`, `polls`, `nodes`,
    `files`, `bytes`. `source` здесь может быть только `прогнали` или
    `не смогли`: мока в этом модуле нет по построению, кадры приходят с чужой
    машины, и подделать их нечем.
    """
    log = (lambda s: None) if log is None else log
    url = base_url(env=env) if url is None else base_url(url)
    client_id = str(uuid.uuid4()) if client_id is None else str(client_id)
    nodes = graph_nodes(graph)
    started = now()

    def result(outcome, *, files=0, size=0, waited=None, polls=0, problems=(),
               note=""):
        waited = (now() - started) if waited is None else waited
        numbers = (f"ждали {waited:.1f} с, опросов {polls}, узлов в графе "
                   f"{nodes}, файлов {files}, байт {size}")
        return {"outcome": outcome,
                "source": RAN if outcome == PASS else NOTHING,
                "problems": list(problems), "waited_s": waited, "polls": polls,
                "nodes": nodes, "files": files, "bytes": size,
                "client_id": client_id, "url": url,
                "note": f"{note} [{numbers}]"}

    log(f"адрес {url}, узлов в графе {nodes}, client_id {client_id}")
    accepted = submit(graph, transport=transport, url=url, client_id=client_id)
    if accepted["outcome"] != PASS:
        return result(accepted["outcome"], problems=accepted["problems"],
                      note=accepted["note"])

    prompt_id = accepted["prompt_id"]
    log(f"задача принята: {prompt_id}")
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    waited = wait(prompt_id, transport=transport, url=url, timeout_s=timeout_s,
                  poll_s=poll_s, now=now, sleep=sleep, log=log)
    if waited["outcome"] != PASS:
        return result(waited["outcome"], waited=waited["waited_s"],
                      polls=waited["polls"], problems=waited["problems"],
                      note=waited["note"])

    problems: list[str] = []
    size = 0
    taken = 0
    left: list[str] = []
    for spec in waited["files"]:
        data, why = fetch(spec, transport=transport, url=url)
        if data is None:
            problems.append(why)
            left.append(spec[FIELD_FILENAME])
            continue
        put = save(Path(out_dir) / prompt_id, spec[FIELD_FILENAME], data)
        log(f"{spec['kind']} от узла {spec['node']}: {put['note']}")
        if put["outcome"] != PASS:
            problems.append(put["note"])
            left.append(spec[FIELD_FILENAME])
            continue
        taken += 1
        size += put["bytes"]

    if problems:
        return result(UNMEASURED, files=taken, size=size,
                      waited=waited["waited_s"], polls=waited["polls"],
                      problems=problems,
                      note=(f"{UNMEASURED}: рендер состоялся, забрано {taken} "
                            f"из {len(waited['files'])} файлов. Осталось на "
                            f"сервере: {', '.join(left)}. Чужой выход не "
                            f"перетёрт — это отказ доставки, а не приговор "
                            f"графу"))
    return result(PASS, files=taken, size=size, waited=waited["waited_s"],
                  polls=waited["polls"],
                  note=(f"{PASS}: задача {prompt_id} отработала, забрано "
                        f"{taken} файлов в {Path(out_dir) / prompt_id}"))


#: Код возврата по вердикту. Три исхода — три кода (Р2): «не смогли» обязано
#: отличаться от успеха, иначе таймаут в скрипте отгрузки читается как готовый
#: ролик. Ноль ошибок при нуле опросов кодом 0 не бывает.
EXIT_CODES = {PASS: 0, FAIL: 1, UNMEASURED: 2}


def main(argv=None) -> int:
    """Точка входа: `python3 -m ball_reel.fork_backend граф.json выход/`."""
    import argparse

    p = argparse.ArgumentParser(description="запуск графа в ComfyUI")
    p.add_argument("graph", help="граф в формате API (json)")
    p.add_argument("out_dir", help="куда класть выходные файлы")
    p.add_argument("--address", default=None,
                   help=f"адрес ComfyUI (иначе ${ENV_ADDRESS}, иначе "
                        f"{DEFAULT_ADDRESS})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--client-id", default=None)
    args = p.parse_args(argv)

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    out = run(graph, out_dir=args.out_dir, transport=HttpTransport(),
              url=args.address, client_id=args.client_id,
              timeout_s=args.timeout, log=print)
    # Вердикт печатается ОДИН раз: `note` уже начинается со слова исхода, и
    # первая же пара глаз на живом прогоне прочитала «годно: годно: ...» (П3).
    print(f"ВЕРДИКТ: {out['note']}")
    for problem in out["problems"]:
        print(f"  нарушение: {problem}")
    return EXIT_CODES[out["outcome"]]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
