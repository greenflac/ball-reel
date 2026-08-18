"""Набор проверок бэкенда ComfyUI. В СЕТЬ НЕ ХОДИТ, и это обеспечено устройством.

КАК ИМЕННО ОБЕСПЕЧЕНО (Т4). Каждая проверка, которой нужен сервер, поднимает
`http.server` в потоке на `127.0.0.1` со СВОБОДНЫМ ПОРТОМ, полученным у ядра, и
говорит с ним ШТАТНЫМ `HttpTransport` — тем самым, что поедет на карту. То есть
HTTP разбирается по-настоящему (коды, тела, query у `/view`), но адрес назначения
физически не может указывать наружу: его выдало ядро секунду назад. Проверка,
зеленеющая при поднятом ComfyUI и краснеющая без него, здесь невозможна — своего
ComfyUI набор поднимает сам, а чужого не ищет.

Два места говорят с подставным транспортом, и оба названы: проверка умолчания
таймаута (751 опрос по HTTP — это минуты вместо миллисекунд) и проверка того,
что негодный формат графа НЕ ДОЕЗЖАЕТ до сети (транспорт, который падает при
любом вызове, — единственный способ показать отсутствие вызова).

ЧАСЫ ПОДМЕНЯЮТСЯ ВСЕГДА. Настоящее ожидание — это проверка длиной в таймаут,
то есть проверка, которую никто не будет гонять.

НЕГАТИВНЫЕ КОНТРОЛИ (И5) ЗАВЕДЕНЫ ЯВНО И НАЗВАНЫ:
  * `test_negative_control_dead_server_is_unmeasured` — вход, на котором прибор
    ОБЯЗАН сказать «не смогли проверить»: порт, на котором никто не слушает;
  * `test_negative_control_rejected_graph_is_a_failure` — вход, на котором он
    ОБЯЗАН сказать «не годно»: сервер отвечает 400.
Без обоих зелёный набор означал бы только, что прибор умеет говорить «годно».
"""

from __future__ import annotations

import json
import socket
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from ball_reel import fork_backend as fb


# ---------------------------------------------------------------------------
# ЛОКАЛЬНЫЙ COMFYUI: НАСТОЯЩИЙ HTTP, НО СВОЙ
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # без этого набор печатает лог сервера
        pass

    @property
    def script(self):
        return self.server.script

    def _send(self, status, payload, *, raw=False):
        body = payload if raw else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",
                         "application/octet-stream" if raw else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        self.script.posted.append((self.path, body))
        status, payload = self.script.prompt_reply
        self._send(status, payload)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path)
        self.script.got.append(self.path)
        if path.path.startswith("/history/"):
            status, payload = self.script.next_history()
            self._send(status, payload)
        elif path.path == "/queue":
            self._send(200, self.script.queue)
        elif path.path == "/view":
            q = dict(urllib.parse.parse_qsl(path.query, keep_blank_values=True))
            self.script.views.append(q)
            data = self.script.files.get(q.get("filename"))
            if data is None:
                self._send(404, {"error": "нет такого"})
            else:
                self._send(200, data, raw=True)
        else:
            self._send(404, {"error": "нет такого пути"})


class Script:
    """Что именно отвечает наш ComfyUI. Ответы — литералы (Т2), списанные с
    документации ComfyUI и НЕ импортированные из проверяемого модуля."""

    def __init__(self, *, prompt_reply=None, history=None, queue=None,
                 files=None):
        self.prompt_reply = prompt_reply or (200, {"prompt_id": "abc-123",
                                                   "number": 1})
        self.history = list(history or [])
        self.queue = queue or {"queue_running": [], "queue_pending": []}
        self.files = dict(files or {})
        self.posted, self.got, self.views = [], [], []

    def next_history(self):
        if not self.history:
            return 200, {}
        item = self.history[0] if len(self.history) == 1 else self.history.pop(0)
        return item


class Comfy:
    """Сервер на свободном порту, выданном ядром. Наружу указать не может."""

    def __init__(self, script):
        self.script = script
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.script = script
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.httpd.server_address[:2]
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class Clock:
    """Часы под подмену: `sleep` не спит, а двигает время."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class BlackHole:
    """Сокет, который ПРИНИМАЕТ соединение и молчит. Отдельно от `dead_port`:
    отказ в соединении мгновенен, а вот молчащий сервер отличает «не отвечает»
    только таймаут — то есть проверять `CONNECT_TIMEOUT_S` больше нечем."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.held = []
        self.stop = threading.Event()

    def _serve(self):
        self.sock.settimeout(0.2)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.held.append(conn)  # принято и не отвечено

    def __enter__(self):
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        for conn in self.held:
            conn.close()
        self.sock.close()


def dead_port() -> int:
    """Порт, на котором ГАРАНТИРОВАННО никто не слушает: занят и отпущен."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


#: Граф в формате API — литералом, а не сборкой из проверяемого модуля (Т2).
GOOD_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 3}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
}

#: Ответ истории отработавшей задачи — форма списана с документации ComfyUI.
DONE_HISTORY = (200, {"abc-123": {
    "status": {"status_str": "success", "completed": True,
               "messages": [["execution_start", {}]]},
    "outputs": {"9": {"images": [{"filename": "ball_00001_.png",
                                 "subfolder": "", "type": "output"}]}},
}})

RUNNING_HISTORY = (200, {})


def render(script, *, graph=None, out_dir, timeout_s=100.0, poll_s=None,
           lines=None):
    """Один прогон против локального сервера штатным транспортом."""
    clock = Clock()
    with Comfy(script) as comfy:
        kw = {} if poll_s is None else {"poll_s": poll_s}
        out = fb.run(GOOD_GRAPH if graph is None else graph,
                     out_dir=out_dir, transport=fb.HttpTransport(),
                     url=comfy.url, timeout_s=timeout_s, now=clock.now,
                     sleep=clock.sleep,
                     log=(lines.append if lines is not None else None), **kw)
    return out


# ---------------------------------------------------------------------------
# ГОДНО
# ---------------------------------------------------------------------------


class TheHappyPath(unittest.TestCase):
    def test_a_finished_job_is_pass_and_the_file_lands_on_disk(self):
        script = Script(history=[RUNNING_HISTORY, DONE_HISTORY],
                        files={"ball_00001_.png": b"PNG" * 40})
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
            self.assertEqual(out["outcome"], "годно", out["note"])
            self.assertEqual(out["source"], "прогнали")
            self.assertEqual(out["files"], 1)
            self.assertEqual(out["bytes"], 120)
            self.assertEqual(out["nodes"], 2)
            self.assertEqual(out["polls"], 2)
            self.assertEqual(out["problems"], [])
            got = Path(tmp) / "abc-123" / "ball_00001_.png"
            self.assertTrue(got.exists(), f"файла нет: {got}")
            self.assertEqual(got.read_bytes(), b"PNG" * 40)

    def test_the_verdict_carries_numbers_and_not_only_a_word(self):
        # Р2/Е3: «годно» без чисел неотличимо от «годно на нуле проверок».
        script = Script(history=[DONE_HISTORY], files={"ball_00001_.png": b"x"})
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        for word in ("ждали", "опросов", "узлов в графе", "файлов", "байт"):
            self.assertIn(word, out["note"], out["note"])
        for key in ("waited_s", "polls", "nodes", "files", "bytes"):
            self.assertIn(key, out)

    def test_the_graph_travels_in_the_documented_envelope(self):
        script = Script(history=[DONE_HISTORY], files={"ball_00001_.png": b"x"})
        with TemporaryDirectory() as tmp:
            render(script, out_dir=tmp)
        path, body = script.posted[0]
        self.assertEqual(path, "/prompt")
        self.assertEqual(sorted(body), ["client_id", "prompt"])
        self.assertEqual(body["prompt"], GOOD_GRAPH)
        self.assertTrue(body["client_id"])

    def test_the_file_is_asked_for_by_name_subfolder_and_type(self):
        script = Script(history=[DONE_HISTORY], files={"ball_00001_.png": b"x"})
        with TemporaryDirectory() as tmp:
            render(script, out_dir=tmp)
        self.assertEqual(script.views, [{"filename": "ball_00001_.png",
                                         "subfolder": "", "type": "output"}])

    def test_progress_is_printed_for_every_poll_with_place_and_seconds(self):
        script = Script(
            history=[RUNNING_HISTORY, RUNNING_HISTORY, DONE_HISTORY],
            queue={"queue_running": [[1, "abc-123", {}, {}, []]],
                   "queue_pending": []},
            files={"ball_00001_.png": b"x"})
        lines = []
        with TemporaryDirectory() as tmp:
            render(script, out_dir=tmp, lines=lines)
        polls = [l for l in lines if l.startswith("опрос ")]
        self.assertEqual(len(polls), 3, lines)
        self.assertIn("исполняется", polls[0])
        self.assertIn("прошло", polls[0])

    def test_a_job_waiting_behind_another_says_it_is_queued(self):
        script = Script(
            history=[RUNNING_HISTORY, DONE_HISTORY],
            queue={"queue_running": [[1, "чужая", {}, {}, []]],
                   "queue_pending": [[2, "abc-123", {}, {}, []]]},
            files={"ball_00001_.png": b"x"})
        lines = []
        with TemporaryDirectory() as tmp:
            render(script, out_dir=tmp, lines=lines)
        self.assertIn("в очереди, впереди 0", lines[2])


# ---------------------------------------------------------------------------
# НЕ ГОДНО — И НЕГАТИВНЫЙ КОНТРОЛЬ НА НЕГО (И5)
# ---------------------------------------------------------------------------


class TheFailures(unittest.TestCase):
    def test_negative_control_rejected_graph_is_a_failure(self):
        """Вход, на котором прибор ОБЯЗАН сказать «не годно»."""
        script = Script(prompt_reply=(400, {
            "error": {"type": "prompt_outputs_failed_validation",
                      "message": "Prompt outputs failed validation"},
            "node_errors": {"3": "Required input is missing: model"}}))
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не годно", out["note"])
        self.assertEqual(out["source"], "не смогли")
        self.assertIn("Prompt outputs failed validation", out["note"])
        self.assertTrue(any("Required input is missing" in p
                            for p in out["problems"]), out["problems"])

    def test_a_node_that_blew_up_is_a_failure_with_the_nodes_own_text(self):
        blown = (200, {"abc-123": {
            "status": {"completed": False, "status_str": "error", "messages": [
                ["execution_error", {"node_id": "3", "node_type": "KSampler",
                                     "exception_type": "torch.OutOfMemoryError",
                                     "exception_message": "CUDA out of memory"}]]},
            "outputs": {}}})
        script = Script(history=[blown])
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не годно", out["note"])
        self.assertIn("KSampler", out["note"])
        self.assertIn("CUDA out of memory", out["note"])

    def test_a_job_that_finished_with_no_files_is_a_failure_not_a_success(self):
        empty = (200, {"abc-123": {"status": {"completed": True}, "outputs": {}}})
        script = Script(history=[empty])
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не годно", out["note"])
        self.assertEqual(out["files"], 0)
        self.assertEqual(out["bytes"], 0)

    def test_a_ui_format_graph_is_refused_before_any_network_call(self):
        """П2: дешёвая проверка раньше дорогой. Транспорт падает при вызове —
        то есть отсутствие вызова доказано, а не заявлено."""

        class Explodes:
            def post(self, *a, **k):
                raise AssertionError("сетевой вызов на негодном формате графа")

            get = post

        ui_graph = {"nodes": [{"id": 1, "type": "KSampler"}], "links": []}
        with TemporaryDirectory() as tmp:
            out = fb.run(ui_graph, out_dir=tmp, transport=Explodes(),
                         url="http://127.0.0.1:1")
        self.assertEqual(out["outcome"], "не годно", out["note"])
        self.assertIn("UI", out["note"])

    def test_a_node_without_class_type_is_named_by_its_id(self):
        bad = {"3": {"inputs": {}}, "9": {"class_type": "SaveImage",
                                          "inputs": {}}}
        self.assertEqual(fb.check_graph(bad), ["узел 3: нет class_type"])

    def test_an_empty_graph_is_a_failure_and_says_zero(self):
        self.assertEqual(fb.check_graph({}), ["граф пуст: узлов 0"])


# ---------------------------------------------------------------------------
# НЕ СМОГЛИ — И НЕГАТИВНЫЙ КОНТРОЛЬ НА НЕГО (И5)
# ---------------------------------------------------------------------------


class TheUnmeasured(unittest.TestCase):
    def test_negative_control_dead_server_is_unmeasured(self):
        """Вход, на котором прибор ОБЯЗАН сказать «не смогли проверить»:
        порт занят и отпущен, слушать на нём некому."""
        with TemporaryDirectory() as tmp:
            out = fb.run(GOOD_GRAPH, out_dir=tmp, transport=fb.HttpTransport(),
                         url=f"http://127.0.0.1:{dead_port()}", timeout_s=5.0)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertEqual(out["files"], 0)
        self.assertIn("не отвечает", out["note"])

    def test_a_server_answering_500_is_unmeasured_and_not_a_bad_graph(self):
        script = Script(prompt_reply=(500, {"error": "внутренняя"}))
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertIn("500", out["note"])

    def test_an_accepted_job_without_a_prompt_id_is_unmeasured(self):
        script = Script(prompt_reply=(200, {"number": 1}))
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertIn("prompt_id", out["note"])

    def test_a_job_hanging_past_the_timeout_is_unmeasured(self):
        script = Script(history=[RUNNING_HISTORY],
                        queue={"queue_running": [],
                               "queue_pending": [[2, "abc-123", {}, {}, []]]})
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp, timeout_s=10.0)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertEqual(out["waited_s"], 10.0)
        self.assertEqual(out["polls"], 6)  # опросы на 0,2,4,6,8,10 с
        self.assertIn("в очереди", out["note"])
        self.assertIn("НЕ отменена", out["note"])

    def test_a_server_that_dies_mid_render_is_unmeasured_after_three_tries(self):
        script = Script(history=[RUNNING_HISTORY])
        clock = Clock()
        with Comfy(script) as comfy:
            url = comfy.url
            out_first = None
        # сервер закрыт: задача принята была, история недоступна
        with TemporaryDirectory() as tmp:
            out = fb.wait("abc-123", transport=fb.HttpTransport(), url=url,
                          timeout_s=1000.0, now=clock.now, sleep=clock.sleep)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertEqual(out["polls"], 3)
        self.assertIn("перестал отвечать", out["note"])
        self.assertIsNone(out_first)

    def test_a_single_blip_does_not_kill_a_running_render(self):
        script = Script(history=[(500, {}), (500, {}), DONE_HISTORY],
                        files={"ball_00001_.png": b"x"})
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "годно", out["note"])
        self.assertEqual(out["polls"], 3)

    def test_a_file_the_server_will_not_hand_over_is_unmeasured(self):
        script = Script(history=[DONE_HISTORY], files={})  # /view отдаст 404
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertEqual(out["files"], 0)
        self.assertIn("ball_00001_.png", out["note"])


# ---------------------------------------------------------------------------
# ИДЕМПОТЕНТНОСТЬ И УБОРКА
# ---------------------------------------------------------------------------


class TheOutputIsNotClobbered(unittest.TestCase):
    def test_the_same_run_twice_writes_once_and_stays_pass(self):
        with TemporaryDirectory() as tmp:
            for _ in range(2):
                script = Script(history=[DONE_HISTORY],
                                files={"ball_00001_.png": b"PNG"})
                out = render(script, out_dir=tmp)
                self.assertEqual(out["outcome"], "годно", out["note"])
            self.assertEqual(
                sorted(p.name for p in (Path(tmp) / "abc-123").iterdir()),
                ["ball_00001_.png"])

    def test_someone_elses_output_at_the_same_path_is_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            mine = Path(tmp) / "abc-123"
            mine.mkdir(parents=True)
            (mine / "ball_00001_.png").write_bytes("ЧУЖОЙ ВЫХОД".encode("utf-8"))
            script = Script(history=[DONE_HISTORY],
                            files={"ball_00001_.png": "наш выход".encode("utf-8")})
            out = render(script, out_dir=tmp)
            self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
            self.assertEqual((mine / "ball_00001_.png").read_bytes(),
                             "ЧУЖОЙ ВЫХОД".encode("utf-8"))
            self.assertIn("не перетёр", out["note"])

    def test_save_says_which_of_the_three_things_happened(self):
        with TemporaryDirectory() as tmp:
            first = fb.save(tmp, "a.bin", b"12345")
            self.assertEqual(first["outcome"], "годно")
            self.assertEqual(first["bytes"], 5)
            again = fb.save(tmp, "a.bin", b"12345")
            self.assertEqual(again["outcome"], "годно")
            self.assertIn("уже лежит", again["note"])
            other = fb.save(tmp, "a.bin", "другое".encode("utf-8"))
            self.assertEqual(other["outcome"], "не смогли проверить")
            self.assertIn("занят", other["note"])
            self.assertEqual((Path(tmp) / "a.bin").read_bytes(), b"12345")

    def test_two_jobs_do_not_share_a_directory(self):
        with TemporaryDirectory() as tmp:
            for job in ("abc-123", "def-456"):
                script = Script(prompt_reply=(200, {"prompt_id": job}),
                                history=[(200, {job: {
                                    "status": {"completed": True},
                                    "outputs": {"9": {"images": [
                                        {"filename": "ball_00001_.png",
                                         "subfolder": "", "type": "output"}]}}}}),],
                                files={"ball_00001_.png": "кадр".encode("utf-8")})
                out = render(script, out_dir=tmp)
                self.assertEqual(out["outcome"], "годно", out["note"])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["abc-123", "def-456"])


# ---------------------------------------------------------------------------
# КОНСТАНТЫ-РЕШЕНИЯ. Ожидаемое — ЛИТЕРАЛ (Т2)
# ---------------------------------------------------------------------------


class TheDecisionConstants(unittest.TestCase):
    def test_the_default_address_is_the_local_comfy_port(self):
        self.assertEqual(fb.address({}), "127.0.0.1:8188")
        self.assertEqual(fb.base_url(env={}), "http://127.0.0.1:8188")

    def test_the_environment_wins_over_the_default(self):
        self.assertEqual(fb.address({"BALL_REEL_COMFY": "10.0.0.7:9000"}),
                         "10.0.0.7:9000")
        self.assertEqual(fb.base_url(env={"BALL_REEL_COMFY": "10.0.0.7:9000"}),
                         "http://10.0.0.7:9000")

    def test_an_empty_environment_value_falls_back_and_does_not_break(self):
        self.assertEqual(fb.address({"BALL_REEL_COMFY": ""}), "127.0.0.1:8188")

    def test_the_timeout_is_computed_from_named_parts_and_not_guessed(self):
        # 420 + 4 окна * 3 шага * 45 с * 2 = 1500 с на десятисекундный ролик.
        self.assertEqual(fb.timeout_for(300), 1500.0)
        self.assertEqual(fb.timeout_for(150), 960.0)
        self.assertEqual(fb.DEFAULT_TIMEOUT_S, 1500.0)

    def test_windows_are_counted_up_not_down(self):
        self.assertEqual(fb.windows_for(77), 1)
        self.assertEqual(fb.windows_for(78), 2)
        self.assertEqual(fb.windows_for(150), 2)
        self.assertEqual(fb.windows_for(300), 4)
        with self.assertRaises(ValueError):
            fb.windows_for(0)

    def test_the_default_timeout_is_the_one_actually_used(self):
        """Умолчание проверяется поведением, а не только равенством числа:
        константа, которую никто не читает, — не константа-решение."""

        class Never:
            def get(self, url):
                return fb.Reply(200, b"{}")

        clock = Clock()
        out = fb.wait("abc-123", transport=Never(), url="http://127.0.0.1:1",
                      now=clock.now, sleep=clock.sleep)
        self.assertEqual(out["waited_s"], 1500.0)
        self.assertEqual(out["polls"], 751)  # 1500 / 2 + 1

    def test_the_poll_interval_sets_how_often_we_ask(self):
        script = Script(history=[RUNNING_HISTORY])
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp, timeout_s=20.0)
        self.assertEqual(out["polls"], 11)  # 20 с / 2 с + 1

    def test_four_hundred_is_the_only_code_that_means_a_bad_graph(self):
        bad = fb.classify_submit(fb.Reply(400, '{"error": {"message": "нет"}}'.encode("utf-8")))
        self.assertEqual(bad["outcome"], "не годно")
        for code in (401, 403, 404, 500, 503):
            with self.subTest(code=code):
                other = fb.classify_submit(fb.Reply(code, b"{}"))
                self.assertEqual(other["outcome"], "не смогли проверить")

    def test_a_silent_server_is_not_a_status_code(self):
        quiet = fb.classify_submit(fb.Reply(None, b"", error="ConnectionRefused"))
        self.assertEqual(quiet["outcome"], "не смогли проверить")
        self.assertIn("ConnectionRefused", quiet["note"])

    def test_the_connect_timeout_is_the_one_written_down(self):
        """Мутация этой константы в обе стороны пережила первый прогон набора:
        её никто не сторожил. Значение проверяется литералом, а его доезд до
        транспорта — следующим тестом, иначе сторожилась бы запись, а не ручка.
        """
        self.assertEqual(fb.CONNECT_TIMEOUT_S, 20.0)
        self.assertEqual(fb.HttpTransport().timeout, 20.0)
        self.assertEqual(fb.HttpTransport(timeout=0.5).timeout, 0.5)

    def test_a_server_that_accepts_and_stays_silent_is_unmeasured(self):
        """Сервер, ПРИНЯВШИЙ соединение и замолчавший, — самый неприятный из
        трёх видов «не отвечает»: он не отказывает, он висит. Без таймаута
        транспорта команда висела бы вместе с ним."""
        with BlackHole() as hole, TemporaryDirectory() as tmp:
            out = fb.run(GOOD_GRAPH, out_dir=tmp,
                         transport=fb.HttpTransport(timeout=0.3),
                         url=hole.url, timeout_s=5.0)
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])
        self.assertIn("не отвечает", out["note"])

    def test_the_exit_code_separates_all_three_outcomes(self):
        self.assertEqual(fb.EXIT_CODES["годно"], 0)
        self.assertEqual(fb.EXIT_CODES["не годно"], 1)
        self.assertEqual(fb.EXIT_CODES["не смогли проверить"], 2)
        self.assertEqual(len(set(fb.EXIT_CODES.values())), 3)


class TheProtocolFieldNames(unittest.TestCase):
    """Каждое имя поля — утверждение о чужом контракте. Тела здесь литеральные,
    поэтому переименование константы в модуле краснеет здесь."""

    def test_the_prompt_id_is_read_from_the_documented_field(self):
        ok = fb.classify_submit(fb.Reply(200, b'{"prompt_id": "z-9", "number": 1}'))
        self.assertEqual(ok["outcome"], "годно")
        self.assertEqual(ok["prompt_id"], "z-9")

    def test_completion_is_read_from_status_and_not_from_outputs(self):
        still_going = {"status": {"completed": False},
                       "outputs": {"9": {"images": [{"filename": "a.png"}]}}}
        self.assertIsNone(fb.classify_history(still_going))
        self.assertFalse(fb.is_completed(still_going))
        self.assertTrue(fb.is_completed({"status": {"completed": True}}))

    def test_files_are_found_by_the_filename_key_in_any_output_kind(self):
        entry = {"status": {"completed": True}, "outputs": {
            "9": {"images": [{"filename": "a.png", "subfolder": "s",
                              "type": "output"}]},
            "12": {"gifs": [{"filename": "b.mp4", "subfolder": "",
                             "type": "output"}]},
            "13": {"text": ["не файл"]}}}
        files = fb.output_files(entry)
        self.assertEqual([f["filename"] for f in files], ["a.png", "b.mp4"])
        self.assertEqual(files[0]["subfolder"], "s")
        self.assertEqual(files[1]["kind"], "gifs")

    def test_the_error_message_is_found_among_the_status_messages(self):
        entry = {"status": {"completed": False, "messages": [
            ["execution_start", {}],
            ["execution_error", {"node_type": "WanVideoSampler",
                                 "exception_message": "no such lora"}]]}}
        self.assertEqual(fb.node_error(entry), "WanVideoSampler: no such lora")
        self.assertIsNone(fb.node_error({"status": {"messages": []}}))

    def test_the_queue_is_read_by_both_its_lists_and_the_id_slot(self):
        queue = {"queue_running": [[1, "бегу", {}, {}, []]],
                 "queue_pending": [[2, "жду", {}, {}, []],
                                   [3, "жду2", {}, {}, []]]}
        self.assertEqual(fb.queue_where(queue, "бегу"), "исполняется")
        self.assertEqual(fb.queue_where(queue, "жду"), "в очереди, впереди 0")
        self.assertEqual(fb.queue_where(queue, "жду2"), "в очереди, впереди 1")
        self.assertEqual(fb.queue_where(queue, "чужая"), "в очереди нет")
        self.assertEqual(fb.queue_where(None, "любая"), "очередь не прочитана")

    def test_the_paths_are_the_ones_comfy_serves(self):
        script = Script(history=[DONE_HISTORY], files={"ball_00001_.png": b"x"})
        with TemporaryDirectory() as tmp:
            render(script, out_dir=tmp)
        self.assertEqual(script.posted[0][0], "/prompt")
        self.assertIn("/history/abc-123", script.got)
        self.assertTrue(any(g.startswith("/view?") for g in script.got),
                        script.got)


class TheInstrumentCanGoRed(unittest.TestCase):
    """Прибор, не умеющий покраснеть, — украшение (И5, применённое к набору)."""

    def test_a_pass_is_impossible_without_a_file_on_disk(self):
        script = Script(history=[DONE_HISTORY], files={"ball_00001_.png": b""})
        with TemporaryDirectory() as tmp:
            out = render(script, out_dir=tmp)
        # Пустое тело — это не файл: сервер ответил 200 и не дал ничего.
        self.assertEqual(out["outcome"], "не смогли проверить", out["note"])

    def test_the_word_pass_is_never_a_substring_answer(self):
        # На этом проекте тринадцать сторожей зеленели на провале, потому что
        # «годно» — подстрока «не годно». Здесь сравнение только строгое.
        self.assertIn("годно", "не годно")
        self.assertNotEqual("не годно", "годно")




class AnInterruptedJobIsNotABadGraph(unittest.TestCase):
    """Найдено сверкой с исходником ComfyUI (`execution.py:686`), а не догадкой.

    На `InterruptProcessingException` сервер кладёт `execution_interrupted` —
    БЕЗ полей `exception_message`/`exception_type`. Прежняя версия искала
    только `execution_error`, и снятая снаружи задача читалась как
    «отработала и не оставила файлов», то есть как НАШ негодный граф.
    Следующая смена искала бы дефект там, где его нет.
    """

    def _entry(self, event, data=None, completed=True, outputs=None):
        return {"status": {"completed": completed, "status_str": "error",
                           "messages": [[event, data or {}]]},
                "outputs": outputs or {}}

    def test_an_interrupted_job_is_unmeasured(self):
        got = fb.classify_history(
            self._entry("execution_interrupted", {"node_type": "WanVideoSampler"}))
        self.assertEqual(got["outcome"], fb.UNMEASURED)
        self.assertIn("СНЯЛИ снаружи", got["note"])
        self.assertIn("WanVideoSampler", got["note"])

    def test_a_node_that_actually_crashed_is_still_a_failure(self):
        """Негативный контроль (И5): различие обязано работать в обе стороны."""
        got = fb.classify_history(self._entry(
            "execution_error",
            {"node_type": "WanVideoSampler", "exception_message": "OOM"}))
        self.assertEqual(got["outcome"], fb.FAIL)
        self.assertIn("OOM", got["note"])

    def test_completed_with_no_files_stays_a_failure(self):
        """Третий случай не должен был поехать вслед за первыми двумя."""
        got = fb.classify_history(
            {"status": {"completed": True, "messages": []}, "outputs": {}})
        self.assertEqual(got["outcome"], fb.FAIL)

    def test_the_interrupt_event_name_is_the_one_the_server_sends(self):
        # Литерал (Т2): так это поле зовётся в execution.py, переименовать его
        # у себя мы не вправе.
        self.assertEqual(fb.MSG_EXECUTION_INTERRUPTED,
                         "execution_interrupted")



if __name__ == "__main__":
    unittest.main()
