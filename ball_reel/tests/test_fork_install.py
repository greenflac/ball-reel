"""Установщик стека: он обязан уметь сказать все три вещи и не тронуть ничего.

ЧТО НЕСЁТ ЭТОТ ФАЙЛ, четырьмя строками:

* **сухой прогон НИЧЕГО НЕ ТРОГАЕТ** — ни одного файла на диске, ни одного
  вызова `git`, ни одного обращения к транспорту. Это проверяется счётчиками
  вызовов, а не доверием: «оно же не должно» — не наблюдение;
* **не смогли** там, где спросить нечем: сети нет, соединение оборвалось,
  сервер ответил 5xx. Ни одно из этого НЕ ЕСТЬ «не годно» (Р1);
* **не годно** там, где ответ получен и он неверный: sha256 разошёлся, размер
  больше эталона, адрес ответил 404. И тогда целевого имени НЕ ПОЯВЛЯЕТСЯ;
* **числа рядом с вердиктом** — сколько на месте, сколько качать, сколько байт
  скачано за запуск (Р2).

В СЕТЬ НЕ ХОДИМ ВООБЩЕ, И ЭТО ОБЕСПЕЧЕНО РАННЕРОМ, А НЕ ДОГОВОРЁННОСТЬЮ (Т4).
Транспорт подставляется параметром `fetch`, `git` и `pip` — параметром `git`.
Отдельный тест держит эту границу: он проходит по исходнику модуля разбором
дерева и требует, чтобы `urlopen` и `subprocess.run` встречались РОВНО в двух
объявленных точках внедрения. Без него первая же новая функция могла бы
позвать сеть напрямую, и весь набор молча стал бы зависеть от того, есть ли
она на машине.

ОЖИДАЕМОЕ ЗДЕСЬ — ЛИТЕРАЛ (Т2). Размер 11496331072, sha `43d720f2…`, адрес
`https://github.com/kijai/ComfyUI-WanVideoWrapper.git` вписаны РУКАМИ. Импорт
ожидаемого из проверяемого модуля поехал бы вместе с ним и промолчал ровно
тогда, когда ловить и надо: смена ступени квантования меняет и размер, и хэш.
Отдельный тест сверяет литерал с настоящим лок-файлом и краснеет ОСМЫСЛЕННО:
«лок изменился, посмотри, нарочно ли». Правило нарушалось на этом проекте
трижды за один день — именно потому, что импорт выглядит аккуратнее литерала.

ФИКСТУРЫ С ОБОИХ КРАЁВ И ИЗ СЕРЕДИНЫ (Т3), четыре состояния одного файла:
файла нет вовсе; временный на байт короче эталона; файл НУЖНОГО РАЗМЕРА С
ЧУЖИМ СОДЕРЖИМЫМ (размер сходится, хэш нет); файл на месте и годен. Плюс
пятое, которое ловит приёмка и которое установщик не имеет права создавать:
правильное имя при неправильном размере.

НЕГАТИВНЫЙ КОНТРОЛЬ В ОБЕ СТОРОНЫ (И5). У докачки он не один: сервер, который
Range УВАЖИЛ (206, дописываем), и сервер, который его ПРОИГНОРИРОВАЛ (200,
начинаем сначала). Второй — тот самый вход, на котором наивная докачка молча
собирает файл правильной длины со сдвинутым содержимым.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_install as fi
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

# --- ЛИТЕРАЛЫ (Т2). Ни одно значение ниже не импортировано из fork_install.
DIFF_NAME = "Wan2.2-Animate-14B-Q4_K_M.gguf"
DIFF_BYTES = 11496331072
DIFF_SHA = "43d720f243c3cdf5346ca05b525ec662f89bc5ebeb8e998dc347b840406cdfa6"
DIFF_URL = ("https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/"
            "resolve/main/Wan2.2-Animate-14B-Q4_K_M.gguf")
VAE_NAME = "wan_2.1_vae.safetensors"
VAE_BYTES = 253815318
LOCK_TOTAL_BYTES = 19334922850

WRAP_CLONE = "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
GGUF_CLONE = "https://github.com/city96/ComfyUI-GGUF.git"
COMFY_CLONE = "https://github.com/comfyanonymous/ComfyUI.git"
#: sha256 тела `nodes.py` обёртки — то, чем пришпилена ревизия. Литерал.
WRAP_NODES_SHA = ("6b3bb6a619e5a928259e6e8400c73ffe7140f17439c5007dd8596b900"
                  "55f0666")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sha_of(data: bytes) -> str:
    """Ожидаемый хэш считается ЗДЕСЬ, а не берётся у проверяемого модуля."""
    return hashlib.sha256(data).hexdigest()


def a_lock(tmp: Path, weights: list) -> Path:
    """Лок-файл нужной формы. Форма взята с настоящего, содержимое — своё."""
    p = tmp / "fork_stack.lock.json"
    p.write_text(json.dumps({"schema": "fork-stack-lock/1", "weights": weights},
                            ensure_ascii=False), encoding="utf-8")
    return p


def a_weight(role: str, path: str, data: bytes, *, url: str = "https://x/y",
             license: str | None = "apache-2.0") -> dict:
    return {"role": role, "path": path, "bytes": len(data),
            "sha256": sha_of(data), "url": url, "license": license}


class Server:
    """Подставной транспорт (Т4). Считает вызовы и умеет врать всеми способами.

    `honor_range=False` — сервер, ИГНОРИРУЮЩИЙ заголовок Range: отвечает 200 и
    отдаёт тело целиком. Это не выдумка ради полноты, а штатное поведение части
    зеркал, и на нём наивная докачка собирает мусор правильной длины.
    """

    def __init__(self, blob: bytes, *, status: int = 200, honor_range: bool = True,
                 cut_after: int | None = None, boom: bool = False,
                 error: str | None = None):
        self.blob = blob
        self.status = status
        self.honor_range = honor_range
        self.cut_after = cut_after
        self.boom = boom
        self.error = error
        self.calls = []

    def open(self, url: str, *, offset: int = 0) -> dict:
        self.calls.append((url, offset))
        if self.error is not None:
            return {"status": None, "length": None, "chunks": iter(()),
                    "error": self.error}
        if self.status >= 400:
            return {"status": self.status, "length": None, "chunks": iter(()),
                    "error": None}
        if offset and self.honor_range:
            body, status = self.blob[offset:], 206
        else:
            body, status = self.blob, 200
        if self.cut_after is not None:
            body = body[:self.cut_after]

        def _chunks():
            step = 7  # мелкими кусками, чтобы обрыв приходился на середину
            for i in range(0, len(body), step):
                yield body[i:i + step]
            if self.boom:
                raise ConnectionResetError("соединение оборвано на середине")

        return {"status": status, "length": len(body), "chunks": _chunks(),
                "error": None}


class Runner:
    """Подставной раннер внешних команд (Т4): и `git`, и `pip`."""

    def __init__(self, code: int = 0, err: str = "", make: bool = True,
                 write: dict | None = None):
        self.code = code
        self.err = err
        self.make = make
        #: что «приезжает» в клоне: {относительный путь: байты}
        self.write = write or {}
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if self.make and self.code == 0 and "clone" in argv:
            dest = Path(argv[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "README.md").write_text("клон", encoding="utf-8")
            for rel, data in self.write.items():
                f = dest / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
        return {"code": self.code, "out": "", "err": self.err}


# ---------------------------------------------------------------------------


class TheLockIsTheOnlySourceOfNamesAndNumbers(unittest.TestCase):
    """Е1: имена, размеры, хэши и адреса не живут в коде установщика."""

    def test_the_real_lock_still_says_what_the_literals_here_say(self):
        """Краснеет ОСМЫСЛЕННО: «лок изменился, посмотри, нарочно ли»."""
        doc = json.loads((REPO_ROOT / "workflows" / "fork_stack.lock.json")
                         .read_text(encoding="utf-8"))
        by_name = {Path(w["path"]).name: w for w in doc["weights"]}
        self.assertIn(DIFF_NAME, by_name)
        self.assertEqual(by_name[DIFF_NAME]["bytes"], DIFF_BYTES)
        self.assertEqual(by_name[DIFF_NAME]["sha256"], DIFF_SHA)
        self.assertEqual(by_name[DIFF_NAME]["url"], DIFF_URL)
        self.assertEqual(by_name[VAE_NAME]["bytes"], VAE_BYTES)
        self.assertEqual(doc["totals"]["bytes"], LOCK_TOTAL_BYTES)

    def test_no_weight_name_size_hash_or_address_is_written_in_the_module(self):
        """Второй способ узнать известное — дефект (Е1). Проверяется текстом."""
        src = (REPO_ROOT / "ball_reel" / "fork_install.py").read_text(
            encoding="utf-8")
        for forbidden in (str(DIFF_BYTES), DIFF_SHA, str(VAE_BYTES),
                          "huggingface.co/", DIFF_NAME, VAE_NAME):
            with self.subTest(value=forbidden):
                self.assertNotIn(forbidden, src,
                                 f"{forbidden} записан в код — это второй "
                                 f"источник истины помимо лок-файла")

    def test_the_address_and_licence_are_carried_over_from_the_lock(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            lock = a_lock(tmp, [a_weight("vae", "v/a.safetensors", b"abc",
                                         url="https://host/a", license="mit")])
            got = fi.read_lock(lock)
            self.assertEqual(got["outcome"], "годно")
            self.assertEqual(got["entries"][0]["url"], "https://host/a")
            self.assertEqual(got["entries"][0]["license"], "mit")

    def test_a_weight_without_an_address_is_named_and_never_downloaded(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            w = a_weight("vae", "v/a.safetensors", b"abc")
            del w["url"]
            got = fi.read_lock(a_lock(tmp, [w]))
            self.assertIsNone(got["entries"][0]["url"])
            self.assertIn("БЕЗ АДРЕСА", got["note"])

    def test_a_missing_registry_is_unmeasured_and_never_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as d:
            got = fi.find_lock(d)
            self.assertEqual(got["outcome"], "не смогли проверить")
            self.assertIn("НЕИЗВЕСТНО, ЧТО КАЧАТЬ", got["note"])

    def test_two_registries_are_a_finding_and_not_a_choice(self):
        with tempfile.TemporaryDirectory() as d:
            wf = Path(d) / "workflows"
            wf.mkdir()
            (wf / "fork_a.lock.json").write_text("{}", encoding="utf-8")
            (wf / "fork_b.lock.json").write_text("{}", encoding="utf-8")
            self.assertEqual(fi.find_lock(d)["outcome"], "не смогли проверить")


class LicencesAreCheckedBeforeAnythingIsFetched(unittest.TestCase):
    """Ц5: минута чтения до загрузки против переделки после неё."""

    def test_a_non_commercial_licence_stops_the_install(self):
        rows = [{"name": "x", "role": "vae", "repo": "r",
                 "license": "cc-by-nc-4.0", "license_note": None}]
        got = fi.licenses(rows)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("ЗАПРЕЩАЮЩАЯ ЛИЦЕНЗИЯ", got["note"])

    def test_a_research_only_licence_stops_it_too(self):
        rows = [{"name": "x", "role": "vae", "repo": "r",
                 "license": "Research-Only Licence", "license_note": None}]
        self.assertEqual(fi.licenses(rows)["outcome"], "не годно")

    def test_an_undeclared_licence_is_unmeasured_and_not_a_blocker(self):
        """Решение владельца §10: строка в учёт к отгрузке, а не отказ."""
        rows = [{"name": "лора", "role": "lora_1", "repo": "Kijai/WanVideo_comfy",
                 "license": None, "license_note": "cardData.license отсутствует"}]
        got = fi.licenses(rows)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("лора", got["note"])
        self.assertNotIn("ЗАПРЕЩАЮЩАЯ", got["note"])

    def test_the_comfyui_engine_licence_is_admitted_as_unread(self):
        """Движок коммерческого продукта с непрочитанной лицензией — находка."""
        got = fi.licenses([])
        names = [r["what"] for r in got["rows"] if r["state"] == "не объявлена"]
        self.assertIn("ComfyUI", names)

    def test_the_detector_can_also_say_yes(self):
        """И5: прибор, который умеет только «нет», ничего не меряет."""
        rows = [{"name": "x", "role": "vae", "repo": "r",
                 "license": "apache-2.0", "license_note": None}]
        patched = dict(fi.LICENSES)
        try:
            fi.LICENSES.clear()
            fi.LICENSES["ПАК"] = {"license": "mit"}
            got = fi.licenses(rows)
        finally:
            fi.LICENSES.clear()
            fi.LICENSES.update(patched)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["checked"], 2)

    def test_zero_records_checked_is_not_a_success(self):
        """Р2: ноль нарушений при нуле проверок — молчание, а не успех."""
        patched = dict(fi.LICENSES)
        try:
            fi.LICENSES.clear()
            got = fi.licenses([])
        finally:
            fi.LICENSES.update(patched)
        self.assertEqual(got["outcome"], "не смогли проверить")


class WhatIsOnDiskAndWhatHasToBeFetched(unittest.TestCase):
    """Т3: четыре состояния одного файла плюс пятое, которого не должно быть."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)
        self.models = self.tmp / "models"
        (self.models / "vae").mkdir(parents=True)
        self.body = b"x" * 64
        self.entry = {"role": "vae", "path": "vae/a.safetensors",
                      "name": "a.safetensors", "bytes": len(self.body),
                      "sha256": sha_of(self.body), "url": "https://host/a"}

    def tearDown(self):
        self._d.cleanup()

    def test_a_missing_file_is_work_to_do_and_not_a_failure(self):
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["rows"][0]["state"], "missing")
        self.assertEqual(got["rows"][0]["todo_bytes"], 64)
        self.assertEqual(got["outcome"], "не смогли проверить",
                         "отсутствующий вес — нормальное состояние голой "
                         "машины, а не провал")

    def test_a_part_file_one_byte_short_is_resumed_not_refetched(self):
        (self.models / "vae" / "a.safetensors.part").write_bytes(self.body[:63])
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["rows"][0]["state"], "partial")
        self.assertEqual(got["rows"][0]["todo_bytes"], 1,
                         "докачать надо один байт, а не весь файл заново")

    def test_the_right_name_with_the_wrong_size_is_the_one_real_failure(self):
        (self.models / "vae" / "a.safetensors").write_bytes(self.body[:63])
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["rows"][0]["state"], "wrong_size")
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("ПРАВИЛЬНОЕ ИМЯ ПРИ НЕПРАВИЛЬНОМ РАЗМЕРЕ", got["note"])

    def test_a_file_of_the_right_size_with_a_foreign_body_passes_the_survey(self):
        """Т3: размер сходится, содержимое чужое. Осмотр его НЕ ловит, и это
        не дефект осмотра: `stat` стоит микросекунду, а хэш — минуты. Ловит
        загрузка (см. соседний класс) и приёмка ключом --sha."""
        (self.models / "vae" / "a.safetensors").write_bytes(b"y" * 64)
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["rows"][0]["state"], "ok")

    def test_a_file_that_is_there_and_good_means_nothing_to_fetch(self):
        (self.models / "vae" / "a.safetensors").write_bytes(self.body)
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["todo_bytes"], 0)
        self.assertIn("КАЧАТЬ НЕЧЕГО", got["note"])

    def test_a_weight_with_no_reference_size_is_no_size_even_with_no_file(self):
        """ДЕФЕКТ, ЗАКРЫТЫЙ ЗДЕСЬ: `no_size` присваивался ТОЛЬКО найденному
        файлу. У отсутствующего размер терялся молча — запись уходила в
        `missing` с `todo_bytes = None`, сумма читала `None` как ноль, и шесть
        весов к загрузке печатались как «без эталона размера 0» и
        «К загрузке 0.0 ГиБ». Наблюдалось прогоном до починки."""
        e = {**self.entry, "bytes": None, "sha256": None}
        got = fi.survey([e], self.models)
        self.assertEqual(got["rows"][0]["state"], "no_size")
        self.assertEqual(got["no_size"], 1)
        self.assertEqual(got["missing"], 0,
                         "запись без эталона размера сосчитана как «качать с "
                         "нуля» — тогда объём к загрузке заведомо неверен")
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_the_unknown_volume_is_counted_and_never_read_as_nothing(self):
        """Е3: рядом с объёмом — сколько записей в него НЕ ВОШЛИ."""
        e = {**self.entry, "bytes": None, "sha256": None}
        got = fi.survey([e, e, e], self.models)
        self.assertEqual(got["todo_bytes"], 0)
        self.assertEqual(got["todo_unknown"], 3)
        self.assertEqual(got["todo_known"], 0)
        self.assertIn("ОБЪЁМ НЕИЗВЕСТЕН", got["note"])
        self.assertIn("без эталона размера 3", got["note"])

    def test_a_part_file_without_a_reference_size_is_no_size_too(self):
        """Т3, середина диапазона: файл частично есть, эталона нет. Докачивать
        не с чем сверяться, и `download` такую запись не качает вовсе."""
        (self.models / "vae" / "a.safetensors.part").write_bytes(self.body[:63])
        e = {**self.entry, "bytes": None, "sha256": None}
        got = fi.survey([e], self.models)
        self.assertEqual(got["rows"][0]["state"], "no_size")
        self.assertIsNone(got["rows"][0]["todo_bytes"])

    def test_when_every_size_is_known_nothing_is_called_unknown(self):
        """И5, негативный контроль: вход, на котором счётчик обязан смолчать.
        Без него «неизвестно» печаталось бы всегда и не значило бы ничего."""
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["todo_unknown"], 0)
        self.assertEqual(got["todo_known"], 1)
        self.assertEqual(got["todo_bytes"], 64)
        self.assertNotIn("ОБЪЁМ НЕИЗВЕСТЕН", got["note"])
        self.assertIn("по 1 весам из 1", got["note"])

    def test_zero_entries_is_not_nothing_to_fetch(self):
        """Р2: пустой список — «неизвестно, что качать»."""
        self.assertEqual(fi.survey([], self.models)["outcome"],
                         "не смогли проверить")

    def test_the_target_subdirectory_follows_the_comfyui_convention(self):
        got = fi.target_for(self.entry, self.models)
        self.assertEqual(got, self.models / "vae" / "a.safetensors")

    def test_an_unknown_role_falls_back_to_the_path_from_the_registry(self):
        e = {**self.entry, "role": "какая-то-новая-роль"}
        self.assertEqual(fi.target_for(e, self.models),
                         self.models / "vae" / "a.safetensors")

    def test_a_weight_already_lying_elsewhere_is_found_and_not_refetched(self):
        (self.models / "loras").mkdir()
        (self.models / "loras" / "a.safetensors").write_bytes(self.body)
        got = fi.survey([self.entry], self.models)
        self.assertEqual(got["rows"][0]["state"], "ok",
                         "раскладка models/ — конвенция, а не приговор: "
                         "качать заново уже лежащие 10 ГиБ дороже всего")


class DirectoriesAreCheckedBeforeTheFirstByte(unittest.TestCase):
    """П2: отсутствующий каталог стоит микросекунду, а не 39-ю минуту."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def _rows(self, base: Path):
        return [{"state": "missing", "target": str(base / "vae" / "a.bin")}]

    def test_a_dry_run_does_not_create_anything(self):
        got = fi.dirs(self._rows(self.tmp), dry_run=True)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["dirs"][0]["state"], "будет создан")
        self.assertFalse((self.tmp / "vae").exists(),
                         "сухой прогон создал каталог — значит он не сухой")

    def test_a_real_run_creates_them(self):
        got = fi.dirs(self._rows(self.tmp), dry_run=False)
        self.assertEqual(got["dirs"][0]["state"], "создан")
        self.assertTrue((self.tmp / "vae").is_dir())

    def test_a_destination_whose_parent_is_a_file_is_a_failure(self):
        """Универсальный отрицательный вход: под каталог занято ФАЙЛОМ.

        Проверяет ту же ветку «писать некуда», что и снятые права, но не
        зависит от того, под кем идут тесты, — а `skip` здесь запрещён (Т6):
        пропущенный тест не отличается от непройденного.
        """
        busy = self.tmp / "занято"
        busy.write_text("это файл, а не каталог", encoding="utf-8")
        got = fi.dirs([{"state": "missing",
                        "target": str(busy / "vae" / "a.bin")}], dry_run=True)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("нет предка", got["note"])

    def test_a_read_only_destination_is_judged_by_the_kernel_and_not_by_us(self):
        """Обе ветки платформы ЧТО-ТО УТВЕРЖДАЮТ, `skip` нет (Т6).

        Под обычным пользователем снятое право на запись — отказ. Под root ядро
        права на каталог не применяет, и ступень обязана сказать «годно»: это
        НЕ дефект ступени, а свойство машины, и молчать о нём нельзя — иначе
        зелёный прогон под root читался бы как проверенная ветка.
        """
        ro = self.tmp / "ro"
        ro.mkdir()
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        try:
            got = fi.dirs(self._rows(ro), dry_run=True)
            if os.geteuid() == 0:
                self.assertEqual(got["outcome"], "годно",
                                 "под root запись разрешена ядром")
            else:
                self.assertEqual(got["outcome"], "не годно")
        finally:
            os.chmod(ro, stat.S_IRWXU)

    def test_no_directories_at_all_is_unmeasured(self):
        self.assertEqual(fi.dirs([], dry_run=True)["outcome"],
                         "не смогли проверить")


class SpaceIsMeasuredAgainstTheRemainder(unittest.TestCase):
    def test_a_resumed_file_only_costs_its_tail(self):
        rows = [{"todo_bytes": 1024}, {"todo_bytes": 0}]
        got = fi.space(rows, ".")
        self.assertEqual(got["todo_gib"], 0.0)

    def test_nothing_to_fetch_still_prints_free_space(self):
        got = fi.space([{"todo_bytes": 0}], ".")
        self.assertIsNotNone(got["free_gib"])


class WhatIsClonedIsDerivedAndNotWritten(unittest.TestCase):
    """Ц10 плюс Е1: адреса выведены из реестра доказанных имён."""

    def test_the_two_packs_go_to_custom_nodes_and_comfyui_is_the_base(self):
        got = fi.packs()
        where = {r["repo"]: r["where"] for r in got["repos"]}
        self.assertEqual(where["ComfyUI-WanVideoWrapper"], "custom_nodes")
        self.assertEqual(where["ComfyUI-GGUF"], "custom_nodes")
        self.assertEqual(where["ComfyUI"], "base")

    def test_the_clone_addresses_are_the_ones_we_expect(self):
        urls = {r["repo"]: r["clone_url"] for r in fi.packs()["repos"]}
        self.assertEqual(urls["ComfyUI-WanVideoWrapper"], WRAP_CLONE)
        self.assertEqual(urls["ComfyUI-GGUF"], GGUF_CLONE)
        self.assertEqual(urls["ComfyUI"], COMFY_CLONE)

    def test_every_repository_carries_at_least_one_content_pin(self):
        for r in fi.packs()["repos"]:
            with self.subTest(repo=r["repo"]):
                self.assertTrue(r["pins"], "репозиторий без пина принимается на "
                                           "любой ревизии, то есть не пришпилен")

    def test_the_wrapper_is_pinned_by_the_body_we_read_the_node_names_from(self):
        wrap = next(r for r in fi.packs()["repos"]
                    if r["repo"] == "ComfyUI-WanVideoWrapper")
        self.assertEqual(wrap["pins"]["nodes.py"], WRAP_NODES_SHA)

    def test_without_a_revision_the_clone_is_shallow(self):
        rec = {"clone_url": "https://x/y.git", "ref": "main", "revision": None}
        cmds = fi.clone_commands(rec, Path("/tmp/y"))
        self.assertEqual(len(cmds), 1)
        self.assertIn("--depth", cmds[0])
        self.assertIn("1", cmds[0])

    def test_a_pinned_revision_is_checked_out_and_the_clone_is_not_shallow(self):
        """Неглубокий клон не умеет переключаться на произвольный коммит."""
        rec = {"clone_url": "https://x/y.git", "ref": "main", "revision": "abc123"}
        cmds = fi.clone_commands(rec, Path("/tmp/y"))
        self.assertEqual(len(cmds), 2)
        self.assertNotIn("--depth", cmds[0])
        self.assertIn("checkout", cmds[1])
        self.assertIn("abc123", cmds[1])

    def test_a_revision_from_the_command_line_reaches_the_plan(self):
        got = fi.packs(revisions={"ComfyUI-GGUF": "deadbeef"})
        rec = next(r for r in got["repos"] if r["repo"] == "ComfyUI-GGUF")
        self.assertEqual(rec["revision"], "deadbeef")

    def test_the_revision_key_is_case_insensitive(self):
        """На регистре имён паков в этом проекте уже один раз разошлись."""
        got = fi.packs(revisions={"comfyui-gguf": "cafe"})
        rec = next(r for r in got["repos"] if r["repo"] == "ComfyUI-GGUF")
        self.assertEqual(rec["revision"], "cafe")

    def test_the_revision_policy_says_out_loud_that_it_is_not_a_commit(self):
        self.assertIn("не коммитом", fi.packs()["policy"])


class ThePinIsCheckedByContent(unittest.TestCase):
    """`main` уехал — и реестр доказанных имён перестал описывать пак."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dest = Path(self._d.name)
        self.body = b"class WanVideoSampler: ...\n"
        self.rec = {"repo": "П", "pins": {"nodes.py": sha_of(self.body)}}

    def tearDown(self):
        self._d.cleanup()

    def test_the_same_body_reads_as_good(self):
        (self.dest / "nodes.py").write_bytes(self.body)
        self.assertEqual(fi.verify_pack(self.rec, self.dest)["outcome"], "годно")

    def test_a_moved_revision_is_not_good_and_says_how_to_pin_it(self):
        (self.dest / "nodes.py").write_bytes(self.body + "# уехало\n".encode("utf-8"))
        got = fi.verify_pack(self.rec, self.dest)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("ревизия уехала", got["note"])
        self.assertIn("rev-parse HEAD", got["note"])

    def test_a_missing_body_is_unmeasured_not_a_mismatch(self):
        got = fi.verify_pack(self.rec, self.dest)
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_a_repository_without_pins_is_unmeasured(self):
        got = fi.verify_pack({"repo": "П", "pins": {}}, self.dest)
        self.assertEqual(got["outcome"], "не смогли проверить")


class CloningTouchesNothingOnADryRun(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def test_a_dry_run_calls_git_zero_times(self):
        git = Runner()
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=True, git=git)
        self.assertEqual(git.calls, [],
                         "сухой прогон позвал git — значит он не сухой")
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertFalse((self.tmp / "ComfyUI").exists())

    def test_a_dry_run_prints_the_exact_commands(self):
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=True, git=Runner())
        printed = " | ".join(c for r in got["repos"] for c in r["commands"])
        self.assertIn(WRAP_CLONE, printed)
        self.assertIn("git clone", printed)

    def test_an_already_cloned_pack_is_not_touched_again(self):
        dest = self.tmp / "ComfyUI" / "custom_nodes" / "ComfyUI-GGUF"
        dest.mkdir(parents=True)
        (dest / "nodes.py").write_text("уже стоит", encoding="utf-8")
        git = Runner()
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=False, git=git)
        rec = next(r for r in got["repos"] if r["repo"] == "ComfyUI-GGUF")
        self.assertEqual(rec["state"], "уже на месте")
        self.assertNotIn([*filter(lambda c: "ComfyUI-GGUF" in c[-1], git.calls)],
                         [[["никогда"]]])
        self.assertFalse(any("ComfyUI-GGUF" in c[-1] and "clone" in c
                             for c in git.calls),
                         "склонировали поверх уже стоящего пака")

    def test_git_that_will_not_start_is_unmeasured_and_never_a_bad_pack(self):
        git = Runner(code=None, err="git не найден", make=False)
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=False, git=git)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("не запустилось", got["note"])

    def test_a_clone_that_fails_is_unmeasured_and_names_the_command(self):
        git = Runner(code=128, err="fatal: repository not found", make=False)
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=False, git=git)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("код 128", got["note"])

    def test_a_clone_whose_bodies_moved_is_not_good(self):
        """Клон удался, а тела не те: `main` уехал, и реестр имён узлов
        перестал описывать установленное. Молча этого допустить нельзя."""
        git = Runner(write={"nodes.py": b"# other revision\n",
                            "nodes_sampler.py": b"# other\n",
                            "nodes_model_loading.py": b"# other\n",
                            "comfy_extras/nodes_mask.py": b"# other\n"})
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=False, git=git)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("ревизия уехала", got["note"])
        self.assertTrue(git.calls)

    def test_a_clone_missing_the_pinned_files_is_unmeasured_not_a_mismatch(self):
        """Файла нет — это «не смогли сверить», а не «содержимое разошлось»."""
        git = Runner()
        got = fi.install_packs(fi.packs(), comfy_dir=self.tmp / "ComfyUI",
                               dry_run=False, git=git)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertTrue(git.calls)


class DependenciesUseTheSameInjectedRunner(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)
        self.installed = {"repos": [{"repo": "П", "dest": str(self.tmp)}]}

    def tearDown(self):
        self._d.cleanup()

    def test_a_dry_run_calls_pip_zero_times(self):
        (self.tmp / "requirements.txt").write_text("gguf\n", encoding="utf-8")
        run = Runner()
        got = fi.deps(self.installed, dry_run=True, runner=run)
        self.assertEqual(run.calls, [])
        self.assertIn("СУХОЙ ПРОГОН", got["note"])

    def test_the_shipped_pip_command_is_a_requirements_install(self):
        """Б2: у отгружаемого значения обязан быть сторож ЛИТЕРАЛОМ (Т2),
        отдельно от тестов, которые константу подменяют.

        МУТАЦИЯ ВЫЖИВАЛА: `("-m","pip","install","-r")` -> `("-m","pip",
        "install")`, 96 тестов, упало 0. Без `-r` команда становится
        `pip install /путь/requirements.txt` — pip читает путь как ИМЯ ПАКЕТА,
        и установки не происходит вовсе.
        """
        self.assertEqual(list(fi.PIP_ARGS), ["-m", "pip", "install", "-r"])
        self.assertEqual(fi.REQUIREMENTS_NAME, "requirements.txt")

    def test_a_real_run_installs_and_reads_as_good(self):
        """СРЕЗ УБРАН НАМЕРЕННО (Б3). Здесь стояло `run.calls[0][:4]`, то есть
        сравнение обрывалось РОВНО ПЕРЕД `-r` — последний элемент константы не
        сторожил никто. Сравнивается ВСЯ команда целиком."""
        (self.tmp / "requirements.txt").write_text("gguf\n", encoding="utf-8")
        run = Runner()
        got = fi.deps(self.installed, dry_run=False, runner=run, python="/py")
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(run.calls[0],
                         ["/py", "-m", "pip", "install", "-r",
                          str(self.tmp / "requirements.txt")])

    def test_the_flag_stands_immediately_before_the_requirements_path(self):
        """Порядок — часть значения: `-r` после пути pip прочитает иначе."""
        (self.tmp / "requirements.txt").write_text("gguf\n", encoding="utf-8")
        run = Runner()
        fi.deps(self.installed, dry_run=False, runner=run, python="/py")
        argv = run.calls[0]
        self.assertEqual(argv[-2], "-r")
        self.assertEqual(argv[-1], str(self.tmp / "requirements.txt"))

    def test_a_pack_without_a_requirements_file_is_unmeasured_not_broken(self):
        """И5, негативный контроль к соседнему тесту: каталог ЕСТЬ, файла в
        нём нет. Здесь строка «без файла зависимостей» законна, и проверка
        обязана смолчать про несклонированный пак."""
        got = fi.deps(self.installed, dry_run=False, runner=Runner())
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("без файла зависимостей 1", got["note"])
        self.assertEqual(got["no_requirements"], 1)
        self.assertEqual(got["unmeasured"], 0)
        self.assertNotIn("каталога пака нет", got["note"])

    def test_a_pack_that_was_never_cloned_is_unmeasured_and_not_file_less(self):
        """ДЕФЕКТ, ЗАКРЫТЫЙ ЗДЕСЬ: каталога пака нет вовсе, а состояние было
        «нет requirements.txt» — то, которое докстринг объявляет безобидным.
        Два разных исхода («пак не встал» и «у пака нет файла зависимостей»)
        схлопывались в безобидный, и сухой прогон докладывал «без файла
        зависимостей 3» про три несуществующих каталога. Наблюдалось прогоном
        до починки."""
        gone = {"repos": [{"repo": "П", "dest": str(self.tmp / "нет-такого")}]}
        got = fi.deps(gone, dry_run=False, runner=Runner())
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(got["rows"][0]["state"][:10], "не смогли:")
        self.assertEqual(got["unmeasured"], 1)
        self.assertEqual(got["no_requirements"], 0,
                         "несклонированный пак сосчитан как «у пака нет файла "
                         "зависимостей» — это разные исходы (Р1)")
        self.assertIn("без файла зависимостей 0", got["note"])

    def test_a_dry_run_does_not_call_a_missing_directory_file_less_either(self):
        """Сухой прогон не клонировал НИЧЕГО, значит про файл зависимостей он
        не знает. «Не смогли», а не «нет файла» (Р1, третий исход)."""
        gone = {"repos": [{"repo": "П", "dest": str(self.tmp / "нет-такого")}]}
        run = Runner()
        got = fi.deps(gone, dry_run=True, runner=run)
        self.assertEqual(run.calls, [])
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(got["unmeasured"], 1)
        self.assertEqual(got["no_requirements"], 0)
        self.assertIn("НЕИЗВЕСТНО", got["note"])

    def test_the_counts_are_a_fraction_and_not_a_bare_number(self):
        """Е3: «поставлено 1 из 2» читается иначе, чем «поставлено 1»."""
        (self.tmp / "requirements.txt").write_text("gguf\n", encoding="utf-8")
        two = {"repos": [{"repo": "П", "dest": str(self.tmp)},
                         {"repo": "Р", "dest": str(self.tmp / "нет-такого")}]}
        got = fi.deps(two, dry_run=False, runner=Runner(), python="/py")
        self.assertEqual((got["installed"], got["unmeasured"],
                          got["no_requirements"], got["total"]), (1, 1, 0, 2))
        self.assertIn("поставлено 1 из 2", got["note"])
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_pip_that_fails_is_unmeasured(self):
        (self.tmp / "requirements.txt").write_text("gguf\n", encoding="utf-8")
        got = fi.deps(self.installed, dry_run=False,
                      runner=Runner(code=1, err="нет сети", make=False))
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("код 1", got["note"])


class DownloadingIsAtomicAndResumable(unittest.TestCase):
    """Самый частый отказ проекта: правильное имя при неправильном размере."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)
        self.body = b"".join(bytes([i % 251]) for i in range(300))
        self.dest = self.tmp / "vae" / "a.safetensors"
        self.dest.parent.mkdir(parents=True)
        self.row = {"name": "a.safetensors", "bytes": len(self.body),
                    "sha256": sha_of(self.body), "url": "https://host/a",
                    "target": str(self.dest),
                    "part": str(self.dest) + ".part"}

    def tearDown(self):
        self._d.cleanup()

    def test_a_clean_download_lands_under_the_right_name(self):
        srv = Server(self.body)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(self.dest.read_bytes(), self.body)
        self.assertFalse(Path(self.row["part"]).exists(),
                         "временный файл остался рядом с готовым")

    def test_the_temporary_file_is_named_part_and_lies_next_to_the_target(self):
        """Литерал `.part` (Т2): имя обязано быть НЕ целевым и НА ТОМ ЖЕ томе."""
        srv = Server(self.body, cut_after=100)
        fi.download(self.row, fetch=srv.open, dry_run=False)
        left = sorted(p.name for p in self.dest.parent.iterdir())
        self.assertEqual(left, ["a.safetensors.part"])

    def test_an_interrupted_download_leaves_no_file_of_the_right_name(self):
        srv = Server(self.body, cut_after=100)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "не смогли проверить",
                         "обрыв — это «не смогли», а не «файл негоден»")
        self.assertFalse(self.dest.exists())
        self.assertEqual(Path(self.row["part"]).stat().st_size, 100)

    def test_a_connection_dropped_mid_body_keeps_what_arrived(self):
        srv = Server(self.body[:150], boom=True)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(Path(self.row["part"]).stat().st_size, 150,
                         "скачанное выброшено — следующий запуск заплатит "
                         "за него ещё раз")

    def test_the_second_run_resumes_from_the_offset_and_finishes(self):
        Path(self.row["part"]).write_bytes(self.body[:150])
        srv = Server(self.body)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(srv.calls, [("https://host/a", 150)],
                         "докачка запросила не тот диапазон")
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(self.dest.read_bytes(), self.body)

    def test_a_server_that_ignores_range_makes_us_start_over(self):
        """И5, вторая сторона: 200 при ненулевом смещении. Наивная докачка
        дописала бы полное тело в хвост и получила мусор правильной длины."""
        Path(self.row["part"]).write_bytes(self.body[:150])
        srv = Server(self.body, honor_range=False)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(self.dest.read_bytes(), self.body,
                         "файл собрался со сдвигом: Range проигнорирован, а "
                         "тело дописано в конец")

    def test_a_temporary_file_longer_than_the_reference_is_started_over(self):
        Path(self.row["part"]).write_bytes(self.body + "лишнее".encode("utf-8"))
        srv = Server(self.body)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(srv.calls, [("https://host/a", 0)])
        self.assertEqual(got["outcome"], "годно")

    def test_a_right_sized_body_with_the_wrong_content_is_not_good(self):
        """Т3: размер сходится, sha256 нет. Целевого имени НЕ ПОЯВЛЯЕТСЯ."""
        srv = Server(b"z" * len(self.body))
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "не годно")
        self.assertFalse(self.dest.exists(),
                         "файл с неверным содержимым уложен под правильным "
                         "именем — приёмка объявит машину готовой")
        self.assertFalse(Path(self.row["part"]).exists())
        self.assertIn(self.row["sha256"], got["note"])

    def test_a_body_longer_than_the_reference_is_not_good(self):
        srv = Server(self.body + "хвост".encode("utf-8"))
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "не годно")
        self.assertFalse(self.dest.exists())

    def test_no_answer_at_all_is_unmeasured_and_never_a_bad_file(self):
        srv = Server(self.body, error="соединение не встало")
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("НЕ «файл негоден»", got["note"])

    def test_a_server_error_is_unmeasured_and_a_client_error_is_not_good(self):
        """5xx — чужая авария, повторяемо; 4xx — наш адрес, находка."""
        self.assertEqual(
            fi.download(self.row, fetch=Server(self.body, status=503).open,
                        dry_run=False)["outcome"], "не смогли проверить")
        self.assertEqual(
            fi.download(self.row, fetch=Server(self.body, status=404).open,
                        dry_run=False)["outcome"], "не годно")

    def test_a_file_already_in_place_is_not_fetched_again(self):
        self.dest.write_bytes(self.body)
        srv = Server(self.body)
        got = fi.download(self.row, fetch=srv.open, dry_run=False)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(srv.calls, [], "повторный запуск полез в сеть")
        self.assertIn("качать нечего", got["note"])

    def test_a_weight_without_a_reference_size_is_never_fetched(self):
        srv = Server(self.body)
        got = fi.download({**self.row, "bytes": None}, fetch=srv.open,
                          dry_run=False)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(srv.calls, [],
                         "качаем без эталона размера — принимаем что дали")

    def test_a_weight_without_a_reference_hash_is_not_laid_down(self):
        srv = Server(self.body)
        got = fi.download({**self.row, "sha256": None}, fetch=srv.open,
                          dry_run=False)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertFalse(self.dest.exists())

    def test_a_dry_run_fetches_nothing_and_writes_nothing(self):
        srv = Server(self.body)
        got = fi.download(self.row, fetch=srv.open, dry_run=True)
        self.assertEqual(srv.calls, [])
        self.assertFalse(self.dest.exists())
        self.assertFalse(Path(self.row["part"]).exists())
        self.assertIn("https://host/a", got["note"])
        self.assertIn("300", got["note"], "план обязан называть число байт")


class AllTheWeightsTogether(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)
        self.models = self.tmp / "models"
        (self.models / "vae").mkdir(parents=True)
        (self.models / "loras").mkdir(parents=True)
        self.small = "м".encode("utf-8") * 10
        self.big = "б".encode("utf-8") * 400
        self.entries = [
            {"role": "lora_1", "path": "loras/big.safetensors",
             "name": "big.safetensors", "bytes": len(self.big),
             "sha256": sha_of(self.big), "url": "https://host/big"},
            {"role": "vae", "path": "vae/small.safetensors",
             "name": "small.safetensors", "bytes": len(self.small),
             "sha256": sha_of(self.small), "url": "https://host/small"},
        ]

    def tearDown(self):
        self._d.cleanup()

    def _fetch(self, blobs):
        calls = []

        def fetch(url, *, offset=0):
            calls.append(url)
            return Server(blobs[url]).open(url, offset=offset)

        return fetch, calls

    def test_the_cheapest_file_is_fetched_first(self):
        """П2 внутри шага: 0.24 ГиБ ловит закрытый прокси за секунды."""
        rows = fi.survey(self.entries, self.models)["rows"]
        fetch, calls = self._fetch({"https://host/big": self.big,
                                    "https://host/small": self.small})
        fi.weights(rows, fetch=fetch, dry_run=False)
        self.assertEqual(calls, ["https://host/small", "https://host/big"])

    def test_one_bad_file_out_of_two_is_printed_as_numbers(self):
        """Е3: частичный результат — числами, а не агрегатным флагом."""
        rows = fi.survey(self.entries, self.models)["rows"]
        fetch, _ = self._fetch({"https://host/big": b"z" * len(self.big),
                                "https://host/small": self.small})
        got = fi.weights(rows, fetch=fetch, dry_run=False)
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual((got["ok"], got["failed"], got["unmeasured"]), (1, 1, 0))

    def test_running_it_twice_downloads_nothing_the_second_time(self):
        rows = fi.survey(self.entries, self.models)["rows"]
        fetch, _ = self._fetch({"https://host/big": self.big,
                                "https://host/small": self.small})
        first = fi.weights(rows, fetch=fetch, dry_run=False)
        self.assertEqual(first["outcome"], "годно")
        again = fi.survey(self.entries, self.models)["rows"]
        fetch2, calls2 = self._fetch({})
        second = fi.weights(again, fetch=fetch2, dry_run=False)
        self.assertEqual(second["outcome"], "годно")
        self.assertEqual(calls2, [])
        self.assertEqual(second["downloaded_bytes"], 0)
        self.assertIn("КАЧАТЬ НЕЧЕГО", second["note"])

    def test_the_downloaded_amount_is_printed_in_bytes_and_not_only_gibibytes(self):
        """Найдено глазами (П3): 200 байт округлялись в «0.0 ГиБ», и отчёт об
        оборванной загрузке был неотличим от отчёта о бездействии."""
        rows = fi.survey(self.entries, self.models)["rows"]
        fetch, _ = self._fetch({"https://host/big": self.big,
                                "https://host/small": self.small})
        got = fi.weights(rows, fetch=fetch, dry_run=False)
        self.assertEqual(got["downloaded_bytes"], 820)
        self.assertIn("820 байт", got["note"])

    def test_zero_weights_is_not_a_success(self):
        self.assertEqual(fi.weights([], fetch=None, dry_run=True)["outcome"],
                         "не смогли проверить")


class TheWholeReportOnADryRun(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = Path(self._d.name)
        (self.tmp / "workflows").mkdir()
        self.body = "тело".encode("utf-8") * 8
        w = a_weight("vae", "vae/a.safetensors", self.body,
                     url="https://host/a")
        p = self.tmp / "workflows" / "fork_stack.lock.json"
        p.write_text(json.dumps({"weights": [w]}, ensure_ascii=False),
                     encoding="utf-8")

    def tearDown(self):
        self._d.cleanup()

    def test_a_dry_run_touches_neither_disk_nor_network(self):
        srv = Server(self.body)
        git = Runner()
        before = sorted(p.name for p in self.tmp.iterdir())
        rep = fi.report(root=str(self.tmp), dry_run=True, fetch=srv.open,
                        git=git)
        self.assertEqual(srv.calls, [])
        self.assertEqual(git.calls, [])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), before)
        self.assertTrue(rep["dry_run"])
        self.assertIn("СУХОЙ ПРОГОН", rep["note"])

    def test_the_steps_run_from_the_cheapest_to_the_most_expensive(self):
        """П2 в чистом виде: порядок здесь и есть прибор."""
        rep = fi.report(root=str(self.tmp), dry_run=True,
                        fetch=Server(self.body).open, git=Runner())
        self.assertEqual([s["name"] for s in rep["steps"]],
                         ["реестр", "лицензии", "осмотр", "каталоги", "место",
                          "паки", "зависимости", "веса"])

    def test_every_step_prints_its_duration(self):
        rep = fi.report(root=str(self.tmp), dry_run=True,
                        fetch=Server(self.body).open, git=Runner())
        for s in rep["steps"]:
            with self.subTest(step=s["name"]):
                self.assertIsInstance(s["seconds"], float)

    def test_a_real_run_installs_everything_and_the_second_says_nothing_to_do(self):
        srv = Server(self.body)
        git = Runner()
        first = fi.report(root=str(self.tmp), dry_run=False, fetch=srv.open,
                          git=git, models_dir=str(self.tmp / "models"))
        weights = next(s for s in first["steps"] if s["name"] == "веса")
        self.assertEqual(weights["outcome"], "годно")
        srv2 = Server(self.body)
        second = fi.report(root=str(self.tmp), dry_run=False, fetch=srv2.open,
                           git=Runner(), models_dir=str(self.tmp / "models"))
        again = next(s for s in second["steps"] if s["name"] == "веса")
        self.assertEqual(again["outcome"], "годно")
        self.assertEqual(srv2.calls, [], "идемпотентности нет: полезли в сеть")

    def test_the_verdict_is_a_number_of_steps_and_not_a_flag(self):
        rep = fi.report(root=str(self.tmp), dry_run=True,
                        fetch=Server(self.body).open, git=Runner())
        self.assertEqual(rep["passed"] + rep["failed"] + rep["unmeasured"],
                         len(rep["steps"]))

    def test_the_rendered_plan_names_the_addresses_and_the_byte_counts(self):
        rep = fi.report(root=str(self.tmp), dry_run=True,
                        fetch=Server(self.body).open, git=Runner())
        text = fi.render(rep)
        self.assertIn("СУХОЙ ПРОГОН", text)
        self.assertIn("https://host/a", text)
        self.assertIn("git clone", text)
        self.assertIn("код возврата", text)

    def test_the_three_outcomes_are_three_exit_codes(self):
        self.assertEqual(fi.EXIT_CODES["годно"], 0)
        self.assertEqual(fi.EXIT_CODES["не годно"], 1)
        self.assertEqual(fi.EXIT_CODES["не смогли проверить"], 2)


class TheEntryPoint(unittest.TestCase):
    def test_the_default_is_a_dry_run_and_never_an_install(self):
        """Случайный запуск не имеет права начать качать 18 ГиБ."""
        self.assertIs(fi.DEFAULT_DRY_RUN, True)
        with tempfile.TemporaryDirectory() as d:
            rep = fi.report(root=d, fetch=None, git=Runner())
            self.assertTrue(rep["dry_run"])

    def test_install_and_dry_run_together_are_refused(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fi.main(["--install", "--dry-run"]), 2)

    def test_a_malformed_revision_is_refused_before_anything_runs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fi.main(["--rev", "безравно"]), 2)

    def test_the_revision_parser_is_outside_the_entry_point(self):
        """Т5: развилка внутри main() недостижима для теста."""
        self.assertEqual(fi.parse_revisions(["a=b", "c = d "]),
                         {"a": "b", "c": "d"})
        with self.assertRaises(ValueError):
            fi.parse_revisions(["="])

    def test_the_dry_run_on_this_repository_prints_a_plan(self):
        """Точка входа на НАСТОЯЩЕМ локе. Вывод перехвачен: план длинный, и
        в протоколе прогона он читался бы как отказ."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = fi.main(["--root", str(REPO_ROOT), "--comfy",
                            str(REPO_ROOT / "нет-такого-ComfyUI")])
        text = buf.getvalue()
        self.assertIn(code, (0, 1, 2))
        self.assertIn("СУХОЙ ПРОГОН", text)
        self.assertIn("18.007", text, "план обязан называть объём загрузки")


class TheShippedTransportIsCoveredWithoutAnyNetwork(unittest.TestCase):
    """`urlopen` подменяется целиком, наружу не уходит ни один пакет.

    ЗАЧЕМ ОТДЕЛЬНЫЙ КЛАСС. Точка внедрения `fetch` избавляет ВЕСЬ модуль от
    сети, но сам штатный транспорт остаётся тогда непроверенным — и его
    константа тоже. Ровно так `CONNECT_TIMEOUT_S` ПЕРЕЖИЛ обе стороны мутации
    на первом заходе: значение уезжало на арендованную машину, не будучи
    сторожимым ничем. Тот же класс дыры уже находился в `fork_backend`.
    """

    def setUp(self):
        self.seen = []
        self.real = fi.urllib.request.urlopen

    def tearDown(self):
        fi.urllib.request.urlopen = self.real

    def _patch(self, behave):
        def fake(req, timeout=None):
            self.seen.append({"url": req.full_url, "timeout": timeout,
                              "headers": dict(req.header_items())})
            return behave()
        fi.urllib.request.urlopen = fake

    def test_the_shipped_timeout_is_sixty_seconds(self):
        """Литерал (Т2): именно это число уедет на арендованную машину."""
        self.assertEqual(fi.HttpFetcher().timeout, 60)

    def test_the_shipped_timeout_actually_reaches_urlopen(self):
        self._patch(lambda: (_ for _ in ()).throw(
            fi.urllib.error.URLError("нет сети")))
        fi.HttpFetcher().open("http://х/у")
        self.assertEqual(self.seen[0]["timeout"], 60,
                         "константа объявлена и до вызова не доезжает")

    def test_an_explicit_timeout_overrides_the_default(self):
        self._patch(lambda: (_ for _ in ()).throw(
            fi.urllib.error.URLError("нет сети")))
        fi.HttpFetcher(timeout=3.5).open("http://х/у")
        self.assertEqual(self.seen[0]["timeout"], 3.5)

    def test_the_shipped_chunk_is_one_mebibyte(self):
        """Литерал (Т2): 1 МиБ = 1048576. МУТАЦИЯ ВЫЖИВАЛА — `CHUNK_BYTES`
        подменялся на 4096 при 96 зелёных тестах: во всех прогонах размер
        куска задавался ПАРАМЕТРОМ (`HttpFetcher(chunk=2)`), а отгружаемое
        умолчание не сторожил никто (тот же класс, что Б2)."""
        self.assertEqual(fi.HttpFetcher().chunk, 1048576)

    def test_the_shipped_chunk_actually_reaches_the_reader(self):
        """Объявлена — не значит доезжает: у `CONNECT_TIMEOUT_S` этот второй
        тест уже ловил разрыв между объявлением и вызовом."""
        seen = []

        class Resp:
            status = 200
            headers = {"Content-Length": "3"}

            def __init__(self):
                self.left = b"abc"

            def read(self, n):
                seen.append(n)
                out, self.left = self.left[:n], self.left[n:]
                return out

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        self._patch(Resp)
        list(fi.HttpFetcher().open("http://х/у")["chunks"])
        self.assertEqual(seen[0], 1048576,
                         "константа объявлена и до чтения не доезжает")

    def test_a_nonzero_offset_asks_for_an_open_range(self):
        self._patch(lambda: (_ for _ in ()).throw(
            fi.urllib.error.URLError("нет сети")))
        fi.HttpFetcher().open("http://х/у", offset=1024)
        headers = {k.lower(): v for k, v in self.seen[0]["headers"].items()}
        self.assertEqual(headers.get("range"), "bytes=1024-")

    def test_a_zero_offset_asks_for_no_range_at_all(self):
        """И5, вторая сторона: заголовка быть НЕ ДОЛЖНО."""
        self._patch(lambda: (_ for _ in ()).throw(
            fi.urllib.error.URLError("нет сети")))
        fi.HttpFetcher().open("http://х/у")
        headers = {k.lower() for k in self.seen[0]["headers"]}
        self.assertNotIn("range", headers)

    def test_no_answer_at_all_reads_as_status_none(self):
        self._patch(lambda: (_ for _ in ()).throw(OSError("соединение отбито")))
        got = fi.HttpFetcher().open("http://х/у")
        self.assertIsNone(got["status"])
        self.assertIn("соединение отбито", got["error"])

    def test_a_refusal_carries_the_code_the_server_gave(self):
        def boom():
            raise fi.urllib.error.HTTPError("http://х/у", 404, "нет", {}, None)
        self._patch(boom)
        self.assertEqual(fi.HttpFetcher().open("http://х/у")["status"], 404)

    def test_a_body_arrives_in_chunks_and_the_length_is_read(self):
        class Resp:
            status = 206
            headers = {"Content-Length": "5"}

            def __init__(self):
                self.left = b"12345"

            def read(self, n):
                out, self.left = self.left[:n], self.left[n:]
                return out

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        self._patch(Resp)
        got = fi.HttpFetcher(chunk=2).open("http://х/у", offset=1)
        self.assertEqual(got["status"], 206)
        self.assertEqual(got["length"], 5)
        self.assertEqual(list(got["chunks"]), [b"12", b"34", b"5"])


class TheRunnerGuaranteesNoNetwork(unittest.TestCase):
    """Т4: границу держит проверка, а не договорённость."""

    def _tree(self):
        return ast.parse((REPO_ROOT / "ball_reel" / "fork_install.py")
                         .read_text(encoding="utf-8"))

    def _functions_calling(self, needle: str) -> set:
        out = set()
        tree = self._tree()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    if needle in ast.unparse(inner.func):
                        out.add(node.name)
        return out

    def test_urlopen_lives_only_in_the_declared_transport(self):
        self.assertEqual(self._functions_calling("urlopen"), {"open"},
                         "в сеть ходит что-то помимо HttpFetcher.open — "
                         "значит подмена транспорта больше ничего не гарантирует")

    def test_subprocess_lives_only_in_the_declared_runner(self):
        self.assertEqual(self._functions_calling("subprocess.run"), {"run_git"},
                         "внешние команды запускаются мимо точки внедрения")

    def test_the_module_never_imports_a_download_helper(self):
        names = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        for forbidden in ("requests", "huggingface_hub", "hf_transfer", "aiohttp"):
            with self.subTest(name=forbidden):
                self.assertNotIn(forbidden, names)

    def test_the_detector_itself_can_find_something(self):
        """И5: проверка, которая не умеет находить, ничего не утверждает."""
        self.assertTrue(self._functions_calling("Path"))
        self.assertEqual(self._functions_calling("совершенно_несуществующее"),
                         set())


class NoAssertionComparesATruncatedCommand(unittest.TestCase):
    """И7: чинить по месту — чинить пятую часть. Сгрепано по форме дефекта.

    Дефект был один: `self.assertEqual(run.calls[0][:4], [...])` — срез
    обрывал сравнение РОВНО ПЕРЕД последним элементом `PIP_ARGS`, и мутация
    `("-m","pip","install","-r")` -> `("-m","pip","install")` выживала на 96
    зелёных тестах. Форма дефекта — СРАВНЕНИЕ СРЕЗА ЗАПИСАННОЙ КОМАНДЫ: всё,
    что за срезом, не сторожит никто.

    Найдено этой формой в `test_fork_install.py`: 1 место, оно и починено.
    Соседние файлы не правились (Ц2, один писатель на модуль); что найдено в
    них — в отчёте смены.

    # DEBT(2026-08-20): связь `fork_video.PROBE_TIMEOUT_S = 20` с литералом
    # `timeout=20` в `fork_template._ffprobe_fps` (Е1) НЕ ЗАСТОРОЖЕНА. Тест на
    # равенство двух модулей здесь написать нечем: в `fork_template` это голый
    # литерал, а не константа, и сторож пришлось бы ставить в чужие файлы.
    """

    def _sliced_command_comparisons(self, src: str) -> list:
        """Срез записанной команды внутри сравнения. Возвращает номера строк."""
        out = []
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            if not ast.unparse(node.func).endswith(("assertEqual", "assertIn",
                                                    "assertNotEqual")):
                continue
            for arg in node.args:
                text = ast.unparse(arg)
                if ".calls[" in text and text.endswith("]") and ":" in \
                        text.rsplit("[", 1)[-1]:
                    out.append(node.lineno)
        return sorted(set(out))

    def test_this_file_compares_commands_whole(self):
        src = (REPO_ROOT / "ball_reel" / "tests"
               / "test_fork_install.py").read_text(encoding="utf-8")
        self.assertEqual(self._sliced_command_comparisons(src), [],
                         "сравнение среза записанной команды: всё, что за "
                         "срезом, не сторожит никто")

    def test_the_detector_can_find_the_defect_it_was_written_for(self):
        """И5: вход, на котором сторож ОБЯЗАН сработать, и вход, на котором он
        обязан смолчать. Без первого сторож утверждает «чисто» ни о чём."""
        bad = "self.assertEqual(run.calls[0][:4], ['/py', '-m', 'pip'])\n"
        good = "self.assertEqual(run.calls[0], ['/py', '-m', 'pip', '-r'])\n"
        self.assertEqual(self._sliced_command_comparisons(bad), [1])
        self.assertEqual(self._sliced_command_comparisons(good), [])


if __name__ == "__main__":
    unittest.main()
