"""Предполёт: подбирает конфигурацию, а не отвергает машину.

Два требования несут файл. Первое — арифметика ступеней обязана краснеть, если
её подделать. Второе, важнее: **пропускная способность без замера не
подставляется из паспорта**. У A16 TFLOPS в даташите нет вообще, и функция,
которая всё-таки выдала бы минуты на ролик, выдала бы их из воздуха.

Третье и четвёртое добавлены позже, по ХЭНДОФ §6 H:

* **веса по хэшу** — три состояния на файл, и «хэш не сошёлся» не сваливается в
  «файла нет»; реестр эталонов приходит ВХОДОМ, а без него исход «не смогли»;
* **живость Comfy** — в сеть тест не ходит (Т4): подделан и ответ, и отказ, а
  закрытый порт берётся на loopback, где отказ приходит сразу и без сети.
"""

from __future__ import annotations

import inspect
import json
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path

from ball_reel import fork_preflight as fp
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

SMI = ("NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n"
       "NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n")


class TheSmiOutputIsParsedWithoutACard(unittest.TestCase):
    """Т5: разбор вынесен из функции, которой нужна видеокарта."""

    def test_every_row_becomes_a_gpu(self):
        got = fp.parse_smi(SMI)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["count"], 2)

    def test_mib_is_converted_to_gigabytes(self):
        got = fp.parse_smi(SMI)
        self.assertAlmostEqual(got["gpus"][0]["memory_gb"], 15.99, places=2)

    def test_ampere_reads_as_native_bf16(self):
        self.assertTrue(fp.parse_smi(SMI)["gpus"][0]["bf16_native"])

    def test_turing_reads_as_not_native_bf16(self):
        """Негативный контроль (И5): признак обязан уметь говорить «нет»."""
        got = fp.parse_smi("NVIDIA T4, 15360 MiB, 7.5, 550.90.07\n")
        self.assertFalse(got["gpus"][0]["bf16_native"])

    def test_the_capability_bar_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        original = fp.BF16_MIN_CAPABILITY
        try:
            fp.BF16_MIN_CAPABILITY = 7.0
            self.assertTrue(
                fp.parse_smi("T4, 15360 MiB, 7.5, 550\n")["gpus"][0]["bf16_native"])
            fp.BF16_MIN_CAPABILITY = 9.0
            self.assertFalse(fp.parse_smi(SMI)["gpus"][0]["bf16_native"])
        finally:
            fp.BF16_MIN_CAPABILITY = original

    def test_empty_output_is_unmeasured_not_zero_cards(self):
        got = fp.parse_smi("\n\n")
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_missing_utility_is_not_the_same_as_no_cards(self):
        """Разные состояния: одно чинится драйвером, другое — другой машиной."""
        import shutil

        original = shutil.which
        shutil.which = lambda name: None
        try:
            got = fp.gpus()
        finally:
            shutil.which = original
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ «карт нет»", got["note"])

    def test_an_unparseable_capability_does_not_crash(self):
        got = fp.parse_smi("NVIDIA A16, 16376 MiB, N/A, 550\n")
        self.assertIsNone(got["gpus"][0]["compute_cap"])
        self.assertIsNone(got["gpus"][0]["bf16_native"])


class TheBudgetPicksAStepInsteadOfRejectingTheCard(unittest.TestCase):
    """Т2: числа — литералы из ХЭНДОФ §3.1, не пересчёт формулой модуля."""

    def test_sixteen_gigabytes_at_seventy_seven_frames_chooses_q3(self):
        """Не «что влезает», а «что влезает С ЗАПАСОМ».

        Первая редакция брала Q4_K_M: 15.97 из 16.0 арифметически влезает.
        ХЭНДОФ §3.1 называет это «впритык» и берёт Q3_K_M, потому что ECC
        включён по умолчанию и забирает НЕИЗМЕРЕННУЮ часть 16 ГБ. Остаток
        0.03 ГБ против неизвестного расхода — совпадение, а не запас.
        """
        got = fp.budget(16.0, length=77)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["chosen"], "Q3_K_M")

    def test_the_step_that_only_just_fits_is_called_tight_not_fitting(self):
        got = fp.budget(16.0, length=77)
        self.assertEqual(got["tight"], ["Q4_K_M"])
        self.assertNotIn("Q4_K_M", got["fits"])
        self.assertIn("впритык", got["note"])
        self.assertIn("ECC", got["note"])

    def test_the_headroom_bar_is_guarded_in_both_directions(self):
        """Т1: сняв запас, обязаны получить Q4; задрав — ни одной ступени."""
        original = fp.MIN_HEADROOM_GB
        try:
            fp.MIN_HEADROOM_GB = 0.0
            self.assertEqual(fp.budget(16.0, length=77)["chosen"], "Q4_K_M")
            fp.MIN_HEADROOM_GB = 10.0
            self.assertIsNone(fp.budget(16.0, length=77)["chosen"])
        finally:
            fp.MIN_HEADROOM_GB = original

    def test_the_arithmetic_matches_the_handoff_to_the_hundredth(self):
        rows = {r["step"]: r for r in fp.budget(16.0, length=77)["rows"]}
        self.assertAlmostEqual(rows["Q3_K_M"]["total_gb"], 13.30, places=2)
        self.assertAlmostEqual(rows["Q4_K_M"]["total_gb"], 15.97, places=2)
        self.assertAlmostEqual(rows["Q3_K_M"]["headroom_gb"], 2.70, places=2)

    def test_forty_nine_frames_matches_the_handoff_arithmetic(self):
        rows = {r["step"]: r for r in fp.budget(16.0, length=49)["rows"]}
        self.assertAlmostEqual(rows["Q4_K_M"]["total_gb"], 15.26, places=2)

    def test_the_report_shows_the_rejected_step_too(self):
        """«А почему не Q4» должно отвечаться тем же отчётом."""
        got = fp.budget(16.0, length=77)
        self.assertIn("Q4_K_M", got["note"])
        self.assertEqual(len(got["rows"]), 2)

    def test_a_tiny_card_fails_and_names_the_levers(self):
        got = fp.budget(8.0, length=77)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIsNone(got["chosen"])
        self.assertIn("ECC", got["note"])
        self.assertIn("49", got["note"])

    def test_a_big_card_takes_the_bigger_step(self):
        """Негативный контроль: подбор умеет и повышать, а не только резать."""
        self.assertEqual(fp.budget(24.0, length=77)["chosen"], "Q4_K_M")

    def test_an_unknown_length_is_unmeasured_rather_than_extrapolated(self):
        got = fp.budget(16.0, length=61)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("выдуманный расход", got["note"])

    def test_unknown_memory_is_unmeasured(self):
        self.assertEqual(fp.budget(None)["outcome"], UNMEASURED)

    def test_the_step_weights_are_guarded(self):
        """Т1: раздув вес ступени, обязаны потерять её из годных."""
        original = dict(fp.STEPS_GB)
        try:
            fp.STEPS_GB["Q3_K_M"] = 99.0
            got = fp.budget(16.0, length=77)
            self.assertNotIn("Q3_K_M", got["fits"],
                             "ступень весом 99 ГБ влезла в 16 — вес не "
                             "участвует в расчёте")
        finally:
            fp.STEPS_GB.clear()
            fp.STEPS_GB.update(original)


class ThroughputRefusesToGuessFromASpecSheet(unittest.TestCase):
    """Главный тест файла. У A16 TFLOPS в даташите нет вообще."""

    def test_without_a_measurement_it_says_unmeasured(self):
        got = fp.throughput()
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["minutes"])

    def test_it_names_why_the_spec_sheet_is_not_used(self):
        got = fp.throughput()
        self.assertIn("НЕ ЗАМЕРЕНА", got["note"])
        self.assertIn("TFLOPS в даташите нет", got["note"])

    def test_the_flops_per_clip_is_still_reported(self):
        """Что посчитать можно — считается: 1.54 PFLOPs x 6 шагов."""
        self.assertAlmostEqual(fp.throughput()["pflops_per_clip"], 9.24,
                               places=2)

    def test_with_a_measurement_it_gives_minutes(self):
        got = fp.throughput(measured_tflops=5.0)
        self.assertEqual(got["outcome"], PASS)
        self.assertAlmostEqual(got["minutes"], 30.8, places=1)

    def test_a_faster_measurement_gives_fewer_minutes(self):
        slow = fp.throughput(measured_tflops=5.0)["minutes"]
        fast = fp.throughput(measured_tflops=10.0)["minutes"]
        self.assertLess(fast, slow)
        self.assertAlmostEqual(fast * 2, slow, places=1)

    def test_a_nonsense_measurement_is_refused(self):
        for bad in (0.0, -3.0):
            with self.subTest(value=bad):
                self.assertEqual(fp.throughput(measured_tflops=bad)["outcome"],
                                 UNMEASURED)


class TheSpecSheetNumberNeverLeaksIntoTheAnswer(unittest.TestCase):
    """ХЭНДОФ §3.1a: «около 18 TFLOPS» — припоминание, а не даташит.

    Проверить его было бы нечем: страница спецификаций лежит на `nvidia.com`,
    домен закрыт прокси, обходить запрещено (Ц3). Значит единственный
    допустимый источник эффективной полосы — замер, и это сторожится тестом, а
    не обещанием в комментарии.
    """

    def test_there_is_no_spec_number_in_the_module(self):
        self.assertIsNone(fp.SPEC_TFLOPS)

    def test_the_throughput_default_is_no_measurement_at_all(self):
        default = inspect.signature(fp.throughput).parameters[
            "measured_tflops"].default
        self.assertIsNone(default)

    def test_a_spec_number_does_not_leak_into_the_answer(self):
        """Т1 в обе стороны: подстановка паспорта не обязана менять ответ.

        Первая сторона — вписав 18.0 в `SPEC_TFLOPS`, обязаны по-прежнему
        получить «не смогли»: паспорт не считается замером. Вторая — тот же
        18.0, поданный ЗАМЕРОМ, обязан дать минуты: функция не сломана, она
        разборчива к источнику.
        """
        original = fp.SPEC_TFLOPS
        try:
            fp.SPEC_TFLOPS = 18.0
            self.assertEqual(fp.throughput()["outcome"], UNMEASURED,
                             "паспортное число подставилось вместо замера")
            self.assertIsNone(fp.throughput()["minutes"])
            self.assertEqual(fp.throughput(measured_tflops=18.0)["outcome"],
                             PASS)
        finally:
            fp.SPEC_TFLOPS = original


def _lock_doc(entries):
    return json.dumps({"weights": entries}, ensure_ascii=False)


class TheWeightsAreCheckedByHash(unittest.TestCase):
    """Три состояния на файл. «Не тот файл» — находка, а не «файла нет»."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "workflows").mkdir()
        self.addCleanup(self.tmp.cleanup)

    def _weight(self, name="model.gguf", body=b"weights-here"):
        p = self.root / name
        p.write_bytes(body)
        return p

    def _lock(self, entries, name="fork_stack.lock"):
        p = self.root / "workflows" / name
        p.write_text(_lock_doc(entries), encoding="utf-8")
        return p

    def test_a_hash_of_a_known_input_is_the_known_literal(self):
        """Т2: ожидаемое — литерал, а не то же самое hashlib из модуля."""
        empty = self.root / "empty.bin"
        empty.write_bytes(b"")
        self.assertEqual(
            fp.sha256_of(empty),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_the_chunk_size_does_not_change_the_hash(self):
        """Размер куска — ручка скорости, а не решения: хэш обязан совпасть."""
        p = self._weight(body=b"x" * 5000)
        original = fp.HASH_CHUNK_BYTES
        try:
            fp.HASH_CHUNK_BYTES = 7
            small = fp.sha256_of(p)
        finally:
            fp.HASH_CHUNK_BYTES = original
        self.assertEqual(small, fp.sha256_of(p))

    def test_a_matching_file_is_ok(self):
        p = self._weight()
        self._lock([{"path": "model.gguf", "sha256": fp.sha256_of(p)}])
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["ok"], 1)
        self.assertEqual(got["files"][0]["state"], "ok")

    def test_a_present_file_with_a_wrong_hash_is_a_finding_not_a_missing_file(self):
        """Главное различие потока: две беды чинятся разными командами."""
        self._weight()
        self._lock([{"path": "model.gguf", "sha256": "00" * 32}])
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["mismatch"], 1)
        self.assertEqual(got["missing"], 0)
        self.assertEqual(got["files"][0]["state"], "mismatch")
        self.assertIn("хэш не сошёлся", got["note"])

    def test_one_changed_byte_turns_ok_into_mismatch(self):
        """Тест, который умеет краснеть: сверка обязана ловить подмену."""
        p = self._weight(body=b"weights-here")
        self._lock([{"path": "model.gguf", "sha256": fp.sha256_of(p)}])
        self.assertEqual(fp.weights(str(self.root))["outcome"], PASS)
        p.write_bytes(b"weights-herf")
        after = fp.weights(str(self.root))
        self.assertEqual(after["outcome"], FAIL)
        self.assertEqual(after["mismatch"], 1)

    def test_an_absent_file_is_missing_not_mismatch(self):
        self._lock([{"path": "model.gguf", "sha256": "00" * 32}])
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["missing"], 1)
        self.assertEqual(got["mismatch"], 0)

    def test_a_present_file_without_a_reference_hash_is_unmeasured(self):
        """Р2: ноль нарушений при нуле сверок — не успех."""
        self._weight()
        self._lock([{"path": "model.gguf"}])
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["no_hash"], 1)
        self.assertEqual(got["ok"], 0)

    def test_one_good_file_does_not_hide_one_bad_one(self):
        good = self._weight("good.gguf", b"good")
        self._weight("bad.gguf", b"bad")
        self._lock([{"path": "good.gguf", "sha256": fp.sha256_of(good)},
                    {"path": "bad.gguf", "sha256": "11" * 32}])
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual((got["ok"], got["mismatch"]), (1, 1))
        self.assertIn("bad.gguf", got["note"])

    def test_without_a_lock_file_the_answer_is_unmeasured_not_fine(self):
        self._weight()
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["ok"], 0)
        self.assertIn("поток E", got["note"])

    def test_two_lock_files_are_ambiguous_rather_than_guessed(self):
        self._lock([{"path": "model.gguf", "sha256": "00" * 32}],
                   name="fork_stack.lock")
        self._lock([{"path": "model.gguf", "sha256": "11" * 32}],
                   name="fork_other.lock")
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("fork_other.lock", got["note"])

    def test_the_lock_mask_is_guarded_in_both_directions(self):
        """Т1: разойдись маска с именем — обязаны получить «не смогли»."""
        p = self._weight()
        self._lock([{"path": "model.gguf", "sha256": fp.sha256_of(p)}])
        original = fp.LOCK_GLOB
        try:
            self.assertEqual(fp.weights(str(self.root))["outcome"], PASS)
            fp.LOCK_GLOB = "nothing_*.lock"
            self.assertEqual(fp.weights(str(self.root))["outcome"], UNMEASURED)
        finally:
            fp.LOCK_GLOB = original

    def test_a_broken_registry_is_a_finding_not_an_empty_result(self):
        (self.root / "workflows" / "fork_stack.lock").write_text(
            "{не json", encoding="utf-8")
        got = fp.weights(str(self.root))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НАХОДКА", got["note"])

    def test_a_registry_without_any_weights_list_is_unmeasured(self):
        (self.root / "workflows" / "fork_stack.lock").write_text(
            json.dumps({"workflow": {"sha256": "00" * 32}}), encoding="utf-8")
        self.assertEqual(fp.weights(str(self.root))["outcome"], UNMEASURED)

    def test_a_path_to_hash_mapping_is_accepted_too(self):
        """Форма записи потока E заранее неизвестна — обе очевидные приняты."""
        p = self._weight()
        (self.root / "workflows" / "fork_stack.lock").write_text(
            json.dumps({"weights": {"model.gguf": fp.sha256_of(p)}}),
            encoding="utf-8")
        self.assertEqual(fp.weights(str(self.root))["outcome"], PASS)

    def test_the_registry_is_an_input_and_can_be_pointed_at_directly(self):
        p = self._weight()
        elsewhere = self.root / "elsewhere.lock"
        elsewhere.write_text(
            _lock_doc([{"path": "model.gguf", "sha256": fp.sha256_of(p)}]),
            encoding="utf-8")
        got = fp.weights(str(self.root), lock_path=elsewhere)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["lock"], str(elsewhere))


class TheRegistryOfStreamEIsActuallyReadable(unittest.TestCase):
    """Реестр — ВХОД, и вход надо проверять на настоящем файле, не только на своём.

    Подделанный реестр проверяет наш разбор, а не договорённость с потоком E.
    ХЭНДОФ обещал `workflows/fork_*.lock`, поток E положил
    `fork_stack.lock.json` — расхождение поймано именно этим тестом.
    """

    ROOT = Path(__file__).resolve().parents[2]

    def test_the_mask_finds_the_registry_that_actually_lies_in_the_tree(self):
        """Т1 работает и здесь: сузив маску до обещанного `fork_*.lock`,
        обязаны получить красное — файл в дереве называется иначе.

        Без этого теста маска, разошедшаяся с именем, дала бы тихое «не
        смогли»: худший ответ из трёх, потому что он выглядит законным.
        """
        d = self.ROOT / "workflows"
        lying = sorted(p.name for p in d.iterdir()
                       if p.is_file() and "lock" in p.name)
        seen = sorted(p.name for p in d.glob(fp.LOCK_GLOB))
        self.assertEqual(
            lying, seen,
            f"в {d} лежит {lying}, а маска {fp.LOCK_GLOB} видит {seen}. "
            f"Разошедшаяся маска отвечает «реестра нет» вместо сверки.")

    def test_either_we_read_it_or_we_say_we_could_not(self):
        found = sorted((self.ROOT / "workflows").glob(fp.LOCK_GLOB))
        got = fp.weights(str(self.ROOT))
        if not found:
            self.assertEqual(got["outcome"], UNMEASURED,
                             "реестра нет, а предполёт не сказал «не смогли»")
            self.assertIn("поток E", got["note"])
            return
        lock = fp.read_lock(found[0])
        self.assertEqual(lock["outcome"], PASS,
                         f"реестр {found[0].name} потока E нашей стороной не "
                         f"разбирается: {lock['note']}")
        self.assertTrue(lock["entries"], "в реестре нет ни одной записи с путём")
        self.assertTrue(all(e["sha256"] for e in lock["entries"]),
                        "в реестре есть запись без хэша — сверять нечем")

    def test_the_weights_are_not_on_this_disk_and_that_is_said_out_loud(self):
        """Веса стека сюда не качались: 40 ГБ. Значит «нет файла», не «ok»."""
        got = fp.weights(str(self.ROOT))
        self.assertNotEqual(got["outcome"], PASS)
        self.assertEqual(got["ok"], 0)


class _Answer:
    """Подделанный ответ HTTP. Тест в сеть не ходит (Т4)."""

    def __init__(self, body, status=200):
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def close(self):
        pass


GOOD_BODY = json.dumps({
    "system": {"comfyui_version": "0.3.40", "python_version": "3.12.0"},
    "devices": [{"name": "cuda:0"}],
})


class TheComfyProbeSaysCouldNotRatherThanCrashing(unittest.TestCase):
    """Comfy в среде нет и ставить его запрещено (§10) — значит «не смогли»."""

    def test_a_believable_answer_reads_as_alive(self):
        got = fp.comfy_alive(opener=lambda url, timeout: _Answer(GOOD_BODY))
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["version"], "0.3.40")
        self.assertEqual(got["devices"], 1)

    def test_a_closed_port_is_unmeasured_not_failed(self):
        """Настоящий loopback: свободный порт берётся и сразу отпускается.

        Ветка по умолчанию (`urllib` без прокси) обязана проверяться живьём,
        иначе подделан весь путь целиком. Сети здесь нет — только 127.0.0.1.
        """
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        got = fp.comfy_alive(f"http://127.0.0.1:{port}")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("§10", got["note"])

    def test_a_timeout_is_unmeasured_too(self):
        def slow(url, timeout):
            raise TimeoutError("timed out")

        got = fp.comfy_alive(opener=slow)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["status"])

    def test_somebody_elses_service_on_the_port_is_a_failure_not_absence(self):
        """Третий исход: порт занят чужим — «поднять Comfy сюда» не выйдет."""
        got = fp.comfy_alive(
            opener=lambda url, timeout: _Answer("<html>nginx</html>"))
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("не Comfy", got["note"])

    def test_valid_json_that_is_not_comfy_is_also_a_failure(self):
        got = fp.comfy_alive(
            opener=lambda url, timeout: _Answer(json.dumps({"ok": True})))
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("system", got["note"])

    def test_an_http_error_names_the_code(self):
        def refuse(url, timeout):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

        got = fp.comfy_alive(opener=refuse)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["status"], 403)

    def test_the_address_is_a_parameter_with_a_default(self):
        seen = []

        def spy(url, timeout):
            seen.append((url, timeout))
            return _Answer(GOOD_BODY)

        fp.comfy_alive(opener=spy)
        fp.comfy_alive("http://10.0.0.5:9000/", opener=spy)
        self.assertEqual(seen[0][0], "http://127.0.0.1:8188/system_stats")
        self.assertEqual(seen[1][0], "http://10.0.0.5:9000/system_stats")
        self.assertEqual(seen[0][1], 2.0)

    def test_the_probe_path_is_guarded(self):
        """Т1: сменив ручку, обязаны увидеть другой запрос."""
        seen = []

        def spy(url, timeout):
            seen.append(url)
            return _Answer(GOOD_BODY)

        original = fp.COMFY_PROBE_PATH
        try:
            fp.COMFY_PROBE_PATH = "/prompt"
            fp.comfy_alive(opener=spy)
        finally:
            fp.COMFY_PROBE_PATH = original
        self.assertEqual(seen[0], "http://127.0.0.1:8188/prompt")


class TheReportCountsThreeOutcomes(unittest.TestCase):

    def test_it_runs_without_a_card_and_says_what_it_could_not_do(self):
        got = fp.report()
        self.assertIn(got["outcome"], (UNMEASURED, FAIL))
        self.assertGreater(got["unmeasured"], 0)

    def test_it_says_out_loud_that_licences_are_not_checked(self):
        self.assertIn("ЛИЦЕНЗИИ НЕ ПРОВЕРЯЮТСЯ", fp.report()["note"])

    def test_the_counts_add_up_to_the_number_of_checks(self):
        got = fp.report()
        self.assertEqual(got["passed"] + got["failed"] + got["unmeasured"],
                         len(got["checks"]))

    def test_disk_is_checked_and_is_cheap(self):
        got = fp.disk()
        self.assertIn(got["outcome"], (PASS, FAIL, UNMEASURED))
        self.assertEqual(got["needed_gb"], 40)

    def test_all_four_checks_of_the_handoff_are_present(self):
        """ХЭНДОФ §6 H: карты, диск, веса по хэшу, живость Comfy."""
        checks = fp.report()["checks"]
        for name in ("карты", "диск", "веса", "comfy"):
            self.assertIn(name, checks)

    def test_the_weights_and_comfy_checks_do_not_pass_by_default_here(self):
        """Ни весов, ни Comfy здесь нет — PASS не имеет права появиться.

        Негативный контроль (И5) на весь отчёт: если бы эти две проверки
        отдавали PASS без файлов и без сервера, отчёт врал бы ровно там, где
        его читают перед арендой машины.

        Состояние «весов» в этой среде за одну смену уже переехало: пока
        реестра потока E не было — «не смогли», как только он лёг — «не годно,
        нет файла ×6». Оба ответа верны, и оба не PASS; поэтому тест
        сторожит именно это, а не одно из двух.
        """
        checks = fp.report()["checks"]
        self.assertIn(checks["веса"]["outcome"], (UNMEASURED, FAIL))
        self.assertEqual(checks["веса"]["ok"], 0)
        self.assertEqual(checks["comfy"]["outcome"], UNMEASURED)


class TheTrainingBudgetIsCountedNotRecalled(unittest.TestCase):
    """Бюджет обучения: расход считается, а не пересказывается.

    Нужда возникла из того, что комментарий в чужом скрипте («1*80G cannot
    train») чуть не стал нашим бюджетом. Владелец оспорил, и оказалось, что 80
    — артефакт ИХ конфигурации: база bf16, видео 81 кадр, без квантизации.
    """

    def test_four_bit_base_on_frames_fits_sixteen_gigabytes(self):
        got = fp.training_budget(16.0, base="uint4", mode="кадры 512")
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(got["headroom_gb"], 0)

    def test_fp8_base_does_not_fit_sixteen(self):
        """Негативный контроль: если влезает ВСЁ, функция ничего не меряет."""
        got = fp.training_budget(16.0, base="fp8", mode="кадры 512")
        self.assertEqual(got["outcome"], FAIL, got["note"])

    def test_video_mode_does_not_fit_sixteen_even_at_four_bits(self):
        got = fp.training_budget(16.0, base="uint4", mode="видео 81 кадр")
        self.assertEqual(got["outcome"], FAIL)

    def test_the_tight_band_is_its_own_outcome_not_a_failure(self):
        """Р1, и эта мутация ПЕРЕЖИЛА первый прогон: свёртывание «впритык» в
        провал не краснело нигде, потому что полосу 0 <= запас < 1 не проверял
        ни один тест.

        Состояние отдельное именно потому, что ECC забирает неизмеренную часть
        памяти, и остаток в 0.7 ГБ от неё не защищает: это не «влезло» и не
        «не влезло», это «мы не знаем».
        """
        total = fp.training_budget(16.0)["total_gb"]
        tight = fp.training_budget(total + 0.7)
        self.assertEqual(tight["outcome"], UNMEASURED, tight["note"])
        self.assertIn("ВПРИТЫК", tight["note"])
        self.assertGreaterEqual(tight["headroom_gb"], 0)

    def test_the_three_bands_are_all_reachable(self):
        """Негативный контроль к предыдущему: если достижимы не все три,
        третий исход существует только на бумаге."""
        total = fp.training_budget(16.0)["total_gb"]
        outcomes = [fp.training_budget(total + d)["outcome"]
                    for d in (-1.0, 0.5, 5.0)]
        self.assertEqual(outcomes, [FAIL, UNMEASURED, PASS])

    def test_an_unknown_card_size_gives_the_cost_but_no_verdict(self):
        """Р1: расход посчитан, вердикта нет — это третий исход, не провал."""
        got = fp.training_budget(None)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNotNone(got["total_gb"])
        self.assertIn("вердикт — нет", got["note"])

    def test_an_unknown_mode_refuses_to_extrapolate(self):
        got = fp.training_budget(16.0, mode="видео 300 кадров")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("выдуманный расход", got["note"])

    def test_the_parts_are_printed_not_just_the_total(self):
        """Человек, которому сказали «10.3 ГБ», следующим спросит «из чего»."""
        got = fp.training_budget(16.0)
        for part in ("база", "LoRA с оптимизатором", "активации", "контекст CUDA"):
            self.assertIn(part, got["parts"])
        self.assertIn("РАСЧЁТ, НЕ ЗАМЕР", got["note"])

    def test_the_rank_moves_the_trainable_parameters_linearly(self):
        a = fp.training_budget(16.0, rank=32)["trainable_params"]
        b = fp.training_budget(16.0, rank=64)["trainable_params"]
        self.assertEqual(b, a * 2)

    def test_a_high_rank_can_break_the_budget(self):
        """Т1 в обе стороны: ранг обязан двигать вердикт, иначе он декорация."""
        self.assertEqual(fp.training_budget(16.0, rank=32)["outcome"], PASS)
        self.assertEqual(fp.training_budget(16.0, rank=2048)["outcome"], FAIL)

    def test_moving_the_base_table_moves_the_verdict(self):
        saved = dict(fp.TRAIN_BASE_GB)
        try:
            fp.TRAIN_BASE_GB["uint4"] = 100.0
            self.assertEqual(fp.training_budget(16.0)["outcome"], FAIL,
                             "подмена таблицы базы не доехала до вердикта")
        finally:
            fp.TRAIN_BASE_GB.clear()
            fp.TRAIN_BASE_GB.update(saved)

    def test_the_model_records_that_it_fails_its_own_control(self):
        """И6: отрицательный результат записан числом и условиями.

        Наш расчёт вендорской конфигурации даёт 54.3 ГБ, а они пишут, что 80 не
        хватает. Модель НЕ воспроизводит известный контрольный вход, и это её
        свойство, а не их ошибка."""
        d = fp.TRAIN_MODEL_DISAGREES_WITH_VENDOR
        self.assertEqual(d["наш расчёт, ГБ"], 54.3)
        self.assertIn("8x80G", d["их утверждение"])
        self.assertIn("занижает", d["вывод"])

    def test_the_recorded_disagreement_matches_what_the_function_computes(self):
        """Записанное число обязано СЛЕДОВАТЬ из функции, иначе это надпись."""
        got = fp.training_budget(80.0, base="bf16", mode="видео 81 кадр")
        self.assertAlmostEqual(
            got["total_gb"],
            fp.TRAIN_MODEL_DISAGREES_WITH_VENDOR["наш расчёт, ГБ"],
            places=1,
            msg="в реестре расхождения лежит число, которого функция не даёт")




class BlockswapIsPartOfTheMemoryModel(unittest.TestCase):
    """Модель, не знающая про то, что делает граф, судит не тот прогон.

    До 18.08 `budget` считал всю модель резидентной и браковал Q4_K_M на
    16 ГиБ (остаток 0.02 при требуемом 1.0). Граф при этом свопит 38 блоков
    из 40. Ошибка была не в числе, а в том, что мерилась конфигурация,
    которую мы не запускаем, — и точно так же она пропустила бы негодную.
    """

    def test_the_block_count_is_the_measured_one(self):
        # Литерал (Т2): 40 блоков получены разбором заголовков четырёх шардов
        # safetensors Range-запросами, без скачивания весов.
        self.assertEqual(fp.BLOCKS_TOTAL, 40)

    def test_the_block_share_is_the_measured_one(self):
        # 16 153 538 560 из 17 274 817 108 параметров.
        self.assertAlmostEqual(fp.BLOCK_SHARE, 0.935092, places=6)

    def test_swapping_nothing_leaves_the_whole_model_on_the_card(self):
        """Негативный контроль (И5): при нулевом свопе модель обязана
        выродиться в прежнюю, иначе она чинит не то."""
        got = fp.blockswap_weights(10.71, 0)
        self.assertAlmostEqual(got["vram_gb"], 10.71, places=2)

    def test_swapping_everything_still_leaves_the_unswappable_part(self):
        """6.49% параметров блоксвоп не трогает вовсе.

        Обнулить их значило бы занижение расхода — ошибка в опасную сторону:
        отчёт обещает память, которой на карте нет.
        """
        got = fp.blockswap_weights(10.71, 40)
        self.assertGreater(got["vram_gb"], 0.6)
        self.assertLess(got["vram_gb"], 1.2)

    def test_more_blocks_than_the_model_has_is_unmeasured_not_zero(self):
        got = fp.blockswap_weights(10.71, 41)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["vram_gb"])

    def test_the_in_flight_block_is_counted_and_the_number_is_guarded(self):
        """Пережила первый заход мутаций в обе стороны — значит её не
        сторожил никто, и отгружаемое значение можно было менять молча.

        Арифметика литералами (Т2): 40 блоков, свопнуто 38, остаются
        резидентными 2, и ещё один лежит на карте в момент своего счёта —
        итого 3. Единица здесь ВЫБРАНА консервативно: занизить значит
        пообещать памяти больше, чем есть, и получить отказ на карте, а не
        в отчёте.
        """
        got = fp.blockswap_weights(10.71, 38)
        self.assertEqual(got["resident_blocks"], 3)
        self.assertAlmostEqual(got["vram_gb"], 1.45, places=2)

    def test_the_in_flight_allowance_is_never_optimistic(self):
        """Ноль означал бы, что блок появляется на карте бесплатно."""
        self.assertGreaterEqual(fp.IN_FLIGHT_BLOCKS, 1)
        self.assertEqual(
            fp.blockswap_weights(10.71, 40)["resident_blocks"],
            fp.IN_FLIGHT_BLOCKS,
            "при полном свопе на карте обязан оставаться ровно блок в полёте")

    def test_what_leaves_the_card_arrives_in_system_memory(self):
        """Блоксвоп не уменьшает расход, он его переносит — и это в отчёте."""
        got = fp.blockswap_weights(10.71, 38)
        self.assertGreater(got["ram_gb"], 9.0)
        self.assertIn("ОПЕРАТИВНУЮ", got["note"])

    def test_the_owners_step_fits_the_card_once_blockswap_is_counted(self):
        """Число, из-за которого поднялся вопрос к владельцу, — и его ответ."""
        got = fp.budget(15.99, length=77, blocks_to_swap=38)
        row = next(r for r in got["rows"] if r["step"] == "Q4_K_M")
        self.assertTrue(row["fits"])
        self.assertGreater(row["headroom_gb"], 9.0)

    def test_without_blockswap_the_same_step_does_not_fit(self):
        """Обе стороны развилки проверены, иначе тест сторожит одну."""
        got = fp.budget(15.99, length=77)
        row = next(r for r in got["rows"] if r["step"] == "Q4_K_M")
        self.assertFalse(row["fits"])
        self.assertLess(row["headroom_gb"], 0.1)

    def test_the_report_says_out_loud_which_of_the_two_was_computed(self):
        with_swap = fp.budget(15.99, blocks_to_swap=38)["note"]
        without = fp.budget(15.99)["note"]
        self.assertIn("блоксвопе 38", with_swap)
        self.assertIn("БЕЗ БЛОКСВОПА", without,
                      "два разных расчёта неразличимы в отчёте — читатель "
                      "решит, что видит тот, которого ждёт")

class BlockswapCostsTimeAndTheTimeIsCounted(unittest.TestCase):
    """Вторая половина размена «память на время».

    Расчёт памяти сказал «Q4_K_M влезает» и замолчал. Но 9.51 ГиБ едут по
    шине на каждом шаге и каждом окне, а NVLink у A16 нет. Пока это не
    посчитано, «влезает» читается как «и ничего не стоит».

    ГЛАВНОЕ ТРЕБОВАНИЕ ЭТОГО КЛАССА: без замера прибор обязан отдавать ВИЛКУ
    и исход «не смогли», а не одно число, выглядящее замером.
    """

    def test_the_pcie_ceiling_is_the_spec_arithmetic_not_a_recollection(self):
        """Т2: ожидаемое пересчитано литералами, а не взято из модуля.

        8 GT/s и 16 GT/s на линию, 16 линий, кодирование 128b/130b:
        8e9 * 16 * 128/130 / 8 = 15.754e9 Б/с; вдвое больше у 4.0.
        """
        gen3 = 8e9 * 16 * (128 / 130) / 8 / 1e9
        gen4 = 16e9 * 16 * (128 / 130) / 8 / 1e9
        self.assertAlmostEqual(fp.PCIE_X16_GBPS["3.0"], gen3, places=1)
        self.assertAlmostEqual(fp.PCIE_X16_GBPS["4.0"], gen4, places=1)
        self.assertAlmostEqual(fp.PCIE_X16_GBPS["3.0"], 15.75, places=2)
        self.assertAlmostEqual(fp.PCIE_X16_GBPS["4.0"], 31.51, places=2)

    def test_the_bus_generation_is_not_chosen_because_it_is_not_known(self):
        """Редакция шины у A16 не подтверждена: nvidia.com закрыт (Ц3).

        Выбрать одну значило бы выдать догадку за паспорт, поэтому в оценку
        идут обе, и вилка обязана быть НЕВЫРОЖДЕННОЙ.
        """
        self.assertEqual(set(fp.PCIE_GEN_UNVERIFIED), {"3.0", "4.0"})
        got = fp.blockswap_seconds(10.71, 38)
        self.assertGreater(got["high_s"], got["low_s"])

    def test_without_a_measurement_the_verdict_is_unmeasured_with_a_fork(self):
        """Р1: третий исход не сворачивается в число."""
        got = fp.blockswap_seconds(10.71, 38)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNotNone(got["low_s"])
        self.assertIn("НЕ ЗАМЕРЕНО", got["note"])

    def test_the_time_is_built_from_the_named_parts(self):
        """Из ЧАСТЕЙ, а не одним числом с потолка — все четыре в отчёте.

        Литералами (Т2): 10 с при 30 к/с прижимаются к 297 кадрам, окнами по
        77 это 4 окна; шагов у графа 3; за перенос едет 9.51 ГиБ.
        Итого 4*3 = 12 переносов и 114.12 ГиБ за ролик.
        """
        got = fp.blockswap_seconds(10.71, 38)
        self.assertEqual(got["parts"]["окон"], 4)
        self.assertEqual(got["parts"]["шагов сэмплера"], 3)
        self.assertEqual(got["parts"]["переносов"], 12)
        self.assertAlmostEqual(got["parts"]["ГиБ за перенос"], 9.51, places=2)
        self.assertAlmostEqual(got["parts"]["ГиБ за ролик"], 114.12, places=1)

    def test_the_fork_is_the_spec_arithmetic_and_it_is_checked_by_hand(self):
        """Низ вилки: 114.12 ГиБ по 31.51 ГБ/с на полном паспорте.

        114.12 * 2**30 / (31.51 * 1e9) = 3.89 с. Верх — та же масса по
        15.75 ГБ/с на половине паспорта, то есть ровно вчетверо больше.
        """
        got = fp.blockswap_seconds(10.71, 38)
        self.assertAlmostEqual(got["low_s"], 3.89, places=1)
        self.assertAlmostEqual(got["high_s"], 15.56, places=1)
        self.assertAlmostEqual(got["high_s"] / got["low_s"], 4.0, places=1)

    def test_swapping_nothing_costs_no_time_and_that_is_knowledge(self):
        """Негативный контроль (И5): вход, где прибор обязан сказать «нет».

        При нулевом свопе по шине не едет ничего, и это ЕДИНСТВЕННЫЙ случай,
        когда без замера полосы можно ответить числом: ноль байт делится на
        любую полосу одинаково.
        """
        got = fp.blockswap_seconds(10.71, 0)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["low_s"], 0.0)
        self.assertEqual(got["high_s"], 0.0)

    def test_a_bigger_swap_costs_strictly_more_time(self):
        """Второй край контроля (И5): вход, где прибор обязан шевельнуться."""
        small = fp.blockswap_seconds(10.71, 5)
        big = fp.blockswap_seconds(10.71, 38)
        self.assertGreater(big["high_s"], small["high_s"])
        self.assertGreater(small["high_s"], 0.0)

    def test_a_measured_bandwidth_collapses_the_fork_to_one_number(self):
        """Как у `throughput`: замер подаётся параметром и меняет исход.

        Литералом: 114.12 ГиБ при 20 ГБ/с = 114.12*2**30/20e9 = 6.13 с.
        """
        got = fp.blockswap_seconds(10.71, 38, measured_gbps=20.0)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["low_s"], got["high_s"])
        self.assertAlmostEqual(got["low_s"], 6.13, places=1)
        self.assertIn("ЗАМЕРЕНО", got["note"])

    def test_four_cards_share_one_bus_and_the_time_grows(self):
        """Находка, которую прибор обязан печатать: NVLink нет, шина одна.

        Четыре карты A16 — четыре независимых потребителя одного корневого
        комплекса PCIe. Считать четырёхкратную выработку при неизменном
        времени шины значило бы обещать масштабирование, которого нет.
        """
        one = fp.blockswap_seconds(10.71, 38)
        four = fp.blockswap_seconds(10.71, 38, concurrent_cards=4)
        # ОБА края вилки, а не один: деление, забытое на одном краю, оставило
        # бы прибор наполовину врущим и мутацию — выжившей (так и было).
        self.assertAlmostEqual(four["high_s"] / one["high_s"], 4.0, places=1)
        self.assertAlmostEqual(four["low_s"] / one["low_s"], 4.0, places=1)
        self.assertAlmostEqual(four["low_s"], 15.56, places=1)

    def test_the_worst_case_efficiency_is_guarded_in_both_directions(self):
        """Т1: константа-решение мутируется строже и слабее.

        Ниже — прибор обязан назвать больше секунд, выше — меньше.
        """
        original = fp.BUS_EFFICIENCY_WORST
        try:
            fp.BUS_EFFICIENCY_WORST = 0.25
            slower = fp.blockswap_seconds(10.71, 38)["high_s"]
            fp.BUS_EFFICIENCY_WORST = 0.90
            faster = fp.blockswap_seconds(10.71, 38)["high_s"]
        finally:
            fp.BUS_EFFICIENCY_WORST = original
        self.assertAlmostEqual(slower, 31.12, places=1)
        self.assertAlmostEqual(faster, 8.64, places=1)
        self.assertGreater(slower, faster)

    def test_the_best_case_efficiency_is_guarded_in_both_directions(self):
        """Т1 для верхнего края: он тоже отгружаемое значение."""
        original = fp.BUS_EFFICIENCY_BEST
        try:
            fp.BUS_EFFICIENCY_BEST = 0.50
            slower = fp.blockswap_seconds(10.71, 38)["low_s"]
            fp.BUS_EFFICIENCY_BEST = 2.00
            faster = fp.blockswap_seconds(10.71, 38)["low_s"]
        finally:
            fp.BUS_EFFICIENCY_BEST = original
        self.assertAlmostEqual(slower, 7.78, places=1)
        self.assertAlmostEqual(faster, 1.95, places=1)

    def test_gibibytes_and_gigabytes_are_not_silently_mixed(self):
        """7% ошибки ниоткуда: веса в ГиБ, полоса в десятичных ГБ."""
        self.assertEqual(fp.GIB_BYTES, 1073741824)
        self.assertEqual(fp.GB_BYTES, 1000000000)

    def test_a_bad_input_is_unmeasured_without_numbers_not_zero_seconds(self):
        """Р1: «не смогли посчитать» и «посчитали, вышло мало» — разные ответы."""
        for bad in (fp.blockswap_seconds(10.71, 41),
                    fp.blockswap_seconds(10.71, 38, windows=0),
                    fp.blockswap_seconds(10.71, 38, steps=0),
                    fp.blockswap_seconds(10.71, 38, concurrent_cards=0),
                    fp.blockswap_seconds(10.71, 38, seconds=99.0)):
            self.assertEqual(bad["outcome"], UNMEASURED)
            self.assertIsNone(bad["low_s"])

    def test_the_graph_parameters_are_imported_not_copied(self):
        """Е1: шаги и окно живут в `fork_comfy`, здесь их копий нет.

        Признак настоящего дубля: изменил одно — обязано измениться второе.
        """
        from ball_reel import fork_comfy as fc

        original = fc.WRAP_STEPS
        try:
            fc.WRAP_STEPS = 6
            self.assertEqual(
                fp.blockswap_seconds(10.71, 38)["parts"]["переносов"], 24,
                "число шагов скопировано в предполёт — оно разъедется с графом")
        finally:
            fc.WRAP_STEPS = original


class HowManyBlocksCanStayHome(unittest.TestCase):
    """Каждый несвопнутый блок — сэкономленное время шины.

    Граф ставит 38 из 40, и это значение владельца, а не подбор под нашу
    карту. Между «влезает при 38» и «нужно 38» — разница в разы по шине.
    """

    def test_the_floor_is_the_minimum_swap_that_still_fits(self):
        """Литералами (Т2): блок Q4_K_M весит 10.71*0.935092/40 = 0.2504 ГиБ.

        Без свопа Q4_K_M на 15.99 ГиБ даёт остаток 0.02, а нужен 1.0. Уехать
        с карты должно 0.98 ГиБ. Но блок в полёте возвращает один обратно:
        при свопе N резидентных блоков остаётся 41-N, то есть свопнуть надо
        N, чтобы уехало N-1 блоков. 4 свопа уносят 3 блока (0.75 — мало),
        5 свопов уносят 4 (1.00 — хватает). Порог 5.
        """
        got = fp.swap_floor(15.99, length=77)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["floor"], 5)
        self.assertEqual(got["step"], "Q4_K_M")

    def test_the_floor_row_really_fits_and_the_one_below_really_does_not(self):
        """Негативный контроль порога (И5): проверены обе стороны границы."""
        below = next(r for r in fp.budget(15.99, length=77, blocks_to_swap=4)["rows"]
                     if r["step"] == "Q4_K_M")
        at = next(r for r in fp.budget(15.99, length=77, blocks_to_swap=5)["rows"]
                  if r["step"] == "Q4_K_M")
        self.assertFalse(below["fits"])
        self.assertTrue(at["fits"])

    def test_the_graph_swaps_thirty_three_blocks_more_than_memory_asks(self):
        """Находка: 33 блока свопаются сверх нужного, и это чистое время шины."""
        got = fp.swap_floor(15.99, length=77)
        self.assertEqual(got["graph_blocks_to_swap"], 38)
        self.assertEqual(got["can_stay_resident"], 33)

    def test_the_floor_is_cheaper_on_the_bus_by_the_ratio_of_blocks(self):
        """5 блоков против 38 — по шине едет в 7.6 раза меньше.

        Литералом: 38/5 = 7.6, и время обязано идти ровно пропорционально
        массе, потому что полоса от числа блоков не зависит.
        """
        got = fp.swap_floor(15.99, length=77)
        self.assertAlmostEqual(
            got["seconds_at_graph"][1] / got["seconds_at_floor"][1], 7.6,
            places=1)
        self.assertGreater(got["seconds_saved_worst_case"], 13.0)

    def test_a_card_where_nothing_fits_is_fail_not_a_floor_of_forty(self):
        """Р1: «не влезает даже при полном свопе» — это НЕ ГОДНО, а не порог 40.

        Свернуть это в «свопай всё» значило бы выдать негодную конфигурацию
        за рабочую: рычаг здесь не блоки, а ступень или длина окна.
        """
        got = fp.swap_floor(4.0, length=77)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIsNone(got["floor"])

    def test_a_big_card_needs_no_swap_at_all(self):
        """Второй край (И5): на просторной карте порог обязан быть нулевым."""
        got = fp.swap_floor(40.0, length=77)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["floor"], 0)
        self.assertEqual(got["seconds_at_floor"], (0.0, 0.0))

    def test_an_unknown_step_is_unmeasured_not_a_floor(self):
        got = fp.swap_floor(15.99, step="Q8_0")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["floor"])

    def test_the_headroom_bar_moves_the_floor_in_both_directions(self):
        """Т1: порог годности — константа-решение, от неё зависит ответ."""
        original = fp.MIN_HEADROOM_GB
        try:
            fp.MIN_HEADROOM_GB = 0.1
            # 2, а не 1: своп одного блока не освобождает НИЧЕГО — его место
            # тут же занимает блок в полёте. Ручка начинает работать со
            # второго блока, и прибор это печатает, а не сглаживает.
            self.assertEqual(fp.swap_floor(15.99, length=77)["floor"], 2)
            fp.MIN_HEADROOM_GB = 4.0
            self.assertEqual(fp.swap_floor(15.99, length=77)["floor"], 17)
        finally:
            fp.MIN_HEADROOM_GB = original


class DoesSwapChangeActivations(unittest.TestCase):
    """Догадку владельца полагается ПРОВЕРИТЬ, а не подтвердить.

    `ACTIVATIONS_GB` снималось без свопа. Рассуждение «своп двигает веса, а
    активации — тензоры счёта» правдоподобно, но у нас уже был случай, когда
    неизмеренная гипотеза подавалась как решённое.
    """

    def test_without_a_measurement_the_answer_is_unmeasured_not_yes(self):
        """Р1 и главное требование задачи: рассуждение в вердикт не идёт."""
        got = fp.activations_under_swap()
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["delta_gb"])
        self.assertIn("НЕ ЗАСЧИТЫВАЕТСЯ", got["note"])

    def test_it_says_what_exactly_to_measure_and_how(self):
        """«Не смогли» без процедуры замера — это отписка, а не третий исход."""
        how = fp.activations_under_swap()["how"]
        self.assertIn("blocks_to_swap", how)
        self.assertIn("nvidia-smi", how)
        self.assertIn("вычесть веса", how)

    def test_a_measurement_that_agrees_confirms_the_owners_guess(self):
        """Вход, где прибор обязан шевельнуться в «годно» (И5).

        Литералом: базовые 2.03 ГиБ, порог 0.15 — 2.10 внутри полосы.
        """
        got = fp.activations_under_swap(2.10)
        self.assertEqual(got["outcome"], PASS)
        self.assertAlmostEqual(got["delta_gb"], 0.07, places=2)

    def test_a_measurement_that_disagrees_is_a_defect_in_the_budget(self):
        """Вход, где прибор обязан сказать «не годно» (И5)."""
        got = fp.activations_under_swap(3.50)
        self.assertEqual(got["outcome"], FAIL)
        self.assertAlmostEqual(got["delta_gb"], 1.47, places=2)
        self.assertIn("ACTIVATIONS_GB", got["note"])

    def test_activations_falling_under_swap_is_a_disagreement_too(self):
        """Расхождение ВНИЗ — такое же расхождение (пережило мутацию `abs`).

        Если при свопе активаций стало заметно МЕНЬШЕ, значит мерилось не то
        же самое, и бюджет всё равно построен на числе не той конфигурации.
        Прибор, сравнивающий без модуля, молча звал бы это согласием.
        """
        got = fp.activations_under_swap(1.50)
        self.assertEqual(got["outcome"], FAIL)
        self.assertAlmostEqual(got["delta_gb"], -0.53, places=2)

    def test_the_tolerance_is_guarded_in_both_directions(self):
        """Т1: строже и слабее, и оба края обязаны менять вердикт.

        2.20 против 2.03 — разница 0.17: при пороге 0.15 это «не годно», при
        пороге 0.30 — «годно».
        """
        original = fp.ACTIVATIONS_SWAP_TOLERANCE_GB
        try:
            fp.ACTIVATIONS_SWAP_TOLERANCE_GB = 0.05
            self.assertEqual(fp.activations_under_swap(2.10)["outcome"], FAIL)
            fp.ACTIVATIONS_SWAP_TOLERANCE_GB = 0.30
            self.assertEqual(fp.activations_under_swap(2.20)["outcome"], PASS)
        finally:
            fp.ACTIVATIONS_SWAP_TOLERANCE_GB = original

    def test_the_tolerance_is_smaller_than_one_swapped_block(self):
        """Порог обязан различать шум замера и лишний блок на карте.

        Блок Q4_K_M весит 0.250 ГиБ (литерал: 10.71*0.935092/40). Порог,
        равный блоку или больше, не заметил бы ровно того, ради чего он есть.
        """
        self.assertLess(fp.ACTIVATIONS_SWAP_TOLERANCE_GB, 0.25)

    def test_an_unknown_length_is_unmeasured_not_agreement(self):
        got = fp.activations_under_swap(2.03, length=61)
        self.assertEqual(got["outcome"], UNMEASURED)


class BlockswapDoesNotExplainTheTrainingGap(unittest.TestCase):
    """Отрицательный результат с числом и условиями (И6).

    Провалившийся негативный контроль модели обучения: 54.3 ГБ там, где
    вендор пишет «1×80G не тянет». Блоксвоп проверен как кандидат в
    объяснение и отвергнут — знак расхождения обратный.
    """

    def test_the_gap_grows_instead_of_closing(self):
        """Числами (Т2, литералы): 80 - 54.3 = 25.7 было, стало 80 - 30.08."""
        got = fp.blockswap_explains_training_gap()
        self.assertEqual(got["outcome"], FAIL)
        self.assertFalse(got["explains"])
        self.assertAlmostEqual(got["ours_resident_gb"], 54.3, places=1)
        self.assertAlmostEqual(got["ours_swapped_gb"], 30.08, places=1)
        self.assertAlmostEqual(got["gap_before_gb"], 25.7, places=1)
        self.assertAlmostEqual(got["gap_after_gb"], 49.92, places=1)

    def test_the_negative_result_is_written_with_the_number_zero(self):
        """И6: «не объясняет» без числа — мнение, а не результат."""
        self.assertIn("объяснено 0 ГБ",
                      fp.blockswap_explains_training_gap()["note"])

    def test_the_instrument_can_also_say_yes(self):
        """Негативный контроль прибора (И5): вход, где он обязан шевельнуться.

        При пороге 30 ГБ вместо 80 своп разрыв действительно закрывает — с
        24.3 ГБ до 0.08. Прибор, умеющий только «нет», не отличим от заглушки.
        """
        got = fp.blockswap_explains_training_gap(vendor_gb=30.0)
        self.assertEqual(got["outcome"], PASS)
        self.assertTrue(got["explains"])

    def test_a_bad_vendor_threshold_is_unmeasured_not_no(self):
        got = fp.blockswap_explains_training_gap(vendor_gb=0)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["explains"])

    def test_the_rejection_is_written_into_the_recorded_disagreement(self):
        """Знание, не попавшее в запись, следующая смена добудет заново."""
        rec = fp.TRAIN_MODEL_DISAGREES_WITH_VENDOR
        self.assertIn("блоксвоп проверен и отвергнут", rec)
        self.assertIn("0 ГБ", rec["блоксвоп проверен и отвергнут"])

    def test_the_swap_amount_moves_the_answer_in_both_directions(self):
        """Т1 на входе-константе: нулевой своп обязан ничего не менять."""
        none_swapped = fp.blockswap_explains_training_gap(blocks_to_swap=0)
        self.assertAlmostEqual(none_swapped["gap_after_gb"],
                               none_swapped["gap_before_gb"], places=1)
        full = fp.blockswap_explains_training_gap(blocks_to_swap=40)
        self.assertGreater(full["gap_after_gb"], 49.92)


class ThePreflightJudgesTheRunWeActuallyLaunch(unittest.TestCase):
    """Отчёт, считающий всю модель резидентной, судит не тот прогон."""

    def test_the_report_counts_the_swap_the_graph_sets(self):
        got = fp.report()
        self.assertEqual(got["blocks_to_swap"], 38)
        self.assertEqual(
            got["checks"]["время шины"]["parts"]["свопнуто блоков"], 38)

    def test_the_report_carries_the_bus_time_and_calls_it_an_estimate(self):
        check = fp.report()["checks"]["время шины"]
        self.assertEqual(check["outcome"], UNMEASURED)
        self.assertIn("НЕ ЗАМЕРЕНО", check["note"])

    def test_the_swap_can_be_switched_off_and_the_report_says_so(self):
        """Обе стороны развилки достижимы, иначе сторожится одна."""
        got = fp.report(blocks_to_swap=None if False else 0)
        self.assertEqual(got["checks"]["время шины"]["outcome"], PASS)
        self.assertEqual(got["checks"]["время шины"]["low_s"], 0.0)




class TheFirstRealTimingInTheProject(unittest.TestCase):
    """Разбор двух точек замера владельца. До них у проекта не было ни одной.

    Считать надо ПРОХОДЫ модели, а не шаги: cfg больше единицы даёт два
    прохода на шаг. На шагах две точки не сходятся вовсе — получается
    отрицательная постоянная часть, и это первый признак, что разбор неверен.
    """

    def test_cfg_above_one_doubles_the_passes(self):
        self.assertEqual(fp.passes_for(20, 3.5), 40)
        self.assertEqual(fp.passes_for(4, 1), 4)

    def test_cfg_exactly_one_is_a_single_pass(self):
        """Граница, а не «больше-меньше»: cfg=1 — наш рабочий режим."""
        self.assertEqual(fp.passes_for(3, 1.0), 3)

    def test_the_two_observations_reconcile_exactly(self):
        """Литералы (Т2) — это наблюдения владельца, они не наши и не поедут.

        40*p + f = 513 и 4*p + f = 71. Сходимость точная; независимая
        проверка — разница первого и второго прогонов (23 и 26 с) должна
        совпасть с постоянной частью, и совпадает.
        """
        p, f = fp.MEASURED_PASS_S, fp.MEASURED_FIXED_S
        self.assertAlmostEqual(40 * p + f, 513.0, places=0)
        self.assertAlmostEqual(4 * p + f, 71.0, places=0)
        self.assertTrue(21.0 <= f <= 27.0,
                        f"постоянная часть {f} разошлась со временем загрузки "
                        f"весов (23 и 26 с) — значит разбор неверен")

    def test_our_own_number_is_unmeasured_because_it_is_another_card(self):
        got = fp.render_seconds(seconds=10.0)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("ИЗМЕРЕНА НЕ У НАС", got["note"])
        self.assertIn("ЗАНИЖЕНИЕ", got["note"],
                      "линейный пересчёт подан как точный — а он оптимистичен")

    def test_a_measured_pass_makes_it_a_verdict(self):
        """Негативный контроль (И5): прибор обязан уметь и говорить «годно»."""
        got = fp.render_seconds(seconds=10.0, pass_s=30.0)
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("ЗАМЕРЕНА на нашей карте", got["note"])

    def test_longer_clips_cost_more_windows_and_more_time(self):
        short = fp.render_seconds(seconds=5.0)
        long = fp.render_seconds(seconds=10.0)
        self.assertGreater(long["windows"], short["windows"])
        self.assertGreater(long["compute_s"], short["compute_s"])

    def test_the_geometry_of_the_measurement_is_the_one_it_was_taken_at(self):
        """Пережило первый заход: пересчёт никем не сторожился.

        Литералы (Т2) — это условия чужого замера, 640x640 и 81 кадр; поедут
        они только если владелец пришлёт другой замер, и тогда тест обязан
        покраснеть, а не поехать следом.
        """
        self.assertEqual(fp.MEASURED_PIXELS, 640 * 640)
        self.assertEqual(fp.MEASURED_FRAMES, 81)

    def test_our_smaller_window_scales_the_cost_down(self):
        """Мутация «пересчёт снят» пережила сьют — значит его не было видно."""
        got = fp.render_seconds(seconds=10.0)
        self.assertLess(got["scale"], 1.0,
                        "наше окно меньше замерного и по пикселям, и по "
                        "кадрам — множитель обязан быть меньше единицы")
        self.assertGreater(got["scale"], 0.8)

    def test_the_fixed_part_is_paid_once_not_per_window(self):
        """Иначе десятисекундный ролик стоил бы вчетверо больше постоянной."""
        got = fp.render_seconds(seconds=10.0)
        self.assertAlmostEqual(got["total_s"] - got["compute_s"],
                               fp.MEASURED_FIXED_S, places=0)


class TheBusShareHasADenominator(unittest.TestCase):
    """«В 7.6 раза дороже» без знаменателя подталкивает к решению, которого
    само число не обосновывает: 7.6 раза от малого — это малое."""

    def test_the_graph_setting_costs_a_small_share_of_the_time(self):
        got = fp.bus_share(seconds=10.0, blocks_to_swap=38)
        self.assertLess(got["high"], 0.20,
                        "доля шины оказалась велика — тогда своп 38 надо "
                        "менять, и это уже другой разговор")
        self.assertGreater(got["high"], 0.0)

    def test_the_threshold_setting_is_cheaper_but_the_gap_is_small(self):
        few = fp.bus_share(seconds=10.0, blocks_to_swap=5)
        many = fp.bus_share(seconds=10.0, blocks_to_swap=38)
        self.assertLess(few["high"], many["high"])
        self.assertLess(many["high"] - few["high"], 0.15,
                        "разница долей велика — вывод «своп почти бесплатен» "
                        "перестал держаться")

    def test_the_share_is_taken_from_compute_not_from_total(self):
        """Пережило первый заход. Знаменателем обязан быть СЧЁТ, а не полное
        время: постоянная часть (VAE-декод, кодирование текста) шину не
        занимает и не занимала бы её и без свопа. Делить на неё значит
        занизить долю — то есть ошибиться в успокаивающую сторону.
        """
        got = fp.bus_share(seconds=10.0, blocks_to_swap=38)
        calc = fp.render_seconds(seconds=10.0)
        # ОБА края, а не один: первая версия сторожила только верхний, и
        # подмена знаменателя на нижнем крае пережила сьют. Прибор врал бы
        # наполовину — ровно та же дыра, что нашлась у деления на карты.
        for край, i in (("low", 0), ("high", 1)):
            with self.subTest(край=край):
                self.assertAlmostEqual(
                    got[край], got["bus_s"][i] / calc["compute_s"], places=4)
                self.assertNotAlmostEqual(
                    got[край], got["bus_s"][i] / calc["total_s"], places=4)

    def test_no_swap_at_all_costs_nothing_on_the_bus(self):
        """Негативный контроль: ноль байт делится на любую полосу одинаково."""
        got = fp.bus_share(seconds=10.0, blocks_to_swap=0)
        self.assertEqual(got["high"], 0.0)

    def test_the_verdict_stays_unmeasured_and_says_why(self):
        got = fp.bus_share(seconds=10.0)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("ОЦЕНКИ", got["note"])


class TheMemoryModelAgreesWithAnOutsideObservation(unittest.TestCase):
    """Первая внешняя сверка модели памяти. Раньше её не с чем было сверять.

    Наблюдение владельца: fp8_scaled на 640x640x81 занимает 84% от 24 ГБ,
    то есть 20.16 ГБ. Наша модель складывает веса, LoRA, активации и резерв
    Comfy. Сходимость в пределах 5% — это не доказательство модели, но это
    первое, что её вообще проверяет извне.
    """

    def test_the_model_lands_within_five_percent_of_the_observation(self):
        observed = 0.84 * 24.0                      # литералы наблюдения (Т2)
        ours = 14.0 + fp.LORA_GB + fp.ACTIVATIONS_GB[77] + fp.COMFY_RESERVE_GB
        self.assertLess(abs(ours - observed) / observed, 0.05,
                        f"модель {ours:.2f} против наблюдённых {observed:.2f}")

    def test_the_model_is_not_above_the_observation(self):
        """Занижение опаснее завышения: обещанная память кончается на карте."""
        observed = 0.84 * 24.0
        ours = 14.0 + fp.LORA_GB + fp.ACTIVATIONS_GB[77] + fp.COMFY_RESERVE_GB
        self.assertLess(ours, observed,
                        "модель обещает БОЛЬШЕ свободной памяти, чем видно "
                        "в наблюдении — ошибка в опасную сторону")



if __name__ == "__main__":
    unittest.main()
