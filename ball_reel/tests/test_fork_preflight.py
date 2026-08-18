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


if __name__ == "__main__":
    unittest.main()
