"""Приёмка арендованной машины: прибор обязан уметь сказать все три вещи.

ЧТО НЕСЁТ ЭТОТ ФАЙЛ, тремя строками:

* **негодно** на битом входе — файл на байт короче эталона и хэш не тот (И5);
* **не смогли** там, где спросить нечем — `nvidia-smi` отсутствует, реестра
  нет, ComfyUI не найден. Ни одно из этого НЕ ЕСТЬ «негодно» (Р1);
* **числа рядом с вердиктом** — проверено, провалено, не смогли, пропущено (Р2).

ОЖИДАЕМОЕ ЗДЕСЬ — ЛИТЕРАЛ (Т2). Размер 11496331072 и sha `43d720f2…` ниже
вписаны РУКАМИ, а не импортированы из лок-файла, который читает проверяемый
код. Импортированный эталон поехал бы вместе с локом и промолчал ровно тогда,
когда лок и надо ловить: смена ступени квантования меняет и размер, и хэш.
Отдельный тест сверяет литерал с настоящим локом — и краснеет он ОСМЫСЛЕННО:
«лок изменился, посмотри, нарочно ли».

В СЕТЬ НЕ ХОДИМ И КАРТЫ НЕ ТРЕБУЕМ (Т4). `nvidia-smi` подменяется через
параметр `smi` — явную точку внедрения; ни одна ветка карты не зависит от того,
есть ли утилита на машине, где идут тесты. HTTP здесь нет вовсе: живость Comfy
— чужая ступень, и этот файл её не трогает.

ФИКСТУРЫ С ОБОИХ КРАЁВ И ИЗ СЕРЕДИНЫ (Т3): файл ровного размера, на байт
меньше, на байт больше, файла нет вовсе.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_stand as st
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

# --- ЛИТЕРАЛЫ (Т2). Ни одно число ниже не импортировано из проверяемого кода.
DIFF_NAME = "Wan2.2-Animate-14B-Q4_K_M.gguf"
DIFF_BYTES = 11496331072
DIFF_SHA = "43d720f243c3cdf5346ca05b525ec662f89bc5ebeb8e998dc347b840406cdfa6"
VAE_NAME = "wan_2.1_vae.safetensors"
VAE_BYTES = 253815318

#: Вывод `nvidia-smi` для утверждённой машины: четыре A16 по 16376 MiB.
#: Строка СОБРАНА ЗДЕСЬ, а не снята с карты — карты нет (Ц4).
SMI_A16 = "\n".join(["NVIDIA A16, 16376 MiB, 8.6, 550.90.07"] * 4) + "\n"
SMI_ONE_A16 = "NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n"
SMI_SMALL = "\n".join(["NVIDIA T4, 15360 MiB, 7.5, 550.90.07"] * 4) + "\n"


def smi_says(text):
    """Точка внедрения вместо утилиты (Т4)."""
    return lambda: {"text": text, "why": ""}


def smi_absent():
    """НЕГАТИВНЫЙ КОНТРОЛЬ (И5): утилиты нет, спросить нечем."""
    return lambda: {"text": None, "why": "nvidia-smi не найден: спросить нечем"}


def write_lock(dirpath: Path, entries=None) -> Path:
    """Реестр из литералов, в том виде, в каком его пишет поток E."""
    if entries is None:
        entries = [
            {"role": "diffusion", "path": DIFF_NAME,
             "bytes": DIFF_BYTES, "sha256": DIFF_SHA},
            {"role": "vae", "path": f"split_files/vae/{VAE_NAME}",
             "bytes": VAE_BYTES, "sha256": "0" * 64},
        ]
    wf = dirpath / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    p = wf / "fork_stack.lock.json"
    p.write_text(json.dumps({"weights": entries}), encoding="utf-8")
    return p


def sparse(path: Path, size: int) -> Path:
    """Файл нужного РАЗМЕРА без записи гигабайтов: 11 ГиБ никто не пишет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        if size:
            fh.seek(size - 1)
            fh.write(b"\0")
    return path


class TheSizeLadderIsWhatCatchesATornDownload(unittest.TestCase):
    """Т3: ровно, на байт меньше, на байт больше, нет вовсе."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.models = self.root / "models"
        self.entries = st.read_lock(write_lock(self.root))["entries"]
        self.addCleanup(self.tmp.cleanup)

    def _put(self, size):
        sparse(self.models / "unet" / DIFF_NAME, size)
        sparse(self.models / "vae" / VAE_NAME, VAE_BYTES)

    def test_exact_size_reads_as_good(self):
        self._put(DIFF_BYTES)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["ok"], 2)

    def test_one_byte_short_reads_as_not_good(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ (И5): оборванная докачка обязана краснеть."""
        self._put(DIFF_BYTES - 1)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["size"], 1)

    def test_one_byte_long_reads_as_not_good(self):
        self._put(DIFF_BYTES + 1)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["size"], 1)

    def test_the_mismatch_is_printed_in_bytes_and_not_rounded_gibibytes(self):
        """Найдено ГЛАЗАМИ на прогоне (П3): «0.0 вместо 0.0 ГиБ».

        Недокачанный хвост бывает короче килобайта, и в ГиБ с тремя знаками он
        исчезает — то есть отчёт скрывал ровно тот отказ, ради которого ступень
        написана.
        """
        self._put(DIFF_BYTES - 1)
        note = st.sizes(self.entries, self.models)["note"]
        self.assertIn("байт вместо", note)
        self.assertIn(str(DIFF_BYTES - 1), note)
        self.assertIn("-1", note)

    def test_a_missing_file_is_not_the_same_state_as_a_wrong_size(self):
        """«Нет файла» и «файл битый» чинятся разными командами."""
        sparse(self.models / "vae" / VAE_NAME, VAE_BYTES)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["missing"], 1)
        self.assertEqual(got["size"], 0)

    def test_zero_files_checked_is_not_a_success(self):
        """Р2: пустая сверка — «не смогли», а не «нарушений ноль»."""
        got = st.sizes([], self.models)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_weight_without_a_reference_size_is_unmeasured(self):
        p = write_lock(self.root, [{"role": "vae", "path": VAE_NAME,
                                    "sha256": "0" * 64}])
        entries = st.read_lock(p)["entries"]
        sparse(self.models / "vae" / VAE_NAME, VAE_BYTES)
        got = st.sizes(entries, self.models)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["no_size"], 1)

    def test_a_weight_in_an_unexpected_subdirectory_is_still_found(self):
        """Е2: верим найденному, а не ожидаемой раскладке."""
        sparse(self.models / "куда-то-ещё" / DIFF_NAME, DIFF_BYTES)
        sparse(self.models / "vae" / VAE_NAME, VAE_BYTES)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], PASS)

    def test_the_role_directory_wins_over_a_stray_copy_of_the_same_name(self):
        """Сторож раскладки: `ROLE_DIRS = {}` пережил первый аудит.

        Рядом с настоящим весом лежит забытая копия с тем же именем и битым
        размером. Прибор обязан взять тот файл, который загрузчик и возьмёт, —
        из каталога роли, — а не первый попавшийся. Каталог-приманка назван
        так, чтобы стоять в обходе РАНЬШЕ настоящего.
        """
        sparse(self.models / "aaa_старое" / DIFF_NAME, DIFF_BYTES - 1)
        sparse(self.models / "unet" / DIFF_NAME, DIFF_BYTES)
        sparse(self.models / "vae" / VAE_NAME, VAE_BYTES)
        got = st.sizes(self.entries, self.models)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertIn("unet", got["files"][0]["found"])

    def test_a_missing_models_directory_is_unmeasured_not_missing_weights(self):
        got = st.sizes(self.entries, self.root / "нет-такого")
        self.assertEqual(got["outcome"], UNMEASURED)


class TheHashLadderIsSeparateAndCatchesWhatSizeCannot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _one(self, body: bytes, sha: str):
        (self.root / "models" / "vae").mkdir(parents=True)
        p = self.root / "models" / "vae" / VAE_NAME
        p.write_bytes(body)
        entries = [{"role": "vae", "path": VAE_NAME, "name": VAE_NAME,
                    "bytes": len(body), "sha256": sha}]
        return st.sizes(entries, self.root / "models")["files"]

    def test_a_wrong_body_of_the_right_size_is_caught_only_by_the_hash(self):
        """И5: вход, на котором прибор ОБЯЗАН сказать «негодно».

        Размер сходится байт в байт, содержимое чужое — ровно тот отказ,
        который ступень размеров пропускает по устройству.
        """
        files = self._one(b"x" * 9, "0" * 64)
        by_size = files[0]["state"]
        got = st.hashes(files)
        self.assertEqual(by_size, "ok", "размер обязан сойтись")
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["mismatch"], 1)

    def test_a_matching_hash_reads_as_good(self):
        import hashlib
        body = b"ball-reel"
        files = self._one(body, hashlib.sha256(body).hexdigest())
        got = st.hashes(files)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["ok"], 1)

    def test_hashing_a_missing_file_is_unmeasured_not_good(self):
        got = st.hashes([{"name": DIFF_NAME, "found": None, "sha256": DIFF_SHA}])
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_zero_hashes_checked_is_not_a_success(self):
        self.assertEqual(st.hashes([])["outcome"], UNMEASURED)


class TheCardStepMustSayCannotRatherThanNoCard(unittest.TestCase):
    def test_a_missing_utility_is_unmeasured(self):
        """И5, Р1: главный негативный контроль этого модуля."""
        got = st.card(smi=smi_absent())
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertNotEqual(got["outcome"], FAIL)

    def test_four_a16_are_seen_as_four(self):
        got = st.card(smi=smi_says(SMI_A16))
        self.assertEqual(got["count"], 4)
        self.assertEqual(got["smallest_gib"], 15.99)

    def test_a_fitting_configuration_reads_as_good(self):
        """Q3_K_M на 49 кадрах — единственная ступень, влезающая в A16."""
        got = st.card(smi=smi_says(SMI_A16), step="Q3_K_M", length=49)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(got["headroom_gib"], 1.0)

    def test_the_project_default_does_not_fit_the_approved_card(self):
        """НАХОДКА ПРИБОРА, а не особенность теста, и число здесь главное.

        Умолчание проекта — Q4_K_M на 77 кадрах. По модели памяти
        `fork_preflight` это 15.97 ГиБ из 15.99, то есть ЗАПАС 0.02 ГиБ при
        требуемом 1.0. Прибор обязан сказать это ЧИСЛОМ и на арендованной
        машине, а не после OOM на первом кадре. Литералы ниже посчитаны
        отдельно: 10.71 + 2.03 + 2.03 + 1.20 = 15.97 (Т2).
        """
        got = st.card(smi=smi_says(SMI_A16))
        self.assertEqual(got["need_gib"], 15.97)
        self.assertEqual(got["headroom_gib"], 0.02)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("впритык", got["note"])

    def test_the_headroom_is_a_number_and_not_a_flag(self):
        got = st.card(smi=smi_says(SMI_A16), step="Q3_K_M", length=49)
        self.assertIsInstance(got["headroom_gib"], float)

    def test_fewer_cards_than_ordered_is_a_finding(self):
        got = st.card(smi=smi_says(SMI_ONE_A16), step="Q3_K_M", length=49)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("утверждено", got["note"])

    def test_a_smaller_card_is_a_finding(self):
        got = st.card(smi=smi_says(SMI_SMALL), step="Q3_K_M", length=49)
        self.assertEqual(got["outcome"], FAIL)

    def test_garbage_from_the_utility_is_unmeasured(self):
        got = st.card(smi=smi_says("\n\n"))
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_an_unknown_step_does_not_silently_pass(self):
        got = st.card(smi=smi_says(SMI_A16), step="Q9_K_XXL")
        self.assertEqual(got["outcome"], FAIL)


class TheDecisionConstantsAreGuardedInBothDirections(unittest.TestCase):
    """Т1: каждая константа-решение мутируется строже И слабее."""

    def test_the_expected_card_count_is_guarded(self):
        original = st.EXPECTED_GPU_COUNT
        try:
            st.EXPECTED_GPU_COUNT = 1
            self.assertEqual(
                st.card(smi=smi_says(SMI_ONE_A16), step="Q3_K_M",
                        length=49)["outcome"], PASS)
            st.EXPECTED_GPU_COUNT = 8
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16), step="Q3_K_M",
                        length=49)["outcome"], FAIL)
        finally:
            st.EXPECTED_GPU_COUNT = original

    def test_the_memory_bar_is_guarded(self):
        original = st.MIN_GPU_MEMORY_GIB
        try:
            st.MIN_GPU_MEMORY_GIB = 14.0
            self.assertEqual(
                st.card(smi=smi_says(SMI_SMALL), step="Q3_K_M",
                        length=49)["outcome"], PASS)
            st.MIN_GPU_MEMORY_GIB = 16.5
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16), step="Q3_K_M",
                        length=49)["outcome"], FAIL)
        finally:
            st.MIN_GPU_MEMORY_GIB = original

    def test_the_default_step_is_guarded(self):
        original = st.DEFAULT_STEP
        try:
            st.DEFAULT_STEP = "Q3_K_M"
            lighter = st.card(smi=smi_says(SMI_A16))["headroom_gib"]
            st.DEFAULT_STEP = "Q4_K_M"
            heavier = st.card(smi=smi_says(SMI_A16))["headroom_gib"]
            self.assertGreater(lighter, heavier,
                               "ступень полегче обязана оставлять больше запаса")
        finally:
            st.DEFAULT_STEP = original

    def test_the_default_length_is_guarded(self):
        original = st.DEFAULT_LENGTH
        try:
            st.DEFAULT_LENGTH = 49
            short = st.card(smi=smi_says(SMI_A16))["headroom_gib"]
            st.DEFAULT_LENGTH = 77
            long = st.card(smi=smi_says(SMI_A16))["headroom_gib"]
            self.assertGreater(short, long,
                               "короткое окно обязано оставлять больше запаса")
            st.DEFAULT_LENGTH = 63
            self.assertEqual(st.card(smi=smi_says(SMI_A16))["outcome"], FAIL,
                             "длины без посчитанных активаций не должно быть "
                             "молчаливого запаса")
        finally:
            st.DEFAULT_LENGTH = original

    def test_the_disk_margin_is_guarded(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        done = [{"state": "ok", "bytes": VAE_BYTES}]
        free = st.disk(done, tmp.name)["free_gib"]
        original = st.DISK_MARGIN_GIB
        try:
            st.DISK_MARGIN_GIB = 0.0
            self.assertEqual(st.disk(done, tmp.name)["outcome"], PASS)
            st.DISK_MARGIN_GIB = free + 1000.0
            self.assertEqual(st.disk(done, tmp.name)["outcome"], FAIL)
        finally:
            st.DISK_MARGIN_GIB = original

    def test_the_required_pack_list_is_guarded(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        comfy = Path(tmp.name) / "ComfyUI"
        for name in ("ComfyUI-WanVideoWrapper", "ComfyUI-GGUF"):
            d = comfy / "custom_nodes" / name
            d.mkdir(parents=True)
            (d / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}",
                                           encoding="utf-8")
        original = dict(st.REQUIRED_PACKS)
        try:
            self.assertEqual(st.nodes(comfy)["outcome"], PASS)
            st.REQUIRED_PACKS = {**original, "ComfyUI-НетТакого": "мутация"}
            self.assertEqual(st.nodes(comfy)["outcome"], FAIL)
            st.REQUIRED_PACKS = {}
            self.assertEqual(st.nodes(comfy)["outcome"], PASS)
        finally:
            st.REQUIRED_PACKS = original

    def test_the_pack_marker_is_guarded(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        comfy = Path(tmp.name) / "ComfyUI"
        for name in st.REQUIRED_PACKS:
            d = comfy / "custom_nodes" / name
            d.mkdir(parents=True)
            (d / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}",
                                           encoding="utf-8")
        original = st.PACK_MARK
        try:
            st.PACK_MARK = "НЕТ_ТАКОГО_СИМВОЛА"
            self.assertEqual(st.nodes(comfy)["outcome"], FAIL)
            st.PACK_MARK = "NODE_CLASS_MAPPINGS"
            self.assertEqual(st.nodes(comfy)["outcome"], PASS)
        finally:
            st.PACK_MARK = original


class TheNodePacksAreCheckedOnFilesWithoutNetwork(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.comfy = Path(self.tmp.name) / "ComfyUI"
        self.addCleanup(self.tmp.cleanup)

    def _pack(self, name, *, init=True, mark=True):
        d = self.comfy / "custom_nodes" / name
        d.mkdir(parents=True, exist_ok=True)
        if init:
            (d / "__init__.py").write_text(
                "NODE_CLASS_MAPPINGS = {}" if mark else "x = 1",
                encoding="utf-8")
        return d

    def test_all_packs_present_and_importable_reads_as_good(self):
        for name in st.REQUIRED_PACKS:
            self._pack(name)
        self.assertEqual(st.nodes(self.comfy)["outcome"], PASS)

    def test_a_missing_pack_is_not_good(self):
        self._pack("ComfyUI-WanVideoWrapper")
        got = st.nodes(self.comfy)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("comfyui-gguf", got["note"])

    def test_a_directory_without_init_is_not_importable(self):
        for name in st.REQUIRED_PACKS:
            self._pack(name)
        (self.comfy / "custom_nodes" / "ComfyUI-WanVideoWrapper"
         / "__init__.py").unlink()
        got = st.nodes(self.comfy)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("no_init", got["note"])

    def test_a_pack_registering_no_nodes_is_not_good(self):
        """Пак, который импортируется и не даёт ни одной ноды."""
        for name in st.REQUIRED_PACKS:
            self._pack(name, mark=(name != "comfyui-gguf"))
        got = st.nodes(self.comfy)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("no_mappings", got["note"])

    def test_the_directory_name_is_matched_case_insensitively(self):
        self._pack("comfyui-wanvideowrapper")
        self._pack("ComfyUI-GGUF")
        self.assertEqual(st.nodes(self.comfy)["outcome"], PASS)

    def test_no_comfy_at_all_is_unmeasured_not_missing_packs(self):
        """Р1: «не нашли ComfyUI» ≠ «паков нет»."""
        got = st.nodes(None, root=self.tmp.name)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_comfy_is_found_by_the_candidate_list_when_not_named(self):
        """Сторож списка мест поиска: пустой список пережил первый аудит."""
        for name in st.REQUIRED_PACKS:
            self._pack(name)
        got = st.nodes(None, root=self.tmp.name)
        self.assertEqual(got["outcome"], PASS, got["note"])

    def test_a_forbidden_pack_is_reported_but_is_not_a_failure(self):
        for name in st.REQUIRED_PACKS:
            self._pack(name)
        self._pack("comfyui-kjnodes")
        got = st.nodes(self.comfy)
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("comfyui-kjnodes", got["forbidden_present"])


class TheDiskStepCountsWhatIsStillToDownload(unittest.TestCase):
    def test_files_already_present_do_not_count_towards_the_need(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        done = st.disk([{"state": "ok", "bytes": DIFF_BYTES}], tmp.name)
        todo = st.disk([{"state": "missing", "bytes": DIFF_BYTES}], tmp.name)
        self.assertEqual(done["todo_gib"], 0.0)
        self.assertEqual(todo["todo_gib"], round(DIFF_BYTES / 1024 ** 3, 3))

    def test_the_remainder_is_printed_in_gibibytes(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        got = st.disk([{"state": "missing", "bytes": 11496331072}], tmp.name)
        # 11496331072 / 2**30 = 10.707… — литерал, посчитанный отдельно (Т2).
        self.assertEqual(got["todo_gib"], 10.707)

    def test_an_unopenable_path_is_unmeasured(self):
        got = st.disk([{"state": "missing", "bytes": VAE_BYTES}],
                      "/нет/такого/пути/совсем")
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_the_shipped_margin_rejects_a_disk_that_only_just_fits(self):
        """Сторож на ОТГРУЖАЕМОМ значении запаса, а не на подменённом.

        Мутация `DISK_MARGIN_GIB = 0.0` пережила первый аудит: единственный
        тест этой константы сам её и подменял, поэтому исходное значение не
        сторожил никто. Здесь остатка ровно на 2 ГиБ меньше свободного — при
        запасе 5.0 это отказ, при нуле было бы «годно».
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        import shutil as _sh
        free = _sh.disk_usage(tmp.name).free
        todo = free - 2 * 1024 ** 3
        if todo <= 0:
            self.skipTest("на этом диске меньше 2 ГиБ свободно")
        got = st.disk([{"state": "missing", "bytes": todo}], tmp.name)
        self.assertEqual(got["outcome"], FAIL, got["note"])

    def test_an_empty_weight_list_is_unmeasured_not_enough_space(self):
        """Р2: ноль байт к докачке при нуле известных весов — не успех."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        got = st.disk([], tmp.name)
        self.assertEqual(got["outcome"], UNMEASURED)


class TheLockIsReadForSizesAndNotOnlyHashes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_bytes_survive_the_parse(self):
        entries = st.read_lock(write_lock(self.root))["entries"]
        self.assertEqual(entries[0]["bytes"], DIFF_BYTES)
        self.assertEqual(entries[0]["sha256"], DIFF_SHA)

    def test_no_lock_is_unmeasured_not_good(self):
        got = st.find_lock(self.root)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_two_locks_are_unmeasured(self):
        write_lock(self.root)
        (self.root / "workflows" / "fork_other.lock").write_text("[]",
                                                                 encoding="utf-8")
        self.assertEqual(st.find_lock(self.root)["outcome"], UNMEASURED)

    def test_broken_json_is_a_finding_not_absent_weights(self):
        write_lock(self.root)
        (self.root / "workflows" / "fork_stack.lock.json").write_text(
            "{не json", encoding="utf-8")
        got = st.find_lock(self.root)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НАХОДКА", got["note"])

    def test_the_real_lock_still_carries_the_numbers_this_file_was_written_against(self):
        """Т2 наоборот: литерал сторожит лок, а не лок — литерал.

        Краснеет осмысленно: «ступень весов сменили — проверь, нарочно ли, и
        перепиши литералы здесь». Импорт числа из лока такого сказать не может.
        """
        repo = Path(__file__).resolve().parent.parent.parent
        got = st.find_lock(repo)
        self.assertEqual(got["outcome"], PASS, got["note"])
        by_name = {e["name"]: e for e in got["entries"]}
        self.assertIn(DIFF_NAME, by_name,
                      "ступень диффузии в локе больше не Q4_K_M")
        self.assertEqual(by_name[DIFF_NAME]["bytes"], DIFF_BYTES)
        self.assertEqual(by_name[DIFF_NAME]["sha256"], DIFF_SHA)
        self.assertEqual(by_name[VAE_NAME]["bytes"], VAE_BYTES)


class TheVerdictIsNumbersAndThreeOutcomes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_lock(self.root)
        self.addCleanup(self.tmp.cleanup)

    def _good_machine(self):
        """Машина, на которой ВСЁ на месте: веса, паки, четыре карты."""
        models = self.root / "models"
        sparse(models / "unet" / DIFF_NAME, DIFF_BYTES)
        sparse(models / "vae" / VAE_NAME, VAE_BYTES)
        for name in st.REQUIRED_PACKS:
            d = self.root / "ComfyUI" / "custom_nodes" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}",
                                           encoding="utf-8")
        return dict(root=str(self.root), models_dir=str(models),
                    comfy_root=str(self.root / "ComfyUI"),
                    smi=smi_says(SMI_A16), step="Q3_K_M", length=49)

    def test_a_good_machine_is_good_and_the_exit_code_is_zero(self):
        rep = st.report(**self._good_machine())
        self.assertEqual(rep["outcome"], PASS, rep["note"])
        self.assertEqual(st.EXIT_CODES[rep["outcome"]], 0)

    def test_a_torn_download_is_not_good_and_the_exit_code_is_one(self):
        args = self._good_machine()
        sparse(self.root / "models" / "unet" / DIFF_NAME, DIFF_BYTES - 1)
        rep = st.report(**args)
        self.assertEqual(rep["outcome"], FAIL)
        self.assertEqual(st.EXIT_CODES[rep["outcome"]], 1)
        self.assertIn("размеры", rep["failed_names"])

    def test_a_machine_without_the_utility_is_unmeasured_and_the_code_is_two(self):
        """И5 сквозь весь прибор: карты не видно — это НЕ «негодно»."""
        args = self._good_machine()
        args["smi"] = smi_absent()
        rep = st.report(**args)
        self.assertEqual(rep["outcome"], UNMEASURED)
        self.assertEqual(st.EXIT_CODES[rep["outcome"]], 2)
        self.assertIn("карта", rep["unmeasured_names"])
        self.assertEqual(rep["failed"], 0)

    def test_the_counters_are_printed_next_to_the_verdict(self):
        """Р2: ноль нарушений при нуле проверок — не успех."""
        rep = st.report(**self._good_machine())
        for key in ("passed", "failed", "unmeasured", "skipped"):
            self.assertIn(key, rep)
        self.assertEqual(rep["passed"] + rep["failed"] + rep["unmeasured"]
                         + rep["skipped"], len(rep["steps"]))
        text = st.render(rep)
        for word in ("пройдено", "провалено", "не смогли", "пропущено"):
            self.assertIn(word, text)

    def test_the_skipped_hash_step_does_not_pretend_to_be_checked(self):
        rep = st.report(**self._good_machine())
        self.assertIn("sha256", rep["skipped_names"])
        self.assertNotIn("sha256", rep["passed_names"])
        self.assertIn("НЕ ЗАПУСКАЛАСЬ",
                      next(s for s in rep["steps"] if s["name"] == "sha256")["note"])

    def test_the_hash_step_runs_when_asked(self):
        args = self._good_machine()
        rep = st.report(sha=True, **args)
        sha = next(s for s in rep["steps"] if s["name"] == "sha256")
        self.assertFalse(sha["skipped"])
        # Разреженные файлы — не настоящие веса, хэш обязан НЕ сойтись.
        self.assertEqual(sha["outcome"], FAIL)
        self.assertEqual(rep["outcome"], FAIL)

    def test_every_step_prints_its_duration(self):
        """П2: длительность каждого шага печатается."""
        rep = st.report(**self._good_machine())
        for s in rep["steps"]:
            self.assertIsInstance(s["seconds"], float)
        self.assertIn(" с  ", st.render(rep))

    def test_the_cheap_steps_come_before_the_expensive_ones(self):
        """Порядок — часть прибора, а не оформление."""
        rep = st.report(**self._good_machine())
        order = [s["name"] for s in rep["steps"]]
        self.assertEqual(order,
                         ["реестр", "размеры", "место", "узлы", "карта", "sha256"])

    def test_an_empty_machine_says_cannot_rather_than_good(self):
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        rep = st.report(root=empty.name, smi=smi_absent())
        self.assertEqual(rep["outcome"], UNMEASURED)
        self.assertEqual(rep["passed"], 0, rep["note"])
        self.assertGreater(rep["unmeasured"], 0)


class TheEntryPointIsUsableAsAFirstCommand(unittest.TestCase):
    def test_the_module_returns_a_code_and_prints_a_report(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = st.main(["--root", tempfile.gettempdir(),
                            "--models", "/нет/такого", "--comfy", "/нет/такого"])
        self.assertIn(code, (0, 1, 2))
        self.assertIn("ПРИЁМКА", buf.getvalue())
        self.assertIn("код возврата", buf.getvalue())

    def test_the_three_outcomes_map_to_three_codes(self):
        """Р1 в коде возврата: двойка не сворачивается в ноль."""
        self.assertEqual([st.EXIT_CODES[PASS], st.EXIT_CODES[FAIL],
                          st.EXIT_CODES[UNMEASURED]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
