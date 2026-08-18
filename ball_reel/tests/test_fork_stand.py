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
меньше, на байт больше, файла нет вовсе. Для оперативной памяти — четыре:
вдоволь, впритык (влезает, но меньше запаса), не хватает, `/proc/meminfo` нет.

ОПЕРАТИВНАЯ ПАМЯТЬ ТОЖЕ НЕ БЕРЁТСЯ С МАШИНЫ (Т4). `/proc/meminfo` подменяется
параметром `meminfo` — такой же явной точкой внедрения, как `smi`. Без неё
тест «памяти хватает» зеленел бы на большой машине и краснел на маленькой,
то есть мерил бы раннер, а не код. Числа ОЗУ ниже — литералы, посчитанные
отдельно: 10.71 ГиБ Q4_K_M × 0.935092 / 40 блоков × 38 свопнутых = 9.51 ГиБ,
плюс запас 4.0 = 13.51 ГиБ (Т2).
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

#: Вывод `nvidia-smi` для утверждённой машины. СТРОКА СОЧИНЕНА ЗДЕСЬ ЦЕЛИКОМ,
#: и «16376 MiB» в ней — НЕ НАБЛЮДЕНИЕ (Ц4): `nvidia-smi` на A16 не запускался
#: ни разу, а до 18.08.2026 это число ходило по проекту как замер. Здесь оно
#: годится ровно на одно — быть ОПТИМИСТИЧНЫМ КРАЕМ вилки допущений; ниже есть
#: такая же сочинённая строка с пессимистичным краем.
SMI_A16 = "\n".join(["NVIDIA A16, 16376 MiB, 8.6, 550.90.07"] * 4) + "\n"
SMI_ONE_A16 = "NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n"

#: ПЕССИМИСТИЧНЫЙ КРАЙ ТОЙ ЖЕ ВИЛКИ: настоящая A16, которая печатает 15360 MiB
#: = 15.0 ГиБ. Тоже сочинено. Фикстура нужна именно потому, что мы НЕ ЗНАЕМ,
#: какой из двух краёв правда: прибор обязан не забраковать эту машину.
SMI_A16_LOW = "\n".join(["NVIDIA A16, 15360 MiB, 8.6, 550.90.07"] * 4) + "\n"

#: ПОДМЕНА КАРТЫ ПРИ АРЕНДЕ: T4 вместо A16, тот же класс «16 ГБ».
SMI_SMALL = "\n".join(["NVIDIA T4, 15360 MiB, 7.5, 550.90.07"] * 4) + "\n"

#: Карта с именем утверждённой, но с памятью, которой не бывает у 16 ГБ.
#: Четыре штуки, чтобы ступень числа карт не подмешивала свою находку.
SMI_A16_TINY = "\n".join(["NVIDIA A16, 8192 MiB, 8.6, 550.90.07"] * 4) + "\n"


def smi_says(text):
    """Точка внедрения вместо утилиты (Т4)."""
    return lambda: {"text": text, "why": ""}


def smi_absent():
    """НЕГАТИВНЫЙ КОНТРОЛЬ (И5): утилиты нет, спросить нечем."""
    return lambda: {"text": None, "why": "nvidia-smi не найден: спросить нечем"}


#: Литералы ОЗУ (Т2). Ни одно число не импортировано из проверяемого кода:
#: 10.71 × 0.935092 = 10.0148; / 40 = 0.25037 на блок; × 38 = 9.5141 -> 9.51.
RAM_NEED_GIB = 9.51
RAM_NEED_WITH_MARGIN_GIB = 13.51
KB_PER_GIB = 1048576  # 2**30 / 1024: единица «kB» в /proc/meminfo — кибибайт


def meminfo_text(total_kb: int, avail_kb: int, free_kb: int | None = None,
                 *, with_available: bool = True) -> str:
    """`/proc/meminfo` в том виде, в каком его печатает ядро."""
    free = total_kb // 8 if free_kb is None else free_kb
    lines = [f"MemTotal:       {total_kb} kB",
             f"MemFree:        {free} kB"]
    if with_available:
        lines.append(f"MemAvailable:   {avail_kb} kB")
    lines += ["Buffers:           54796 kB", "Cached:          2318640 kB"]
    return "\n".join(lines) + "\n"


def meminfo_says(total_gib: float, avail_gib: float, free_gib=None,
                 **kw):
    """Точка внедрения вместо чтения /proc/meminfo (Т4)."""
    text = meminfo_text(round(total_gib * KB_PER_GIB),
                        round(avail_gib * KB_PER_GIB),
                        None if free_gib is None else round(free_gib * KB_PER_GIB),
                        **kw)
    return lambda: {"text": text, "why": ""}


def meminfo_raw(text):
    return lambda: {"text": text, "why": ""}


def meminfo_absent():
    """НЕГАТИВНЫЙ КОНТРОЛЬ (И5): /proc/meminfo нет — спросить нечем."""
    return lambda: {"text": None, "why": "/proc/meminfo не прочитан"}


#: Машина, на которой памяти ВДОВОЛЬ. Отдельным именем, потому что её
#: подставляют все сквозные проверки отчёта: без неё они мерили бы раннер.
MEMINFO_PLENTY = lambda: meminfo_says(128.0, 100.0)()


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

    def test_the_project_default_is_judged_at_the_blockswap_the_graph_sets(self):
        """ИСПРАВЛЕННЫЙ ДЕФЕКТ: ступень мерила конфигурацию, которую не запускают.

        Умолчание проекта — Q4_K_M на 77 кадрах, и граф ставит своп 38 из 40.
        Литералы (Т2): 10.71 × 0.935092 / 40 = 0.2504 на блок; резидентно
        40-38+1 = 3 блока = 0.751; несвопаемое 10.71 × 0.064908 = 0.695; итого
        весов 1.45. Плюс LoRA 2.03, активации 2.03, резерв Comfy 1.20 = 6.71.
        """
        got = st.card(smi=smi_says(SMI_A16))
        self.assertEqual(got["need_gib"], 6.71)
        self.assertEqual(got["headroom_gib"], 9.28)
        self.assertEqual(got["outcome"], PASS, got["note"])

    def test_without_blockswap_the_same_card_is_tight(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ ТОЙ ЖЕ ПРАВКИ (И5): своп обязан менять ответ.

        При `blocks=0` вся модель лежит на карте, и Q4_K_M на 77 кадрах даёт
        15.97 из 15.99 — запас 0.02 при требуемом 1.0. Прибор, отвечающий
        одинаково при 0 и при 38, не мерил бы блоксвоп вовсе.
        """
        got = st.card(smi=smi_says(SMI_A16), blocks=0)
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

    def test_a_substituted_card_is_caught_by_its_name_not_by_a_guessed_byte(self):
        """Подмена A16 на T4 ловится ИМЕНЕМ ОТ ДРАЙВЕРА (Е2).

        Раньше её ловила планка 15.5 ГиБ, обоснованная разностью двух чисел,
        которых никто не наблюдал. Имя — свидетельство: оно есть в том же
        выводе и не требует знать раскладку памяти ни одной из карт.
        """
        got = st.card(smi=smi_says(SMI_SMALL), step="Q3_K_M", length=49)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("не ту карту", got["note"])

    def test_an_approved_card_with_the_pessimistic_layout_is_not_rejected(self):
        """ТРЕТИЙ ИСХОД (Р1) НА ГЛАВНОМ НЕЗНАНИИ ЭТОГО МОДУЛЯ.

        Если утверждённая A16 печатает 15360 MiB (нижний край вилки, и он
        ровно так же не наблюдался, как верхний), приёмка НЕ ВПРАВЕ сказать
        «не та карта»: имя то самое, а по объёму мы отличить не можем.
        Ответ — «не смогли», и это не «годно» и не «негодно».
        """
        got = st.card(smi=smi_says(SMI_A16_LOW))
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertNotEqual(got["outcome"], FAIL)
        self.assertEqual(got["smallest_gib"], 15.0)
        self.assertIn("НЕ СМОГЛИ ОТЛИЧИТЬ", got["note"])

    def test_a_card_below_any_sixteen_gigabyte_layout_is_a_finding(self):
        """И5, другая сторона: 8 ГиБ — это уже не незнание, а находка."""
        got = st.card(smi=smi_says(SMI_A16_TINY))
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["count"], 4, "число карт не должно мешать")
        self.assertIn("ни одна карта класса 16 ГБ", got["note"])

    def test_garbage_from_the_utility_is_unmeasured(self):
        got = st.card(smi=smi_says("\n\n"))
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_an_unknown_step_does_not_silently_pass(self):
        got = st.card(smi=smi_says(SMI_A16), step="Q9_K_XXL")
        self.assertEqual(got["outcome"], FAIL)


class TheRamStepIsTheRequirementBlockswapCreated(unittest.TestCase):
    """Блоксвоп не убрал 9.51 ГиБ — он перенёс их с карты в ОЗУ машины.

    Т3, четыре фикстуры: вдоволь / впритык / не хватает / спросить нечем.
    И5 в обе стороны: есть вход, где ступень ОБЯЗАНА сказать «годно», и вход,
    где ОБЯЗАНА сказать «негодно».
    """

    def test_a_machine_with_plenty_of_ram_reads_as_good(self):
        """И5, сторона «обязана шевельнуться в плюс»."""
        got = st.ram(meminfo=meminfo_says(128.0, 100.0))
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["need_gib"], RAM_NEED_GIB)
        self.assertEqual(got["total_gib"], 128.0)
        self.assertEqual(got["available_gib"], 100.0)

    def test_a_machine_without_enough_ram_reads_as_not_good(self):
        """И5, сторона «обязана сказать нет»: 4 ГиБ против требуемых 9.51."""
        got = st.ram(meminfo=meminfo_says(8.0, 4.0))
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertIn("НЕ ХВАТАЕТ", got["note"])
        self.assertEqual(got["left_gib"], round(4.0 - RAM_NEED_GIB, 3))

    def test_ram_that_fits_but_leaves_less_than_the_margin_is_not_good(self):
        """СЕРЕДИНА (Т3): 11 ГиБ больше требуемых 9.51 и меньше 13.51.

        Это и есть машина, ради которой ступень написана: арифметически
        влезает, а на деле уйдёт в своп на диск. «Влезает» здесь не годно.
        """
        got = st.ram(meminfo=meminfo_says(16.0, 11.0))
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertGreater(got["available_gib"], got["need_gib"])
        self.assertLess(got["available_gib"], RAM_NEED_WITH_MARGIN_GIB)

    def test_no_meminfo_is_unmeasured_and_never_no_memory(self):
        """Р1/И5: главный негативный контроль ступени. Не Linux — не отказ."""
        got = st.ram(meminfo=meminfo_absent())
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertNotEqual(got["outcome"], FAIL)
        # Требование посчитано даже там, где машину спросить нечем: человек
        # обязан узнать ЧИСЛО, под которое ему выбирать машину.
        self.assertEqual(got["need_gib"], RAM_NEED_GIB)

    def test_the_available_field_is_read_and_not_the_free_one(self):
        """Разница между MemFree и MemAvailable — гигабайты, и она решает.

        Сервер, который только что скачал 18 ГиБ весов, показывает почти
        нулевой MemFree при полностью свободной памяти: весь кэш забит этими
        весами. По MemFree такая машина была бы забракована.
        """
        cached = st.ram(meminfo=meminfo_says(128.0, 100.0, free_gib=0.5))
        self.assertEqual(cached["outcome"], PASS, cached["note"])
        self.assertEqual(cached["available_gib"], 100.0)
        # И обратная сторона: большой MemFree не спасает малый MemAvailable.
        lying = st.ram(meminfo=meminfo_says(128.0, 4.0, free_gib=100.0))
        self.assertEqual(lying["outcome"], FAIL, lying["note"])

    def test_a_kernel_without_memavailable_is_unmeasured_not_substituted(self):
        """MemAvailable появился в ядре 3.14. Подставить MemFree — соврать."""
        got = st.ram(meminfo=meminfo_says(128.0, 100.0, free_gib=100.0,
                                          with_available=False))
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIsNone(got["available_gib"])
        self.assertIn("MemFree", got["note"])

    def test_an_unknown_step_cannot_be_priced_and_says_so(self):
        got = st.ram(meminfo=meminfo_says(128.0, 100.0), step="Q9_K_XXL")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["need_gib"])

    def test_an_impossible_block_count_is_unmeasured(self):
        got = st.ram(meminfo=meminfo_says(128.0, 100.0), blocks=41)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])

    def test_without_blockswap_the_requirement_collapses_to_zero(self):
        """Негативный контроль формулы (И5): нет свопа — нечему быть в ОЗУ."""
        got = st.ram(meminfo=meminfo_says(128.0, 100.0), blocks=0)
        self.assertEqual(got["need_gib"], 0.0)
        self.assertIn("БЛОКСВОП ВЫКЛЮЧЕН", got["note"])

    def test_a_lighter_step_needs_less_ram(self):
        """Требование обязано ЗАВИСЕТЬ от ступени, а не быть вписанным."""
        heavy = st.ram(meminfo=meminfo_says(128.0, 100.0), step="Q4_K_M")
        light = st.ram(meminfo=meminfo_says(128.0, 100.0), step="Q3_K_M")
        self.assertGreater(heavy["need_gib"], light["need_gib"])
        # Литерал: 8.04 × 0.935092 / 40 × 38 = 7.1418 -> 7.14 (Т2).
        self.assertEqual(light["need_gib"], 7.14)

    def test_the_numbers_are_numbers_and_not_a_flag(self):
        """Р2/Е3: частичный результат печатается числами."""
        got = st.ram(meminfo=meminfo_says(16.0, 14.0))
        for key in ("total_gib", "available_gib", "need_gib", "left_gib",
                    "margin_gib"):
            self.assertIsInstance(got[key], float, key)

    def test_meminfo_kilobytes_are_kibibytes(self):
        """Подпись «kB» в /proc/meminfo врёт, число нет: делитель 1024."""
        vals = st.parse_meminfo("MemTotal:       1048576 kB\n")
        self.assertEqual(vals["MemTotal"], 1073741824)

    def test_the_shipped_meminfo_path_is_the_one_the_kernel_uses(self):
        """Сторож на ОТГРУЖАЕМОМ значении, который сам его НЕ ПОДМЕНЯЕТ.

        Мутация `MEMINFO_PATH` на несуществующий путь ПЕРЕЖИЛА первый заход:
        единственный тест этой константы сам её и мутировал, поэтому исходное
        значение не сторожил никто, и приёмка на любой машине честно говорила
        бы «не смогли» — то есть тихо перестала бы проверять ОЗУ вообще. Это
        ровно тот дефект, что уже был на `DISK_MARGIN_GIB`.

        Здесь два assert, и ни один не skip (Т6): литерал написан руками (Т2),
        а на Linux дополнительно проверяется, что по этому пути действительно
        отвечает ядро.
        """
        import sys
        self.assertEqual(st.MEMINFO_PATH, "/proc/meminfo")
        got = st.read_meminfo()
        if sys.platform.startswith("linux"):
            self.assertIsNotNone(got["text"],
                                 "на Linux по этому пути обязано читаться")
            self.assertIn("MemAvailable", got["text"])
        else:
            self.assertIsNone(got["text"])
            self.assertIn("НЕ «памяти нет»", got["why"])

    def test_the_real_reader_reports_cannot_when_the_path_is_absent(self):
        """Ветка «файла нет» — проверяется подменой, без ухода с машины."""
        original = st.MEMINFO_PATH
        try:
            st.MEMINFO_PATH = "/нет/такого/meminfo"
            got = st.read_meminfo()
            self.assertIsNone(got["text"])
            self.assertIn("НЕ «памяти нет»", got["why"])
        finally:
            st.MEMINFO_PATH = original


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
        """Т1 в обе стороны. Карта во всех трёх ветках НАЗЫВАЕТСЯ A16, иначе

        мутация планки была бы неотличима от проверки имени.
        """
        original = st.MIN_GPU_MEMORY_GIB
        try:
            st.MIN_GPU_MEMORY_GIB = 14.5   # слабее: 15.0 попадает выше планки
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16_LOW), step="Q3_K_M",
                        length=49)["outcome"], PASS)
            st.MIN_GPU_MEMORY_GIB = 16.5   # строже: даже 15.99 сомнительна
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16), step="Q3_K_M",
                        length=49)["outcome"], UNMEASURED)
        finally:
            st.MIN_GPU_MEMORY_GIB = original

    def test_the_doubt_floor_is_guarded(self):
        """Т1: нижняя граница полосы сомнения — тоже константа-решение."""
        original = st.GPU_MEMORY_DOUBT_GIB
        try:
            st.GPU_MEMORY_DOUBT_GIB = 15.5   # строже: 15.0 уже находка
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16_LOW), step="Q3_K_M",
                        length=49)["outcome"], FAIL)
            st.GPU_MEMORY_DOUBT_GIB = 4.0    # слабее: 8 ГиБ проходит в сомнение
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16_TINY), step="Q3_K_M",
                        length=49)["outcome"], UNMEASURED)
        finally:
            st.GPU_MEMORY_DOUBT_GIB = original

    def test_the_expected_card_name_is_guarded(self):
        """Т1 в обе стороны по имени утверждённой карты.

        Судим по находке, а не по итоговому исходу: у T4 объём 15.0 ГиБ и без
        того попадает в полосу сомнения, и итог там «не смогли» по другой
        причине. Мутация обязана переставить именно НАХОДКУ ПРО ИМЯ.
        """
        original = st.EXPECTED_GPU_NAME
        try:
            self.assertNotIn("не ту карту",
                             st.card(smi=smi_says(SMI_A16))["note"])
            self.assertIn("не ту карту",
                          st.card(smi=smi_says(SMI_SMALL))["note"])
            st.EXPECTED_GPU_NAME = "T4"      # обе находки обязаны поменяться
            self.assertIn("не ту карту",
                          st.card(smi=smi_says(SMI_A16))["note"])
            self.assertNotIn("не ту карту",
                             st.card(smi=smi_says(SMI_SMALL))["note"])
            self.assertEqual(
                st.card(smi=smi_says(SMI_A16), step="Q3_K_M",
                        length=49)["outcome"], FAIL)
        finally:
            st.EXPECTED_GPU_NAME = original

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

    def test_the_ram_margin_is_guarded(self):
        """Т1: запас ОЗУ мутируется слабее И строже, оба раза наблюдаемо."""
        original = st.RAM_MARGIN_GIB
        try:
            # Слабее: без запаса машина «впритык» становится годной.
            st.RAM_MARGIN_GIB = 0.0
            self.assertEqual(st.ram(meminfo=meminfo_says(16.0, 11.0))["outcome"],
                             PASS)
            # Строже: машина, годная при отгружаемом запасе, перестаёт быть.
            st.RAM_MARGIN_GIB = 100.0
            self.assertEqual(st.ram(meminfo=meminfo_says(128.0, 100.0))["outcome"],
                             FAIL)
        finally:
            st.RAM_MARGIN_GIB = original

    def test_the_shipped_ram_margin_is_bracketed_from_both_sides(self):
        """Сторож на ОТГРУЖАЕМОМ значении, который сам его НЕ ПОДМЕНЯЕТ.

        Ровно та дыра, что уже была на `DISK_MARGIN_GIB`: единственный тест
        константы сам её и мутировал, поэтому исходные 5.0 не сторожил никто, и
        мутация в ноль пережила аудит. Здесь запас зажат с двух сторон
        литералами: при 13.4 ГиБ доступных ступень ОБЯЗАНА сказать «негодно»
        (значит запас больше 3.89), при 13.6 — «годно» (значит не больше 4.09).
        Любая мутация 4.0 наружу этого коридора красит один из двух assert.
        """
        self.assertEqual(st.ram(meminfo=meminfo_says(16.0, 13.4))["outcome"],
                         FAIL, "13.4 ГиБ — меньше 9.51 + запас")
        self.assertEqual(st.ram(meminfo=meminfo_says(16.0, 13.6))["outcome"],
                         PASS, "13.6 ГиБ — больше 9.51 + запас")

    def test_the_swapped_block_count_comes_from_the_graph(self):
        """Е1: число блоков не вписано в приёмку, а взято из `fork_comfy`.

        Литерал 38 (Т2). Краснеет ОСМЫСЛЕННО: граф сменил blocks_to_swap —
        требование к ОЗУ изменилось, посмотри, нарочно ли.
        """
        got = st.ram(meminfo=meminfo_says(128.0, 100.0))
        self.assertEqual(got["blocks"], 38)
        self.assertIn("38 из 40", got["note"])

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
                    smi=smi_says(SMI_A16), meminfo=meminfo_says(128.0, 100.0),
                    step="Q3_K_M", length=49)

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
                         ["оперативка", "реестр", "размеры", "место", "узлы",
                          "карта", "sha256"])

    def test_a_machine_short_on_ram_fails_the_whole_acceptance(self):
        """Ступень доезжает до вердикта, а не остаётся числом внутри.

        Машина с четырьмя A16, всеми весами и всеми паками — и 11 ГиБ ОЗУ.
        До этой ступени она проходила приёмку целиком и падала на первом
        прогоне. Теперь это код возврата 1 и имя ступени в отчёте.
        """
        args = self._good_machine()
        args["meminfo"] = meminfo_says(16.0, 11.0)
        rep = st.report(**args)
        self.assertEqual(rep["outcome"], FAIL, rep["note"])
        self.assertEqual(st.EXIT_CODES[rep["outcome"]], 1)
        self.assertIn("оперативка", rep["failed_names"])

    def test_a_machine_without_meminfo_is_unmeasured_not_failed(self):
        """Р1 сквозь весь прибор: не-Linux — это двойка, а не единица."""
        args = self._good_machine()
        args["meminfo"] = meminfo_absent()
        rep = st.report(**args)
        self.assertEqual(rep["outcome"], UNMEASURED, rep["note"])
        self.assertEqual(st.EXIT_CODES[rep["outcome"]], 2)
        self.assertIn("оперативка", rep["unmeasured_names"])
        self.assertEqual(rep["failed"], 0)

    def test_the_report_counts_seven_steps_now(self):
        """Р2: арифметика счётчиков считает ступени, а не помнит их число."""
        rep = st.report(**self._good_machine())
        self.assertEqual(len(rep["steps"]), 7)
        self.assertEqual(rep["passed"] + rep["failed"] + rep["unmeasured"]
                         + rep["skipped"], 7)
        self.assertIn("ступеней 7", rep["note"])
        self.assertIn("оперативка", st.render(rep))

    def test_an_empty_machine_says_cannot_rather_than_good(self):
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        rep = st.report(root=empty.name, smi=smi_absent(),
                        meminfo=meminfo_absent())
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
