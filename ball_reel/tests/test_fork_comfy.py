"""Поток E: вырезание кастомных нод проверяется СПИСКОМ, а не глазами.

«Посмотрел, вроде их нет» — то же утверждение без прогона, против которого
написан весь проект. Здесь оно заменено перечислением: в произведённом графе не
должно остаться ни одного типа из трёх кастомных наборов, и проверяет это код.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_reel import fork_comfy as fk

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / fk.UPSTREAM


class TheUpstreamIsTheOneThatWasRecorded(unittest.TestCase):
    """Хэш сверяется КОДОМ до производства, а не человеком при чтении."""

    def test_the_template_is_present_and_matches_its_hash(self):
        self.assertTrue(UPSTREAM.exists())
        fk.load_upstream()

    def test_a_changed_template_is_refused_and_called_a_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            p.write_text('{"nodes": []}', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                fk.load_upstream(p)
            self.assertIn("НАХОДКА", str(caught.exception))

    def test_the_hash_check_can_be_waived_only_explicitly(self):
        """Негативный контроль (И5): сторож обязан уметь и пропускать."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            p.write_text('{"nodes": []}', encoding="utf-8")
            self.assertEqual(fk.load_upstream(p, check_hash=False),
                             {"nodes": []})

    def test_a_missing_template_says_why_it_is_in_the_repo(self):
        with self.assertRaises(FileNotFoundError) as caught:
            fk.load_upstream(ROOT / "нет-такого.json")
        self.assertIn("арендованной машине", str(caught.exception))


class TheCustomPacksAreActuallyInTheUpstream(unittest.TestCase):
    """Иначе следующий блок зеленеет на пустом множестве.

    Форма частая и незаметная: «все найденные вырезаны» проходит, когда не
    найдено ничего, и молчит ровно тогда, когда сломался поиск.
    """

    def setUp(self):
        self.src = fk.load_upstream()

    def test_the_upstream_really_carries_custom_nodes(self):
        found = fk.custom_nodes(self.src)
        self.assertGreater(len(found), 3,
                           "в темплейте не нашлось кастомных нод — скорее "
                           "всего сломался разбор, а не темплейт исправился")

    def test_all_three_packs_are_represented(self):
        packs = {n["pack"].lower() for n in fk.custom_nodes(self.src)}
        self.assertEqual(packs, set(fk.CUSTOM_PACKS),
                         f"наборы в темплейте разошлись со списком: {packs}")

    def test_the_node_that_waits_for_a_mouse_is_there(self):
        types = {n["type"] for n in fk.custom_nodes(self.src)}
        self.assertIn(fk.MANUAL_NODE, types,
                      "PointsEditor исчез из темплейта — тогда вторая, "
                      "непреодолимая причина вырезания больше не про этот файл")

    def test_the_pack_names_are_matched_case_insensitively(self):
        """Записка пишет `ComfyUI-KJNodes`, узлы — `comfyui-kjnodes`."""
        node = {"properties": {fk.PACK_KEY: "ComfyUI-KJNodes"}, "id": 1,
                "type": "BlockifyMask"}
        self.assertEqual(len(fk.custom_nodes({"nodes": [node]})), 1)


class TheDerivedGraphCarriesNoCustomNode(unittest.TestCase):
    """Главная проверка потока."""

    @classmethod
    def setUpClass(cls):
        cls.src = fk.load_upstream()
        cls.derived = fk.derive(cls.src)
        cls.report = fk.audit(cls.derived, upstream=cls.src)

    def test_not_one_custom_node_survives(self):
        self.assertEqual(
            self.report["custom_left"], [],
            f"кастомные ноды остались: {self.report['custom_left']}")

    def test_the_mouse_node_is_gone(self):
        types = {n["type"] for n in self.derived["graph"]["nodes"]}
        self.assertNotIn(fk.MANUAL_NODE, types)

    def test_the_removal_list_names_the_reason_for_each(self):
        whys = {n["why"] for n in self.derived["removed"]}
        self.assertIn("ждёт человека с мышью", whys)
        self.assertIn("кастомный набор — лицензионная цепочка", whys)

    def test_no_link_points_at_a_removed_node(self):
        self.assertEqual(self.report["dangling"], [],
                         "оборванная связь: загрузчик Comfy пойдёт искать "
                         "узел, которого нет")

    def test_the_graph_actually_shrank(self):
        self.assertLess(self.report["nodes_after"], self.report["nodes_before"])

    def test_the_verdict_is_pass_and_the_note_counts_everything(self):
        self.assertEqual(self.report["outcome"], fk.PASS, self.report["note"])
        self.assertIn("ЧИСТО", self.report["note"])

    def test_the_note_says_out_loud_that_the_graph_was_not_run(self):
        self.assertIn("НЕПРОВЕРЕНО", self.report["note"])
        self.assertIn("не исполнялся", self.report["note"])

    def test_the_derived_graph_records_where_it_came_from(self):
        fork = self.derived["graph"]["extra"]["fork"]
        self.assertEqual(fork["derived_from_sha256"], fk.UPSTREAM_SHA256)
        self.assertIn("character_mask", fork["our_sources"])

    def test_our_three_sequences_are_wired_in(self):
        titles = [n.get("title", "") for n in self.derived["graph"]["nodes"]]
        for target in ("face_video", "pose_video", "character_mask"):
            with self.subTest(target=target):
                self.assertTrue(any(target in t for t in titles),
                                f"нашей последовательности для {target} нет — "
                                f"вырезали и не подставили")


class TheAuditCanActuallyGoRed(unittest.TestCase):
    """Сторож, который не умеет краснеть, — украшение."""

    def test_a_surviving_custom_node_is_reported_as_a_problem(self):
        src = fk.load_upstream()
        derived = fk.derive(src)
        derived["graph"]["nodes"].append(
            {"id": 9999, "type": "BlockifyMask",
             "properties": {fk.PACK_KEY: "comfyui-kjnodes"}})
        got = fk.audit(derived, upstream=src)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("BlockifyMask#9999", got["note"])

    def test_a_surviving_mouse_node_is_called_out_by_name(self):
        src = fk.load_upstream()
        derived = fk.derive(src)
        derived["graph"]["nodes"].append(
            {"id": 9998, "type": fk.MANUAL_NODE,
             "properties": {fk.PACK_KEY: "comfy-core"}})
        got = fk.audit(derived, upstream=src)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("ждёт мыши", got["note"])

    def test_a_dangling_link_is_reported(self):
        src = fk.load_upstream()
        derived = fk.derive(src)
        derived["graph"]["links"].append([12345, 999999, 0, 888888, 0, "IMAGE"])
        got = fk.audit(derived, upstream=src)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("оборванных связей 1", got["note"])


class TheInputsOfTheAnimateNodeAreActuallyFed(unittest.TestCase):
    """Проверка, появившаяся после настоящего дефекта.

    Первая редакция `derive` вырезала кастомные ноды и клала рядом три
    загрузчика, НЕ ПОДКЛЮЧАЯ их. Аудит говорил «ЧИСТО»: кастомных нод нет,
    оборванных связей нет. А четыре входа `WanAnimateToVideo` остались без
    питания — вырезание сняло связи 704, 705, 706, 707, — и граф не мог
    поехать. Отчёт, зеленеющий на неработоспособном графе, хуже отсутствия
    отчёта.
    """

    def test_the_upstream_itself_feeds_all_five(self):
        """Иначе следующий тест зеленел бы на сломанном определении слотов."""
        self.assertEqual(fk.unfed_inputs(fk.load_upstream()), [])

    def test_the_derived_graph_feeds_all_five_too(self):
        derived = fk.derive(fk.load_upstream())
        self.assertEqual(fk.unfed_inputs(derived["graph"]), [],
                         "вырезали и не подключили — граф не поедет")

    def test_the_check_goes_red_when_a_feeding_link_is_cut(self):
        """Сторож обязан уметь краснеть."""
        derived = fk.derive(fk.load_upstream())
        graph = derived["graph"]
        node_id, slots = fk._subgraph_feeding_animate(graph)
        slot = slots["character_mask"]
        graph["links"] = [l for l in graph["links"]
                          if not (l[3] == node_id and l[4] == slot)]
        self.assertEqual(fk.unfed_inputs(graph), ["character_mask"])
        got = fk.audit({"graph": graph})
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("входов без питания 1", got["note"])

    def test_the_slots_come_from_the_declaration_not_the_link_order(self):
        """Связи можно пересортировать; номер, выведенный из их порядка, поедет."""
        src = fk.load_upstream()
        node_id, slots = fk._subgraph_feeding_animate(src)
        self.assertIsNotNone(node_id)
        by_name = {n["id"]: n for n in src["nodes"]}
        declared = [i.get("name") for i in by_name[node_id]["inputs"]]
        for name, slot in slots.items():
            with self.subTest(name=name):
                self.assertEqual(declared[slot], name)

    def test_the_mask_path_ends_in_a_mask_typed_link(self):
        graph = fk.derive(fk.load_upstream())["graph"]
        node_id, slots = fk._subgraph_feeding_animate(graph)
        slot = slots["character_mask"]
        feeding = [l for l in graph["links"]
                   if l[3] == node_id and l[4] == slot]
        self.assertEqual(len(feeding), 1)
        self.assertEqual(feeding[0][5], "MASK",
                         "в character_mask приходит не MASK — Comfy откажет "
                         "по типу")


class TheContractOfTheAnimateNodeIsChecked(unittest.TestCase):
    """Узел лежит в САБГРАФЕ; поиск только по верхнему уровню соврал бы."""

    def test_every_required_input_is_present_in_the_upstream(self):
        got = fk.contract()
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertEqual(got["missing"], [])

    def test_the_search_really_goes_into_subgraphs(self):
        """Иначе предыдущий тест доказывал бы обратное своим же провалом."""
        src = fk.load_upstream()
        top = {n["type"] for n in src["nodes"]}
        self.assertNotIn("WanAnimateToVideo", top,
                         "узел оказался на верхнем уровне — тогда обход "
                         "сабграфов ничем не подтверждён")

    def test_a_graph_without_the_node_is_unmeasured_not_broken(self):
        got = fk.contract({"nodes": []})
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertIn("НЕ «контракт нарушен»", got["note"])

    def test_a_missing_input_is_named(self):
        graph = {"nodes": [{"type": "WanAnimateToVideo",
                            "inputs": [{"name": "reference_image"}]}]}
        got = fk.contract(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("character_mask", got["note"])

    def test_the_required_list_is_the_five_the_readme_read_from_links(self):
        self.assertEqual(len(fk.ANIMATE_INPUTS), 5)
        self.assertIn("character_mask", fk.ANIMATE_INPUTS)


class IntroducedTypesAreProvenByTheTemplateOrFlagged(unittest.TestCase):
    """Ц10: имя, которого может не быть, не вводится молча."""

    def test_the_only_type_outside_the_template_is_imagetomask(self):
        """Ровно одно имя вводится не из темплейта, и оно названо.

        `character_mask` требует MASK, наши маски приезжают картинками, а
        преобразователя IMAGE->MASK в темплейте нет — там маску отдавал
        вырезанный `BlockifyMask`.
        """
        derived = fk.derive(fk.load_upstream())
        flagged = {i["type"] for i in derived["introduced"]}
        self.assertEqual(flagged, {"ImageToMask"},
                         f"список введённых извне типов изменился: {flagged}. "
                         f"Новое имя обязано быть либо доказано командой, "
                         f"либо помечено — молча вводить нельзя.")

    def test_imagetomask_is_now_proven_by_the_downloaded_source(self):
        """~~НЕПРОВЕРЕНО~~ закрыто 17.08.2026 загрузкой `nodes_mask.py`.

        Пометка снята НЕ решением, а замером: Comfy ставить по-прежнему
        нельзя, но исходник читается без установки. Тест сторожит, что
        доказательство названо целиком — URL, sha256 тела, строка, дата, — а
        не заменено словом «проверено».
        """
        derived = fk.derive(fk.load_upstream())
        provenance = {i["type"]: i["provenance"] for i in derived["introduced"]}
        self.assertIn("ДОКАЗАНО исходником", provenance["ImageToMask"])
        self.assertNotIn("НЕПРОВЕРЕНО", provenance["ImageToMask"])
        src = fk.PROVEN_BY_SOURCE["ImageToMask"]
        self.assertEqual(len(src["body_sha256"]), 64)
        self.assertIn("nodes_mask.py", src["url"])
        self.assertIn("red", src["quote"],
                      "виджет ставит 'red' — а доказательство обязано "
                      "показывать, что такое значение у ноды есть")

    def test_a_type_proven_by_nothing_is_still_flagged(self):
        """Сторож обязан уметь краснеть — иначе он украшение.

        Предыдущий тест зеленеет и в том случае, если `provenance_of` начнёт
        объявлять доказанным вообще всё. Здесь подаётся имя, которого нет ни
        в темплейте, ни в реестре.
        """
        got = fk.provenance_of("НетТакойНоды", {"LoadVideo"})
        self.assertIn("НЕПРОВЕРЕНО", got)
        self.assertNotIn("ДОКАЗАНО", got)

    def test_the_three_provenances_are_actually_three(self):
        """Р1: три ответа, а не два. Негативный контроль к обоим положительным."""
        self.assertIn("ДОКАЗАНО темплейтом",
                      fk.provenance_of("LoadVideo", {"LoadVideo"}))
        self.assertIn("ДОКАЗАНО исходником",
                      fk.provenance_of("ImageToMask", set()))
        self.assertIn("НЕПРОВЕРЕНО", fk.provenance_of("Выдумка", set()))

    def test_no_unproven_type_is_left_in_the_derived_graph(self):
        """Их было одно, стало ноль — и число доезжает до отчёта."""
        src = fk.load_upstream()
        got = fk.audit(fk.derive(src), upstream=src)
        self.assertEqual(got["unproven_types"], [], got["note"])
        self.assertIn("введённых недоказанных типов 0", got["note"])

    def test_a_fabricated_type_reaches_the_audit_note(self):
        """Пометка, не доехавшая до отчёта, не меняет ничьего решения."""
        src = fk.load_upstream()
        derived = fk.derive(src)
        derived["introduced"].append(
            {"type": "ВыдуманнаяНода", "id": 4242,
             "provenance": fk.provenance_of("ВыдуманнаяНода", set())})
        got = fk.audit(derived, upstream=src)
        self.assertEqual(len(got["unproven_types"]), 1)
        self.assertIn("ВыдуманнаяНода", got["note"])

    def test_everything_else_we_add_is_proven_by_the_template(self):
        src = fk.load_upstream()
        derived = fk.derive(src)
        added = {n["type"] for n in derived["graph"]["nodes"]
                 if n.get("title")}
        proven = fk.known_types(src)
        self.assertEqual(added - proven - {"ImageToMask"}, set(),
                         "введён тип, которого нет ни в темплейте, ни в "
                         "списке помеченных")

    def test_the_flag_list_is_always_present_even_when_empty(self):
        derived = fk.derive(fk.load_upstream())
        self.assertIn("introduced", derived)
        self.assertIsInstance(derived["introduced"], list)

    def test_known_types_reads_the_template_and_not_a_hardcoded_list(self):
        types = fk.known_types(fk.load_upstream())
        self.assertIn("LoadVideo", types)
        self.assertNotIn("ImageToMask", types,
                         "ImageToMask появился в темплейте — значит его можно "
                         "вводить как доказанный, и пометку снять")


class TheGraphCanBeWrittenAndReadBack(unittest.TestCase):

    def test_written_json_parses_and_keeps_the_provenance(self):
        derived = fk.derive(fk.load_upstream())
        with tempfile.TemporaryDirectory() as tmp:
            p = fk.write(derived, Path(tmp) / "sub" / "fork.json")
            back = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(back["extra"]["fork"]["derived_from"], fk.UPSTREAM)
        self.assertEqual(fk.audit({"graph": back})["outcome"], fk.PASS)


#: Геометрия для тестов адаптера. Не штатная (480x848x77): 77 кадров по
#: 512x512 на канал лица — это сотни мегабайт, и тест, который душит машину,
#: перестают запускать. Кратности при этом настоящие: 32 и 48 кратны 16,
#: (5-1) % 4 == 0. Штатная геометрия проверяется отдельным тестом на числах.
W, H, N = 32, 48, 5


def valid_inputs(**over):
    """Годный вход. Один конструктор на все тесты: подделка получается
    ИЗМЕНЕНИЕМ одного поля, поэтому «что именно уронило проверку» видно."""
    got = {
        "reference_image": np.zeros((1, H, W, 3), np.uint8),
        "face_video": np.zeros((N, fk.FACE_SIDE, fk.FACE_SIDE, 3), np.uint8),
        "pose_video": np.zeros((N, H, W, 3), np.uint8),
        "background_video": np.zeros((N, H, W, 3), np.uint8),
        "character_mask": np.zeros((N, H, W), np.float32),
        "width": W, "height": H, "length": N,
    }
    got.update(over)
    return got


class TheNodeHasNoCompositingAndTheModuleSaysSo(unittest.TestCase):
    """§1a. Утверждение «вне маски пиксели копируются» — ЛОЖНОЕ.

    Оно стояло даже в вендорской записке, скачанной вместе с темплейтом.
    Записку править нельзя — она первоисточник; значит опровержение обязано
    жить в коде, иначе следующая смена прочитает записку и унаследует ошибку.
    """

    def test_the_vendor_note_really_does_say_composite(self):
        """Негативный контроль (И5): без него опровержение опровергает пустоту.

        Если вендор однажды исправит записку, этот тест покраснеет — и это
        правильно: тогда опровержение относится уже не к тому тексту.
        """
        doc = (ROOT / "workflows/upstream/WanAnimateToVideo.doc.md").read_text(
            encoding="utf-8")
        self.assertIn("composite with generated content", doc)

    def test_the_module_names_the_refuted_source_by_file_and_line(self):
        self.assertIn("WanAnimateToVideo.doc.md:27", fk.NO_COMPOSITE["refutes"])
        self.assertIn("noise_mask", fk.NO_COMPOSITE["claim"])

    def test_the_refutation_carries_the_hash_of_the_body_that_was_read(self):
        """«Проверено по коду» без sha256 тела через месяц неотличимо от
        «вроде помню»: master движется, номера строк едут."""
        self.assertEqual(len(fk.NO_COMPOSITE["body_sha256"]), 64)
        self.assertIn("nodes_wan.py", fk.NO_COMPOSITE["source"])
        for line in ("1166", "1219", "1248", "1454"):
            with self.subTest(line=line):
                self.assertIn(line, fk.NO_COMPOSITE["evidence"])

    def test_the_refutation_travels_inside_the_derived_graph(self):
        """Граф уедет на арендованную машину без этого репозитория."""
        derived = fk.derive(fk.load_upstream())
        self.assertEqual(derived["graph"]["extra"]["fork"]["no_composite"],
                         fk.NO_COMPOSITE)

    @staticmethod
    def _unstruck_claims(text: str) -> list:
        """Строки, обещающие сохранность пикселей вне маски и НЕ перечёркнутые.

        Перечёркнутым считается упоминание внутри опровержения: строка с `~~`,
        со словом `ЛОЖНО`, `СНЯТО`, `ошибается`, `не смеет` — либо отрицание
        вплотную перед самим словом («не подложка», «нет композитинга»).

        Правило про отрицание пришлось дописать после того, как проверка
        покраснела на строке `background_video — это ОБУСЛОВЛИВАНИЕ, а не\\n
        подложка`: перенос строки развёл отрицание и слово по разным строкам.
        Это находка про сам прибор, а не про текст, и негативный контроль на
        неё стоит ниже.
        """
        words = ("копиру", "нетронут", "подложк", "композит")
        marks = ("~~", "ЛОЖНО", "СНЯТО", "не смеет", "ошибается")
        bad = []
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            hits = [low.index(w) for w in words if w in low]
            if not hits:
                continue
            if any(m in line for m in marks):
                continue
            # отрицание вплотную перед словом — «не подложка», «нет композита»
            if all(any(low[max(0, at - 5):at].strip().endswith(n)
                       for n in ("не", "нет")) for at in hits):
                continue
            bad.append(f"{i}: {line.strip()}")
        return bad

    def test_the_checker_itself_goes_red(self):
        """Прибор с негативным контролем (И5): вход, где он обязан сказать
        «нет», и вход, где обязан шевельнуться."""
        self.assertEqual(len(self._unstruck_claims(
            "вне маски идёт нетронутая копия драйвинга")), 1)
        self.assertEqual(self._unstruck_claims(
            "~~вне маски идёт нетронутая копия~~ — снято"), [])
        self.assertEqual(self._unstruck_claims("обычный текст без обещаний"),
                         [])
        # отрицание вплотную — перечёркнуто; оторванное переносом — нет
        self.assertEqual(self._unstruck_claims("это не подложка"), [])
        self.assertEqual(len(self._unstruck_claims(
            "это обусловливание, а не\nподложка целиком")), 1)

    def test_no_line_of_the_module_promises_copying_outside_the_mask(self):
        src = (ROOT / "ball_reel/fork_comfy.py").read_text(encoding="utf-8")
        self.assertEqual(self._unstruck_claims(src), [],
                         "в модуле осталось неперечёркнутое обещание "
                         "сохранности пикселей вне маски (§1a)")


class TheInputCheckCatchesWhatComfyWouldCatchNineSecondsLater(
        unittest.TestCase):
    """П2: те же нарушения, но за миллисекунду и все сразу."""

    def test_a_valid_input_passes(self):
        """Иначе всё ниже зеленеет на проверке, которая ругается на всё."""
        self.assertEqual(fk.check_inputs(valid_inputs()), [])

    def test_the_shipped_geometry_of_the_handoff_passes(self):
        """480x848x77 из §3 — штатная. Проверка, бракующая её, бесполезна.

        Ради этого теста массивы не создаются: он про арифметику кратностей,
        и 77 кадров 848x480 в память класть незачем.
        """
        self.assertEqual(480 % fk.SIDE_MULTIPLE, 0)
        self.assertEqual(848 % fk.SIDE_MULTIPLE, 0)
        self.assertEqual((77 - fk.LENGTH_BASE) % fk.LENGTH_STEP, 0)

    def test_seventy_six_is_refused_and_seventy_seven_is_not(self):
        """Мутация константы-решения в обе стороны (Т1).

        §3.2 говорит «кратна 4», первоисточник — «step: 4 от 1». При чтении
        прозы буквально 77 бы забраковалось, а 76 прошло — то есть проверка
        отвергала бы ровно штатную геометрию стека.
        """
        self.assertEqual((76 - fk.LENGTH_BASE) % fk.LENGTH_STEP, 3)
        self.assertEqual((77 - fk.LENGTH_BASE) % fk.LENGTH_STEP, 0)
        bad = fk.check_inputs(valid_inputs(
            length=N + 1, face_video=np.zeros(
                (N + 1, fk.FACE_SIDE, fk.FACE_SIDE, 3), np.uint8),
            pose_video=np.zeros((N + 1, H, W, 3), np.uint8),
            background_video=np.zeros((N + 1, H, W, 3), np.uint8),
            character_mask=np.zeros((N + 1, H, W), np.float32)))
        self.assertTrue(any("шаг 4" in p for p in bad), bad)

    def test_a_width_that_is_not_a_multiple_of_sixteen_is_refused(self):
        bad = fk.check_inputs(valid_inputs(
            width=W + 1,
            reference_image=np.zeros((1, H, W + 1, 3), np.uint8),
            pose_video=np.zeros((N, H, W + 1, 3), np.uint8),
            background_video=np.zeros((N, H, W + 1, 3), np.uint8),
            character_mask=np.zeros((N, H, W + 1), np.float32)))
        self.assertTrue(any("не кратно 16" in p for p in bad), bad)

    def test_a_frame_narrower_than_one_mask_block_is_refused(self):
        """§5.4: расширение меньше блока 32 px для модели не существует."""
        bad = fk.check_inputs(valid_inputs(
            width=16, reference_image=np.zeros((1, H, 16, 3), np.uint8),
            pose_video=np.zeros((N, H, 16, 3), np.uint8),
            background_video=np.zeros((N, H, 16, 3), np.uint8),
            character_mask=np.zeros((N, H, 16), np.float32)))
        self.assertTrue(any("блока маски 32" in p for p in bad), bad)

    def test_a_face_channel_that_is_not_512_is_refused(self):
        """1207 в nodes_wan.py масштабирует лицо в 512 безусловно — подать
        другое значит не знать, что доехало до модели."""
        bad = fk.check_inputs(valid_inputs(
            face_video=np.zeros((N, 256, 256, 3), np.uint8)))
        self.assertTrue(any("256x256 вместо 512x512" in p for p in bad), bad)

    def test_a_missing_required_input_is_named(self):
        got = valid_inputs()
        del got["character_mask"]
        bad = fk.check_inputs(got)
        self.assertTrue(any("character_mask" in p for p in bad), bad)

    def test_a_sequence_of_the_wrong_length_is_named(self):
        bad = fk.check_inputs(valid_inputs(
            pose_video=np.zeros((N - 1, H, W, 3), np.uint8)))
        self.assertTrue(any("pose_video: кадров 4" in p for p in bad), bad)

    def test_a_mask_with_a_channel_axis_is_refused(self):
        """character_mask — (N,H,W). С каналом это уже IMAGE, и Comfy откажет
        по типу, но на девятой секунде загрузки весов."""
        bad = fk.check_inputs(valid_inputs(
            character_mask=np.zeros((N, H, W, 1), np.float32)))
        self.assertTrue(any("без канала" in p for p in bad), bad)

    def test_a_reference_that_is_a_sequence_is_refused(self):
        """§1: вход продукта — ОДНА фотография."""
        bad = fk.check_inputs(valid_inputs(
            reference_image=np.zeros((N, H, W, 3), np.uint8)))
        self.assertTrue(any("одна картинка" in p for p in bad), bad)

    def test_all_violations_come_back_at_once_not_one_per_run(self):
        """Ради этого проверка и написана: пять перезапусков стоили дорого."""
        bad = fk.check_inputs(valid_inputs(
            width=W + 1, face_video=np.zeros((N, 256, 256, 3), np.uint8)))
        self.assertGreaterEqual(len(bad), 2, bad)


class TheMockCannotPassItselfOffAsARun(unittest.TestCase):
    """Главная проверка адаптера. Мок, выдающий себя за прогон, — тот самый
    дефект, против которого написан весь проект."""

    @classmethod
    def setUpClass(cls):
        cls.graph = fk.derive(fk.load_upstream())["graph"]

    def test_without_a_backend_the_source_is_mock_and_says_so_out_loud(self):
        got = fk.render(self.graph, valid_inputs())
        self.assertEqual(got["source"], fk.MOCK)
        self.assertIn("ГЕНЕРАЦИИ НЕ БЫЛО", got["note"])

    def test_a_mock_can_never_return_pass(self):
        """Первый из трёх независимых заслонов."""
        got = fk.render(self.graph, valid_inputs())
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertNotEqual(got["outcome"], fk.PASS)

    def test_the_mock_frames_are_marked_in_the_pixels_themselves(self):
        """Второй заслон: словарь теряется, массив едет дальше."""
        got = fk.render(self.graph, valid_inputs())
        self.assertTrue(fk.is_mock(got["frames"]))
        self.assertEqual(got["frames"].shape, (N, H, W, 3))

    def test_the_mark_survives_a_round_trip_through_a_file(self):
        """Именно тот путь, на котором теряется поле словаря."""
        got = fk.render(self.graph, valid_inputs())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "frames.npy"
            np.save(p, got["frames"])
            self.assertTrue(fk.is_mock(np.load(p)))

    def test_real_frames_are_not_mistaken_for_mock(self):
        """Негативный контроль к метке (И5): прибор обязан и молчать."""
        self.assertFalse(fk.is_mock(np.zeros((N, H, W, 3), np.uint8)))
        self.assertFalse(fk.is_mock(None))
        self.assertFalse(fk.is_mock(np.zeros((H, W, 3), np.uint8)))

    def test_a_backend_returning_marked_frames_is_caught_as_a_forgery(self):
        """ТРЕТИЙ ЗАСЛОН, без которого всё обходится одним аргументом.

        Подсунуть мок как `backend=` — самый дешёвый способ получить `PASS`
        без единого кадра. Проверка идёт в обратную сторону именно поэтому.
        """
        def liar(graph, inputs):
            return fk.mock_frames(inputs["length"], inputs["height"],
                                  inputs["width"])

        got = fk.render(self.graph, valid_inputs(), backend=liar)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertEqual(got["source"], fk.NOTHING)
        self.assertIsNone(got["frames"])
        self.assertIn("подлог", got["note"])

    def test_an_honest_backend_gets_pass_and_ran(self):
        """Негативный контроль к предыдущему: заслон обязан пропускать.

        НЕПРОВЕРЕНО: «настоящий» бэкенд здесь подставной. На ComfyUI ветка не
        исполнялась ни разу — его в этой среде нет и ставить запрещено (§10).
        """
        def honest(graph, inputs):
            return np.full((inputs["length"], inputs["height"],
                            inputs["width"], 3), 7, np.uint8)

        got = fk.render(self.graph, valid_inputs(), backend=honest)
        self.assertEqual(got["source"], fk.RAN)
        self.assertEqual(got["outcome"], fk.PASS)
        self.assertEqual(got["backend"], "honest")

    def test_require_generated_refuses_mock_frames(self):
        with self.assertRaises(ValueError) as caught:
            fk.require_generated(fk.render(self.graph, valid_inputs()))
        self.assertIn(fk.MOCK, str(caught.exception))

    def test_require_generated_believes_the_pixels_over_the_flag(self):
        """Е2: при расхождении флага и свидетельства верь свидетельству."""
        forged = {"source": fk.RAN, "outcome": fk.PASS,
                  "frames": fk.mock_frames(N, H, W)}
        with self.assertRaises(ValueError) as caught:
            fk.require_generated(forged)
        self.assertIn("не флагу", str(caught.exception))

    def test_require_generated_lets_real_frames_through(self):
        def honest(graph, inputs):
            return np.full((inputs["length"], inputs["height"],
                            inputs["width"], 3), 7, np.uint8)

        frames = fk.require_generated(
            fk.render(self.graph, valid_inputs(), backend=honest))
        self.assertEqual(frames.shape, (N, H, W, 3))


class TheAdapterHasThreeOutcomesNotTwo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = fk.derive(fk.load_upstream())["graph"]

    def test_bad_input_returns_no_frames_at_all(self):
        """Даже поддельных: пустой результат заметен, а правдоподобный мок на
        негодном входе — нет."""
        got = fk.render(self.graph, valid_inputs(width=W + 1))
        self.assertEqual(got["source"], fk.NOTHING)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIsNone(got["frames"])

    def test_a_broken_graph_is_refused_before_the_inputs_are_even_looked_at(
            self):
        derived = fk.derive(fk.load_upstream())
        derived["graph"]["nodes"].append(
            {"id": 9997, "type": fk.MANUAL_NODE,
             "properties": {fk.PACK_KEY: "comfy-core"}})
        got = fk.render(derived["graph"], valid_inputs())
        self.assertEqual(got["source"], fk.NOTHING)
        self.assertTrue(any("граф:" in p for p in got["problems"]),
                        got["problems"])

    def test_a_crashing_backend_is_unmeasured_and_not_fail(self):
        """Р1: упавший бэкенд ничего не сказал про граф. Свернуть это в «не
        годно» — соврать в ту же сторону, что и свернуть в «годно»."""
        def crashes(graph, inputs):
            raise RuntimeError("нет карты")

        got = fk.render(self.graph, valid_inputs(), backend=crashes)
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertEqual(got["source"], fk.NOTHING)
        self.assertIn("нет карты", got["note"])

    def test_a_backend_returning_the_wrong_shape_is_fail(self):
        """А вот это уже про сам бэкенд, и это `не годно`, а не «не смогли»."""
        def sloppy(graph, inputs):
            return np.zeros((3, 3), np.uint8)

        got = fk.render(self.graph, valid_inputs(), backend=sloppy)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIsNone(got["frames"])

    def test_the_three_words_are_distinct(self):
        self.assertEqual(len({fk.RAN, fk.MOCK, fk.NOTHING}), 3)
        self.assertEqual(len({fk.PASS, fk.FAIL, fk.UNMEASURED}), 3)


class TheLockFileIsCompleteAndEveryNumberHasACommand(unittest.TestCase):
    """Лок-файл: воркфлоу, веса с URL и размерами, дата и источник КАЖДОГО
    утверждения о контракте."""

    @classmethod
    def setUpClass(cls):
        cls.lock = fk.load_lock()
        cls.report = fk.audit_lock(cls.lock)

    def test_the_lock_file_passes_its_own_audit(self):
        self.assertEqual(self.report["outcome"], fk.PASS, self.report["note"])

    def test_the_audit_prints_how_many_it_checked_next_to_how_many_failed(self):
        """Р2: ноль нарушений при нуле проверенных записей — не успех."""
        self.assertGreater(self.report["weights_checked"], 0)
        self.assertGreater(self.report["claims_checked"], 0)
        self.assertIn("проверено весов", self.report["note"])

    def test_an_empty_lock_is_unmeasured_and_not_clean(self):
        got = fk.audit_lock({"weights": [], "contract_claims": []})
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertNotEqual(got["outcome"], fk.PASS)

    def test_all_six_layers_of_the_stack_are_there(self):
        roles = {w["role"] for w in self.lock["weights"]}
        self.assertEqual(roles, {"diffusion", "text_encoder", "vae",
                                 "clip_vision", "lora_1", "lora_2"},
                         f"слой стека §3 пропал из лок-файла: {roles}")

    def test_every_weight_carries_url_size_hash_and_the_command(self):
        for w in self.lock["weights"]:
            with self.subTest(role=w["role"]):
                self.assertTrue(w["url"].startswith("https://huggingface.co/"))
                self.assertIsInstance(w["bytes"], int)
                self.assertEqual(len(w["sha256"]), 64)
                self.assertIn("paths-info", w["measured_by"])
                self.assertEqual(w["measured_at"], "2026-08-17")

    def test_the_sizes_are_measured_and_not_recalled(self):
        """Припоминание — не замер. Признак замера здесь — байты, а не ГБ:
        число вроде «8.04 ГБ» можно вспомнить, 8630769472 — нет."""
        for w in self.lock["weights"]:
            with self.subTest(role=w["role"]):
                self.assertTrue(w["measured"])
                self.assertIn("ЗАМЕРЕНО", w["provenance"])
                self.assertNotEqual(
                    round(w["bytes"] / 2 ** 30, 2) * 2 ** 30, w["bytes"],
                    "число байт совпало со своим же округлением до сотых ГБ — "
                    "похоже, его не замерили, а развернули обратно из ГБ")

    def test_bytes_and_gib_agree_and_the_checker_notices_when_they_do_not(self):
        """Сторож обязан краснеть: подделываем одно из двух чисел."""
        broken = json.loads(json.dumps(self.lock))
        broken["weights"][0]["gib"] = 99.0
        got = fk.audit_lock(broken)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertTrue(any("разошлись" in p for p in got["problems"]),
                        got["problems"])

    def test_a_weight_without_a_hash_is_caught(self):
        broken = json.loads(json.dumps(self.lock))
        broken["weights"][0]["sha256"] = ""
        got = fk.audit_lock(broken)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertTrue(any("sha256" in p for p in got["problems"]))

    def test_an_unmeasured_weight_must_be_marked_unverified(self):
        """Третий исход у веса: не «есть»/«нет», а «не замерен, и это сказано»."""
        loose = {"weights": [{"role": "выдумка", "measured": False,
                              "provenance": "просто так"}],
                 "contract_claims": self.lock["contract_claims"]}
        self.assertEqual(fk.audit_lock(loose)["outcome"], fk.FAIL)
        loose["weights"][0]["provenance"] = "НЕПРОВЕРЕНО: не нашёлся"
        self.assertEqual(fk.audit_lock(loose)["outcome"], fk.PASS)

    def test_every_contract_claim_names_its_source_and_its_date(self):
        for c in self.lock["contract_claims"]:
            with self.subTest(claim=c["claim"][:40]):
                self.assertTrue(c["source"])
                self.assertEqual(c["checked"], "2026-08-17")

    def test_a_claim_without_a_source_is_caught(self):
        broken = json.loads(json.dumps(self.lock))
        broken["contract_claims"][0]["source"] = ""
        got = fk.audit_lock(broken)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertTrue(any("без поля source" in p for p in got["problems"]))

    def test_the_lock_records_the_hash_of_the_workflow_the_code_checks(self):
        """Е1: одно знание — одно место. Разъехавшийся хэш — дефект, который
        обнаружится ровно тогда, когда вендор обновит файл."""
        self.assertEqual(self.lock["workflow"]["upstream_sha256"],
                         fk.UPSTREAM_SHA256)
        self.assertEqual(self.lock["workflow"]["upstream"], fk.UPSTREAM)

    def test_the_lock_names_the_packs_the_code_actually_cuts(self):
        self.assertEqual(tuple(self.lock["workflow"]["removed_packs"]),
                         fk.CUSTOM_PACKS)

    def test_the_total_is_the_sum_and_fits_the_forty_gigabyte_claim(self):
        """§3: «диск: 40 ГБ достаточно». Теперь это замер, а не оценка."""
        want = sum(w["bytes"] for w in self.lock["weights"])
        self.assertEqual(self.lock["totals"]["bytes"], want)
        self.assertEqual(self.report["total_bytes"], want)
        self.assertLess(want / 2 ** 30, 40.0)

    def test_the_unverified_list_is_present_and_not_empty(self):
        """Пустой список НЕПРОВЕРЕНО на неисполнявшемся графе был бы враньём."""
        unverified = self.lock["unverified"]
        self.assertTrue(unverified)
        self.assertTrue(any("не исполнялся" in u for u in unverified))
        self.assertTrue(any("не скачан" in u for u in unverified))

    def test_the_licence_of_the_two_vendor_loras_is_recorded_not_hidden(self):
        """§10: лицензия не блокирует разработку — но и не замалчивается."""
        loras = [w for w in self.lock["weights"]
                 if w["role"].startswith("lora")]
        self.assertEqual(len(loras), 2)
        for w in loras:
            with self.subTest(role=w["role"]):
                self.assertEqual(w["license"], "НЕ ОБЪЯВЛЕНА")
                self.assertIn("отгрузке", w["license_note"])

    def test_the_open_discrepancy_of_the_two_loaders_is_written_down(self):
        """Находка прогона: произведённый граф грузит fp8-модель темплейта, а
        не Q3_K_M из §3. Незаписанная, она всплыла бы на карте как OOM."""
        claims = [c for c in self.lock["contract_claims"]
                  if "РАСХОЖДЕНИЕ" in str(c.get("status", ""))]
        self.assertEqual(len(claims), 1, "расхождение загрузчиков пропало из "
                                         "лок-файла — а оно не закрыто")
        self.assertIn("UNETLoader", claims[0]["evidence"])

    def test_that_discrepancy_is_still_true_of_the_graph_we_produce(self):
        """Документ, разошедшийся с кодом, — дефект. Здесь он бы разошёлся
        молча: кто-нибудь починит загрузчики и забудет строку в лок-файле."""
        graph = fk.derive(fk.load_upstream())["graph"]
        loaders = {n["type"]: n.get("widgets_values", [None])[0]
                   for n in graph["nodes"]
                   if n["type"] in ("UNETLoader", "CLIPLoader")}
        self.assertIn("fp8", str(loaders.get("UNETLoader")),
                      "загрузчик диффузии починили — снимите запись о "
                      "расхождении из лок-файла")
        self.assertIn("fp8", str(loaders.get("CLIPLoader")),
                      "загрузчик энкодера починили — снимите запись о "
                      "расхождении из лок-файла")

    def test_a_missing_lock_file_says_which_file(self):
        with self.assertRaises(FileNotFoundError) as caught:
            fk.load_lock(ROOT / "нет-такого.json")
        self.assertIn("нет-такого.json", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
