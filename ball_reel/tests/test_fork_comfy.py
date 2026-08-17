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

    def test_the_one_unproven_type_is_flagged_and_not_introduced_silently(self):
        """Ровно одно недоказанное имя, и оно названо.

        `character_mask` требует MASK, наши маски приезжают картинками, а
        преобразователя IMAGE->MASK в темплейте нет — там маску отдавал
        вырезанный `BlockifyMask`. Имя `ImageToMask` взято из встроенного
        набора Comfy и НЕ ПРОВЕРЕНО исполнением. Тест сторожит не отсутствие
        такого имени, а то, что оно ПОМЕЧЕНО.
        """
        derived = fk.derive(fk.load_upstream())
        flagged = {i["type"] for i in derived["introduced"]}
        self.assertEqual(flagged, {"ImageToMask"},
                         f"список недоказанных типов изменился: {flagged}. "
                         f"Новое имя обязано быть либо доказано командой, "
                         f"либо помечено — молча вводить нельзя.")
        for i in derived["introduced"]:
            self.assertIn("НЕПРОВЕРЕНО", i["provenance"])

    def test_the_unproven_type_reaches_the_audit_note(self):
        """Пометка, не доехавшая до отчёта, не меняет ничьего решения."""
        src = fk.load_upstream()
        got = fk.audit(fk.derive(src), upstream=src)
        self.assertIn("ImageToMask", got["note"])
        self.assertEqual(len(got["unproven_types"]), 1)

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


if __name__ == "__main__":
    unittest.main()
