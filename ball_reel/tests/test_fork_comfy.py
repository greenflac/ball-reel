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

        Ступень названа ЯВНО с 18.08.2026: умолчание переехало на `QUANT_GGUF`
        (развилка §1 закрыта владельцем), и на нём вводятся ещё два типа —
        загрузчики GGUF. Тест правлен, а не код: он про ШТАТНУЮ ступень, и
        подставить ей чужой список значило бы измерить другое.
        """
        derived = fk.derive(fk.load_upstream(), quant=fk.QUANT_TEMPLATE)
        flagged = {i["type"] for i in derived["introduced"]}
        self.assertEqual(flagged, {"ImageToMask"},
                         f"список введённых извне типов изменился: {flagged}. "
                         f"Новое имя обязано быть либо доказано командой, "
                         f"либо помечено — молча вводить нельзя.")

    def test_the_default_stage_introduces_exactly_three_named_types(self):
        """Негативный контроль к предыдущему: на умолчании их три, и все три
        названы поимённо. Иначе «ровно одно имя» читалось бы как «всегда одно»."""
        flagged = {i["type"] for i in fk.derive(fk.load_upstream())["introduced"]}
        self.assertEqual(flagged,
                         {"ImageToMask", "UnetLoaderGGUF", "CLIPLoaderGGUF"},
                         f"на умолчании введены другие типы: {flagged}")

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
        # Ступень явно штатная: на умолчании (GGUF) добавляются ещё два типа,
        # и они доказаны ИСХОДНИКОМ, а не темплейтом — это проверяет
        # test_the_gguf_loaders_are_proven_by_downloaded_source.
        src = fk.load_upstream()
        derived = fk.derive(src, quant=fk.QUANT_TEMPLATE)
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
                # Дата проверяется ФОРМАТОМ, а не конкретным днём. Раньше здесь
                # стоял литерал «2026-08-17», и он краснел при каждом ПОВТОРНОМ
                # замере — то есть наказывал за то, ради чего сторож и заведён.
                # Защиты литерал не давал никакой: подменить дату на любую
                # другую он бы позволил ровно так же.
                self.assertRegex(w["measured_at"], r"^20\d\d-\d\d-\d\d$")

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

    def test_the_lock_pins_the_quant_stage_the_owner_chose(self):
        """Решение владельца от 18.08.2026 — Q4_K_M под карту A16.

        Литералом и намеренно: тихая смена ступени меняет и качество, и
        скорость, и объём закачки. Пусть краснеет — тогда смена будет
        осознанной, а не унаследованной.
        """
        diffusion = next(w for w in self.lock["weights"]
                         if w["role"] == "diffusion")
        self.assertEqual(diffusion["path"], "Wan2.2-Animate-14B-Q4_K_M.gguf")
        self.assertEqual(diffusion["bytes"], 11496331072)

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

    def test_that_discrepancy_is_now_true_only_of_the_template_stage(self):
        """~~«расхождение всё ещё верно для графа, который мы производим»~~ —
        18.08.2026 умолчание переведено на ступень из лока, и на умолчании
        расхождения БОЛЬШЕ НЕТ.

        # DEBT(2026-08-18): запись `contract_claims[...] status=РАСХОЖДЕНИЕ`
        # в `workflows/fork_stack.lock.json` описывает теперь только штатную
        # ступень. Лок-файл — не файл этой смены (Ц2), правку делает владелец.

        Тест правлен, а не код: он сторожил, чтобы документ не разошёлся с
        кодом молча, и ровно это и сработало — расхождение названо, а не
        стёрто.
        """
        default = fk.derive(fk.load_upstream())["graph"]
        by_type = {n["type"] for n in default["nodes"]}
        self.assertNotIn("UNETLoader", by_type,
                         "на умолчании снова стоит штатный загрузчик fp8 — "
                         "решение владельца о Q4_K_M до кода не доехало")
        template = fk.derive(fk.load_upstream(), quant=fk.QUANT_TEMPLATE)
        loaders = {n["type"]: n.get("widgets_values", [None])[0]
                   for n in template["graph"]["nodes"]
                   if n["type"] in ("UNETLoader", "CLIPLoader")}
        self.assertIn("fp8", str(loaders.get("UNETLoader")))
        self.assertIn("fp8", str(loaders.get("CLIPLoader")))

    def test_a_missing_lock_file_says_which_file(self):
        with self.assertRaises(FileNotFoundError) as caught:
            fk.load_lock(ROOT / "нет-такого.json")
        self.assertIn("нет-такого.json", str(caught.exception))


class TheGraphAndTheLockAreComparedByFileName(unittest.TestCase):
    """Дефект, который прошёл ОБА аудита и целый прогон с мутациями.

    `audit` мерил структуру, `audit_lock` — полноту лока, и на состоянии, где
    граф грузит fp8 (17.138 ГиБ), а лок объявляет Q3_K_M (8.038 ГиБ), оба
    печатали «годно». Смена на арендованной машине скачала бы по локу 15.338
    ГиБ и запустила граф, просящий другие файлы. Здесь это сторожится.
    """

    def test_the_template_stage_is_red_because_the_fork_is_open(self):
        """~~«развилка открыта»~~ — ЗАКРЫТА 18.08.2026, ступень Q4_K_M.

        Тест оставлен и перенацелен на ЯВНУЮ штатную ступень: сторож обязан
        иметь вход, на котором краснеет, иначе он украшение. Зелёное умолчание
        сторожит соседний тест.
        """
        got = fk.audit_weights(fk.derive(quant=fk.QUANT_TEMPLATE))
        self.assertEqual(got["outcome"], fk.FAIL,
                         "расхождение графа с локом объявлено годным — тот же "
                         "дефект, что прошёл полный аудит")

    def test_it_names_both_directions_not_just_one(self):
        problems = " ".join(fk.audit_weights(
            fk.derive(quant=fk.QUANT_TEMPLATE))["problems"])
        self.assertIn("а в локе такого файла нет", problems,
                      "не назван файл, который граф грузит помимо лока")
        self.assertIn("ни один загрузчик графа его не просит", problems,
                      "не назван файл, который лок объявил зря — а именно его "
                      "и качают на машину")

    def test_the_gguf_stage_makes_the_same_check_green(self):
        got = fk.audit_weights(fk.derive(quant=fk.QUANT_GGUF))
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertEqual(got["problems"], [])

    def test_the_default_stage_is_green_because_the_fork_is_closed(self):
        """Дефект, ради которого правился код (записан в хэндофе как открытое
        расхождение): умолчание производства осталось шаблонным fp8 после того,
        как владелец выбрал Q4_K_M, и проверка была красной при закрытой
        развилке. Красное «не решено» и красное «решение не доехало» — разное.
        """
        got = fk.audit_weights(fk.derive())
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertIn("расхождений 0", got["note"])

    def test_the_numbers_stand_next_to_the_verdict(self):
        got = fk.audit_weights(fk.derive(quant=fk.QUANT_TEMPLATE))
        self.assertEqual(got["loaders_checked"], 6)
        self.assertEqual(got["declared"], 6)
        self.assertIn("разобрано загрузчиков 6", got["note"])

    def test_a_graph_without_loaders_is_unmeasured_not_clean(self):
        """Р2: ноль расхождений при нуле разобранного — не успех."""
        got = fk.audit_weights({"graph": {"nodes": []}})
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertEqual(got["loaders_checked"], 0)

    def test_an_empty_lock_is_unmeasured_too(self):
        got = fk.audit_weights(fk.derive(), lock={"weights": []})
        self.assertEqual(got["outcome"], fk.UNMEASURED)

    def test_only_loader_widgets_count_not_text_mentioned_in_notes(self):
        """Разбор по тексту вернул бы и bf16, и fp8 сразу: ссылки на оба лежат
        в записке темплейта. Мерить надо то, что Comfy пойдёт открывать."""
        files = {r["file"] for r in fk.graph_weights(
            fk.derive(quant=fk.QUANT_TEMPLATE)["graph"])}
        self.assertNotIn("wan2.2_animate_14B_bf16.safetensors", files,
                         "в веса графа попало имя из записки — разбор идёт по "
                         "тексту, а не по виджетам загрузчиков")
        self.assertIn("Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                      files)

    def test_breaking_the_widget_index_is_caught(self):
        """Т1, мутация в обе стороны: «имя файла — нулевой виджет» проверено по
        шаблону, а не предположено, и подмена индекса обязана краснеть.

        Эта мутация ВЫЖИЛА на первом прогоне и стоила правки кода, а не теста:
        разбор брал второй виджет `CLIPLoader`, находил там `wan` — тип
        энкодера — и объявлял его именем файла весов. Расхождений выходило те же
        4, разобранных те же 6, вердикт не двигался.
        """
        saved = dict(fk.WEIGHT_WIDGET)
        try:
            fk.WEIGHT_WIDGET["CLIPLoader"] = ("clip_name", 1)
            # Ступень явно штатная: штатный `CLIPLoader` стоит в графе только
            # на ней — на умолчании он уже переведён в `CLIPLoaderGGUF`.
            got = fk.audit_weights(fk.derive(quant=fk.QUANT_TEMPLATE))
            self.assertEqual(len(got["unparsed"]), 1, got["note"])
            self.assertIn("реестр WEIGHT_WIDGET указывает не туда",
                          " ".join(got["problems"]))
            files = {r["file"] for r in fk.graph_weights(
                fk.derive(quant=fk.QUANT_TEMPLATE)["graph"])}
            self.assertNotIn("wan", files,
                             "тип энкодера принят за имя файла весов")
        finally:
            fk.WEIGHT_WIDGET.clear()
            fk.WEIGHT_WIDGET.update(saved)

    def test_with_the_registry_right_nothing_is_unparsed(self):
        """Негативный контроль к предыдущему (И5): вход, где список обязан быть
        пустым. Без него проверка «нашлось неразобранное» зеленела бы всегда."""
        for quant in (None, fk.QUANT_GGUF, fk.QUANT_TEMPLATE):
            with self.subTest(quant=quant):
                got = fk.audit_weights(fk.derive(quant=quant))
                self.assertEqual(got["unparsed"], [], got["note"])
                self.assertIn("неразобранных 0", got["note"])

    def test_a_loader_carrying_no_file_name_is_named_not_skipped(self):
        """Тихо пропустить такой загрузчик — молчаливое «его в графе нет»."""
        graph = {"nodes": [{"id": 7, "type": "VAELoader",
                            "widgets_values": ["не файл"]}]}
        got = fk.unparsed_loaders(graph)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["got"], "не файл")


class TheQuantStageIsAFlagNotARewrite(unittest.TestCase):
    """Развилка §1 стоит одного слова — иначе решение владельца стоит правки
    графа, а правка под сроком делается на арендованной машине наспех."""

    def test_an_unknown_stage_falls_over_and_names_the_options(self):
        with self.assertRaises(ValueError) as caught:
            fk.derive(quant="q8")
        self.assertIn("q8", str(caught.exception))
        self.assertIn(fk.QUANT_GGUF, str(caught.exception))

    def test_the_stage_is_recorded_inside_the_produced_file(self):
        """Файл уедет на машину без этого репозитория: чем он грузится, должно
        быть видно из него самого."""
        for quant in (None, fk.QUANT_GGUF, fk.QUANT_TEMPLATE):
            with self.subTest(quant=quant):
                graph = fk.derive(quant=quant)["graph"]
                self.assertEqual(graph["extra"]["fork"]["quant"],
                                 fk.QUANT_DEFAULT if quant is None else quant)

    def test_the_default_is_not_bound_in_the_signature(self):
        """И7: умолчание-константа в сигнатуре связывается на импорте, и
        мутация константы модуля до неё не доходит. Форму уже выгребали в семи
        местах — проверяется, что она не вернулась."""
        import inspect

        for func, param in ((fk.derive, "quant"),
                            (fk.derive_wrapper, "pose_strength"),
                            (fk.derive_wrapper, "seconds"),
                            (fk.derive_wrapper, "width"),
                            (fk.derive_wrapper, "height"),
                            (fk.derive_wrapper, "blocks_to_swap"),
                            (fk.frames_for_seconds, "fps"),
                            (fk.window_plan, "window")):
            with self.subTest(func=func.__name__, param=param):
                default = inspect.signature(func).parameters[param].default
                self.assertIsNone(default,
                                  "умолчание снова стоит в сигнатуре — подмена "
                                  "константы перестанет доходить до вызова")

    def test_mutating_the_default_stage_reaches_the_call(self):
        """Продолжение предыдущего: сторож проверяет не форму, а следствие.

        Мутируется `QUANT_DEFAULT`, а не `QUANT_TEMPLATE`. Прежняя редакция
        подменяла `QUANT_TEMPLATE` на `QUANT_GGUF` и проверяла, что диффузия
        стала GGUF — после переезда умолчания на GGUF она стала ПУСТОЙ:
        зеленела бы и без всякой подмены. Мутационный тест, который зеленеет
        на невыполненной мутации, — самый дорогой сорт украшения, и найден он
        здесь ровно потому, что мутацию прогнали, а не вспомнили.
        """
        before = {r["file"] for r in fk.graph_weights(fk.derive()["graph"])}
        self.assertTrue(any(f.endswith(".gguf") for f in before), before)
        saved = fk.QUANT_DEFAULT
        try:
            fk.QUANT_DEFAULT = fk.QUANT_TEMPLATE
            files = {r["file"] for r in fk.graph_weights(fk.derive()["graph"])}
            self.assertIn("Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                          files,
                          "подмена константы умолчания не доехала до вызова")
            self.assertFalse(any(f.endswith(".gguf") for f in files),
                             f"после подмены умолчания диффузия всё ещё "
                             f"GGUF: {sorted(files)}")
        finally:
            fk.QUANT_DEFAULT = saved

    def test_the_gguf_loaders_are_proven_by_downloaded_source(self):
        """Ц10: ~20% предлагаемых моделью имён не существует. Оба имени взяты
        из NODE_CLASS_MAPPINGS скачанного файла, а не из памяти."""
        for node_type, line in (("UnetLoaderGGUF", 135),
                                ("CLIPLoaderGGUF", 200)):
            with self.subTest(node_type=node_type):
                src = fk.PROVEN_BY_SOURCE[node_type]
                self.assertIn("ComfyUI-GGUF", src["url"])
                self.assertEqual(src["line"], line)
                self.assertEqual(len(src["body_sha256"]), 64)

    def test_nothing_introduced_by_the_gguf_stage_is_unproven(self):
        derived = fk.derive(quant=fk.QUANT_GGUF)
        got = fk.audit(derived)
        self.assertEqual(got["unproven_types"], [], got["note"])
        self.assertEqual(got["outcome"], fk.PASS)

    def test_the_clip_loader_gets_two_widgets_not_three(self):
        """У GGUF-варианта виджетов два: имя и тип. Перенести третий `device`
        значило бы отдать ноде лишнее значение, и Comfy прочёл бы его как тип."""
        graph = fk.derive(quant=fk.QUANT_GGUF)["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "CLIPLoaderGGUF")
        self.assertEqual(len(node["widgets_values"]), 2, node["widgets_values"])
        self.assertEqual(node["widgets_values"][1], "wan",
                         "тип энкодера потерян при переносе виджетов")

    def test_the_unet_loader_gets_exactly_one_widget(self):
        graph = fk.derive(quant=fk.QUANT_GGUF)["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "UnetLoaderGGUF")
        self.assertEqual(len(node["widgets_values"]), 1, node["widgets_values"])

    def test_the_file_names_come_from_the_lock_not_from_source_strings(self):
        """Е1: строка, скопированная в код, — второй способ узнать известное,
        то есть тот самый дефект, который эта развилка и закрывает."""
        src = Path(fk.__file__).read_text(encoding="utf-8")
        for name in ("Wan2.2-Animate-14B-Q3_K_M.gguf",
                     "umt5-xxl-encoder-Q5_K_M.gguf"):
            with self.subTest(name=name):
                self.assertNotIn(f'"{name}"', src,
                                 "имя файла весов вписано в модуль строкой — "
                                 "оно обязано читаться из лок-файла")

    def test_the_gguf_stage_still_feeds_every_required_input(self):
        """Перевод загрузчиков не смеет уронить проводку: узел меняет тип, а
        связи у него остаются те же."""
        derived = fk.derive(quant=fk.QUANT_GGUF)
        self.assertEqual(fk.unfed_inputs(derived["graph"]), [])

    def test_the_fourth_pack_is_named_out_loud_in_the_audit(self):
        """`custom_left` мерит другое — осталось ли что-то из трёх вырезанных.
        После перевода он честно печатает 0, а сторонний пак снова один."""
        got = fk.audit(fk.derive(quant=fk.QUANT_GGUF))
        self.assertEqual(got["custom_left"], [])
        self.assertIn(fk.GGUF_PACK, got["packs_required"])
        self.assertIn("сторонних паков к установке 1", got["note"])

    def test_the_template_stage_requires_no_third_party_pack(self):
        """Негативный контроль (И5): вход, где та же проверка обязана молчать."""
        got = fk.audit(fk.derive(quant=fk.QUANT_TEMPLATE))
        self.assertEqual(got["packs_required"], [])
        self.assertIn("сторонних паков к установке 0", got["note"])


# ===========================================================================
# ЧАСТЬ II. ГРАФ НА ОБЁРТКЕ `kijai/ComfyUI-WanVideoWrapper`
# ===========================================================================
#
# Что здесь НЕ проверяется (Ц4): граф обёртки не исполнялся ни разу. ComfyUI в
# этой среде нет. Всё ниже — структура, арифметика и сверка с исходником,
# скачанным командой.

#: Геометрия для тестов адаптера обёртки. Маленькая по той же причине, что и в
#: части I: 149 кадров по 480x832 в память класть незачем, а кратности
#: настоящие — 32 и 48 кратны 16, (5-1) % 4 == 0. Продуктовая геометрия
#: проверяется отдельно, на числах.
WW, WH, WN = 32, 48, 5


def wrapper_inputs(**over):
    got = {
        "ref_images": np.zeros((1, WH, WW, 3), np.uint8),
        "pose_images": np.zeros((WN, WH, WW, 3), np.uint8),
        "face_images": np.zeros((WN, fk.FACE_SIDE, fk.FACE_SIDE, 3), np.uint8),
        "mask": np.zeros((WN, WH, WW), np.float32),
        "width": WW, "height": WH, "num_frames": WN,
    }
    got.update(over)
    return got


class EveryWrapperNodeNameWasProvenByACommand(unittest.TestCase):
    """Ц10: у моделей примерно пятая часть предлагаемых имён не существует.

    Здесь имя ноды обёртки не может попасть в граф, не будучи доказанным
    скачанным исходником: `derive_wrapper` спрашивает `provenance_of` про
    КАЖДЫЙ вводимый тип, а аудит роняет вердикт, если хоть одно происхождение
    начинается с НЕПРОВЕРЕНО.
    """

    @classmethod
    def setUpClass(cls):
        cls.derived = fk.derive_wrapper()
        cls.types = {n["type"] for n in cls.derived["graph"]["nodes"]}

    def test_not_one_type_in_the_graph_is_unproven(self):
        for node_type in sorted(self.types):
            with self.subTest(node_type=node_type):
                got = fk.provenance_of(node_type, fk.known_types(
                    fk.load_upstream()))
                self.assertNotIn("НЕПРОВЕРЕНО", got, node_type)

    def test_every_wrapper_node_carries_url_hash_line_and_date(self):
        """«Проверено» без URL, sha256 тела и номера строки через месяц
        неотличимо от «вроде помню»."""
        wrapper = [t for t in self.types if t.startswith("WanVideo")]
        self.assertGreaterEqual(len(wrapper), 8, sorted(wrapper))
        for node_type in sorted(wrapper):
            with self.subTest(node_type=node_type):
                src = fk.PROVEN_BY_SOURCE[node_type]
                self.assertIn("ComfyUI-WanVideoWrapper", src["url"])
                self.assertEqual(len(src["body_sha256"]), 64)
                self.assertIsInstance(src["line"], int)
                self.assertIsInstance(src["mapping_line"], int)
                self.assertRegex(src["checked"], r"^20\d\d-\d\d-\d\d$")

    def test_each_name_was_found_twice_class_and_mapping(self):
        """Класса мало: незарегистрированного класса в графе не существует так
        же, как несуществующего. Номера строк разные — значит смотрели оба."""
        for node_type, src in fk.PROVEN_BY_SOURCE.items():
            if "WanVideoWrapper" not in src["url"]:
                continue
            with self.subTest(node_type=node_type):
                self.assertNotEqual(src["line"], src["mapping_line"])
                self.assertGreater(src["mapping_line"], src["line"])

    def test_the_three_downloaded_bodies_are_three_distinct_files(self):
        hashes = {src["body_sha256"] for src in fk.PROVEN_BY_SOURCE.values()
                  if "WanVideoWrapper" in src["url"]}
        self.assertEqual(len(hashes), 3, hashes)

    def test_a_plausible_but_fabricated_wrapper_name_is_still_unproven(self):
        """Негативный контроль (И5). `WanVideoAnimateLoader` звучит ровно как
        те, что существуют, — и его нет ни в одном из трёх файлов."""
        got = fk.provenance_of("WanVideoAnimateLoader", fk.known_types(
            fk.load_upstream()))
        self.assertIn("НЕПРОВЕРЕНО", got)
        self.assertNotIn(("WanVideoAnimateLoader"), fk.PROVEN_BY_SOURCE)

    def test_the_cached_text_encoder_is_proven_and_deliberately_unused(self):
        """Имя доказано, а узел отвергнут замером: его загрузчик T5 читает
        `load_torch_file` и .gguf не понимает, а лок объявляет .gguf."""
        self.assertIn("WanVideoTextEncodeCached", fk.PROVEN_BY_SOURCE)
        self.assertNotIn("WanVideoTextEncodeCached", self.types)
        self.assertIn("В ГРАФ НЕ СТАВИТСЯ",
                      fk.PROVEN_BY_SOURCE["WanVideoTextEncodeCached"]["note"])
        self.assertIn("WanVideoTextEmbedBridge", self.types)

    def test_create_video_is_proven_by_the_template_though_not_at_top_level(self):
        """Правка `known_types` 18.08.2026, и вот её негативный контроль:
        типа НЕТ в списке верхнего уровня, но он есть в файле — значит
        существует, и пометка «не подтверждён ничем» была бы ложной тревогой."""
        src = fk.load_upstream()
        self.assertNotIn("CreateVideo", {n["type"] for n in src["nodes"]})
        self.assertIn("CreateVideo", fk.known_types(src))
        self.assertIn("ДОКАЗАНО темплейтом",
                      fk.provenance_of("CreateVideo", fk.known_types(src)))

    def test_known_types_did_not_start_proving_everything(self):
        """Негативный контроль к той же правке: обход стал глубже, но не
        начал считать доказанным что попало."""
        types = fk.known_types(fk.load_upstream())
        self.assertNotIn("ImageToMask", types)
        self.assertNotIn("WanVideoSampler", types)

    def test_the_licence_of_the_wrapper_was_checked_before_it_was_used(self):
        """Ц5: лицензия проверяется ДО встраивания, командой."""
        self.assertEqual(fk.WRAP_LICENSE["license"], "apache-2.0")
        self.assertIn("curl", fk.WRAP_LICENSE["checked_by"])
        self.assertRegex(fk.WRAP_LICENSE["checked"], r"^20\d\d-\d\d-\d\d$")


class TheWrapperGraphIsWiredAndTheAuditCanGoRed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.derived = fk.derive_wrapper()
        cls.report = fk.audit_wrapper(cls.derived)

    def test_the_audit_is_green_and_prints_its_numbers(self):
        self.assertEqual(self.report["outcome"], fk.PASS, self.report["note"])
        self.assertIn("ЧИСТО", self.report["note"])
        self.assertIn("входов без питания 0", self.report["note"])
        self.assertIn("НЕПРОВЕРЕНО", self.report["note"])
        self.assertIn("не исполнялся", self.report["note"])

    def test_every_required_input_of_every_node_is_fed(self):
        checked, bad = fk.wrap_unfed_inputs(self.derived["graph"])
        self.assertEqual(bad, [], bad)
        self.assertEqual(checked, len(fk.WRAP_REQUIRED),
                         "проверено узлов меньше, чем в таблице обязательных — "
                         "какой-то узел в граф не попал вовсе")

    def test_cutting_one_feeding_link_goes_red_and_names_the_input(self):
        """Сторож обязан уметь краснеть — иначе он украшение."""
        derived = fk.derive_wrapper()
        graph = derived["graph"]
        node = next(n for n in graph["nodes"]
                    if n["type"] == "WanVideoAnimateEmbeds")
        slot = [i["name"] for i in node["inputs"]].index("mask")
        graph["links"] = [l for l in graph["links"]
                          if not (l[3] == node["id"] and l[4] == slot)]
        checked, bad = fk.wrap_unfed_inputs(graph)
        self.assertEqual(checked, len(fk.WRAP_REQUIRED))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("mask без питания", bad[0])
        got = fk.audit_wrapper({"graph": graph})
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("входов без питания 1", got["note"])

    def test_an_empty_graph_is_unmeasured_and_not_clean(self):
        """Р2: ноль нарушений при нуле проверенных узлов — не успех."""
        got = fk.audit_wrapper({"graph": {"nodes": [], "links": []}})
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertNotEqual(got["outcome"], fk.PASS)
        self.assertIn("не успех", got["note"])

    def test_a_dangling_link_is_reported(self):
        derived = fk.derive_wrapper()
        derived["graph"]["links"].append([9999, 77777, 0, 88888, 0, "IMAGE"])
        got = fk.audit_wrapper(derived)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("оборванных связей 1", got["note"])

    def test_a_pack_the_owner_refused_is_caught_by_name_and_reason(self):
        """Решение «KJNodes не берём» проверяется кодом, а не памятью."""
        derived = fk.derive_wrapper()
        derived["graph"]["nodes"].append(
            {"id": 9998, "type": "ImageResizeKJv2",
             "properties": {fk.PACK_KEY: "ComfyUI-KJNodes"},
             "inputs": [], "outputs": [], "widgets_values": []})
        got = fk.audit_wrapper(derived)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertEqual(len(got["forbidden"]), 1)
        self.assertIn("GPL-3.0", " ".join(got["problems"]))

    def test_the_clean_graph_carries_no_forbidden_pack(self):
        """Негативный контроль к предыдущему: вход, где проверка молчит."""
        self.assertEqual(self.report["forbidden"], [])
        self.assertIn("запрещённых паков 0", self.report["note"])

    def test_exactly_two_third_party_packs_are_named_out_loud(self):
        """Расход, а не нарушение, — но он обязан быть НАЗВАН: обёртка и
        GGUF-загрузчик энкодера. Всё остальное штатное."""
        self.assertEqual(self.report["packs_required"],
                         [fk.WRAP_PACK, fk.GGUF_PACK])

    def test_the_refutation_and_the_resize_facts_travel_inside_the_file(self):
        """Граф уедет на машину без этого репозитория и без хэндофа."""
        fork = self.derived["graph"]["extra"]["fork"]
        self.assertEqual(fork["no_composite"], fk.NO_COMPOSITE)
        self.assertEqual(fork["resize_facts"], fk.WRAP_RESIZE_FACTS)
        self.assertEqual(fork["license"], fk.WRAP_LICENSE)
        self.assertTrue(any("DEBT" in d for d in fork["deferred"]))

    def test_the_graph_survives_a_round_trip_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fk.write(self.derived, Path(tmp) / "sub" / "wrap.json")
            back = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(fk.audit_wrapper({"graph": back})["outcome"], fk.PASS)

    def test_a_link_to_a_nonexistent_input_name_falls_over_at_build_time(self):
        """Связи кладутся по ИМЕНИ входа: номер, выведенный из порядка, поедет
        молча при первом же изменении чужой ноды."""
        wire = fk._Wire(set())
        a = wire.node("A", pack="p", outputs=[("out", "IMAGE")])
        b = wire.node("B", pack="p", inputs=[("in", "IMAGE")])
        wire.link(a, "out", b, "in")
        with self.assertRaises(KeyError):
            wire.link(a, "out", b, "нет-такого-входа")


class TheGeometryAndTheLengthAreTheOnesTheOwnerChose(unittest.TestCase):
    """480x832 вертикально, 30 к/с, 5-10 с. Числа — литералами (Т2)."""

    def test_the_default_frame_is_vertical_four_eighty_by_eight_thirty_two(self):
        params = fk.derive_wrapper()["params"]
        self.assertEqual((params["width"], params["height"]), (480, 832))
        self.assertLess(params["width"], params["height"],
                        "кадр перестал быть вертикальным")

    def test_the_widgets_of_the_graph_carry_that_geometry(self):
        graph = fk.derive_wrapper()["graph"]
        node = next(n for n in graph["nodes"]
                    if n["type"] == "WanVideoAnimateEmbeds")
        self.assertEqual(node["widgets_values"][:3], [480, 832, 149])
        self.assertEqual(node["widgets_values"][4], 77)

    def test_the_output_is_thirty_frames_per_second_not_the_template_sixteen(
            self):
        """Умолчание виджета CreateVideo в темплейте — 16, и его однажды уже
        приняли за свойство модели. Здесь стоит решение владельца."""
        graph = fk.derive_wrapper()["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "CreateVideo")
        self.assertEqual(node["widgets_values"], [30])

    def test_the_length_table_matches_the_one_counted_independently(self):
        """Т2: ожидаемое — литералы, а не импорт из проверяемого модуля.
        Числа сверены с таблицей хэндофа, посчитанной другим человеком."""
        for seconds, frames, windows, generated in ((5, 149, 2, 153),
                                                    (7, 209, 3, 229),
                                                    (10, 297, 4, 305)):
            with self.subTest(seconds=seconds):
                got = fk.frames_for_seconds(seconds)
                self.assertEqual(got["frames"], frames, got["note"])
                plan = fk.window_plan(frames)
                self.assertEqual(plan["windows"], windows, plan["note"])
                self.assertEqual(plan["generated"], generated, plan["note"])

    def test_the_length_band_is_refused_on_both_sides(self):
        """Мутация порога в обе стороны (Т1): 5.0 и 10.0 годятся, 4.9 и 10.1 —
        нет. Порог, у которого проверена одна сторона, сторожит половину."""
        fk.frames_for_seconds(5.0)
        fk.frames_for_seconds(10.0)
        for bad in (4.9, 10.1):
            with self.subTest(seconds=bad):
                with self.assertRaises(ValueError) as caught:
                    fk.frames_for_seconds(bad)
                self.assertIn("вне полосы", str(caught.exception))

    def test_the_snapping_of_the_length_is_computed_not_guessed(self):
        """150 кадров молча становятся 149 внутри обёртки (nodes.py:1230).
        Негативный контроль: на числе, уже стоящем на решётке, прибор молчит."""
        self.assertEqual(fk.snap_frames(150), 149)
        self.assertEqual(fk.snap_frames(149), 149)
        self.assertEqual(fk.snap_frames(1), 1)
        self.assertEqual(fk.frames_for_seconds(5)["snapped_away"], 1)
        self.assertEqual(fk.frames_for_seconds(10)["snapped_away"], 3)

    def test_a_shorter_clip_than_one_window_needs_one_window(self):
        """Негативный контроль к плану окон: вход, где цикла нет вовсе."""
        got = fk.window_plan(77)
        self.assertEqual((got["windows"], got["discarded"]), (1, 0))
        self.assertEqual(fk.window_plan(78)["windows"], 2)

    def test_mutating_the_frame_rate_reaches_both_the_plan_and_the_widget(self):
        """Т1: подмена константы-решения обязана доехать до вывода."""
        saved = fk.WRAP_FPS
        try:
            fk.WRAP_FPS = 16
            self.assertEqual(fk.frames_for_seconds(5)["frames"], 77)
            graph = fk.derive_wrapper()["graph"]
            node = next(n for n in graph["nodes"] if n["type"] == "CreateVideo")
            self.assertEqual(node["widgets_values"], [16])
        finally:
            fk.WRAP_FPS = saved
        self.assertEqual(fk.frames_for_seconds(5)["frames"], 149)

    def test_mutating_the_window_reaches_the_plan_in_both_directions(self):
        saved = fk.WRAP_WINDOW
        try:
            fk.WRAP_WINDOW = 41
            self.assertEqual(fk.window_plan(149)["windows"], 4)
            fk.WRAP_WINDOW = 149
            self.assertEqual(fk.window_plan(149)["windows"], 1)
        finally:
            fk.WRAP_WINDOW = saved
        self.assertEqual(fk.window_plan(149)["windows"], 2)

    def test_mutating_the_frame_geometry_reaches_the_produced_graph(self):
        saved_w, saved_h = fk.WRAP_WIDTH, fk.WRAP_HEIGHT
        try:
            fk.WRAP_WIDTH, fk.WRAP_HEIGHT = 832, 480
            params = fk.derive_wrapper()["params"]
            self.assertEqual((params["width"], params["height"]), (832, 480))
        finally:
            fk.WRAP_WIDTH, fk.WRAP_HEIGHT = saved_w, saved_h

    def test_a_frame_side_off_the_grid_is_refused_on_both_sides(self):
        """Обёртка округляет сторону ВНИЗ до 16 молча (nodes.py:1223), поэтому
        480 годится, а 481 обязано падать при сборке, а не на карте."""
        fk.derive_wrapper(width=480)
        with self.assertRaises(ValueError) as caught:
            fk.derive_wrapper(width=481)
        self.assertIn("округлит вниз молча", str(caught.exception))

    def test_the_block_swap_limit_is_the_one_the_node_declares(self):
        """Т1 в обе стороны: 48 — предел ноды, 49 обязано падать."""
        self.assertEqual(fk.BLOCKS_MAX, 48)
        fk.derive_wrapper(blocks_to_swap=48)
        with self.assertRaises(ValueError):
            fk.derive_wrapper(blocks_to_swap=49)

    def test_the_sampler_carries_the_settings_measured_in_the_workflow(self):
        """Литералами (Т2), а не импортом из модуля: 3 шага, cfg 1, shift 5,
        dpm++_sde — снято из узла 27 боевого воркфлоу владельца. Зерно у нас
        фиксированное: `randomize` владельца делает два прогона несравнимыми,
        а замеры нам нужнее удобства.
        """
        graph = fk.derive_wrapper()["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "WanVideoSampler")
        self.assertEqual(node["widgets_values"][:4], [3, 1, 5, 0])
        self.assertEqual(node["widgets_values"][6], "dpm++_sde")
        self.assertEqual(node["widgets_values"][4], "fixed")

    def test_the_block_swap_widget_carries_the_measured_thirty_eight(self):
        graph = fk.derive_wrapper()["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "WanVideoBlockSwap")
        self.assertEqual(node["widgets_values"][0], 38)
        graph = fk.derive_wrapper(blocks_to_swap=20)["graph"]
        node = next(n for n in graph["nodes"] if n["type"] == "WanVideoBlockSwap")
        self.assertEqual(node["widgets_values"][0], 20)


class PoseStrengthIsALeverAndItReachesTheGraph(unittest.TestCase):
    """Рычаг синхронности: у владельца 1.1 против умолчания ноды 1.0."""

    @staticmethod
    def _embeds(**kw):
        graph = fk.derive_wrapper(**kw)["graph"]
        return next(n for n in graph["nodes"]
                    if n["type"] == "WanVideoAnimateEmbeds")

    def test_the_default_is_the_value_measured_in_the_owners_workflow(self):
        self.assertEqual(fk.POSE_STRENGTH, 1.1)
        self.assertEqual(self._embeds()["widgets_values"][6], 1.1)

    def test_the_lever_can_actually_be_turned(self):
        self.assertEqual(self._embeds(pose_strength=0.7)["widgets_values"][6],
                         0.7)

    def test_mutating_the_default_reaches_the_widget(self):
        """Т1: умолчание разрешается в теле, значит мутация обязана доехать."""
        saved = fk.POSE_STRENGTH
        try:
            fk.POSE_STRENGTH = 1.0
            self.assertEqual(self._embeds()["widgets_values"][6], 1.0)
        finally:
            fk.POSE_STRENGTH = saved
        self.assertEqual(self._embeds()["widgets_values"][6], 1.1)

    def test_the_face_lever_is_separate_from_the_pose_one(self):
        """Негативный контроль: подмена одного рычага не двигает другой.

        Литерал 1.0, а не `fk.FACE_STRENGTH` (Т2): импортированное ожидание
        поедет вместе с кодом и промолчит. Найдено прогоном мутаций —
        подмена FACE_STRENGTH на 0.8 не роняла НИ ОДНОГО теста.
        """
        node = self._embeds(pose_strength=0.5)
        self.assertEqual(node["widgets_values"][6], 0.5)
        self.assertEqual(node["widgets_values"][7], 1.0,
                         "сила лица уехала с силой позы или сменилась молча")
        self.assertEqual(self._embeds(face_strength=0.4)["widgets_values"][7],
                         0.4)

    def test_colormatch_between_windows_is_off_as_measured(self):
        self.assertEqual(self._embeds()["widgets_values"][5], "disabled")
        self.assertEqual(self._embeds(colormatch="mkl")["widgets_values"][5],
                         "mkl")


class TheReferenceIsFittedByPaddingAndNothingIsCropped(unittest.TestCase):
    """Дефект 17.08.2026: штатная нода режет референс `center` и срезает голову.

    Обёртка вместо обрезки растягивает (nodes.py:1288) — то есть ни одно из
    двух готовых поведений нам не годится, и референс приводится своей
    функцией ДО графа.
    """

    @staticmethod
    def _portrait(height=800, width=600):
        """Портрет с меткой в самой верхней строке — это «голова»."""
        img = np.zeros((height, width, 3), np.uint8)
        img[0, :, 0] = 255
        return img

    @classmethod
    def _tall(cls):
        """Фото ВО ВЕСЬ РОСТ: 600x1600, то есть уже кадра по пропорции.

        Фикстура выбрана не наугад, и первая редакция теста была на 600x800 —
        там обрезка по центру голову НЕ трогает, потому что режет по бокам, и
        негативный контроль краснел не по делу. Дефект (голова уезжает) живёт
        ровно на снимках ВЫШЕ кадра по пропорции, а именно такие и приходят
        при кадрировке во весь рост. Т3: край диапазона, а не середина.
        """
        return cls._portrait(height=1600, width=600)

    @staticmethod
    def _center_crop(img, width, height):
        """Как режет штатная нода. Живёт в тесте, а не в модуле: это НЕ наш
        способ, это то, с чем сравниваемся."""
        scale = max(width / img.shape[1], height / img.shape[0])
        new_h = max(1, int(round(img.shape[0] * scale)))
        new_w = max(1, int(round(img.shape[1] * scale)))
        ys = (np.arange(new_h) * (img.shape[0] / new_h)).astype(int)
        xs = (np.arange(new_w) * (img.shape[1] / new_w)).astype(int)
        scaled = img[ys.clip(0, img.shape[0] - 1)][:, xs.clip(
            0, img.shape[1] - 1)]
        top = (new_h - height) // 2
        left = (new_w - width) // 2
        return scaled[top:top + height, left:left + width]

    def test_the_head_of_a_full_height_photo_survives_the_padding(self):
        out, record = fk.pad_reference(self._tall())
        self.assertEqual(out.shape, (1, 832, 480, 3))
        row = out[0, 0, record["pad_left"]:480 - record["pad_right"], 0]
        self.assertTrue((row == 255).all(),
                        "верхняя строка референса не доехала до кадра")
        self.assertEqual(record["pad_top"], 0)
        self.assertEqual(record["rows_dropped"], 0)
        self.assertGreater(record["pad_left"], 0)

    def test_the_same_head_does_not_survive_the_center_crop(self):
        """Негативный контроль к самому дефекту (И5): без него «голова цела»
        доказывает, что мы вообще не трогали картинку."""
        cropped = self._center_crop(self._tall(), 480, 832)
        self.assertEqual(cropped.shape, (832, 480, 3))
        self.assertFalse((cropped[0, :, 0] == 255).any(),
                         "обрезка по центру сохранила верхнюю строку — тогда "
                         "сравнивать не с чем, и дефект здесь не показан")

    def test_a_photo_wider_than_the_frame_gets_its_padding_below(self):
        """Второй край диапазона (Т3): 600x800 шире кадра по пропорции, и поля
        ложатся вниз, а не по бокам."""
        out, record = fk.pad_reference(self._portrait())
        self.assertEqual(out.shape, (1, 832, 480, 3))
        self.assertEqual((record["pad_top"], record["pad_left"]), (0, 0))
        self.assertGreater(record["pad_bottom"], 0)
        self.assertEqual(record["rows_dropped"], 0)

    def test_an_image_already_in_the_frame_aspect_gets_no_padding_at_all(self):
        """Прибор обязан уметь и НЕ ДЕЛАТЬ НИЧЕГО."""
        out, record = fk.pad_reference(np.zeros((832, 480, 3), np.uint8))
        self.assertEqual(out.shape, (1, 832, 480, 3))
        self.assertEqual((record["pad_top"], record["pad_bottom"],
                          record["pad_left"], record["pad_right"]),
                         (0, 0, 0, 0))

    def test_a_wide_photo_is_padded_left_and_right_and_still_not_cropped(self):
        """Фикстура с другого края диапазона (Т3): не портрет, а панорама."""
        wide = np.zeros((400, 1600, 3), np.uint8)
        wide[:, 0, 1] = 255
        out, record = fk.pad_reference(wide)
        self.assertEqual(out.shape, (1, 832, 480, 3))
        self.assertEqual(record["cols_dropped"], 0)
        self.assertGreater(record["pad_bottom"], 0)

    def test_the_alignment_is_a_real_choice_and_it_changes_the_output(self):
        """Мутация константы-решения в обе стороны: `top` кладёт поля вниз,
        `center` — поровну."""
        top, rec_top = fk.pad_reference(self._portrait(), align="top")
        mid, rec_mid = fk.pad_reference(self._portrait(), align="center")
        self.assertEqual(rec_top["pad_top"], 0)
        self.assertGreater(rec_mid["pad_top"], 0)
        self.assertFalse(np.array_equal(top, mid))
        with self.assertRaises(ValueError):
            fk.pad_reference(self._portrait(), align="bottom")

    def test_mutating_the_declared_alignment_reaches_the_call(self):
        saved = dict(fk.REFERENCE_FIT)
        try:
            fk.REFERENCE_FIT["align"] = "center"
            _, record = fk.pad_reference(self._portrait())
            self.assertGreater(record["pad_top"], 0,
                               "подмена объявленного выравнивания не доехала")
        finally:
            fk.REFERENCE_FIT.clear()
            fk.REFERENCE_FIT.update(saved)

    def test_the_padding_uses_the_edge_pixel_and_not_a_black_bar(self):
        """Ровная чёрная полоса — сильный контур, модель вправе принять её за
        часть сцены. У владельца стоит `pad_edge_pixel`, у нас то же."""
        img = np.full((800, 600, 3), 200, np.uint8)
        out, record = fk.pad_reference(img)
        self.assertEqual(int(out[0, -1, 0, 0]), 200)

    def test_the_produced_graph_says_how_the_reference_was_fitted(self):
        fit = fk.derive_wrapper()["graph"]["extra"]["fork"]["reference_fit"]
        self.assertEqual((fit["mode"], fit["align"]), ("pad", "top"))
        self.assertIn("pad_reference", fit["by"])

    def test_a_reference_not_in_frame_geometry_is_refused_by_the_input_check(
            self):
        """Иначе обёртка растянет его молча (nodes.py:1288)."""
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            ref_images=np.zeros((1, WH + 16, WW, 3), np.uint8)))
        self.assertTrue(any("pad_reference" in p for p in bad), bad)

    def test_a_padded_reference_passes(self):
        """Негативный контроль: вход, где та же проверка обязана молчать."""
        out, _ = fk.pad_reference(self._portrait(), width=WW, height=WH)
        self.assertEqual(fk.check_wrapper_inputs(
            wrapper_inputs(ref_images=out)), [])


class TheWrapperInputCheckCatchesSilentCoercion(unittest.TestCase):
    """Отличие от части I: обёртка на негодном входе не падает, а МОЛЧА
    приводит. Разбираться пришлось бы по готовому ролику."""

    def test_a_valid_input_passes(self):
        self.assertEqual(fk.check_wrapper_inputs(wrapper_inputs()), [])

    def test_the_shipped_geometry_passes_the_arithmetic(self):
        self.assertEqual(480 % fk.SIDE_MULTIPLE, 0)
        self.assertEqual(832 % fk.SIDE_MULTIPLE, 0)
        self.assertEqual((149 - fk.LENGTH_BASE) % fk.LENGTH_STEP, 0)

    def test_a_length_off_the_grid_is_named_with_the_number_it_becomes(self):
        """Т1 в обе стороны: 5 годится, 6 — нет, и в сообщении стоит 5."""
        self.assertEqual(fk.check_wrapper_inputs(wrapper_inputs()), [])
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            num_frames=WN + 1,
            pose_images=np.zeros((WN + 1, WH, WW, 3), np.uint8),
            face_images=np.zeros(
                (WN + 1, fk.FACE_SIDE, fk.FACE_SIDE, 3), np.uint8),
            mask=np.zeros((WN + 1, WH, WW), np.float32)))
        self.assertTrue(any("прижмёт к 5 молча" in p for p in bad), bad)

    def test_a_side_off_the_grid_is_refused(self):
        bad = fk.check_wrapper_inputs(wrapper_inputs(width=WW + 1))
        self.assertTrue(any("округлит вниз молча" in p for p in bad), bad)

    def test_a_face_channel_that_is_not_512_is_refused(self):
        """1324 в nodes.py режет лицо по центру до 512 — тот же дефект, что у
        штатной ноды, и он молчит так же."""
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            face_images=np.zeros((WN, 256, 256, 3), np.uint8)))
        self.assertTrue(any("дорежет по центру" in p for p in bad), bad)

    def test_a_mask_with_a_channel_axis_is_refused(self):
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            mask=np.zeros((WN, WH, WW, 1), np.float32)))
        self.assertTrue(any("без канала" in p for p in bad), bad)

    def test_a_sequence_of_the_wrong_length_is_named(self):
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            pose_images=np.zeros((WN - 1, WH, WW, 3), np.uint8)))
        self.assertTrue(any("pose_images: кадров 4" in p for p in bad), bad)

    def test_a_missing_input_is_named(self):
        got = wrapper_inputs()
        del got["mask"]
        bad = fk.check_wrapper_inputs(got)
        self.assertTrue(any("mask" in p for p in bad), bad)

    def test_all_violations_come_back_at_once(self):
        bad = fk.check_wrapper_inputs(wrapper_inputs(
            width=WW + 1, face_images=np.zeros((WN, 256, 256, 3), np.uint8)))
        self.assertGreaterEqual(len(bad), 2, bad)

    def test_mutating_the_face_side_reaches_the_check(self):
        """Т1: константа-решение, которая ничего не сторожит, — украшение."""
        # Вход строится ДО подмены: `wrapper_inputs` сам читает FACE_SIDE, и
        # мутация, сдвинувшая заодно и фикстуру, не проверяла бы ничего.
        got = wrapper_inputs()
        saved = fk.FACE_SIDE
        try:
            fk.FACE_SIDE = 256
            self.assertTrue(any("вместо 256x256" in p
                                for p in fk.check_wrapper_inputs(got)),
                            "мутация не доехала")
        finally:
            fk.FACE_SIDE = saved
        self.assertEqual(fk.check_wrapper_inputs(got), [])

    def test_mutating_the_length_step_reaches_the_check(self):
        saved = fk.LENGTH_STEP
        try:
            fk.LENGTH_STEP = 2
            self.assertEqual(fk.snap_frames(4), 3)
            fk.LENGTH_STEP = 8
            self.assertEqual(fk.snap_frames(4), 1)
        finally:
            fk.LENGTH_STEP = saved
        self.assertEqual(fk.snap_frames(4), 1)
        self.assertEqual(fk.snap_frames(5), 5)


class TheWrapperWeightsComeFromTheLockAndTheLoaderCanReadThem(
        unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.derived = fk.derive_wrapper()
        cls.graph = cls.derived["graph"]

    def test_the_graph_and_the_lock_agree_on_all_six_files(self):
        got = fk.audit_weights({"graph": self.graph})
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertEqual(got["loaders_checked"], 6)
        self.assertEqual(got["declared"], 6)

    def test_the_diffusion_stage_is_the_one_the_owner_chose(self):
        """Q4_K_M, и имя приходит из лока — в модуль оно не вписано (Е1)."""
        files = {r["file"] for r in fk.graph_weights(self.graph)}
        self.assertIn("Wan2.2-Animate-14B-Q4_K_M.gguf", files)
        src = Path(fk.__file__).read_text(encoding="utf-8")
        for name in ("Wan2.2-Animate-14B-Q4_K_M.gguf",
                     "umt5-xxl-encoder-Q5_K_M.gguf",
                     "WanAnimate_relight_lora_fp16.safetensors"):
            with self.subTest(name=name):
                self.assertNotIn(f'"{name}"', src,
                                 "имя файла весов вписано в модуль строкой")

    def test_a_lock_without_a_role_falls_over_instead_of_inventing_a_name(self):
        """Ц10: придумать имя файла — ровно тот дефект, который на этом проекте
        уже стоил кода против несуществующего репозитория."""
        lock = json.loads(json.dumps(fk.load_lock()))
        lock["weights"] = [w for w in lock["weights"]
                           if w["role"] != "clip_vision"]
        with self.assertRaises(KeyError) as caught:
            fk.derive_wrapper(lock=lock)
        self.assertIn("clip_vision", str(caught.exception))

    def test_both_adapters_are_parsed_out_of_the_single_multi_node(self):
        """Пять пар в одном узле: «один загрузчик — один файл» здесь неверно."""
        loras = [r for r in fk.graph_weights(self.graph)
                 if r["type"] == "WanVideoLoraSelectMulti"]
        self.assertEqual(len(loras), 2, loras)
        self.assertEqual({r["index"] for r in loras}, {0, 2})

    def test_the_empty_adapter_slots_are_not_called_broken(self):
        """Негативный контроль (И5): три слота `none` — исправное состояние,
        и сторож, красный на каждом исправном графе, снимают целиком.

        Слово `none` — ЛИТЕРАЛ (Т2), а не `fk.EMPTY_LORA_SLOT`. Прогон мутаций
        показал, почему: подмена константы на «пусто» не роняла ничего —
        и граф, и разбор читали одну и ту же константу, то есть сходились
        между собой в чём угодно. Литерал взят из исходника ноды
        (`lora_files = ["none"] + ...`, nodes_model_loading.py:507).
        """
        self.assertEqual(fk.unparsed_loaders(self.graph), [])
        node = next(n for n in self.graph["nodes"]
                    if n["type"] == "WanVideoLoraSelectMulti")
        self.assertEqual(node["widgets_values"][4], "none")
        self.assertEqual(node["widgets_values"][10:], [False, False],
                         "merge_loras обязан быть False: WanVideoSetLoRAs "
                         "иначе роняет прогон, а GGUF слить всё равно нельзя")

    def test_garbage_in_an_adapter_slot_is_still_caught(self):
        """А вот это уже поломка, и молчать про неё нельзя."""
        graph = json.loads(json.dumps(self.graph))
        node = next(n for n in graph["nodes"]
                    if n["type"] == "WanVideoLoraSelectMulti")
        node["widgets_values"][4] = "не файл"
        got = fk.unparsed_loaders(graph)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["got"], "не файл")

    def test_every_loader_can_read_the_format_it_is_handed(self):
        got = fk.audit_loader_formats(self.graph)
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertEqual(got["checked"], 6)

    def test_the_gguf_encoder_in_the_wrappers_own_node_is_caught(self):
        """НАЙДЕНО ЧТЕНИЕМ ИСХОДНИКА, и без этой проверки уехало бы на карту.

        `WanVideoTextEncodeCached` зовёт `LoadWanVideoT5TextEncoder`, а тот
        читает `load_torch_file`. Лок объявляет энкодер в .gguf — узел упал бы
        на загрузке, и `audit_weights` при этом был бы зелёным: имена-то
        сходятся.
        """
        graph = json.loads(json.dumps(self.graph))
        node = next(n for n in graph["nodes"] if n["type"] == "CLIPLoaderGGUF")
        node["type"] = "WanVideoTextEncodeCached"
        self.assertEqual(fk.audit_weights({"graph": graph})["outcome"], fk.PASS,
                         "имена по-прежнему сходятся — значит конфликт формата "
                         "ловит не эта проверка, и вторая нужна")
        got = fk.audit_loader_formats(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("читать умеет только .safetensors", got["note"])
        self.assertEqual(fk.audit_wrapper({"graph": graph})["outcome"], fk.FAIL)

    def test_a_gguf_handed_to_the_vae_loader_is_caught_too(self):
        """Второй вход того же сторожа, и он появился после прогона мутаций:
        расширение списка форматов VAE до .gguf не роняло ничего, потому что
        .gguf в VAE никто не подавал. Проверка, у которой нет входа, где она
        краснеет, ничего не сторожит."""
        graph = json.loads(json.dumps(self.graph))
        node = next(n for n in graph["nodes"] if n["type"] == "WanVideoVAELoader")
        node["widgets_values"][0] = "umt5-xxl-encoder-Q5_K_M.gguf"
        got = fk.audit_loader_formats(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("WanVideoVAELoader", got["note"])

    def test_a_graph_without_loaders_is_unmeasured_not_clean(self):
        """Р2: ноль конфликтов при нуле проверенного — не успех."""
        got = fk.audit_loader_formats({"nodes": []})
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertEqual(got["checked"], 0)
        self.assertIn("не успех", got["note"])

    def test_the_gguf_diffusion_needs_no_third_party_unet_loader(self):
        """Находка чтением: `WanVideoModelLoader` принимает .gguf сам
        (nodes_model_loading.py:1131). На один сторонний пак меньше."""
        types = {n["type"] for n in self.graph["nodes"]}
        self.assertNotIn("UnetLoaderGGUF", types)
        self.assertIn("WanVideoModelLoader", types)
        self.assertIn(".gguf", fk.LOADER_FORMATS["WanVideoModelLoader"])


class TheRenderPicksTheRulesInsteadOfBeingWrittenTwice(unittest.TestCase):
    """Е1: три исхода происхождения, метка в пикселях и обратная проверка на
    подлог у обоих производств одни и те же."""

    @classmethod
    def setUpClass(cls):
        cls.graph = fk.derive_wrapper()["graph"]

    def test_without_a_backend_the_wrapper_path_says_it_is_a_mock(self):
        got = fk.render(self.graph, wrapper_inputs(), kind="wrapper")
        self.assertEqual(got["source"], fk.MOCK)
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertIn("ГЕНЕРАЦИИ НЕ БЫЛО", got["note"])
        self.assertTrue(fk.is_mock(got["frames"]))

    def test_a_backend_returning_marked_frames_is_still_caught(self):
        def liar(graph, inputs):
            return fk.mock_frames(inputs["num_frames"], inputs["height"],
                                  inputs["width"])

        got = fk.render(self.graph, wrapper_inputs(), backend=liar,
                        kind="wrapper")
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("подлог", got["note"])

    def test_an_honest_backend_gets_pass_and_ran(self):
        """НЕПРОВЕРЕНО: «настоящий» бэкенд подставной. На ComfyUI эта ветка не
        исполнялась ни разу — его в этой среде нет."""
        def honest(graph, inputs):
            return np.full((inputs["num_frames"], inputs["height"],
                            inputs["width"], 3), 7, np.uint8)

        got = fk.render(self.graph, wrapper_inputs(), backend=honest,
                        kind="wrapper")
        self.assertEqual((got["source"], got["outcome"]), (fk.RAN, fk.PASS))

    def test_bad_input_returns_no_frames_at_all(self):
        got = fk.render(self.graph, wrapper_inputs(width=WW + 1),
                        kind="wrapper")
        self.assertEqual(got["source"], fk.NOTHING)
        self.assertIsNone(got["frames"])

    def test_the_two_rule_sets_are_not_interchangeable(self):
        """Негативный контроль (И5): граф обёртки, проверенный правилами части
        I, обязан провалиться — иначе `kind` ничего не выбирает."""
        got = fk.render(self.graph, wrapper_inputs())
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertTrue(any("вход:" in p for p in got["problems"]),
                        got["problems"])

    def test_an_unknown_rule_set_falls_over_and_names_the_options(self):
        with self.assertRaises(ValueError) as caught:
            fk.render(self.graph, wrapper_inputs(), kind="выдумка")
        self.assertIn("wrapper", str(caught.exception))




class TheGraphIsTranslatedIntoWhatTheServerAccepts(unittest.TestCase):
    """Дыра, из-за которой e2e не запускался: обе половины зелёные, стыка нет.

    `fork_backend.check_graph` на нашем графе отвечал «формат UI, а /prompt
    принимает формат API». Сборщик и клиент были готовы каждый по себе.
    """

    def _api(self, graph=None):
        return fk.to_api(
            fk.derive_wrapper() if graph is None else graph)

    def test_the_whole_graph_converts(self):
        got = self._api()
        self.assertEqual(got["outcome"], fk.PASS, got["note"])
        self.assertEqual(got["converted"], got["checked"])

    def test_the_backend_no_longer_refuses_it(self):
        """Свидетельство, а не намерение (Е2): судит тот же прибор, что отказал."""
        from ball_reel import fork_backend

        self.assertEqual(fork_backend.check_graph(self._api()["api"]), [])

    def test_every_node_carries_class_type_and_inputs(self):
        for nid, node in self._api()["api"].items():
            with self.subTest(node=nid):
                self.assertIn("class_type", node)
                self.assertIsInstance(node["inputs"], dict)

    def test_links_become_pairs_of_source_and_slot(self):
        """Проверяется КЛАСС УЗЛА-ИСТОЧНИКА, а не то, что ссылка на что-то есть.

        Мутация «развернуть концы связи» пережила первую версию теста: у
        сэмплера вход `model` идёт нулевым слотом, поэтому подстановка
        приёмника вместо источника давала ту же пару чисел и та же проверка
        зеленела. Ссылка, указывающая на сам сэмплер, — это цикл в графе,
        и узнали бы мы о нём на карте.
        """
        api = self._api()["api"]
        sid, sampler = next((k, v) for k, v in api.items()
                            if v["class_type"] == "WanVideoSampler")
        src_id, slot = sampler["inputs"]["model"]
        self.assertNotEqual(src_id, sid, "связь указывает на сам узел")
        self.assertEqual(api[src_id]["class_type"], "WanVideoSetBlockSwap")
        self.assertEqual(slot, 0)
        emb_id, _ = sampler["inputs"]["image_embeds"]
        self.assertEqual(api[emb_id]["class_type"], "WanVideoAnimateEmbeds")

    def test_the_dropped_widgets_are_named_in_the_report(self):
        """Иначе константа с именем дорисованного виджета — украшение.

        Первая версия сверяла отчёт с `fk.UI_ONLY_AFTER_SEED` — то есть с
        импортом из проверяемого модуля, — и мутация имени пережила её ровно
        поэтому: ожидаемое поехало вместе с кодом (Т2). Здесь литерал: так
        зовут виджет во фронтенде ComfyUI, и переименовать его у себя мы не
        вправе.
        """
        dropped = " ".join(self._api()["dropped_ui_widgets"])
        self.assertIn("control_after_generate", dropped)

    def test_only_the_file_taking_nodes_carry_an_upload_button(self):
        """Литералы (Т2): расширить список молча значит съесть чужое значение."""
        self.assertEqual(set(fk.UI_UPLOAD_NODES), {"LoadImage", "LoadVideo"})

    def test_a_string_where_an_integer_is_declared_is_a_failure(self):
        graph = fk.derive_wrapper()
        node = next(n for n in graph["graph"]["nodes"]
                    if n["type"] == "WanVideoDecode")
        node["widgets_values"][1] = "272"          # tile_x объявлен INT
        got = fk.to_api(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIn("tile_x", got["problems"][0])

    def test_a_boolean_where_an_integer_is_declared_is_a_failure(self):
        """True в Python — целое; без явной проверки это пролезает."""
        graph = fk.derive_wrapper()
        node = next(n for n in graph["graph"]["nodes"]
                    if n["type"] == "WanVideoDecode")
        node["widgets_values"][1] = True
        self.assertEqual(fk.to_api(graph)["outcome"], fk.FAIL)

    def test_the_widget_values_land_under_their_own_names(self):
        """Литералы (Т2). Съедет разбор — тест покраснеет, а не поедет следом."""
        api = self._api()["api"]
        sampler = next(n for n in api.values()
                       if n["class_type"] == "WanVideoSampler")
        self.assertEqual(sampler["inputs"]["steps"], 3)
        self.assertEqual(sampler["inputs"]["scheduler"], "dpm++_sde")
        self.assertEqual(sampler["inputs"]["riflex_freq_index"], 0)

    def test_the_interface_only_widgets_are_dropped(self):
        """Не выкинув их, мы сдвинули бы ВСЕ последующие значения на позицию.

        Две штуки дорисовывает фронтенд: `control_after_generate` после
        целого виджета `seed` (у нас всплыло на сэмплере — 14 значений против
        13 объявленных) и кнопка загрузки у нод, принимающих файл.
        """
        got = self._api()
        self.assertEqual(len(got["dropped_ui_widgets"]), 6)
        api = got["api"]
        sampler = next(n for n in api.values()
                       if n["class_type"] == "WanVideoSampler")
        self.assertNotIn("fixed", sampler["inputs"].values())
        self.assertIs(sampler["inputs"]["force_offload"], True,
                      "после зерна не выкинуто дорисованное — значение "
                      "уехало на позицию вправо")

    def test_a_type_that_does_not_match_the_declaration_is_a_failure(self):
        """Негативный контроль (И5) на настоящем дефекте.

        Именно так нашлось, что `batched_cfg` (BOOLEAN) получал пустую строку:
        в формате UI имён нет, и значение просто лежало в списке.
        """
        graph = fk.derive_wrapper()
        node = next(n for n in graph["graph"]["nodes"]
                    if n["type"] == "WanVideoSampler")
        node["widgets_values"][9] = "не булево"
        got = fk.to_api(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertIsNone(got["api"], "негодный перевод отдан наружу")
        self.assertIn("batched_cfg", got["problems"][0])

    def test_an_unknown_node_type_is_unmeasured_not_guessed(self):
        """Позиционная догадка стоит не ошибки сборки, а неверного ролика.

        ИМЯ ТЕСТА БЫЛО ПРАВИЛЬНЫМ, А ПРОВЕРКА — НЕТ: она требовала «не годно»
        там, где по смыслу «не смогли перевести». Разница не словесная. Любая
        нода из нового пака реестру неизвестна, и вердикт «граф НЕ ГОДЕН»
        отправлял бы следующую смену искать дефект в графе, которого там нет,
        — тогда как правильный ответ «пополнить реестр имён».
        """
        graph = fk.derive_wrapper()
        graph["graph"]["nodes"][0]["type"] = "ЧегоТакогоНетВРеестре"
        got = fk.to_api(graph)
        self.assertEqual(got["outcome"], fk.UNMEASURED)
        self.assertIn("не разобран", got["unknown_types"][0])
        self.assertEqual(got["problems"], [],
                         "неизвестный тип попал в список НЕГОДНОГО — тогда он "
                         "неотличим от значения не того типа")
        self.assertIsNone(got["api"], "недопереведённый граф отдан наружу")

    def test_a_bad_value_is_still_a_failure_not_an_unknown(self):
        """Негативный контроль к предыдущему: два случая обязаны различаться."""
        graph = fk.derive_wrapper()
        node = next(n for n in graph["graph"]["nodes"]
                    if n["type"] == "WanVideoDecode")
        node["widgets_values"][1] = "не целое"
        got = fk.to_api(graph)
        self.assertEqual(got["outcome"], fk.FAIL)
        self.assertEqual(got["unknown_types"], [])

    def test_too_many_values_are_refused_rather_than_truncated(self):
        graph = fk.derive_wrapper()
        node = next(n for n in graph["graph"]["nodes"]
                    if n["type"] == "WanVideoDecode")
        node["widgets_values"].extend([1, 2, 3])
        got = fk.to_api(graph)
        self.assertEqual(got["outcome"], fk.FAIL)

    def test_something_that_is_not_a_ui_graph_is_unmeasured(self):
        got = fk.to_api({"нет": "узлов"})
        self.assertEqual(got["outcome"], fk.UNMEASURED,
                         "ноль ошибок при нуле переведённого прочтено как "
                         "успех — Р2 нарушено")

    def test_the_widget_names_come_from_hashed_sources(self):
        """Ц10: внешнее имя доказывается командой, а не памятью модели."""
        self.assertGreaterEqual(len(fk.WIDGET_SOURCES), 7)
        for name, sha in fk.WIDGET_SOURCES.items():
            with self.subTest(source=name):
                self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_every_node_type_the_builder_uses_has_proven_names(self):
        """Иначе новый узел проедет в граф, а перевод его молча не осилит."""
        used = {n["type"] for n in fk.derive_wrapper()["graph"]["nodes"]}
        missing = sorted(used - set(fk.API_WIDGETS))
        self.assertEqual(missing, [], f"имён виджетов нет для: {missing}")





class TheWidgetNamesAgreeWithARealComfyExport(unittest.TestCase):
    """ВТОРОЙ независимый источник имён, и он обязан сойтись с первым.

    Первый — разбор исходников нод. Второй — настоящий экспорт ComfyUI,
    присланный владельцем: его фронтенд кладёт рядом с позиционным списком
    ещё и `widgets_values_named`, то есть сам называет каждое значение.
    Два источника, снятых разными способами; расхождение между ними — это
    находка, а не шум. Так и вышло: `clip` у `CLIPTextEncode` разбор принял
    за виджет, а эталон знает у него ровно один — `text`.
    """

    REF = Path("workflows/fork_widget_names.reference.json")

    def setUp(self):
        if not self.REF.exists():
            self.skipTest(f"нет эталона {self.REF} — сверять не с чем")
        self.ref = json.loads(self.REF.read_text(encoding="utf-8"))["имена"]

    def test_no_declared_widget_is_missing_from_the_real_export(self):
        bad = []
        for node_type, (pairs, _req) in fk.API_WIDGETS.items():
            theirs = self.ref.get(node_type)
            if theirs is None:
                continue
            for name, _kind in pairs:
                if name not in theirs:
                    bad.append(f"{node_type}.{name}")
        self.assertEqual(bad, [], f"в реестре есть имена, которых настоящий "
                                  f"ComfyUI не знает: {bad}")

    def test_the_export_confirms_the_upload_pseudo_widget(self):
        """Именно это правило выкидывает лишнее значение у нод с файлом."""
        self.assertIn("upload", self.ref["LoadImage"])

    def test_the_export_confirms_control_after_generate_follows_the_seed(self):
        """Независимое подтверждение правила, найденного на нашем сэмплере."""
        ksampler = self.ref["KSamplerAdvanced"]
        self.assertEqual(ksampler[ksampler.index("noise_seed") + 1],
                         "control_after_generate")

    def test_the_reference_covers_enough_to_be_worth_calling_a_check(self):
        self.assertGreaterEqual(len(self.ref), 10,
                                "эталон слишком мал — сверка ничего не ловит")





class TheApiFormatMatchesARealComfyExport(unittest.TestCase):
    """ЭТАЛОН формата API: настоящий экспорт `Export (API)`, присланный
    владельцем 18.08.2026 (`workflows/fork_api_format.reference.json`,
    sha256 9ac395ac57fc0e66…, 32 узла).

    До него формат был известен по документации и по чтению `server.py`.
    Теперь есть образец, сделанный самим ComfyUI, — и три наших правила
    подтверждаются им НЕЗАВИСИМО, а не нашими же рассуждениями.
    """

    REF = Path("workflows/fork_api_format.reference.json")

    def setUp(self):
        if not self.REF.exists():
            self.skipTest(f"нет эталона {self.REF}")
        self.ref = json.loads(self.REF.read_text(encoding="utf-8"))

    def test_our_records_have_the_same_shape_as_the_real_ones(self):
        theirs = {k for node in self.ref.values() for k in node}
        ours = {k for node in fk.to_api(fk.derive_wrapper())["api"].values()
                for k in node}
        self.assertEqual(ours, theirs)

    def test_the_real_export_has_no_control_after_generate(self):
        """Подтверждение выкидывания дорисованного — не наше рассуждение.

        В формате интерфейса эта строка у `KSamplerAdvanced` стоит сразу за
        зерном (видно в `fork_widget_names.reference.json`). В формате API её
        нет ни у одного узла. Значит правило верное, и проверено оно образцом,
        а не нами.
        """
        self.assertNotIn("control_after_generate",
                         {k for n in self.ref.values() for k in n["inputs"]})

    def test_the_real_export_has_no_upload_widget_either(self):
        loaders = [n for n in self.ref.values()
                   if n["class_type"] == "LoadImage"]
        self.assertTrue(loaders, "в эталоне нет LoadImage — проверка ищет не то")
        self.assertEqual(set(loaders[0]["inputs"]), {"image"})

    def test_links_are_written_the_same_way_we_write_them(self):
        pairs = [v for n in self.ref.values() for v in n["inputs"].values()
                 if isinstance(v, list) and len(v) == 2]
        self.assertTrue(pairs)
        for src, slot in pairs:
            with self.subTest(link=(src, slot)):
                self.assertIsInstance(src, str)
                self.assertIsInstance(slot, int)

    def test_the_backend_accepts_the_real_export(self):
        """Негативный контроль наоборот: прибор обязан пропускать годное.

        `check_graph` мы писали, глядя на свой граф. Если бы он браковал
        настоящий экспорт ComfyUI, это значило бы, что он проверяет наши
        привычки, а не формат.
        """
        from ball_reel import fork_backend

        self.assertEqual(fork_backend.check_graph(self.ref), [])

    def test_no_widget_in_the_real_export_is_unknown_to_our_registry(self):
        """Расхождение здесь — находка: либо у нас лишнее имя, либо не хватает."""
        bad = []
        for nid, node in self.ref.items():
            spec = fk.API_WIDGETS.get(node["class_type"])
            if spec is None:
                continue
            ours = {n for n, _ in spec[0]}
            theirs = {k for k, v in node["inputs"].items()
                      if not (isinstance(v, list) and len(v) == 2)}
            extra = theirs - ours
            if extra:
                bad.append(f"{node['class_type']}#{nid}: {sorted(extra)}")
        self.assertEqual(bad, [])





class TheOutputRateFollowsTheDrivingBelowThirty(unittest.TestCase):
    """Правило владельца 20.08.2026: ниже 30 наследуем, 30 и выше фиксируем.

    ~~«Драйвинг не ниже 30, вверх не приводим»~~ снято: то правило выросло из
    посылки «все сурсы будут выше 24 кадра», а первый же боевой ролик проекта
    снят на 24. Правило отвергало собственный материал.

    Литералы здесь — числа решения владельца (Т2), импортировать их из
    проверяемого модуля значило бы сверять значение само с собой.
    """

    def test_below_thirty_is_inherited(self):
        for src in (12, 24, 25, 29.97):
            with self.subTest(fps=src):
                got = fk.output_fps(src)
                self.assertEqual(got["fps"], src)
                self.assertTrue(got["inherited"])

    def test_thirty_exactly_is_fixed_not_inherited(self):
        """Граница именно здесь, и она названа в правиле словом «и выше»."""
        got = fk.output_fps(30)
        self.assertEqual(got["fps"], 30.0)
        self.assertFalse(got["inherited"])

    def test_above_thirty_is_capped(self):
        for src in (50, 60, 120):
            with self.subTest(fps=src):
                self.assertEqual(fk.output_fps(src)["fps"], 30.0)

    def test_an_unknown_rate_is_unmeasured_and_never_thirty(self):
        """Пережило первый заход мутаций: свёртка None в 30 не краснела.

        Подставленная частота молча растянет или ускорит движение — и в
        отчёте это сойдётся, потому что все дальнейшие числа посчитаны по
        ней же. Единственный честный ответ — «назвать нечем».
        """
        got = fk.output_fps(None)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["fps"])
        self.assertIn("НЕ «берём 30»", got["note"])

    def test_zero_and_negative_are_refused_not_inherited(self):
        """Тоже пережило первый заход: ноль проходил как годная частота.

        Ноль означает «метаданные не прочитаны», а не «ролик мгновенный», и
        унаследовать его значит поделить на него ниже по пути.
        """
        for bad in (0, -1, -30):
            with self.subTest(fps=bad):
                with self.assertRaises(ValueError):
                    fk.output_fps(bad)

    def test_a_string_is_refused(self):
        for bad in ("тридцать", True):
            with self.subTest(fps=bad):
                with self.assertRaises(TypeError):
                    fk.output_fps(bad)

    def test_the_note_says_what_the_capping_costs(self):
        """Оператор должен видеть, за что платит, а не только что решено."""
        self.assertIn("2.00x", fk.output_fps(60)["note"])

    def test_the_note_of_an_exact_match_does_not_talk_about_thinning(self):
        """Негативный контроль (И5): при 30 прореживать нечего."""
        self.assertNotIn("прореживаются", fk.output_fps(30)["note"])



if __name__ == "__main__":
    unittest.main()
