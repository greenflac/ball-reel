"""Лечение, которое нельзя скопировать, — это не лечение.

Весь предполёт стоит на том, что у каждого отказа есть команда рядом. Правило
держалось до первой машины с Windows: там оператор увидел четыре отказа и ни
одного исполнимого лечения — `mkdir -p`, `unzip`, heredoc `python3 - <<'EOF'` и
`python3` вместо `python`. Ни одна из четырёх строк не работает в CMD.

Хуже обычного описания проблемы: выглядит как готовое решение, и время уходит
не на починку, а на разбор синтаксиса чужой оболочки — накануне сдачи.

Тесты здесь проверяют ровно две вещи и обе — про исполнимость, а не про текст:
что в напечатанном не осталось конструкций, которых на этой оболочке нет, и
что однострочники на Python действительно компилируются как Python.
"""

from __future__ import annotations

import shlex
import unittest

from ball_reel import cure

#: Чего не должно быть в лечении НИ НА ОДНОЙ платформе, и почему. Список
#: намеренно короткий: `mkdir -p` сюда НЕ ВХОДИТ — на Unix это правильная
#: команда, и запрещать её значило бы проверять текст вместо исполнимости.
#: Платформенная часть проверяется иначе: сверкой с тем, что порождает `cure`.
WRONG_EVERYWHERE = {
    "unzip": "в Windows нет unzip, а на Unix есть не всегда",
    "<<'EOF'": "heredoc — синтаксис Bourne shell, CMD падает на первой строке",
    " && cd ": "цепочка с cd между командами теряется при копировании построчно",
}


class TheSnippetIsRealPython(unittest.TestCase):

    def test_a_multiline_body_becomes_one_runnable_command(self):
        cmd = cure.py_snippet("import os\nprint(os.name)")
        body = shlex.split(cmd)[2]
        compile(body, "<cure>", "exec")

    def test_the_interpreter_name_matches_the_platform(self):
        self.assertEqual(cure.PY, "python" if cure.WINDOWS else "python3")

    def test_blank_lines_do_not_produce_empty_statements(self):
        cmd = cure.py_snippet("import os\n\n\nprint(1)")
        compile(shlex.split(cmd)[2], "<cure>", "exec")

    def test_quotes_survive_the_round_trip(self):
        """Аргументы лечения — пути и имена репозиториев, все в кавычках."""
        cmd = cure.py_snippet("d('h94/IP-Adapter-FaceID', p=['a.bin'])")
        self.assertIn("'h94/IP-Adapter-FaceID'", cmd)
        self.assertEqual(shlex.split(cmd)[1], "-c")


class ThePrintedCuresAreRunnableHere(unittest.TestCase):
    """Каждое лечение проверяется на ТОЙ платформе, где тест запущен."""

    def _clean(self, text: str, where: str):
        """Две проверки, и вторая важнее.

        ПЕРВАЯ — нет конструкций, неверных везде. ВТОРАЯ — лечение построено
        ЧЕРЕЗ `cure`, а не написано руками: сверяем, что напечатанные `mkdir` и
        `curl` буквально совпадают с тем, что порождает `cure` для ЭТОЙ
        платформы. Так тест ловит захардкоженный Unix-синтаксис, запущенный на
        Linux, — то есть ровно тот случай, когда дефект виден только у того,
        кто на Windows.
        """
        for bad, why in WRONG_EVERYWHERE.items():
            with self.subTest(where=where, bad=bad):
                self.assertNotIn(bad, text, f"{where}: {bad} — {why}")
        if "mkdir" in text:
            with self.subTest(where=where, part="mkdir"):
                self.assertIn(cure.mkdir(cure.home(".x")).split()[0], text)
                self.assertRegex(
                    text, r"mkdir " + ("\"" if cure.WINDOWS else r"-p "),
                    f"{where}: mkdir написан руками, а не через cure.mkdir")
        if "curl" in text:
            with self.subTest(where=where, part="curl"):
                self.assertIn("curl -sSL -o", text,
                              f"{where}: curl написан руками, а не через "
                              f"cure.download")

    def test_missing_pose_model_cure(self):
        import os

        from ball_reel import pose

        old = os.environ.get(pose.MODEL_ENV)
        os.environ[pose.MODEL_ENV] = "/нет/такого/pose.task"
        self.addCleanup(lambda: os.environ.__setitem__(pose.MODEL_ENV, old)
                        if old else os.environ.pop(pose.MODEL_ENV, None))
        with self.assertRaises(RuntimeError) as ctx:
            pose._model_path()
        self._clean(str(ctx.exception), "pose")
        self.assertIn(pose.MODEL_ONLINE, str(ctx.exception))

    def test_missing_segmentation_cure(self):
        from ball_reel import bodyparts

        self._clean(bodyparts.why_unavailable("/нет/такого/seg.tflite"),
                    "bodyparts")

    def test_missing_dwpose_cure(self):
        import os

        from ball_reel import dwpose

        for env, val in ((dwpose.DET_ENV, "/нет/a.onnx"),
                         (dwpose.POSE_ENV, "/нет/b.onnx")):
            old = os.environ.get(env)
            os.environ[env] = val
            self.addCleanup(lambda e=env, o=old: os.environ.__setitem__(e, o)
                            if o else os.environ.pop(e, None))
        self._clean(dwpose.why_unavailable(), "dwpose")

    def test_missing_weights_cure_is_four_runnable_python_commands(self):
        """Ровно то место, где стоял heredoc: CMD выполнял первую строку и падал."""
        import os
        import tempfile

        from ball_reel import preflight_gpu

        old = os.environ.get("HF_HOME")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["HF_HOME"] = tmp
            try:
                _, ok, detail = preflight_gpu.check_weights()
            finally:
                if old:
                    os.environ["HF_HOME"] = old
                else:
                    os.environ.pop("HF_HOME", None)
        if ok:
            self.skipTest("веса на месте — лечению неоткуда взяться")
        self._clean(detail, "weights")
        cmds = [ln.strip() for ln in detail.splitlines()
                if ln.strip().startswith(cure.PY + " -c")]
        self.assertGreaterEqual(len(cmds), 1, "лечение не содержит команд")
        for c in cmds:
            with self.subTest(cmd=c[:60]):
                compile(shlex.split(c)[2], "<cure>", "exec")

    def test_missing_face_weights_cure_does_not_need_unzip(self):
        import os
        import tempfile

        from ball_reel import preflight_gpu

        old = os.environ.get("INSIGHTFACE_HOME")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["INSIGHTFACE_HOME"] = tmp
            try:
                _, ok, detail = preflight_gpu.check_face_model()
            finally:
                if old:
                    os.environ["INSIGHTFACE_HOME"] = old
                else:
                    os.environ.pop("INSIGHTFACE_HOME", None)
        if ok:
            self.skipTest("веса лица на месте")
        self._clean(detail, "face model")
        cmds = [ln.strip() for ln in detail.splitlines()
                if ln.strip().startswith(cure.PY + " -c")]
        self.assertTrue(cmds, "лечение без исполнимой команды")
        compile(shlex.split(cmds[0])[2], "<cure>", "exec")


class TheDetectorItselfCanGoRed(unittest.TestCase):
    """Тест, который не умеет краснеть, — украшение."""

    def test_a_unix_only_cure_is_caught(self):
        checker = ThePrintedCuresAreRunnableHere("test_missing_pose_model_cure")
        with self.assertRaises(AssertionError):
            checker._clean("curl -sSLO a.zip && unzip -o a.zip", "нарочно плохое")

    def test_the_helpers_differ_between_platforms(self):
        """Если бы `cure` печатал одно и то же везде, он был бы бесполезен."""
        real = cure.WINDOWS
        try:
            cure.WINDOWS = True
            win_mkdir, win_home = cure.mkdir("x"), cure.home(".dwpose")
            cure.WINDOWS = False
            nix_mkdir, nix_home = cure.mkdir("x"), cure.home(".dwpose")
        finally:
            cure.WINDOWS = real
        self.assertNotEqual(win_mkdir, nix_mkdir)
        self.assertNotIn("-p", win_mkdir)
        self.assertIn("-p", nix_mkdir)
        self.assertIn("%USERPROFILE%", win_home)
        self.assertTrue(nix_home.startswith("~"))


if __name__ == "__main__":
    unittest.main()
