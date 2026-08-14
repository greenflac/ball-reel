"""Кодировка файлов задаётся явно. Иначе она задаётся локалью машины.

НАЙДЕНО НА ЖИВОЙ МАШИНЕ, и найдено дорого: предполёт прошёл девять проверок и
встал на десятой — `манифест demo/kit/conditions/manifest.json не читается
(UnicodeDecodeError)`. Файл был цел, лежал в git, открывался на машине автора.

Причина в том, что `Path.read_text()` без аргумента берёт кодировку из локали.
На Linux это UTF-8, и всё работает. На Windows — cp1251, и первая же русская
строка в манифесте роняет чтение. Кодировка при этом НЕ хранится в файле: она
свойство того, кто открывает, и разойтись две машины могут молча.

Форма дефекта та же, что у Unix-команд в лечениях предполёта: на машине автора
невидим ПО УСТРОЙСТВУ. Проверяющий получает отказ там, где автор видел успех, и
никакой прогон у автора этого не покажет.

Поэтому тест не про поведение, а про исходники: ни один вызов чтения или записи
текста в пакете не имеет права молчать о кодировке. Разбор по AST, а не по
подстроке: `read_text()` встречается в комментариях и докстрингах, и подстрочная
версия зеленела бы на них.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent

#: Методы, которые молча берут кодировку из локали. `open()` сюда же, когда его
#: зовут в текстовом режиме.
TEXTUAL = ("read_text", "write_text")


def _sources() -> list:
    """Файлы пакета, кроме тестов: тесты работают на временных файлах."""
    return [p for p in PKG.rglob("*.py") if "tests" not in p.parts]


def _calls_without_encoding(path: Path) -> list:
    """[(строка, метод)] — вызовы, не назвавшие кодировку."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else "")
        if name in TEXTUAL:
            if not any(k.arg == "encoding" for k in node.keywords):
                bad.append((node.lineno, name))
        elif name == "open" and isinstance(fn, ast.Name):
            # ТОЛЬКО встроенный `open`, а не `Image.open`, `io.open` или любой
            # чужой метод с тем же именем. Первая редакция этого не различала и
            # выдала 29 ложных срабатываний подряд — все на `Image.open(path)`,
            # который двоичный и кодировки не принимает вовсе. Ложная тревога
            # здесь стоит дороже пропуска: тест, который кричит на исправном
            # коде, выключают целиком.
            # Двоичный режим кодировки не имеет — и не должен её называть.
            args = [a for a in node.args[1:2]
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            mode = args[0].value if args else "r"
            if "b" not in mode and not any(k.arg == "encoding"
                                           for k in node.keywords):
                bad.append((node.lineno, f"open(mode={mode!r})"))
    return bad


class EveryTextFileNamesItsEncoding(unittest.TestCase):

    def test_no_call_leaves_the_encoding_to_the_locale(self):
        offenders = []
        for p in _sources():
            for line, what in _calls_without_encoding(p):
                offenders.append(f"{p.relative_to(PKG.parent)}:{line} {what}")
        self.assertEqual(
            offenders, [],
            "кодировка отдана локали машины — на Windows это cp1251, и файл с "
            f"русским текстом не прочитается: {offenders}")

    def test_the_detector_finds_a_deliberate_omission(self):
        """Тест, который не умеет краснеть, — украшение."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.py"
            bad.write_text(
                "from pathlib import Path\n"
                "def f(p):\n"
                "    return Path(p).read_text()\n", encoding="utf-8")
            self.assertEqual(len(_calls_without_encoding(bad)), 1)

    def test_the_detector_accepts_a_correct_call(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.py"
            good.write_text(
                "from pathlib import Path\n"
                "def f(p):\n"
                "    return Path(p).read_text(encoding='utf-8')\n",
                encoding="utf-8")
            self.assertEqual(_calls_without_encoding(good), [])

    def test_image_open_is_not_the_builtin_open(self):
        """29 ложных срабатываний первой редакции — все на `Image.open`."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            i = Path(tmp) / "i.py"
            i.write_text("from PIL import Image\n"
                         "def f(p):\n    return Image.open(p).convert('RGB')\n",
                         encoding="utf-8")
            self.assertEqual(_calls_without_encoding(i), [])

    def test_binary_open_is_not_flagged(self):
        """У двоичного режима кодировки нет, и требовать её — ложная тревога."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp) / "b.py"
            b.write_text("def f(p):\n    return open(p, 'rb').read()\n",
                         encoding="utf-8")
            self.assertEqual(_calls_without_encoding(b), [])

    def test_a_substring_check_would_have_missed_it(self):
        """Почему AST: имя метода встречается в докстрингах и комментариях."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "d.py"
            d.write_text('"""Здесь про read_text() в докстринге."""\n'
                         "# и в комментарии: write_text(x)\n", encoding="utf-8")
            self.assertEqual(_calls_without_encoding(d), [])


class TheShippedManifestsAreReadableAnywhere(unittest.TestCase):
    """Мало объявить кодировку — файлы в git должны ей соответствовать."""

    def test_kit_manifests_decode_as_utf8(self):
        import json

        found = 0
        for name in ("kit", "kit_waist"):
            p = PKG.parent / "demo" / name / "conditions" / "manifest.json"
            if not p.exists():
                continue
            found += 1
            with self.subTest(kit=name):
                data = json.loads(p.read_text(encoding="utf-8"))
                self.assertIn("conditions", data)
        if not found:
            self.skipTest("китов нет в этом дереве")


if __name__ == "__main__":
    unittest.main()
