"""Аудит кода: тесты, покрытие и проверка того, что тесты вообще кусаются.

    python3 -m ball_reel.codeaudit            # всё
    python3 -m ball_reel.codeaudit --quick    # без мутаций

Зелёные тесты не доказывают, что тесты хорошие. Набор из ста проверок, каждая
из которых утверждает «функция не упала», даёт то же самое «OK», что и набор,
который действительно сторожит поведение, — и отличить их можно только одним
способом: сломать код нарочно и посмотреть, покраснеет ли что-нибудь.

Это та же дисциплина, что и в остальном проекте. Гейт не верит генератору на
слово и меряет результат; аудит не верит тестам на слово и меряет их. Мутация,
которая выжила, — это место, где мы думаем, что защищены, а на самом деле нет.

Мутации выбраны не случайно: это ПОРОГИ И МЕРЫ, на которых стоят вердикты
пайплайна. Если снятый порог не роняет ни одного теста, значит этот порог
нигде не проверяется, и завтра его можно молча сдвинуть.

Ничего не генерирует, в сеть не ходит, денег не тратит.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: Порог покрытия для модулей, которые несут вердикты. Для остальных (CLI,
#: обёртки над сетью) покрытие мало о чём говорит, и требовать его — значит
#: писать тесты ради числа.
CORE_MODULES = ("identity_arcface", "identity", "motion", "pose", "intake",
                "router", "subject", "skeleton", "driving")

#: Порог намеренно не 90 и не 100. Остаток непокрытого в `driving`, `pose` и
#: `identity_arcface` — это обёртки над моделью и ffmpeg (`extract`,
#: `face_detail`, `landmarks`, `_analyzer`): офлайн они не исполняются вообще,
#: и «покрыть» их можно только заглушкой, которая проверяет заглушку.
#: Гнать число вверх такими тестами — обманывать себя ровно тем способом,
#: против которого написан весь этот модуль.
#:
#: Поэтому строчное покрытие здесь — гигиенический минимум («модуль вообще
#: запускается тестами»), а настоящий сигнал даёт мутационная секция ниже:
#: она проверяет не сколько строк исполнилось, а сторожит ли их хоть кто-то.
MIN_CORE_COVERAGE = 50

#: (что ломаем, на что, чем это притворяется). Каждая мутация — правдоподобная
#: ошибка, а не абсурд: сдвинутый порог, выключенная проверка, мера, которая
#: всегда говорит «хорошо».
MUTATIONS = [
    ("ball_reel.motion", "SEAMLESS_MAX", 999.0,
     "порог стыка лупа снят — «зациклилось» становится всегда истиной"),
    ("ball_reel.motion", "JUMP_MAX", 1e9,
     "детектор телепорта выключен — рваное движение проходит"),
    ("ball_reel.motion", "STILL_MIN", -1.0,
     "порог «ничего не происходит» снят — статика считается движением"),
    ("ball_reel.pose", "SAME_POSE_MAX", 999.0,
     "поза старт-кадра больше не сверяется с референсом"),
    ("ball_reel.pose", "WORST_JOINT_MAX", 999.0,
     "уехавшая рука перестаёт ловиться (среднее её размывает)"),
    ("ball_reel.pose", "POSE_WANDER_MAX", 999.0,
     "клип может уйти в другую позу — не заметим"),
    ("ball_reel.pose", "LIMB_WOBBLE_MAX", 999.0,
     "резиновые конечности объявляются анатомией"),
    ("ball_reel.pose", "MIN_VISIBILITY", -1.0,
     "невидимые суставы начинают участвовать в измерении"),
    ("ball_reel.identity_arcface", "SAME_PERSON_MAX", 999.0,
     "любое лицо признаётся тем же человеком"),
    ("ball_reel.identity_arcface", "HARD_DRIFT_MAX", 999.0,
     "клип может уехать в другого человека к концу"),
    ("ball_reel.identity_arcface", "MIN_COVERAGE", 0.0,
     "«нечего было судить» превращается в «прошло»"),
    ("ball_reel.intake", "MIN_SOURCE_FACE_PX", 0,
     "фото с нечитаемым лицом принимается в работу"),
    ("ball_reel.intake", "MIN_DETECTOR_SCORE", 0.0,
     "неуверенная детекция больше не отмечается"),
    ("ball_reel.chain", "KEYFRAME_POSE_MAX", 999.0,
     "кейфрейм принимается, даже если позу не воспроизвёл"),
    ("ball_reel.dwpose", "MIN_SCORE", -1.0,
     "ненаблюдаемые точки DWPose начинают попадать в условия"),
    ("ball_reel.identity_arcface", "START_MIN_FACE_PX", 0,
     "старт-кадр с нечитаемым лицом идёт в дорогой видео-вызов"),
    ("ball_reel.intake", "MIN_MESH_FACE_PX", 0,
     "мимика и геометрия объявляются снятыми с нечитаемого лица"),
    ("ball_reel.garment", "GARMENT_DRIFT_MAX", 999.0,
     "поплывшая между кейфреймами одежда объявляется постоянной"),
    ("ball_reel.garment", "CORE_FRACTION", 2.0,
     "цвет одежды меряется вместе с фоном по краям области"),
    ("ball_reel.gpu_keyframes", "CLIP_TOKEN_LIMIT", 10000,
     "промт длиннее энкодера уходит в генерацию и молча обрезается"),
    ("ball_reel.dwpose", "BOX_PADDING", 1.0,
     "кроп без запаса режет конечности у края бокса"),
    ("ball_reel.preflight_gpu", "MIN_VRAM_GB", 0.0,
     "предполёт пропускает карту, на которой генерация не поедет"),
    ("ball_reel.marks", "MIN_MARK_PX", 0,
     "примета судится по пятну 5x5 — там нет контраста, только шум сжатия"),
    ("ball_reel.marks", "MIN_REFERENCE_CONTRAST", 0.0,
     "ровная кожа на референсе объявляется приметой, и её «потерю» вменяют генератору"),
    ("ball_reel.marks", "PRESENT_RATIO", 0.0,
     "исчезнувшая примета объявляется перенесённой"),
    ("ball_reel.marks", "LIMB_HALF_WIDTH", 99.0,
     "вклейка выходит за конечность — примета ложится на фон и одежду"),
    ("ball_reel.marks", "SURROUND_SCALE", 1.0,
     "кольцо кожи совпадает с приметой — контраст меряется сам с собой"),
    ("ball_reel.marks", "NOISE_FLOOR", 0.0,
     "чернилами считается каждый пиксель — coverage перестаёт быть диагностикой"),
    ("ball_reel.marks", "MIN_RING_PX", 0,
     "опорой становится горсть пикселей: медиана пляшет сильнее измеряемого"),
    ("ball_reel.marks", "MAX_BASELINE_SPREAD", 99.0,
     "загрязнённая опора больше не помечается — фон молча работает кожей"),
    ("ball_reel.marks", "SIGN_MIN", 99.0,
     "направление контраста не проверяется никогда: светлое пятно на месте "
     "тёмной татуировки проходит как «примета на месте»"),
    ("ball_reel.bodyparts", "MIN_SKIN_SHARE", 0.0,
     "примета вклеивается поверх рукава: доля кожи в окне больше ничего не решает"),
    ("ball_reel.bodyparts", "NATIVE_SIDE", 100000,
     "грубость маски объявляется нулевой — край вклейки смягчается на ничто"),
    ("ball_reel.fluid", "MIN_REGION_PX", 0,
     "реализм жидкости судится по пятну 30x40 — там нет статистики"),
    ("ball_reel.fluid", "HIGHLIGHT_SIGMAS", 0.0,
     "бликом объявляется каждый пиксель — метрика мерит фон"),
    # Шкалы осей расстояния. Раньше здесь стояло деление на сам эталон, и от
    # этого «жидкости нет вообще» оказывалось БЛИЖЕ к референсу, чем тот же
    # самый гель. Сняв шкалу, ось перестаёт различать что бы то ни было.
    ("ball_reel.fluid", "HIGHLIGHT_SHARE_SCALE", 999.0,
     "площадь блика перестаёт влиять на расстояние — матовое и зеркальное равны"),
    ("ball_reel.fluid", "HIGHLIGHT_RATIO_SCALE", 999.0,
     "яркость блика перестаёт влиять — тусклое и сияющее неразличимы"),
    ("ball_reel.fluid", "HIGHLIGHT_DESAT_SCALE", 999.0,
     "обесцвечивание блика перестаёт влиять — исчезает признак «мокрое»"),
    ("ball_reel.fluid", "DETAIL_SCALE", 999.0,
     "мелкая структура перестаёт влиять — размазанное пятно равно фактуре"),
    ("ball_reel.fluid", "LUMA_QUANTUM", 0.0,
     "деление на почти чёрное больше не ограничено — ratio улетает в миллионы"),
    ("ball_reel.fluid", "FLAT_SPAN_FRACTION", 1.0,
     "однотонная область больше не объявляется неизмеримой, а судится"),
    ("ball_reel.bench", "MIN_SESSIONS_FOR_YIELD", 0,
     "доля по одной сессии подаётся как результат — «1/1, 100%»"),
    ("ball_reel.router", "API_CANNOT", (),
     "траектория маршрутизируется на шлюз, который её не умеет"),
    ("ball_reel.timing", "REALTIME_BUDGET_MS", 1e9,
     "бюджет латентности снят — всё объявляется уложившимся в реальное время"),
    ("ball_reel.timing", "HOPELESS_FACTOR", 1e9,
     "безнадёжный этап выдаётся за поправимый настройками"),
    ("ball_reel.timing", "MIN_TAIL_SAMPLES", 0,
     "p95 по трём замерам объявляется основанием для вердикта"),
    ("ball_reel.timing", "REALTIME_BUDGET_MS", 1e9,
     "первая ступень лестницы объявляется мгновенной при любой задержке"),
    ("ball_reel.timing", "BORDERLINE_BAND", 0.0,
     "этап на самой черте бюджета получает вердикт, который меняется от прогона к прогону"),
]

def _mutate_source(text: str, name: str, value) -> str:
    """Подменить объявление константы В ИСХОДНИКЕ.

    Именно в исходнике, а не через setattr на импортированном модуле. Половина
    порогов здесь используется как значение по умолчанию (`max_wobble: float =
    LIMB_WOBBLE_MAX`), а такие связываются один раз в момент определения
    функции — подмена атрибута модуля после импорта на них не действует.
    Харнесс, который этого не учитывает, объявляет мутанта выжившим там, где на
    самом деле он до кода не доехал, и посылает искать несуществующую дыру
    в тестах. Проверено на LIMB_WOBBLE_MAX.
    """
    import re

    pattern = re.compile(rf"^{re.escape(name)}\s*(:[^=\n]+)?=.*$", re.MULTILINE)
    if not pattern.search(text):
        return ""
    return pattern.sub(f"{name} = {value!r}", text, count=1)


def run_tests() -> tuple:
    r = subprocess.run([sys.executable, "-m", "unittest", "discover",
                        "-s", "ball_reel/tests", "-p", "test_*.py"],
                       capture_output=True, text=True)
    tail = (r.stderr or "").strip().splitlines()
    count = next((ln for ln in tail if ln.startswith("Ran ")), "?")
    return r.returncode == 0, count


def run_coverage() -> tuple:
    """Покрытие по несущим модулям. Возвращает (ok, {модуль: процент})."""
    dev = subprocess.DEVNULL
    subprocess.run([sys.executable, "-m", "coverage", "run",
                    "--source=ball_reel", "--omit=*/tests/*",
                    "-m", "unittest", "discover", "-s", "ball_reel/tests",
                    "-p", "test_*.py"], stdout=dev, stderr=dev)
    r = subprocess.run([sys.executable, "-m", "coverage", "report",
                        "--skip-empty"], capture_output=True, text=True)
    per: dict = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith("%"):
            name = Path(parts[0]).stem
            if name in CORE_MODULES:
                per[name] = int(parts[-1].rstrip("%"))
    low = {k: v for k, v in per.items() if v < MIN_CORE_COVERAGE}
    return (not low), per


def run_mutation(module: str, name: str, value) -> tuple:
    """Прогнать тесты на копии пакета с испорченной константой.

    Возвращает (убит, примечание). «Убит» = хотя бы один тест покраснел, то
    есть порог действительно кем-то сторожится.
    """
    import shutil
    import tempfile

    rel = Path(module.replace(".", "/") + ".py")
    if not rel.exists():
        return False, f"нет файла {rel}"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "ball_reel"
        shutil.copytree("ball_reel", dst,
                        ignore=shutil.ignore_patterns("__pycache__", "fixtures"))
        # fixtures нужны офлайн-тестам, но копировать их дорого — линкуем.
        (dst / "fixtures").symlink_to(Path("ball_reel/fixtures").resolve())
        target = Path(tmp) / rel
        mutated = _mutate_source(target.read_text(), name, value)
        if not mutated:
            return False, f"объявление {name} не найдено в {rel}"
        target.write_text(mutated)
        r = subprocess.run(
            [sys.executable, "-m", "unittest", "discover",
             "-s", "ball_reel/tests", "-p", "test_*.py"],
            cwd=tmp, capture_output=True, text=True)
        return r.returncode != 0, ""


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.codeaudit",
        description="тесты + покрытие + мутационная проверка порогов")
    ap.add_argument("--quick", action="store_true", help="без мутаций")
    args = ap.parse_args(argv)

    print("=" * 72)
    ok_tests, count = run_tests()
    print(f"{'PASS' if ok_tests else 'FAIL'}  тесты          {count}")
    if not ok_tests:
        print("\nОСТАНОВЛЕНО: аудит на красных тестах бессмысленен.")
        return 1

    ok_cov, per = run_coverage()
    worst = sorted(per.items(), key=lambda kv: kv[1])
    detail = ", ".join(f"{k} {v}%" for k, v in worst[:4])
    print(f"{'PASS' if ok_cov else 'FAIL'}  покрытие ядра  "
          f"порог {MIN_CORE_COVERAGE}%; худшие: {detail}")

    if args.quick:
        print("=" * 72)
        print("мутации пропущены (--quick)")
        return 0 if (ok_tests and ok_cov) else 1

    print("-" * 72)
    print("мутации: ломаем порог — тесты ОБЯЗАНЫ покраснеть")
    survived = []
    for module, name, value, meaning in MUTATIONS:
        killed, note = run_mutation(module, name, value)
        short = f"{module.split('.')[-1]}.{name}"
        print(f"  {'убита ' if killed else 'ВЫЖИЛА'}  {short:<38} "
              f"{note or meaning}")
        if not killed:
            survived.append((short, note or meaning))

    print("=" * 72)
    rate = (len(MUTATIONS) - len(survived)) / len(MUTATIONS)
    print(f"мутационное покрытие: {rate:.0%} "
          f"({len(MUTATIONS) - len(survived)}/{len(MUTATIONS)})")
    if survived:
        print("\nВЫЖИВШИЕ — здесь мы думаем, что защищены, но это не так:")
        for short, meaning in survived:
            print(f"  {short}: {meaning}")
        print("\nЭто не обязательно баг кода. Это либо недостающий тест, либо "
              "порог, который никуда не подключён. Разбирать поштучно.")
    ok = ok_tests and ok_cov and not survived
    print(f"\n{'АУДИТ ПРОЙДЕН' if ok else 'АУДИТ НЕ ПРОЙДЕН'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
