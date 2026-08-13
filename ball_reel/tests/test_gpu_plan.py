"""Разрешение как условие работоспособности ГЕЙТА, а не как качество картинки.

Найдено живым прогоном и стоит того, чтобы стоять отдельным файлом: ArcFace
отказывается судить лицо мельче своего порога, а на полноростовом вертикальном
кадре доля лица зафиксирована геометрией. Значит порог идентичности
превращается в требование к разрешению — и это требование надо знать ДО аренды
карты, а не выяснять после.

Числа геометрические, а не модельные: они одинаковы для gateway-моделей и для
того опенсорса, который поедет в прод.
"""

from __future__ import annotations

import unittest


class ResolutionDecidesWhetherIdentityCanBeJudgedAtAll(unittest.TestCase):
    def setUp(self):
        from ball_reel import gpu_keyframes

        self.g = gpu_keyframes

    def test_the_measured_share_reproduces_the_live_numbers(self):
        # Замерено: 1080x1920 дал лицо 181 px, 464x832 дал 78 px.
        self.assertAlmostEqual(self.g.face_px_at(1920), 180, delta=3)
        self.assertAlmostEqual(self.g.face_px_at(832), 78, delta=3)

    def test_both_gpu_plans_fail_full_body_and_say_so(self):
        # 512x768 и 640x960 — обе наши конфигурации. Ни одна не даёт судимого
        # лица на полном росте, и план обязан об этом предупредить, а не
        # обнаружиться на арендованной карте.
        for height in (768, 960):
            ok, why = self.g.identity_verifiable(height)
            self.assertFalse(ok, height)
            self.assertIn("НЕ СМОЖЕТ", why)

    def test_720p_and_above_are_judgeable(self):
        for height in (1280, 1920):
            ok, why = self.g.identity_verifiable(height)
            self.assertTrue(ok, why)

    def test_the_refusal_names_the_three_ways_out(self):
        _, why = self.g.identity_verifiable(768)
        self.assertIn("по высоте", why)      # поднять разрешение
        self.assertIn("по пояс", why)        # сменить кадрирование
        self.assertIn("не проверяется", why)  # принять честно

    def test_a_tighter_framing_rescues_the_small_frame(self):
        # Выход для 4 ГБ: поясная композиция даёт вдвое большую долю лица.
        ok, _ = self.g.identity_verifiable(
            768, share=self.g.WAIST_UP_FACE_SHARE)
        self.assertTrue(ok)

    def test_the_plan_carries_the_warning_where_it_will_be_read(self):
        notes = " ".join(self.g.plan(vram_gb=4.0).notes)
        self.assertIn("ИДЕНТИЧНОСТЬ НА ПОЛНОМ РОСТЕ", notes)

    def test_the_shares_are_read_from_the_module_with_literal_heights(self):
        full, waist = (self.g.FULL_BODY_FACE_SHARE, self.g.WAIST_UP_FACE_SHARE)
        self.assertGreater(waist, full)
        self.assertLess(full, 0.15)
        self.assertFalse(self.g.identity_verifiable(768, share=full)[0])
        self.assertTrue(self.g.identity_verifiable(768, share=waist)[0])


if __name__ == "__main__":
    unittest.main()


class TheSmokeVERDICTUsesTheModulesOwnBars(unittest.TestCase):
    """Гейт целевого пути судил мягче собственных порогов, и это никто не ловил.

    Логика вердикта жила внутри `main()` литеральными числами: 0.35 для лица
    (случайно совпало с `SAME_PERSON_MAX`) и 0.25 для позы, при том что
    `pose.SAME_POSE_MAX` равен 0.15. Расхождение на две трети, на ЦЕЛЕВОМ
    пути, и увидеть его можно было только сличив два файла глазами: тест сюда
    не доставал, мутация тоже — константа, не участвующая в коде, неубиваема.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local

    def _delta(self, mean, worst, joint="l_wrist"):
        return {"mean": mean, "worst": worst, "worst_joint": joint}

    def test_a_pose_between_the_two_bars_is_now_rejected(self):
        # Ровно та щель, что была открыта: 0.20 проходило старый бар 0.25 и не
        # проходит настоящий 0.15. Числа литеральные намеренно — тест, берущий
        # вход из константы, которую сторожит, едет вместе с ней.
        from ball_reel.pose import SAME_POSE_MAX

        self.assertLess(SAME_POSE_MAX, 0.20)
        got = self.r.smoke_verdict(0.1, self._delta(0.20, 0.30))
        self.assertFalse(got["pose_ok"], got["pose_note"])

    def test_a_faithful_start_frame_still_passes(self):
        # Обратная сторона: ужесточение, бракующее всё, — не фикс, а регресс.
        # 0.043 — верх диапазона, замеренного на живых старт-кадрах (0.018-0.043).
        got = self.r.smoke_verdict(0.1, self._delta(0.043, 0.12))
        self.assertTrue(got["pose_ok"], got["pose_note"])
        self.assertTrue(got["face_ok"], got["face_note"])

    def test_one_limb_gone_astray_is_caught_by_the_worst_joint(self):
        # Худший сустав ПЕЧАТАЛСЯ, но не проверялся, хотя сообщение обещало
        # пользователю бар 0.40. Замеренный случай: согнутая рука даёт среднее
        # 0.11 при запястье 0.97 — среднее её размывает полностью.
        got = self.r.smoke_verdict(0.1, self._delta(0.11, 0.97))
        self.assertFalse(got["pose_ok"], got["pose_note"])
        self.assertIn("0.97", got["pose_note"])

    def test_an_unmeasured_pose_is_a_failure_not_a_pass(self):
        got = self.r.smoke_verdict(0.1, None, driving="drive/0001.png")
        self.assertFalse(got["pose_ok"])
        self.assertIn("НЕ ИЗМЕРЕНА", got["pose_note"])
        self.assertIn("drive/0001.png", got["pose_note"])

    def test_a_missing_face_is_not_a_pass_either(self):
        got = self.r.smoke_verdict(None, self._delta(0.02, 0.05))
        self.assertFalse(got["face_ok"])
        self.assertIn("НЕ НАЙДЕНО", got["face_note"])

    def test_the_face_bar_bites_from_both_sides(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        d = self._delta(0.02, 0.05)
        self.assertTrue(self.r.smoke_verdict(SAME_PERSON_MAX - 0.01, d)["face_ok"])
        self.assertFalse(self.r.smoke_verdict(SAME_PERSON_MAX + 0.01, d)["face_ok"])
