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
