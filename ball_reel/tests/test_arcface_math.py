"""The ArcFace path's arithmetic, unit-tested WITHOUT the 300 MB model.

The model-backed embedding cannot run here, but the cosine distance the whole
identity verdict rests on can, and must: it is the number a sceptic recomputes.
Skips cleanly if numpy is absent (the offline default install has no numpy).
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class CosineDistanceIsRight(unittest.TestCase):
    def setUp(self):
        from ball_reel import identity_arcface
        self.cd = identity_arcface.cosine_distance

    def test_identical_vectors_are_zero(self):
        self.assertEqual(self.cd([1, 2, 3], [1, 2, 3]), 0.0)

    def test_scaled_same_direction_is_zero(self):
        # cosine ignores magnitude: same person, different crop scale.
        self.assertEqual(self.cd([1, 0, 0], [5, 0, 0]), 0.0)

    def test_orthogonal_is_one(self):
        self.assertEqual(self.cd([1, 0], [0, 1]), 1.0)

    def test_opposite_is_two_clamped_to_range(self):
        # 1 - (-1) = 2, but a face distance below "identical" is the useful
        # range; the function returns the raw 1 - cos, which for opposites is 2.
        self.assertEqual(self.cd([1, 0], [-1, 0]), 2.0)

    def test_zero_vector_is_max_distance(self):
        self.assertEqual(self.cd([0, 0, 0], [1, 1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
