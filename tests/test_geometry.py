import unittest

import numpy as np

from study.caption_partial import correlations
from study.geometry import adjust, heldout_scale, neighbors, permute_indices, ranks, rdm, score_rank, score_within
from study.roster import freeze_splits


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.generator = np.random.default_rng(87)
        self.points = self.generator.normal(size=(8, 12))
        self.distances = rdm(self.points)

    def test_rotation_preserves_rsa_and_metric(self):
        rotation, _ = np.linalg.qr(self.generator.normal(size=(12, 12)))
        rotated = rdm(self.points @ rotation)
        np.testing.assert_allclose(rotated, self.distances, atol=1e-14)
        x, y = ranks(self.distances), ranks(rotated)
        actual = score_rank([x, x], [y, y], ((0, 0), (1, 1)), np.arange(8)[None])
        self.assertAlmostEqual(actual[0], 1)
        metric = heldout_scale([self.distances] * 2, [2 * self.distances] * 2,
                               ((0, 0), (1, 1)), list("abcdefgh"), 17)
        self.assertFalse(set(metric["development"]) & set(metric["heldout"]))
        self.assertAlmostEqual(metric["relative_rmse"], 0)
        self.assertGreater(metric["constant_rmse"], 0)

    def test_blocked_permutation_changes_neither_concept_roster_nor_group(self):
        groups = np.array(list("aaabbbcc"))
        permutation = permute_indices(len(groups), groups, 25, self.generator, True)
        np.testing.assert_array_equal(np.sort(permutation, axis=1), np.tile(np.arange(8), (25, 1)))
        np.testing.assert_array_equal(groups[permutation], np.tile(groups, (25, 1)))
        within, names = score_within([self.distances] * 2, [self.distances] * 2,
                                     ((0, 0), (1, 1)), np.arange(8)[None], groups)
        self.assertEqual(names, ["a", "b"])
        self.assertAlmostEqual(within[0], 1)

    def test_neighbor_ties_and_missing_holm_slots(self):
        tied = np.ones((5, 5))
        np.testing.assert_allclose(neighbors(tied, 2), .5 * (1 - np.eye(5)))
        present, absent = {"outcomes": {"paired": {"global_rsa": {"p": .01}}}}, {}
        adjust([present, absent])
        self.assertAlmostEqual(present["outcomes"]["paired"]["global_rsa"]["p_holm"], .12)

    def test_source_creators_never_cross_folds(self):
        rows = []
        for name in ["a", "b"]:
            for i in range(6):
                rows.append(dict(source_id=f"{name}-{i}", concept=name, group="g",
                                 author_key=f"author-{i}", uploader_key=f"uploader-{i}"))
        selected, excluded = freeze_splits(rows)
        self.assertFalse(excluded)
        self.assertEqual(len(selected), 12)
        for i in range(6):
            paired = [row for row in selected if row["author_key"] == f"author-{i}"]
            self.assertEqual(len({row["fold"] for row in paired}), 1)
            self.assertEqual(len({row["creator_component"] for row in paired}), 1)

    def test_caption_partial_matches_explicit_residualization(self):
        a = self.generator.normal(size=(5, 12))
        a -= a.mean(axis=1, keepdims=True)
        x, y, z = a[0], a[1:3], a[3:]
        expected = []
        for target, caption in zip(y, z, strict=True):
            projection = np.outer(caption, caption) / (caption @ caption)
            residual = np.eye(len(x)) - projection
            expected.append(np.corrcoef(residual @ x, residual @ target)[0, 1])
        np.testing.assert_allclose(correlations(x, y, z), expected, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
