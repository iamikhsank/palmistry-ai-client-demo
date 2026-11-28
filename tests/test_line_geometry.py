import unittest

import numpy as np

from modules.line_geometry import analyze_ridge_candidates, describe_mask, thin, palm_mask_from_landmarks


class LineGeometryTests(unittest.TestCase):
    def test_straight_line_position_and_relative_length(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 10:81] = True
        candidates = describe_mask(mask)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["length_relative_width"], 0.7)
        self.assertEqual(candidates[0]["junction_clusters"], 0)
        self.assertEqual(candidates[0]["bbox_relative"], (0.1, 0.5, 0.81, 0.51))

    def test_t_junction_reports_one_junction_cluster(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:81, 50] = True
        mask[50, 50:81] = True
        candidate = describe_mask(mask)[0]
        self.assertEqual(candidate["junction_clusters"], 1)

    def test_diagonal_length(self):
        mask = np.eye(60, dtype=bool)
        candidate = describe_mask(mask)[0]
        self.assertAlmostEqual(candidate["length_relative_width"], 59 * np.sqrt(2) / 60)

    def test_uniform_map_has_no_candidates(self):
        self.assertEqual(analyze_ridge_candidates(np.zeros((60, 60))), [])

    def test_tiny_components_filtered(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:6, 2] = True
        self.assertEqual(describe_mask(mask), [])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            thin(np.zeros((2, 2, 2)))
        with self.assertRaises(ValueError):
            analyze_ridge_candidates(np.full((10, 10), np.nan))

    def test_landmark_palm_mask_excludes_fingers_and_background(self):
        points = np.full((21, 2), 0.5)
        points[[0, 1, 5, 9, 13, 17]] = [[.5,.9],[.25,.65],[.2,.3],[.4,.25],[.6,.25],[.8,.3]]
        mask = palm_mask_from_landmarks(points, size=100, margin_fraction=0)
        self.assertTrue(mask[50,50])
        self.assertFalse(mask[10,50])
        self.assertFalse(mask[50,95])
        mirror = points.copy()
        mirror[:,0] = 1 - mirror[:,0]
        flipped = np.fliplr(palm_mask_from_landmarks(mirror, size=100, margin_fraction=0))
        self.assertGreater(np.logical_and(mask, flipped).sum() / np.logical_or(mask, flipped).sum(), 0.97)

    def test_mask_blocks_strong_background_edges(self):
        ridge = np.zeros((100,100), dtype=np.uint8)
        ridge[40, 25:75] = 150
        ridge[10, 10:90] = 255
        mask = np.zeros((100,100), dtype=bool)
        mask[20:80,20:80] = True
        result = analyze_ridge_candidates(ridge, size=100, palm_mask=mask)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_relative'][1], 0.4)

    def test_rejects_clipped_or_invalid_landmarks(self):
        points = np.full((21,2), 0.5)
        points[0,0] = -0.1
        with self.assertRaises(ValueError):
            palm_mask_from_landmarks(points)


if __name__ == '__main__':
    unittest.main()
