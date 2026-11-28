import unittest
import numpy as np

from modules.experimental_line_paths import trace_components, propose_line_zones, trace_ridge_zones


def hand():
    points = np.full((21,2), .5)
    points[5], points[17], points[0] = (.2,.2), (.8,.2), (.5,.8)
    return points


class ExperimentalLinePathTests(unittest.TestCase):
    def test_length_uses_single_path_not_total_branch_length(self):
        mask = np.zeros((100,100), bool)
        mask[50,10:91] = True
        mask[35:51,50] = True
        paths = trace_components(mask)
        self.assertEqual(len(paths),1)
        self.assertAlmostEqual(paths[0]['length_px'], 80.)
        self.assertGreater(paths[0]['junction_pixels'],0)

    def test_empty_and_small_shapes_abstain(self):
        paths, zones = trace_ridge_zones(np.zeros((100,100)), hand(), np.ones((100,100),bool), size=100)
        self.assertEqual(paths,[])
        self.assertEqual(zones,{})

    def test_line_zones_mirror_with_hand(self):
        path = {'xy':np.array([[30.,34.],[70.,34.]]), 'length_px':40.,'junction_pixels':0}
        left = propose_line_zones([path],hand(),100)
        mirrored = hand().copy(); mirrored[:,0] = 1-mirrored[:,0]
        flipped = dict(path,xy=path['xy'].copy()); flipped['xy'][:,0] = 99-flipped['xy'][:,0]
        right = propose_line_zones([flipped],mirrored,100)
        self.assertIn('asmara',left)
        self.assertIn('asmara',right)
        self.assertAlmostEqual(left['asmara']['length_palm_widths'],right['asmara']['length_palm_widths'])

    def test_ambiguous_parallel_candidates_abstain(self):
        paths = [{'xy':np.array([[30.,34.],[70.,34.]]), 'length_px':40.,'junction_pixels':0},
                 {'xy':np.array([[30.,35.],[70.,35.]]), 'length_px':40.,'junction_pixels':0}]
        self.assertNotIn('asmara',propose_line_zones(paths,hand(),100))

    def test_degenerate_landmarks_rejected(self):
        points = np.full((21,2),.5)
        with self.assertRaises(ValueError):
            propose_line_zones([{'xy':np.array([[1.,2.],[3.,4.]]),'length_px':4.,'junction_pixels':0}],points,100)


if __name__ == '__main__':
    unittest.main()
