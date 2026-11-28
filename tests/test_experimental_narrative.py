import unittest
from unittest.mock import patch
import numpy as np
from modules.experimental_narrative import _similar, stable_zones, describe_geometry


class ExperimentalNarrativeTests(unittest.TestCase):
    def test_spatial_stability_distinguishes_shift_and_reversal(self):
        line = np.array([[x,20] for x in range(10,51)],float)
        self.assertTrue(_similar(line,line[::-1],100))
        self.assertFalse(_similar(line,line + [0,15],100))
        self.assertFalse(_similar(line,line[:10],100))

    def test_changed_threshold_with_shifted_line_abstains(self):
        path = np.array([[x,20] for x in range(10,51)],float)
        def result(y):
            return ([{'xy':path + [0,y]}], {'asmara':{'path_index':0, 'length_palm_widths':.7,
                'position_palm':(.5,.2), 'junction_pixels':0}})
        with patch('modules.experimental_narrative.trace_ridge_zones',side_effect=[result(0),result(0),result(20)]):
            _, zones = stable_zones(None,None,None,size=100)
        self.assertEqual(zones,{})

    def test_report_does_not_invent_missing_features_or_age(self):
        report = describe_geometry('kiri', {'kepala': {'length_palm_widths':1.25,
                     'position_palm':(.5,.6), 'junction_pixels':1}})
        self.assertIn('1.25 kali lebar telapak',report)
        self.assertIn('Garis kehidupan: jalur belum cukup konsisten',report)
        self.assertIn('cabang belum dapat dipastikan',report)
        self.assertNotIn('usia 30',report)
        self.assertNotIn('sentimeter',report)

    def test_invalid_measurement_rejected(self):
        with self.assertRaises(ValueError):
            describe_geometry('kiri',{'kepala':{'length_palm_widths':float('nan'),
                                'position_palm':(.5,.5),'junction_pixels':0}})

    def test_synthetic_ridge_flows_through_to_copyable_report(self):
        ridge = np.zeros((256,256),dtype=np.uint8)
        ridge[88,64:192] = 220
        landmarks = np.full((21,2),.5)
        landmarks[5], landmarks[17], landmarks[0] = (.2,.2), (.8,.2), (.5,.8)
        paths, zones = stable_zones(ridge,landmarks,np.ones((256,256),dtype=bool))
        self.assertIn('asmara',zones)
        self.assertGreater(len(paths),0)
        self.assertIn('Garis asmara: kandidat jalur', describe_geometry('kanan',zones))


if __name__ == '__main__':
    unittest.main()
