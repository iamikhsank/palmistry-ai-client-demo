import unittest

from modules.client_report import describe_manual_observations, describe_screening, make_report


class ClientReportTests(unittest.TestCase):
    def test_candidates_are_uncertain_and_ranked(self):
        text = describe_screening("kiri", {"a": 0.6, "b": 0.9, "c": 0.2},
                                  {"a": "A", "b": "B", "c": "C"}, 0.5)
        self.assertIn("B, A", text)
        self.assertNotIn("C,", text)
        self.assertIn("bukan kepastian", text)
        self.assertIn("belum dapat ditentukan", text)

    def test_empty_screen_is_not_negative_diagnosis(self):
        text = describe_screening("kanan", {"a": 0.1}, {"a": "A"})
        self.assertIn("tidak membuktikan", text)

    def test_two_hand_report(self):
        text = make_report({"kiri": {"narrative": "KIRI"}, "kanan": {"narrative": "KANAN"}})
        self.assertIn("KIRI", text)
        self.assertIn("KANAN", text)
        self.assertIn("bukan dasar diagnosis", text)

    def test_manual_observations_are_explicit_and_attributed(self):
        text = describe_manual_observations("kiri", {"kehidupan": "samar", "kepala": "jelas"})
        self.assertIn("diisi peninjau secara manual", text)
        self.assertIn("**Garis kehidupan.** Berdasarkan catatan peninjau, garis ini terlihat samar", text)
        self.assertIn("**Garis kepala.** Berdasarkan catatan peninjau, garis ini terlihat jelas", text)
        self.assertNotIn("Garis asmara:", text)
        self.assertIn("usia tidak disimpulkan otomatis", text)

    def test_unreviewed_not_assumed_absent(self):
        text = describe_manual_observations("kanan", {"nasib": "tidak_terlihat"})
        self.assertIn("belum terlihat pada foto", text)
        self.assertNotIn("tidak ada", text)

    def test_export_keeps_manual_and_model_attribution(self):
        text = make_report({"kiri": {"narrative": "HASIL MODEL", "manual_narrative": "CATATAN MANUAL"}})
        self.assertIn("HASIL MODEL", text)
        self.assertIn("CATATAN MANUAL", text)

    def test_candidate_descriptions_do_not_assert_presence_or_age(self):
        text = describe_screening("kiri", {"mark_island": 0.8, "mark_apollo_star": 0.7},
                                  {"mark_island": "Tanda Pulau", "mark_apollo_star": "Bintang Apollo"})
        self.assertIn("**Tanda Pulau.** Model menandai kategori ini sebagai kandidat", text)
        self.assertIn("masih perlu diperiksa peninjau", text)
        self.assertIn("belum dapat ditentukan", text)

    def test_manual_detail_is_attributed_and_not_invented(self):
        text = describe_manual_observations("kiri", {"kehidupan": "jelas"},
                                            {"kehidupan": "Panjang dengan cabang yang perlu ditinjau."})
        self.assertIn("Catatan tambahan peninjau untuk garis kehidupan", text)
        self.assertIn("Panjang dengan cabang yang perlu ditinjau", text)
        self.assertNotIn("usia 30", text)


if __name__ == "__main__":
    unittest.main()
