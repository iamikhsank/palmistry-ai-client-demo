import unittest

from modules.client_report import describe_manual_observations, describe_screening, make_copyable_report, make_report


class ClientReportTests(unittest.TestCase):
    def test_copyable_report_separates_hands_and_manual_from_model(self):
        report = make_copyable_report({
            "kiri": {"observations": {"kehidupan": "jelas"},
                     "details": {"kehidupan": {"length": "long", "branches": "present"}},
                     "notes": {"kehidupan": "Cabang terlihat di bagian atas."},
                     "model_candidates": ["Tanda Pulau"]},
            "kanan": {"observations": {"kepala": "samar"}},
        })
        self.assertIn("TANGAN KIRI", report)
        self.assertIn("Garis kehidupan: terlihat jelas, panjang, ada percabangan.", report)
        self.assertIn("Cabang terlihat di bagian atas", report)
        self.assertIn("TANGAN KANAN", report)
        self.assertIn("Garis kepala: terlihat samar", report)
        self.assertIn("bukan pemetaan garis utama", report)
        self.assertNotIn("usia 30", report)

    def test_copyable_report_does_not_create_unreviewed_measurements(self):
        report = make_copyable_report({"kiri": {"model_candidates": ["Tanda Pulau"]}})
        self.assertNotIn("belum ada keterangan yang dicatat", report)
        self.assertIn("Tanda Pulau", report)
        self.assertNotIn("Garis kehidupan: panjang", report)

    def test_automatic_narrative_uses_each_hand_and_threshold(self):
        names = {"mark_island": "Tanda Pulau", "branch_terminal_fork": "Cabang Garpu"}
        report = make_copyable_report({
            "kiri": {"model_probabilities": {"mark_island": 0.9, "branch_terminal_fork": 0.1}, "model_names": names},
            "kanan": {"model_probabilities": {"mark_island": float('nan'), "branch_terminal_fork": 0.8}, "model_names": names},
        })
        left, right = report.split("TANGAN KANAN")
        self.assertIn("Tanda Pulau", left)
        self.assertNotIn("Cabang Garpu", left)
        self.assertIn("Cabang Garpu", right)
        self.assertNotIn("Tanda Pulau", right)
        self.assertNotIn("belum ada keterangan yang dicatat", report)
        self.assertNotIn("usia 30", report)

    def test_empty_selection_is_not_invented(self):
        report = make_copyable_report({"kiri": {"model_probabilities": {"mark_island": 0.1},
            "model_names": {"mark_island": "Tanda Pulau"}}})
        self.assertIn("Tidak ada kategori pola yang mencapai ambang", report)
        self.assertNotIn("Tanda Pulau", report)

    def test_candidates_are_uncertain_and_ranked(self):
        text = describe_screening("kiri", {"a": 0.6, "b": 0.9, "c": 0.2},
                                  {"a": "A", "b": "B", "c": "C"}, 0.5)
        self.assertIn("B, A", text)
        self.assertNotIn("C,", text)
        self.assertIn("bukan kepastian bentuk", text)
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
        self.assertIn("**Tanda Pulau.** Hasil model menandai kategori ini", text)
        self.assertIn("bukan kepastian bentuk atau lokasi", text)
        self.assertIn("belum dapat ditentukan", text)

    def test_every_known_feature_has_distinct_guidance(self):
        from modules.client_report import FEATURE_DETAILS, FEATURE_EXPLANATIONS
        self.assertEqual(len(FEATURE_EXPLANATIONS), 16)
        self.assertEqual(set(FEATURE_EXPLANATIONS), set(FEATURE_DETAILS))
        text = describe_screening("kanan", {"line_marriage": 0.8}, {"line_marriage": "Garis Hubungan"})
        self.assertIn("tidak menentukan jumlah hubungan", text)
        self.assertNotIn("usia 30 tahun", text)

    def test_manual_detail_is_attributed_and_not_invented(self):
        text = describe_manual_observations("kiri", {"kehidupan": "jelas"},
                                            {"kehidupan": "Panjang dengan cabang yang perlu ditinjau."})
        self.assertIn("Catatan tambahan peninjau untuk garis kehidupan", text)
        self.assertIn("Panjang dengan cabang yang perlu ditinjau", text)
        self.assertNotIn("usia 30", text)


if __name__ == "__main__":
    unittest.main()
