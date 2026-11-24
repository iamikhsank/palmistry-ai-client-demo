import unittest

from modules.client_report import describe_screening, make_copyable_report


class ScoreValidationTests(unittest.TestCase):
    def test_bad_values_do_not_create_candidates(self):
        names = {"good": "Pola valid", "bad": "Pola tidak valid"}
        text = describe_screening("kiri", {"good": 0.7, "bad": float("inf"),
                                           "other": "invalid"}, names)
        self.assertIn("Pola valid", text)
        self.assertNotIn("Pola tidak valid", text)

    def test_out_of_range_not_in_copyable_report(self):
        result = make_copyable_report({"kiri": {
            "model_names": {"bad": "Pola tidak valid"},
            "model_probabilities": {"bad": 1.2},
        }})
        self.assertNotIn("Pola tidak valid", result)

    def test_invalid_threshold_fails_closed(self):
        for threshold in (float("nan"), -1, 2, "bad"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    describe_screening("kiri", {"a": 0.8}, {"a": "A"}, threshold)

    def test_missing_scores_are_handled(self):
        self.assertIn("Tidak ada kandidat", describe_screening("kanan", None, {}))

    def test_grouped_report_keeps_each_hand_separate(self):
        names = {"mark_island": "Tanda Pulau", "branch_terminal_fork": "Cabang Garpu"}
        text = make_copyable_report({
            "kiri": {"model_names": names, "model_probabilities": {"mark_island": .8}},
            "kanan": {"model_names": names, "model_probabilities": {"branch_terminal_fork": .9}},
        })
        left, right = text.split("TANGAN KANAN")
        right = right.split("KESIMPULAN UTAMA")[0]
        self.assertIn("Pola pada garis", left)
        self.assertIn("Tanda Pulau", left)
        self.assertNotIn("Cabang Garpu", left)
        self.assertIn("Pola cabang dan kesinambungan", right)
        self.assertNotIn("Tanda Pulau", right)
        self.assertIn("bukan pengukuran", text)

    def test_overall_summary_respects_each_hand_and_case(self):
        names = {"mark_island": "Tanda Pulau", "branch_ascending_jupiter": "Cabang Yupiter"}
        text = make_copyable_report({
            "kiri": {"model_names": names, "model_probabilities": {"mark_island": .9}},
            "kanan": {"model_names": names, "model_probabilities": {"branch_ascending_jupiter": .8}},
        })
        summary = text.split("KESIMPULAN UTAMA", 1)[1]
        self.assertIn("tangan kiri: Tanda Pulau", summary)
        self.assertIn("tangan kanan: Cabang Yupiter", summary)
        self.assertIn("bukan hasil pengukuran garis", summary)

    def test_summary_does_not_invent_empty_side(self):
        text = make_copyable_report({"kiri": {"model_names": {}, "model_probabilities": {}}})
        self.assertIn("tangan kiri: tidak ada kategori di atas ambang", text)
        self.assertNotIn("tangan kanan:", text)

    def test_unknown_feature_name_is_preserved(self):
        text = make_copyable_report({"kiri": {"model_names": {"other": "Pola lain"},
                                                "model_probabilities": {"other": .8}}})
        self.assertIn("Kategori lainnya yang ditandai: Pola lain", text)

    def test_summary_uses_only_selected_ranked_scores(self):
        names = {"mark_island": "Tanda Pulau", "branch_terminal_fork": "Cabang Garpu",
                 "mark_jupiter_star": "Bintang Yupiter"}
        text = make_copyable_report({"kiri": {"model_names": names,
            "model_probabilities": {"mark_island": .7, "branch_terminal_fork": .9,
                                    "mark_jupiter_star": .2}}})
        summary = text.split("Ringkasan tangan kiri:", 1)[1].split("\n", 1)[0]
        self.assertIn("Cabang Garpu, Tanda Pulau", summary)
        self.assertNotIn("Bintang Yupiter", summary)
        self.assertIn("bukan kepastian", summary)


if __name__ == "__main__":
    unittest.main()
