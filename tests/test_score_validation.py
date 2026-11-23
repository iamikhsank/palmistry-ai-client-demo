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
        self.assertIn("Pola pada garis", left)
        self.assertIn("Tanda Pulau", left)
        self.assertNotIn("Cabang Garpu", left)
        self.assertIn("Pola cabang dan kesinambungan", right)
        self.assertNotIn("Tanda Pulau", right)
        self.assertIn("bukan pengukuran", text)

    def test_unknown_feature_name_is_preserved(self):
        text = make_copyable_report({"kiri": {"model_names": {"other": "Pola lain"},
                                                "model_probabilities": {"other": .8}}})
        self.assertIn("Kategori lainnya yang ditandai: Pola lain", text)


if __name__ == "__main__":
    unittest.main()
