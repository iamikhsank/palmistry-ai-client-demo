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


if __name__ == "__main__":
    unittest.main()
