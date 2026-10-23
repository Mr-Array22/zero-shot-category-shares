import unittest

import numpy as np

from jevstudy.diagnostics import largest_confusion_cell, mcnemar_counts


class McNemarTests(unittest.TestCase):
    def test_discordant_counts_and_exact_p(self):
        truth = np.zeros(10, dtype=int)
        jev = np.zeros(10, dtype=int)
        tfidf = np.zeros(10, dtype=int)
        jev[[4, 5, 6, 9]] = 1
        tfidf[[0, 1, 2, 3, 9]] = 1
        m = mcnemar_counts(truth, jev, tfidf)
        self.assertEqual(m["jev_right_tfidf_wrong"], 4)
        self.assertEqual(m["tfidf_right_jev_wrong"], 3)
        self.assertEqual(m["both_right"], 2)
        self.assertEqual(m["both_wrong"], 1)
        self.assertAlmostEqual(m["exact_two_sided_p"], 1.0)
        self.assertAlmostEqual(m["continuity_corrected_z"], 0.0)

    def test_one_sided_discordance_matches_exact_binomial(self):
        truth = np.zeros(10, dtype=int)
        jev = np.zeros(10, dtype=int)
        tfidf = np.ones(10, dtype=int)
        m = mcnemar_counts(truth, jev, tfidf)
        self.assertEqual(m["jev_right_tfidf_wrong"], 10)
        self.assertEqual(m["tfidf_right_jev_wrong"], 0)
        self.assertAlmostEqual(m["exact_two_sided_p"], 2 / 1024)
        self.assertAlmostEqual(m["continuity_corrected_z"], 9 / np.sqrt(10))
        self.assertEqual(m["more_accurate_on_discordant_pairs"], "jev")

    def test_identical_predictions_have_no_discordant_pairs(self):
        truth = np.arange(5)
        m = mcnemar_counts(truth, truth.copy(), truth.copy())
        self.assertEqual(m["exact_two_sided_p"], 1.0)
        self.assertIsNone(m["continuity_corrected_z"])
        self.assertEqual(m["more_accurate_on_discordant_pairs"], "neither")


class ThresholdCellTests(unittest.TestCase):
    def test_largest_cell_and_inclusive_threshold(self):
        truth = np.array([3, 3, 3, 3])
        pred = np.array([2, 2, 2, 3])
        probs = np.array([
            [0, 0, .69, .2, .11],
            [0, 0, .68, .2, .12],
            [0, 0, .9, .05, .05],
            [0, 0, 0, 1., 0],
        ])
        cell = largest_confusion_cell(truth, pred, probs, .69)
        self.assertEqual(cell["true"], "debt_collection")
        self.assertEqual(cell["predicted"], "credit_reporting")
        self.assertEqual(cell["count"], 3)
        self.assertEqual(cell["cleared_threshold"], 2)
        self.assertAlmostEqual(cell["mean_true_label_probability"], (.2 + .2 + .05) / 3)
        self.assertAlmostEqual(cell["mean_true_label_probability_cleared"], (.2 + .05) / 2)

    def test_no_cell_items_cleared_reports_none(self):
        truth = np.array([0, 1])
        pred = np.array([1, 1])
        probs = np.array([[.4, .6, 0, 0, 0], [0, 1., 0, 0, 0]])
        cell = largest_confusion_cell(truth, pred, probs, .9)
        self.assertEqual(cell["count"], 1)
        self.assertEqual(cell["cleared_threshold"], 0)
        self.assertIsNone(cell["mean_true_label_probability_cleared"])


if __name__ == "__main__":
    unittest.main()
