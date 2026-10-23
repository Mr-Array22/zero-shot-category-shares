import unittest

import numpy as np

from jevstudy.quantify import (K, em_prior, estimate_all, hard_confusion, select_jev_responses, solve_clip, solve_simplex)
from jevstudy.study import LABELS, PINNED_MODEL, object_hash


def response(label):
    probs = {k: float(k == label) for k in LABELS}
    return {"model": PINNED_MODEL, "answers": {"product": {"type": "choice", "choice": label, "confidence": 1.0,
            "probabilities": probs}}, "usage": {"input_tokens": 10, "output_tokens": 2}}


def labelled(counts_matrix):
    """counts_matrix[j, i] = number of items with true class i predicted as j."""
    y, pred = [], []
    k = counts_matrix.shape[0]
    for i in range(k):
        for j in range(k):
            n = int(counts_matrix[j, i])
            y.extend([i] * n)
            pred.extend([j] * n)
    return np.array(y), np.array(pred)


class ConfusionTests(unittest.TestCase):
    def test_hard_confusion_columns_are_conditional_distributions(self):
        y = np.array([0, 0, 1, 1, 1, 2, 3, 4, 4, 4])
        pred = np.array([0, 1, 1, 1, 0, 2, 3, 4, 4, 0])
        M = hard_confusion(y, pred)
        self.assertEqual(M.shape, (K, K))
        np.testing.assert_allclose(M.sum(axis=0), np.ones(K))
        self.assertAlmostEqual(M[1, 1], 2 / 3)
        self.assertAlmostEqual(M[0, 1], 1 / 3)

    def test_missing_development_category_is_rejected(self):
        with self.assertRaises(ValueError):
            hard_confusion(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))


class ACCPriorShiftTests(unittest.TestCase):
    def test_acc_recovers_known_priors_under_pure_prior_shift(self):
        # Column sums are 100 so every development count is integral.
        N = np.array([[80, 10, 0, 5, 0], [10, 70, 5, 0, 0], [5, 10, 90, 0, 10],
                      [0, 5, 5, 90, 0], [5, 5, 0, 5, 90]])
        M = N / 100.0
        dev_y, dev_pred = labelled(N)
        test_counts = np.array([100, 400, 100, 300, 200])  # new priors, multiples of 100
        test_matrix = (N * test_counts[None, :]) // 100      # same P(pred | true)
        test_y, test_pred = labelled(test_matrix)
        p_true = test_counts / test_counts.sum()
        dev = {"y": dev_y, "pred": dev_pred, "P": None}
        test = {"y": test_y, "pred": test_pred, "P": None}
        est, info = estimate_all(dev, test)
        self.assertFalse(np.allclose(est["CC"], p_true, atol=1e-3))  # shift biases counting
        np.testing.assert_allclose(est["ACC_clip"], p_true, atol=1e-10)
        np.testing.assert_allclose(est["ACC_simplex"], p_true, atol=1e-6)
        self.assertFalse(info["methods"]["ACC_clip"]["clip_needed"])
        np.testing.assert_allclose(M @ p_true, np.bincount(test_pred, minlength=K) / len(test_pred), atol=1e-12)


class SolverTests(unittest.TestCase):
    def test_unclipped_solution_is_returned_when_feasible(self):
        M = np.eye(3)
        q = np.array([.2, .3, .5])
        est, diag = solve_clip(M, q)
        np.testing.assert_allclose(est, q)
        self.assertFalse(diag["clip_needed"])

    def test_clipping_path_produces_valid_distribution(self):
        M = np.array([[.9, .2], [.1, .8]])
        q = np.array([0.0, 1.0])
        est, diag = solve_clip(M, q)
        self.assertTrue(diag["clip_needed"])
        self.assertTrue((est >= 0).all())
        self.assertAlmostEqual(est.sum(), 1.0)
        np.testing.assert_allclose(est, [0.0, 1.0], atol=1e-12)
        simplex, _ = solve_simplex(M, q)
        np.testing.assert_allclose(simplex, [0.0, 1.0], atol=1e-5)

    def test_simplex_least_squares_beats_grid_search(self):
        rng = np.random.default_rng(7)
        M = rng.random((3, 3))
        M /= M.sum(axis=0, keepdims=True)
        q = np.array([0.0, 0.0, 1.0])
        est, _ = solve_simplex(M, q)
        self.assertTrue((est >= 0).all())
        self.assertAlmostEqual(est.sum(), 1.0)
        ours = np.linalg.norm(M @ est - q)
        best = np.inf
        grid = np.arange(0, 1 + 1e-9, .01)
        for a in grid:
            for b in grid:
                if a + b <= 1 + 1e-9:
                    p = np.array([a, b, max(0.0, 1 - a - b)])
                    best = min(best, np.linalg.norm(M @ p - q))
        self.assertLessEqual(ours, best + 1e-6)


class EMTests(unittest.TestCase):
    def test_em_converges_and_recovers_shifted_priors_on_calibrated_toy(self):
        rng = np.random.default_rng(3)
        means = np.array([0., 2., 4.])
        train_prior = np.array([.5, .3, .2])
        test_prior = np.array([.15, .25, .6])
        n = 20000
        y = rng.choice(3, size=n, p=test_prior)
        x = rng.normal(means[y], 1.0)
        likelihood = np.exp(-0.5 * (x[:, None] - means[None, :]) ** 2)
        posterior = likelihood * train_prior
        posterior /= posterior.sum(axis=1, keepdims=True)
        estimate, diag = em_prior(posterior, train_prior)
        self.assertTrue(diag["converged"])
        self.assertLess(diag["iterations"], 1000)
        self.assertLess(np.abs(estimate - test_prior).max(), .02)
        self.assertAlmostEqual(estimate.sum(), 1.0)

    def test_em_on_one_hot_probabilities_reduces_to_counting(self):
        labels = np.array([0, 0, 1, 2, 2, 2, 1, 0])
        P = np.eye(3)[labels]
        estimate, _ = em_prior(P, np.array([.3, .3, .4]))
        np.testing.assert_allclose(estimate, np.bincount(labels, minlength=3) / len(labels), atol=1e-9)


class JevSelectionTests(unittest.TestCase):
    def test_last_successful_response_per_id_is_used(self):
        request = {"model": PINNED_MODEL, "state": {"feedback": "x"}}
        requests = {"7": {"request": request}}
        h = object_hash(request)
        attempts = [{"id": "7", "status": "failed", "request_hash": h, "response": None},
                    {"id": "7", "status": "ok", "request_hash": h, "response": response(LABELS[0])},
                    {"id": "7", "status": "ok", "request_hash": h, "response": response(LABELS[2])}]
        selected, meta = select_jev_responses(attempts, requests, {"7"})
        self.assertEqual(selected["7"]["label"], LABELS[2])
        self.assertEqual(meta["failed_attempt_records"], 1)
        self.assertEqual(meta["ids_with_multiple_successes"], 1)


if __name__ == "__main__":
    unittest.main()
