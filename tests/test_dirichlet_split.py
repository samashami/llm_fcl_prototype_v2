import unittest

import numpy as np

from src.run_llm_fcl_controller import make_dirichlet_client_splits


class DirichletSplitChecks(unittest.TestCase):
    @staticmethod
    def synthetic_data(samples_per_class=200, n_classes=20):
        targets = np.repeat(np.arange(n_classes), samples_per_class)
        indices = np.arange(len(targets), dtype=np.int64)
        return indices, targets

    @staticmethod
    def mean_client_total_variation(splits, targets):
        n_classes = int(targets.max()) + 1
        global_distribution = np.bincount(targets, minlength=n_classes) / len(targets)
        distances = []
        for split in splits:
            client_distribution = np.bincount(
                targets[np.asarray(split)], minlength=n_classes
            ) / len(split)
            distances.append(0.5 * np.abs(client_distribution - global_distribution).sum())
        return float(np.mean(distances))

    def test_same_seed_is_deterministic(self):
        indices, targets = self.synthetic_data()
        first = make_dirichlet_client_splits(indices, targets, 5, 0.5, 42)
        second = make_dirichlet_client_splits(indices, targets, 5, 0.5, 42)
        self.assertEqual(first, second)

    def test_different_seeds_produce_different_splits(self):
        indices, targets = self.synthetic_data()
        first = make_dirichlet_client_splits(indices, targets, 5, 0.5, 41)
        second = make_dirichlet_client_splits(indices, targets, 5, 0.5, 42)
        self.assertNotEqual(first, second)

    def test_assignment_is_complete_and_non_overlapping(self):
        indices, targets = self.synthetic_data()
        splits = make_dirichlet_client_splits(indices, targets, 5, 0.1, 42)
        assigned = np.concatenate([np.asarray(split) for split in splits])
        self.assertEqual(len(assigned), len(indices))
        self.assertEqual(len(np.unique(assigned)), len(indices))
        np.testing.assert_array_equal(np.sort(assigned), indices)

    def test_lower_alpha_has_stronger_class_heterogeneity(self):
        indices, targets = self.synthetic_data(samples_per_class=500)
        low_alpha = make_dirichlet_client_splits(indices, targets, 5, 0.05, 42)
        high_alpha = make_dirichlet_client_splits(indices, targets, 5, 10.0, 42)
        self.assertGreater(
            self.mean_client_total_variation(low_alpha, targets),
            self.mean_client_total_variation(high_alpha, targets),
        )


if __name__ == "__main__":
    unittest.main()
