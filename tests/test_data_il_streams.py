import unittest

import numpy as np
import torch

from src.data_il_streams import (
    CONTROLLED_DOMAIN_TRANSFORMS,
    StageDomainDataset,
    cl_batch_sizes,
    make_controlled_domain_shift_batches,
    make_random_chunks,
)


class _TinyCifar:
    def __init__(self):
        self.data = np.full((3, 32, 32, 3), 127, dtype=np.uint8)
        self.targets = [0, 1, 2]


def legacy_make_cl_batches(indices, num_batches=7, seed=42):
    rng_local = np.random.RandomState(seed)
    idx = np.array(indices, dtype=np.int64)
    rng_local.shuffle(idx)
    init = int(round(0.466 * len(idx)))
    init = max(1, min(len(idx) - (num_batches - 1), init))
    first = idx[:init]
    rem = idx[init:]
    if num_batches <= 1:
        return [idx.tolist()]
    per = len(rem) // (num_batches - 1)
    chunks = [rem[i * per:(i + 1) * per] for i in range(num_batches - 2)]
    chunks.append(rem[(num_batches - 2) * per:])
    return [first.tolist()] + [chunk.tolist() for chunk in chunks]


class ControlledDataILStreamTests(unittest.TestCase):
    def setUp(self):
        self.indices = list(range(1400))
        self.targets = [index % 100 for index in self.indices]

    def test_assignment_is_deterministic_and_complete_without_overlap(self):
        first = make_controlled_domain_shift_batches(self.indices, self.targets, seed=42)
        second = make_controlled_domain_shift_batches(self.indices, self.targets, seed=42)
        self.assertEqual(first, second)
        flattened = [item for batch in first for item in batch]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(self.indices))

    def test_stage_sizes_follow_existing_pattern(self):
        batches = make_controlled_domain_shift_batches(self.indices, self.targets, seed=43)
        sizes = [len(batch) for batch in batches]
        self.assertEqual(sizes, cl_batch_sizes(len(self.indices)))
        self.assertEqual(sizes[0], round(0.466 * len(self.indices)))
        self.assertLessEqual(max(sizes[1:]) - min(sizes[1:]), 5)

    def test_every_class_is_represented_when_each_class_has_seven_examples(self):
        batches = make_controlled_domain_shift_batches(self.indices, self.targets, seed=44)
        for batch in batches:
            self.assertEqual({self.targets[index] for index in batch}, set(range(100)))

    def test_evaluation_domain_transforms_are_deterministic(self):
        dataset = _TinyCifar()
        for stage in range(len(CONTROLLED_DOMAIN_TRANSFORMS)):
            view = StageDomainDataset(dataset, [0], stage=stage, experiment_seed=42, training=False)
            first, first_label = view[0]
            second, second_label = view[0]
            self.assertEqual(first_label, second_label)
            self.assertTrue(torch.equal(first, second), CONTROLLED_DOMAIN_TRANSFORMS[stage]["name"])

    def test_random_chunks_remains_the_legacy_protocol(self):
        indices = list(range(127))
        self.assertEqual(make_random_chunks(indices, num_batches=7, seed=42), legacy_make_cl_batches(indices, num_batches=7, seed=42))


if __name__ == "__main__":
    unittest.main()
