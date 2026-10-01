import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl import Client

from src.policy.bi_memory import BIMemory, IMAGENET_MEAN, IMAGENET_STD


class TinyImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(3, 3)

    def forward(self, x):
        return self.head(self.pool(x).flatten(1))


class BIMemoryTests(unittest.TestCase):
    def images(self, n):
        raw = torch.rand(n, 3, 32, 32)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (raw - mean) / std

    def test_client_preserves_an_empty_bi_memory_instance(self):
        model = TinyImageClassifier()
        memory = BIMemory(capacity=4, seed=3)
        loader = DataLoader(
            TensorDataset(self.images(2), torch.tensor([0, 1])),
            batch_size=2,
        )
        client = Client(
            0,
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            loader,
            device=torch.device("cpu"),
            replay=memory,
        )
        self.assertIs(client.replay, memory)
        self.assertTrue(hasattr(client.replay, "add_domain_batch"))

    def test_bottomk_update_is_capacity_bounded_and_class_balanced(self):
        memory = BIMemory(capacity=4, seed=3, score_batch_size=4)
        x, y = self.images(8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        result = memory.add_domain_batch(x, y, stage_id=0, model=TinyImageClassifier())
        self.assertEqual(result["scored_candidates"], 8)
        self.assertEqual(len(memory), 4)
        self.assertEqual(memory.labels.count(0), 2)
        self.assertEqual(memory.labels.count(1), 2)
        self.assertEqual(set(memory.stage_ids), {0})

    def test_incoming_domain_is_offered_once_and_excluded_from_replay(self):
        memory = BIMemory(capacity=20, seed=2)
        x0, y0 = self.images(4), torch.tensor([0, 0, 1, 1])
        memory.add_domain_batch(x0, y0, stage_id=0, model=TinyImageClassifier())
        x1, y1 = self.images(3), torch.tensor([0, 1, 1])
        result = memory.add_domain_batch(x1, y1, stage_id=1, model=TinyImageClassifier())
        self.assertEqual(result["incoming"], 3)
        self.assertEqual(memory.samples_offered, 7)
        rx, ry = memory.sample_count_excluding_stage(20, "cpu", exclude_stage=1)
        self.assertEqual(len(rx), 4)
        self.assertTrue(torch.equal(ry.sort().values, y0.sort().values))
        with self.assertRaisesRegex(ValueError, "already admitted"):
            memory.add_domain_batch(x1, y1, stage_id=1, model=TinyImageClassifier())

    def test_state_round_trip_preserves_stage_provenance_and_capacity(self):
        memory = BIMemory(capacity=10, seed=12)
        memory.add_domain_batch(self.images(3), torch.tensor([1, 2, 1]), 0,
                                TinyImageClassifier())
        state = memory.state_dict()
        restored = BIMemory(capacity=10, seed=99)
        restored.load_state_dict(state)
        self.assertEqual(restored.stage_ids, (0,))
        self.assertEqual(restored.labels, memory.labels)
        self.assertEqual(len(restored), 3)


if __name__ == "__main__":
    unittest.main()
