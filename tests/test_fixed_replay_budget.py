import unittest
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl import Client
from src.strategies.replay import ReplayBuffer


class FixedReplayBudgetTests(unittest.TestCase):
    def _client(self, replay_items=0):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=6, shuffle=False)
        model = nn.Linear(4, 2)
        replay = ReplayBuffer(capacity=20, seed=7)
        if replay_items:
            replay.add_batch(x[:replay_items], y[:replay_items])
        return Client(
            0,
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            loader,
            replay=replay,
        )

    def test_fixed_budget_reallocates_without_growing_total_batch(self):
        client = self._client(replay_items=6)
        client.train_one_epoch(replay_ratio=0.5, fixed_batch_budget=True)
        self.assertEqual(client._last_current_presentations, 3)
        self.assertEqual(client._last_replay_presentations, 3)
        self.assertEqual(
            client._last_current_presentations + client._last_replay_presentations,
            6,
        )

    def test_empty_buffer_uses_the_complete_current_batch(self):
        client = self._client(replay_items=0)
        client.train_one_epoch(replay_ratio=0.5, fixed_batch_budget=True)
        self.assertEqual(client._last_current_presentations, 6)
        self.assertEqual(client._last_replay_presentations, 0)

    def test_fixed_budget_training_does_not_reinsert_current_or_replay_items(self):
        client = self._client(replay_items=4)
        before = list(client.replay.data)
        client.train_one_epoch(replay_ratio=0.5, fixed_batch_budget=True)
        after = list(client.replay.data)
        self.assertEqual(len(after), len(before))
        for (before_x, before_y), (after_x, after_y) in zip(before, after):
            self.assertTrue(torch.equal(before_x, after_x))
            self.assertTrue(torch.equal(before_y, after_y))

    def test_replay_rng_round_trips_through_checkpoint_state(self):
        buffer = ReplayBuffer(capacity=20, seed=17)
        x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        y = torch.arange(10)
        buffer.add_batch(x, y)
        buffer.sample_count(3, device=torch.device("cpu"))
        state = buffer.state_dict()
        expected_x, expected_y = buffer.sample_count(4, device=torch.device("cpu"))

        restored = ReplayBuffer(capacity=1, seed=999)
        restored.load_state_dict(state)
        actual_x, actual_y = restored.sample_count(4, device=torch.device("cpu"))
        self.assertTrue(torch.equal(expected_x, actual_x))
        self.assertTrue(torch.equal(expected_y, actual_y))

    def test_replay_ratio_changes_allocation_not_steps_or_presentations(self):
        low = self._client(replay_items=6)
        high = self._client(replay_items=6)
        low.train_one_epoch(replay_ratio=0.3, fixed_batch_budget=True)
        high.train_one_epoch(replay_ratio=0.7, fixed_batch_budget=True)
        self.assertEqual(low._optimizer_step, high._optimizer_step)
        low_total = low._last_current_presentations + low._last_replay_presentations
        high_total = high._last_current_presentations + high._last_replay_presentations
        self.assertEqual(low_total, high_total)
        self.assertEqual(low_total, 6)

    def test_unseeded_buffer_preserves_legacy_global_random_stream(self):
        buffer = ReplayBuffer(capacity=20)
        x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        y = torch.arange(10)
        buffer.add_batch(x, y)
        self.assertNotIn("rng_state", buffer.state_dict())
        random.seed(123)
        expected_indices = random.sample(list(range(10)), 4)
        random.seed(123)
        actual_x, actual_y = buffer.sample_count(4, device=torch.device("cpu"))
        self.assertEqual(actual_y.tolist(), expected_indices)
        self.assertEqual(actual_x[:, 0].tolist(), [float(i * 4) for i in expected_indices])


if __name__ == "__main__":
    unittest.main()
