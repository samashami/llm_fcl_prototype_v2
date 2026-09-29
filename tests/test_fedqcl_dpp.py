import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl import Client
from src.policy.fedqcl_dpp import (
    FedQCLMemory,
    FedQCLState,
    fedqcl_objective,
    update_queue,
)


class FedQCLQueueTests(unittest.TestCase):
    def test_positive_violation_increments_and_negative_violation_clamps(self):
        after, violation = update_queue(0.4, 1.8, 1.0, 0.2)
        self.assertAlmostEqual(violation, 0.6)
        self.assertAlmostEqual(after, 1.0)
        after, violation = update_queue(0.1, 0.7, 1.0, 0.2)
        self.assertAlmostEqual(violation, -0.5)
        self.assertEqual(after, 0.0)

    def test_stage_reference_is_fixed_until_a_new_stage(self):
        state = FedQCLState()
        self.assertTrue(state.begin_stage(2, {0: 1.25}, "stage2-hash"))
        self.assertFalse(state.begin_stage(2, {0: 99.0}, "changed-hash"))
        self.assertEqual(state.reference_losses, {0: 1.25})
        self.assertEqual(state.reference_model_sha256, "stage2-hash")
        self.assertTrue(state.begin_stage(3, {0: 0.75, 2: 1.1}, "stage3-hash"))
        self.assertEqual(state.queues, {0: 0.0, 2: 0.0})

    def test_queue_weighted_replay_loss_changes_gradient(self):
        replay_logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
        current_logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
        replay_loss = nn.functional.cross_entropy(
            replay_logits, torch.tensor([1])
        )
        current_loss = nn.functional.cross_entropy(
            current_logits, torch.tensor([0])
        )
        objective = fedqcl_objective(
            current_loss, {4: replay_loss}, {4: 0.0}, {4: 0.5}, penalty_weight=1.0
        )
        grad_without_queue = torch.autograd.grad(
            objective, replay_logits, allow_unused=True
        )[0]
        self.assertIsNone(grad_without_queue)

        replay_logits2 = replay_logits.detach().clone().requires_grad_(True)
        replay_loss2 = nn.functional.cross_entropy(
            replay_logits2, torch.tensor([1])
        )
        objective2 = fedqcl_objective(
            current_loss.detach(),
            {4: replay_loss2},
            {4: 2.0},
            {4: 0.5},
            penalty_weight=1.0,
        )
        grad_with_queue = torch.autograd.grad(objective2, replay_logits2)[0]
        self.assertGreater(float(torch.norm(grad_with_queue)), 0.0)

    def test_client_optimizer_update_changes_when_queue_activates(self):
        torch.manual_seed(0)
        memory = FedQCLMemory(capacity=10, seed=1)
        memory.add_domain_batch(5, torch.tensor([[0.0, 1.0]]), torch.tensor([1]))
        current_x = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        current_y = torch.tensor([0, 0])
        loader = DataLoader(TensorDataset(current_x, current_y), batch_size=2)

        initial = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            initial.weight.zero_()
        base = nn.Linear(2, 2, bias=False)
        base.load_state_dict(initial.state_dict())
        treated = nn.Linear(2, 2, bias=False)
        treated.load_state_dict(initial.state_dict())
        base_client = Client(
            0, base, torch.optim.SGD(base.parameters(), lr=0.1), loader
        )
        treated_client = Client(
            0, treated, torch.optim.SGD(treated.parameters(), lr=0.1), loader
        )
        shared = dict(
            replay_ratio=0.5,
            fixed_batch_budget=True,
            fedqcl_memory=memory,
            fedqcl_reference_losses={5: 0.0},
            fedqcl_v=1.0,
        )
        base_client.train_one_epoch(fedqcl_queues={5: 0.0}, **shared)
        treated_client.train_one_epoch(fedqcl_queues={5: 10.0}, **shared)
        self.assertEqual(base_client._optimizer_step, treated_client._optimizer_step)
        self.assertEqual(base_client._last_current_presentations, 1)
        self.assertEqual(base_client._last_replay_presentations, 1)
        self.assertFalse(
            torch.equal(base.weight.detach(), treated.weight.detach())
        )
        self.assertGreater(abs(treated_client._last_fedqcl_penalty_loss), 0.0)

    def test_domain_memory_is_bounded_and_samples_keep_domain_identity(self):
        memory = FedQCLMemory(capacity=6, seed=12)
        for group_id in range(3):
            x = torch.arange(group_id * 10, group_id * 10 + 10).float().view(-1, 1)
            y = torch.full((10,), group_id, dtype=torch.long)
            memory.add_domain_batch(group_id, x, y)
        self.assertLessEqual(len(memory), 6)
        self.assertEqual(memory.group_ids, (0, 1, 2))
        sampled = memory.sample_by_group(6, "cpu")
        self.assertEqual(set(sampled), {0, 1, 2})
        for group_id, (_, labels) in sampled.items():
            self.assertTrue(torch.all(labels == group_id))


if __name__ == "__main__":
    unittest.main()
